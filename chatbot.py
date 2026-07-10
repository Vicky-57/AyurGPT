from dotenv import load_dotenv
import os
import time
import sys
import io

# Prevent UnicodeEncodeErrors in Windows console
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template_string, session, Response
from flask_cors import CORS
import uuid
from datetime import datetime
import json
import re

import psycopg2
from psycopg2.extras import RealDictCursor

class ChatDatabase:
    def __init__(self):
        self.host = os.getenv("SUPABASE_DB_HOST")
        self.database = os.getenv("SUPABASE_DB_NAME", "postgres")
        self.user = os.getenv("SUPABASE_DB_USER", "postgres")
        self.password = os.getenv("SUPABASE_DB_PASS")
        self.port = os.getenv("SUPABASE_DB_PORT", "5432")
        
        self.enabled = all([self.host, self.password])
        if self.enabled:
            try:
                conn = self.get_connection()
                conn.autocommit = True
                cur = conn.cursor()
                cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    sources JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history(session_id);")
                cur.close()
                conn.close()
                print("[OK] Supabase Chat History Database initialized successfully")
            except Exception as e:
                print(f"[WARNING] Database initialization failed: {e}. Session saving will be disabled.")
                self.enabled = False
        else:
            print("[INFO] Supabase credentials not found in env. Session saving is disabled.")

    def get_connection(self):
        return psycopg2.connect(
            host=self.host,
            database=self.database,
            user=self.user,
            password=self.password,
            port=self.port,
            connect_timeout=5
        )

    def save_message(self, session_id, role, content, sources=None):
        if not self.enabled:
            return
        try:
            conn = self.get_connection()
            conn.autocommit = True
            cur = conn.cursor()
            sources_json = json.dumps(sources) if sources else None
            cur.execute(
                "INSERT INTO chat_history (session_id, role, content, sources) VALUES (%s, %s, %s, %s);",
                (session_id, role, content, sources_json)
            )
            cur.close()
            conn.close()
        except Exception as e:
            print(f"[ERROR] Failed to save message to database: {e}")

    def get_history(self, session_id):
        if not self.enabled:
            return []
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT role, content, sources, created_at FROM chat_history WHERE session_id = %s ORDER BY created_at ASC;",
                (session_id,)
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()
            history = []
            for r in rows:
                history.append({
                    'role': r['role'],
                    'content': r['content'],
                    'sources': r['sources'] if r['sources'] else []
                })
            return history
        except Exception as e:
            print(f"[ERROR] Failed to fetch message history: {e}")
            return []

class AyurvedaBot:
    def __init__(self, db_path="qdrant_db", collection_name="ashtang_collection"):
        load_dotenv()
        self.db = ChatDatabase()
        # Initialize Sarvam API client
        self.client = OpenAI(
            api_key=os.getenv("SARVAM_API_KEY"),
            base_url="https://api.sarvam.ai/v1"
        )
        self.collection_name = collection_name
        
        # Initialize Qdrant and SentenceTransformer
        try:
            self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            if qdrant_url and qdrant_api_key:
                print(f"Connecting to Qdrant Cloud at '{qdrant_url}'...")
                self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            else:
                print(f"Connecting to local QdrantDB at '{db_path}'...")
                self.qdrant_client = QdrantClient(path=db_path)
                
            info = self.qdrant_client.get_collection(self.collection_name)
            self.db_ready = True
            print(f"[OK] Connected to QdrantDB: {info.points_count} chunks available")
        except Exception as e:
            print(f"[ERROR] QdrantDB connection failed: {e}")
            self.db_ready = False
        
        # Store conversation sessions
        self.conversation_sessions = {}

    def clean_response_text(self, text):
        """Remove unwanted asterisks and formatting markers from response"""
        # Remove ** markers but preserve content
        cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        
        # Remove single * markers
        cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)
        
        # Clean up any remaining formatting issues
        cleaned = re.sub(r'_{2,}', '', cleaned)  # Remove multiple underscores
        cleaned = re.sub(r'#{1,6}\s*', '', cleaned)  # Remove markdown headers
        
        # Ensure proper spacing around punctuation and sentences
        cleaned = re.sub(r'([.!?])([A-Z])', r'\1 \2', cleaned)  # Add space after sentence endings
        cleaned = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned)  # Add space between lowercase and uppercase
        cleaned = re.sub(r'(\w)([0-9])', r'\1 \2', cleaned)  # Add space before numbers
        cleaned = re.sub(r'([0-9])(\w)', r'\1 \2', cleaned)  # Add space after numbers
        cleaned = re.sub(r'(\w)(\[)', r'\1 \2', cleaned)  # Add space before brackets
        cleaned = re.sub(r'(\])(\w)', r'\1 \2', cleaned)  # Add space after brackets
        cleaned = re.sub(r'([,;:])([A-Za-z])', r'\1 \2', cleaned)  # Add space after punctuation
        
        # Clean up excessive spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def format_response_text(self, text):
        """Format response text with proper paragraphs and spacing"""
        # Split into sentences and group them into paragraphs
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        formatted_paragraphs = []
        current_paragraph = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            current_paragraph.append(sentence)
            
            # Start new paragraph every 3-4 sentences or at logical breaks
            if (len(current_paragraph) >= 3 and 
                any(marker in sentence.lower() for marker in 
                    ['however', 'therefore', 'furthermore', 'additionally', 'in summary', 
                     'source', 'according to', 'the concept of', 'in ayurveda']) or
                len(current_paragraph) >= 4):
                
                formatted_paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        # Add remaining sentences
        if current_paragraph:
            formatted_paragraphs.append(' '.join(current_paragraph))
        
        return '\n\n'.join(formatted_paragraphs)

    def get_predefined_response(self, query, session_data):
        """Handle predefined responses for greetings and common queries"""
        query_lower = query.lower().strip()
        
        # Greetings - exact matches
        greetings = ['hi','hii', 'heyy','hello', 'hey', 'namaste', 'namaskar', 'good morning', 'good evening', 'good afternoon']
        if query_lower in greetings:
            is_first_interaction = len(session_data['messages']) == 0
            if is_first_interaction:
                return {
                    'answer': '''🙏 Namaste! Welcome to the Ashtanga Ayurveda Knowledge Assistant.

I'm here to help you explore the sacred teachings of Ashtanga from the classical texts that have been trained in my knowledge base.

📚 You can ask me about:
• Fundamental Ayurvedic principles and concepts
• Specific definitions and explanations from the texts
• Treatment modalities and diagnostic methods
• Medicinal plants and their properties
• Philosophical concepts and spiritual aspects

Examples of questions you can ask:
• "What is Ashtanga in Ayurveda?"
• "Explain the concept of Panchakarma"
• "What are the three doshas?"

What would you like to learn about today?''',
                    'sources': [],
                    'processing_time': 0.1
                }
            else:
                return {
                    'answer': '''🙏 Namaste! How can I assist you further with the Ashtanga teachings?

Feel free to ask me anything about the Ayurvedic concepts from the classical texts.''',
                    'sources': [],
                    'processing_time': 0.1
                }
        
        # Help queries
        help_phrases = ['what can you teach', 'what can you help', 'what do you know', 'what topics', 'help me', 'help']
        if any(phrase in query_lower for phrase in help_phrases):
            return {
                'answer': '''📖 I can help you understand Ashtanga Ayurveda based on the classical texts:

🔹 Core Topics Available:
• Fundamental principles and philosophy
• Diagnostic methods and examination techniques  
• Treatment approaches and therapeutic modalities
• Medicinal substances and their properties
• Lifestyle guidance and preventive measures
• Spiritual aspects of healing

💡 How to ask questions:
• Be specific about what you want to know
• Use Sanskrit terms or English equivalents
• Ask for definitions, explanations, or practical applications

Example questions:
• "Define Ashtanga Ayurveda"
• "What is the significance of Tridosha?"
• "Explain Panchakarma procedures"

Simply type your question and I'll search through the classical texts to provide you with accurate information!''',
                'sources': [],
                'processing_time': 0.1
            }
        
        # Goodbye/thanks
        farewell_phrases = ['bye', 'goodbye', 'thank you', 'thanks', 'dhanyawad']
        if query_lower in farewell_phrases:
            return {
                'answer': '''🙏 Thank you for exploring the wisdom of Ashtanga Ayurveda!

May this ancient knowledge guide you toward holistic health and spiritual well-being.

"स्वस्थस्य स्वास्थ्य रक्षणं आतुरस्य विकार प्रशमनं"  
(Preserve the health of the healthy, cure the diseases of the diseased)

Feel free to return anytime to learn more from the sacred texts! 🕉️''',
                'sources': [],
                'processing_time': 0.1
            }
        
        return None

    def is_ayurveda_related(self, query):
        """Check if query is related to Ayurveda/medical topics"""
        query_lower = query.lower()
        
        ayurveda_keywords = [
            'ayurveda', 'dosha', 'vata', 'pitta', 'kapha', 'agni', 'ama', 'ojas', 'tejas', 'prana',
            'panchakarma', 'rasayana', 'chikitsa', 'nadi', 'pulse', 'prakriti', 'vikriti',
            'dinacharya', 'ritucharya', 'ahara', 'vihara', 'dravya', 'rasa', 'virya', 'prabhava',
            'ashtanga', 'sushruta', 'charaka', 'vagbhata', 'srotas', 'dhatu', 'mala', 'marma',
            'yoga', 'meditation', 'herbs', 'treatment', 'medicine', 'health', 'disease',
            'constitution', 'diagnosis', 'therapy', 'healing', 'wellness', 'balance'
        ]
        
        return any(keyword in query_lower for keyword in ayurveda_keywords)

    def search_knowledge_base(self, query, top_k=5):
        """Search the QdrantDB for relevant content"""
        if not self.db_ready:
            return [], []
        
        try:
            # Generate local query embedding
            query_vector = self.embedding_model.encode([query])[0].tolist()
            
            # Query Qdrant
            results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k
            )
            
            retrieved_docs = []
            retrieved_metadata = []
            
            for hit in results.points:
                payload = hit.payload
                # Extract text and metadata fields
                doc = payload.get("document", "")
                meta = {k: v for k, v in payload.items() if k != "document"}
                
                retrieved_docs.append(doc)
                retrieved_metadata.append(meta)
                
            return retrieved_docs, retrieved_metadata
        except Exception as e:
            print(f"[ERROR] Database search failed: {e}")
            return [], []

    def check_context_relevance(self, query, retrieved_docs):
        """Check if retrieved documents are relevant to the query"""
        if not retrieved_docs:
            return False
        
        # Check if any document has substantial content (more than 50 words)
        relevant_docs = [doc for doc in retrieved_docs if len(doc.split()) > 50]
        
        if not relevant_docs:
            return False
        
        # Simple relevance check - you can enhance this with more sophisticated methods
        query_words = set(query.lower().split())
        ayurveda_terms = {'ayurveda', 'dosha', 'vata', 'pitta', 'kapha', 'treatment', 'medicine', 'health'}
        
        # If query contains Ayurveda terms, be more lenient
        if query_words.intersection(ayurveda_terms):
            return True
        
        # Check if retrieved content mentions key terms
        combined_docs = ' '.join(relevant_docs).lower()
        doc_words = set(combined_docs.split())
        
        # If there's reasonable overlap between query and documents
        common_words = query_words.intersection(doc_words)
        return len(common_words) >= 2 or any(word in combined_docs for word in query_words if len(word) > 3)

    def ask_question_with_context_streaming(self, query, session_id, top_k=5):
        """Process question with context from knowledge base - streaming version"""
        print(f"[INFO] Processing streaming query for session {session_id}: '{query[:50]}...'")
        start = time.time()

        # Get or create conversation session
        if session_id not in self.conversation_sessions:
            history_messages = self.db.get_history(session_id)
            memory_messages = []
            for msg in history_messages:
                memory_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": datetime.now()
                })
            self.conversation_sessions[session_id] = {
                'messages': memory_messages,
                'created_at': datetime.now()
            }
        
        session_data = self.conversation_sessions[session_id]
        
        # Check for predefined responses first
        predefined_response = self.get_predefined_response(query, session_data)
        if predefined_response:
            # Store in conversation memory
            session_data['messages'].extend([
                {"role": "user", "content": query, "timestamp": datetime.now()},
                {"role": "assistant", "content": predefined_response['answer'], "timestamp": datetime.now()}
            ])
            # Save user & assistant messages to database
            self.db.save_message(session_id, "user", query)
            self.db.save_message(session_id, "assistant", predefined_response['answer'])
            # For predefined responses, return immediately
            yield f"data: {json.dumps(predefined_response)}\n\n"
            return

        # Check if database is ready
        if not self.db_ready:
            error_response = {
                'answer': '''❌ Knowledge base not available.

The Ashtanga texts haven't been loaded yet. Please contact the administrator to train the system with PDF documents.

For now, I can only respond to basic greetings and help queries.''',
                'sources': [],
                'processing_time': round(time.time() - start, 2),
                'complete': True
            }
            # Save to database
            self.db.save_message(session_id, "user", query)
            self.db.save_message(session_id, "assistant", error_response['answer'])
            yield f"data: {json.dumps(error_response)}\n\n"
            return

        # Search knowledge base
        retrieved_docs, retrieved_metadata = self.search_knowledge_base(query, top_k)
        
        # Check if we found relevant content
        if not self.check_context_relevance(query, retrieved_docs):
            # No relevant content found
            not_found_response = {
                'answer': '''🙏 I apologize, but I don't have information about your query in my knowledge base.

I can only provide information based on the Ashtanga Ayurveda texts that have been trained in my system. Your question seems to be outside the scope of my current knowledge base.

I can help you with:
• Ayurvedic principles and concepts
• Treatment methods from classical texts
• Diagnostic approaches in Ayurveda
• Medicinal substances and their properties
• Philosophical aspects of Ayurveda

Please try asking about:
• "What are the three doshas?"
• "Explain Panchakarma treatments"
• "What is Ashtanga Ayurveda?"
• "How is pulse diagnosis done in Ayurveda?"

Would you like to ask something about these topics instead?''',
                'sources': [],
                'processing_time': round(time.time() - start, 2),
                'complete': True
            }
            
            session_data['messages'].extend([
                {"role": "user", "content": query, "timestamp": datetime.now()},
                {"role": "assistant", "content": not_found_response['answer'], "timestamp": datetime.now()}
            ])
            
            # Save to database
            self.db.save_message(session_id, "user", query)
            self.db.save_message(session_id, "assistant", not_found_response['answer'])
            
            yield f"data: {json.dumps(not_found_response)}\n\n"
            return
        
        # Build context with source information
        context_parts = []
        source_info = []
        
        for i, (doc, metadata) in enumerate(zip(retrieved_docs, retrieved_metadata)):
            page_num = metadata.get('page_number', 'Unknown')
            source_file = metadata.get('source_file', 'Unknown')
            context_parts.append(f"[Source {i+1}, Page {page_num}]: {doc}")
            source_info.append(f"Page {page_num}")
        
        retrieved_text = "\n\n".join(context_parts)
        
        # Build conversation history for context
        conversation_history = ""
        if session_data['messages']:
            recent_messages = session_data['messages'][-4:]  # Last 2 exchanges
            for msg in recent_messages:
                conversation_history += f"{msg['role'].title()}: {msg['content'][:200]}...\n"
        
        # Enhanced system prompt with strict context control
        system_prompt = """You are an expert Ayurvedic scholar specializing in the Ashtanga texts. Your role is to provide accurate, detailed answers based on the provided classical text content while also using your knowledge to enhance explanations.

CRITICAL INSTRUCTIONS:
1. Use the provided book content as your PRIMARY source
2. You can supplement with your general Ayurvedic knowledge to provide better explanations, but always prioritize the provided text
3. Be specific and scholarly in your explanations
4. Reference page numbers naturally in your response when making claims from the provided text
5. Use clear, educational language suitable for serious students of Ayurveda
6. Include relevant Sanskrit terms when appropriate (both from source and your knowledge)
7. Structure your response clearly with proper paragraphs and explanations
8. If the provided content has limited information, you can expand using your knowledge but mention what comes from the source vs. general knowledge
9. NEVER use asterisks (**) or markdown formatting in your response
10. Write in plain text with proper sentences, paragraphs, and clear structure
11. Always maintain proper spacing and formatting - write naturally as you would in an academic paper
12. Break your response into logical paragraphs for better readability

RESPONSE FORMAT:
- Start with the main answer combining source content and your knowledge
- Include specific page references for claims from the provided text
- Provide comprehensive explanations that help the student understand the concepts fully
- Use natural paragraph breaks to separate different concepts or ideas
- Do not use any asterisks, bold markers, or markdown formatting
- Write with proper sentence structure and spacing

CONTEXT FORMAT: Each source shows [Source X, Page Y] followed by the relevant text from the classical manuscripts."""

        # Build the prompt
        user_prompt = f"""Based on the following classical Ashtanga text content and your knowledge of Ayurveda, please provide a comprehensive answer:

REFERENCE MATERIAL FROM ASHTANGA TEXTS:
{retrieved_text}

RECENT CONVERSATION CONTEXT:
{conversation_history}

USER'S QUESTION: {query}

Please provide a detailed answer that combines the reference material with your Ayurvedic knowledge to give the most helpful response. Include page references for specific claims from the provided texts. Do not use any asterisks or formatting markers in your response."""

        try:
            # Start streaming response
            response = self.client.chat.completions.create(
                model="sarvam-105b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.3,  # Slightly higher for more natural responses
                stream=True  # Enable streaming
            )
            
            accumulated_answer = ""
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    accumulated_answer += content  # Keep original for storage
                    
                    # Send the raw chunk (don't clean individual chunks to preserve spacing)
                    chunk_data = {
                        'chunk': content,
                        'sources': list(set(source_info)),
                        'complete': False
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Send completion signal
            total_elapsed = time.time() - start
            
            # Clean and format the full accumulated answer
            cleaned_answer = self.clean_response_text(accumulated_answer)
            formatted_answer = self.format_response_text(cleaned_answer)
            
            final_data = {
                'complete': True,
                'processing_time': round(total_elapsed, 2),
                'sources': list(set(source_info)),
                'formatted_text': formatted_answer  # Send the properly formatted version
            }
            
            # Store the formatted answer in conversation memory
            session_data['messages'].extend([
                {"role": "user", "content": query, "timestamp": datetime.now()},
                {"role": "assistant", "content": formatted_answer, "timestamp": datetime.now()}
            ])
            
            # Save user & assistant messages to database
            self.db.save_message(session_id, "user", query)
            self.db.save_message(session_id, "assistant", formatted_answer, list(set(source_info)))
            
            # Keep only last 20 messages
            if len(session_data['messages']) > 20:
                session_data['messages'] = session_data['messages'][-20:]
            
            yield f"data: {json.dumps(final_data)}\n\n"
            print(f"[DONE] Streaming answer generated in {total_elapsed:.2f} seconds")
            
        except Exception as e:
            print(f"[ERROR] Failed to generate streaming response: {e}")
            error_response = {
                'answer': "I apologize, but I encountered an error while processing your question. Please try again.",
                'sources': [],
                'processing_time': round(time.time() - start, 2),
                'complete': True,
                'error': True
            }
            yield f"data: {json.dumps(error_response)}\n\n"

    def ask_question_with_context(self, query, session_id, top_k=5):
        """Non-streaming version for backward compatibility"""
        # This method can still be used for non-streaming requests
        start = time.time()

        if session_id not in self.conversation_sessions:
            history_messages = self.db.get_history(session_id)
            memory_messages = []
            for msg in history_messages:
                memory_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": datetime.now()
                })
            self.conversation_sessions[session_id] = {
                'messages': memory_messages,
                'created_at': datetime.now()
            }
        
        session_data = self.conversation_sessions[session_id]
        
        predefined_response = self.get_predefined_response(query, session_data)
        if predefined_response:
            session_data['messages'].extend([
                {"role": "user", "content": query, "timestamp": datetime.now()},
                {"role": "assistant", "content": predefined_response['answer'], "timestamp": datetime.now()}
            ])
            self.db.save_message(session_id, "user", query)
            self.db.save_message(session_id, "assistant", predefined_response['answer'])
            return predefined_response

        if not self.db_ready:
            answer_text = '''❌ Knowledge base not available.

The Ashtanga texts haven't been loaded yet. Please contact the administrator to train the system with PDF documents.

For now, I can only respond to basic greetings and help queries.'''
            self.db.save_message(session_id, "user", query)
            self.db.save_message(session_id, "assistant", answer_text)
            return {
                'answer': answer_text,
                'sources': [],
                'processing_time': round(time.time() - start, 2)
            }

        retrieved_docs, retrieved_metadata = self.search_knowledge_base(query, top_k)
        
        if not self.check_context_relevance(query, retrieved_docs):
            not_found_response = '''🙏 I apologize, but I don't have information about your query in my knowledge base.

I can only provide information based on the Ashtanga Ayurveda texts that have been trained in my system. Your question seems to be outside the scope of my current knowledge base.

I can help you with:
• Ayurvedic principles and concepts
• Treatment methods from classical texts
• Diagnostic approaches in Ayurveda
• Medicinal substances and their properties
• Philosophical aspects of Ayurveda

Please try asking about:
• "What are the three doshas?"
• "Explain Panchakarma treatments"
• "What is Ashtanga Ayurveda?"
• "How is pulse diagnosis done in Ayurveda?"

Would you like to ask something about these topics instead?'''
            
            session_data['messages'].extend([
                {"role": "user", "content": query, "timestamp": datetime.now()},
                {"role": "assistant", "content": not_found_response, "timestamp": datetime.now()}
            ])
            
            self.db.save_message(session_id, "user", query)
            self.db.save_message(session_id, "assistant", not_found_response)
            
            return {
                'answer': not_found_response,
                'sources': [],
                'processing_time': round(time.time() - start, 2)
            }
        
        # Build context and generate response (similar to streaming but return complete response)
        context_parts = []
        source_info = []
        
        for i, (doc, metadata) in enumerate(zip(retrieved_docs, retrieved_metadata)):
            page_num = metadata.get('page_number', 'Unknown')
            context_parts.append(f"[Source {i+1}, Page {page_num}]: {doc}")
            source_info.append(f"Page {page_num}")
        
        retrieved_text = "\n\n".join(context_parts)
        
        conversation_history = ""
        if session_data['messages']:
            recent_messages = session_data['messages'][-4:]
            for msg in recent_messages:
                conversation_history += f"{msg['role'].title()}: {msg['content'][:200]}...\n"
        
        system_prompt = """You are an expert Ayurvedic scholar specializing in the Ashtanga texts. Your role is to provide accurate, detailed answers based on the provided classical text content while also using your knowledge to enhance explanations.

CRITICAL INSTRUCTIONS:
1. Use the provided book content as your PRIMARY source
2. You can supplement with your general Ayurvedic knowledge to provide better explanations
3. Be specific and scholarly in your explanations
4. Reference page numbers naturally when making claims from the provided text
5. Use clear, educational language suitable for serious students of Ayurveda
6. Include relevant Sanskrit terms when appropriate
7. Structure your response clearly with proper explanations
8. NEVER use asterisks (**) or markdown formatting in your response
9. Write in plain text with proper paragraphs and clear structure"""

        user_prompt = f"""Based on the following classical Ashtanga text content and your knowledge of Ayurveda, please provide a comprehensive answer:

REFERENCE MATERIAL FROM ASHTANGA TEXTS:
{retrieved_text}

RECENT CONVERSATION CONTEXT:
{conversation_history}

USER'S QUESTION: {query}

Please provide a detailed answer that combines the reference material with your Ayurvedic knowledge. Include page references for specific claims from the provided texts. Do not use any asterisks or formatting markers in your response."""

        try:
            response = self.client.chat.completions.create(
                model="sarvam-105b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            cleaned_answer = self.clean_response_text(answer)
            formatted_answer = self.format_response_text(cleaned_answer)
            
            session_data['messages'].extend([
                {"role": "user", "content": query, "timestamp": datetime.now()},
                {"role": "assistant", "content": formatted_answer, "timestamp": datetime.now()}
            ])
            
            self.db.save_message(session_id, "user", query)
            self.db.save_message(session_id, "assistant", formatted_answer, list(set(source_info)))
            
            if len(session_data['messages']) > 20:
                session_data['messages'] = session_data['messages'][-20:]
            
            total_elapsed = time.time() - start
            print(f"[DONE] Answer generated in {total_elapsed:.2f} seconds")
            
            return {
                'answer': formatted_answer,
                'sources': list(set(source_info)),
                'processing_time': round(total_elapsed, 2)
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to generate response: {e}")
            return {
                'answer': "I apologize, but I encountered an error while processing your question. Please try again.",
                'sources': [],
                'processing_time': round(time.time() - start, 2)
            }

# Flask App Setup
app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Initialize bot
bot = AyurvedaBot()

@app.route('/')
def index():
    # Generate session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    try:
        with open('index.html', 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ashtanga Ayurveda Chatbot</title>
        </head>
        <body>
            <h1>Ashtanga Ayurveda Chatbot</h1>
            <p>Please save the HTML code as index.html in the same directory as this Python file.</p>
            <p>You can use the API endpoints:</p>
            <ul>
                <li>POST /api/ask - Send queries (non-streaming)</li>
                <li>POST /api/ask_stream - Send queries (streaming)</li>
                <li>GET /api/status - Check system status</li>
            </ul>
        </body>
        </html>
        '''

@app.route('/api/ask', methods=['POST'])
def api_ask():
    data = request.get_json()
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    
    try:
        result = bot.ask_question_with_context(query, session_id)
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask_stream', methods=['POST'])
def api_ask_stream():
    data = request.get_json()
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    
    try:
        return Response(
            bot.ask_question_with_context_streaming(query, session_id),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
            }
        )
    except Exception as e:
        print(f"[ERROR] Streaming API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_session', methods=['POST'])
def clear_session():
    session_id = session.get('session_id')
    if session_id in bot.conversation_sessions:
        del bot.conversation_sessions[session_id]
    # Delete from database too
    if bot.db.enabled:
        try:
            conn = bot.db.get_connection()
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("DELETE FROM chat_history WHERE session_id = %s;", (session_id,))
            cur.close()
            conn.close()
        except Exception as e:
            print(f"[ERROR] Failed to clear session from DB: {e}")
    return jsonify({'message': 'Session cleared'})

@app.route('/api/history', methods=['GET'])
def api_history():
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    history = bot.db.get_history(session_id)
    return jsonify({'history': history})

@app.route('/api/status')
def api_status():
    """Check system status"""
    try:
        chunks_count = bot.qdrant_client.get_collection(bot.collection_name).points_count if bot.db_ready else 0
    except Exception:
        chunks_count = 0
    return jsonify({
        'database_ready': bot.db_ready,
        'chunks_available': chunks_count,
        'active_sessions': len(bot.conversation_sessions)
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("[INIT] ASHTANGA AYURVEDA CHATBOT WITH STREAMING")
    print("="*60)
    
    if bot.db_ready:
        try:
            chunks_count = bot.qdrant_client.get_collection(bot.collection_name).points_count
        except Exception:
            chunks_count = 0
        print("[OK] Knowledge base loaded successfully")
        print(f"[DATA] {chunks_count} text chunks available")
        print("[SERVER] Starting web interface with streaming support...")
    else:
        print("[WARNING] Knowledge base not found!")
        print("[INFO] Run: python train_pdf.py <your_pdf_file>")
        print("[SERVER] Starting chatbot anyway (limited functionality)...")
    
    print("="*60)
    print("[URL] Open: http://127.0.0.1:5000")
    print("[URL] Streaming endpoint: /api/ask_stream")
    print("[URL] Non-streaming endpoint: /api/ask")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)