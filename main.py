import fitz  # PyMuPDF
import numpy as np
from dotenv import load_dotenv
import os
import time
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from flask import Flask, request, jsonify, session, Response
from flask_cors import CORS
import uuid
from datetime import datetime
import json

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Store conversation sessions in memory (use Redis/database for production)
conversation_sessions = {}

# Step 1: Enhanced PDF text extraction with page tracking
def extract_text_from_pdf(file_path):
    print(f"[INFO] Extracting text from '{file_path}'...")
    start = time.time()
    doc = fitz.open(file_path)
    pages_text = []
    
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        pages_text.append({
            'page_number': page_num + 1,
            'text': page_text
        })
    
    elapsed = time.time() - start
    print(f"[DONE] Text extraction complete in {elapsed:.2f} seconds.")
    return pages_text

# Step 2: Enhanced chunking with page references
def chunk_text_with_pages(pages_text, chunk_size=1000, overlap=200):
    print("[INFO] Splitting text into chunks with page tracking...")
    start = time.time()
    chunks = []
    chunk_id = 0
    
    for page_data in pages_text:
        page_num = page_data['page_number']
        text = page_data['text']
        
        start_idx = 0
        while start_idx < len(text):
            end_idx = min(start_idx + chunk_size, len(text))
            chunk_text = text[start_idx:end_idx]
            
            chunks.append({
                'id': f"chunk_{chunk_id}",
                'text': chunk_text,
                'page_number': page_num,
                'start_idx': start_idx,
                'end_idx': end_idx
            })
            
            chunk_id += 1
            start_idx += chunk_size - overlap
    
    elapsed = time.time() - start
    print(f"[DONE] Created {len(chunks)} chunks in {elapsed:.2f} seconds.")
    return chunks

# Step 3: Chroma client (persistent)
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(
    name="ashtang_collection",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-ada-002"
    )
)

# Step 4: Enhanced loading with page metadata
def load_or_build_chunks(file_path):
    if collection.count() > 0:
        print("[INFO] Using existing ChromaDB collection.")
        return

    pages_text = extract_text_from_pdf(file_path)
    chunks = chunk_text_with_pages(pages_text)

    print("[INFO] Creating embeddings and storing in ChromaDB...")
    start = time.time()
    
    for chunk in chunks:
        collection.add(
            ids=[chunk['id']],
            documents=[chunk['text']],
            metadatas=[{
                "source": file_path,
                "page_number": chunk['page_number'],
                "start_idx": chunk['start_idx'],
                "end_idx": chunk['end_idx']
            }]
        )
        
        if len([c for c in chunks if c['id'] <= chunk['id']]) % 10 == 0:
            processed = len([c for c in chunks if c['id'] <= chunk['id']])
            print(f"  - Stored {processed}/{len(chunks)} chunks...")
    
    elapsed = time.time() - start
    print(f"[DONE] Stored {len(chunks)} chunks in {elapsed:.2f} seconds.")

# Predefined responses for common queries
def get_predefined_response(query, session_data):
    query_lower = query.lower().strip()
    
    # Exact match greetings - be more specific
    if query_lower in ['hi', 'hello', 'hey', 'namaste', 'namaskar', 'good morning', 'good evening', 'good afternoon']:
        is_first_interaction = len(session_data['messages']) == 0
        if is_first_interaction:
            return {
                'answer': '''🙏 Namaste! Welcome to the Ashtanga Ayurveda Knowledge Assistant. 

I'm here to help you explore the sacred teachings of Ashtanga - the eight-limbed path of Ayurvedic wisdom. You can ask me about:

📚 **Core Topics I can help with:**
• **Fundamental Principles** - Tridosha, Panchamahabhuta, Prakriti-Vikriti
• **Diagnostic Methods** - Nadi Pariksha, Darshana, Sparshana, Prashna
• **Treatment Modalities** - Panchakarma, Rasayana, Satvavajaya Chikitsa
• **Medicinal Plants** - Properties, uses, and preparations
• **Lifestyle Guidelines** - Dinacharya, Ritucharya, Ahara-Vihara
• **Philosophical Concepts** - Consciousness, mind-body connection, spiritual health

Feel free to ask specific questions like "What is Panchakarma?" or "Explain the concept of Agni in Ayurveda"

What would you like to learn about today?''',
                'sources': [],
                'processing_time': 0.1
            }
        else:
            return {
                'answer': '''🙏 Namaste! How can I assist you further with the Ashtanga teachings? 

Feel free to ask me anything about Ayurvedic principles, treatments, or philosophical concepts.''',
                'sources': [],
                'processing_time': 0.1
            }
    
    # Help queries - exact matches
    if any(phrase in query_lower for phrase in ['what can you teach', 'what can you help', 'what do you know', 'what topics', 'what can i learn', 'help me']):
        return {
            'answer': '''📖 **I can guide you through the comprehensive teachings of Ashtanga Ayurveda:**

**🔹 Foundational Concepts:**
• Tridosha Theory (Vata, Pitta, Kapha)
• Panchamahabhuta (Five Elements)
• Prakriti and Vikriti analysis
• Concept of Agni (digestive fire)
• Ojas, Tejas, and Prana

**🔹 Diagnostic Approaches:**
• Ashta Vidha Pariksha (8-fold examination)
• Nadi Pariksha (pulse diagnosis)
• Constitutional assessment methods

**🔹 Treatment Modalities:**
• Panchakarma procedures and benefits
• Rasayana (rejuvenation therapies)
• Satvavajaya Chikitsa (psychotherapy)
• Medicinal plant applications

**🔹 Lifestyle Wisdom:**
• Daily routines (Dinacharya)
• Seasonal adaptations (Ritucharya)
• Dietary guidelines and food combinations
• Yoga and meditation practices

**🔹 Advanced Topics:**
• Spiritual dimensions of healing
• Mind-body integration principles
• Preventive healthcare approaches

Simply ask me about any specific topic, and I'll provide detailed explanations with references to the classical text!''',
            'sources': [],
            'processing_time': 0.1
        }
    
    # Goodbye/thanks - exact matches
    if query_lower in ['bye', 'goodbye', 'thank you', 'thanks', 'dhanyawad', 'thank you so much']:
        return {
            'answer': '''🙏 Thank you for exploring the wisdom of Ashtanga Ayurveda with me. 

May this knowledge serve your journey toward holistic health and spiritual well-being. 

*"Swasthyasya swasthya rakshanam, aturasya vikara prashamanam"* 
(Preserve the health of the healthy, cure the diseases of the diseased)

Feel free to return anytime for more guidance from the sacred texts! 🕉️''',
            'sources': [],
            'processing_time': 0.1
        }
    
    return None

# Check if query is related to Ayurveda/medical topics
def is_ayurveda_related(query):
    query_lower = query.lower()
    
    # Ayurvedic keywords
    ayurveda_keywords = [
        'ayurveda', 'dosha', 'vata', 'pitta', 'kapha', 'agni', 'ama', 'ojas', 'tejas', 'prana',
        'panchakarma', 'rasayana', 'chikitsa', 'nadi', 'pulse', 'prakriti', 'vikriti',
        'dinacharya', 'ritucharya', 'ahara', 'vihara', 'dravya', 'rasa', 'virya', 'prabhava',
        'ashtanga', 'sushruta', 'charaka', 'vagbhata', 'srotas', 'dhatu', 'mala', 'marma',
        'yoga', 'meditation', 'herbs', 'treatment', 'medicine', 'health', 'disease',
        'constitution', 'diagnosis', 'therapy', 'healing', 'wellness', 'balance'
    ]
    
    # Check if any Ayurvedic keyword is present
    if any(keyword in query_lower for keyword in ayurveda_keywords):
        return True
    
    # Check if it's a very short/generic query that might not be Ayurveda related
    if len(query.strip().split()) <= 2 and not any(keyword in query_lower for keyword in ayurveda_keywords):
        return False
    
    return True

# Step 5: Enhanced question answering with context and memory
def ask_question_with_context(query, session_id, top_k=5):
    print(f"[INFO] Processing query for session {session_id}: '{query}'")
    start = time.time()

    # Get or create conversation session
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = {
            'messages': [],
            'created_at': datetime.now()
        }
    
    session_data = conversation_sessions[session_id]
    
    # Check for predefined responses first
    predefined_response = get_predefined_response(query, session_data)
    if predefined_response:
        # Still store in conversation memory
        session_data['messages'].extend([
            {"role": "user", "content": query, "timestamp": datetime.now()},
            {"role": "assistant", "content": predefined_response['answer'], "timestamp": datetime.now()}
        ])
        return predefined_response

    # Search ChromaDB for actual content queries
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    retrieved_docs = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]
    
    # Build context with source information
    context_parts = []
    source_info = []
    
    for i, (doc, metadata) in enumerate(zip(retrieved_docs, retrieved_metadata)):
        page_num = metadata.get('page_number', 'Unknown')
        context_parts.append(f"[Source {i+1}, Page {page_num}]: {doc}")
        source_info.append(f"Page {page_num}")
    
    retrieved_text = "\n\n".join(context_parts)
    
    # Build conversation history for context
    conversation_history = ""
    if session_data['messages']:
        recent_messages = session_data['messages'][-6:]  # Last 3 exchanges
        for msg in recent_messages:
            conversation_history += f"{msg['role'].title()}: {msg['content']}\n"
    
    # Check if retrieved content is relevant (basic relevance filtering)
    if not any(len(doc.strip()) > 50 for doc in retrieved_docs):
        return {
            'answer': '''I apologize, but I couldn't find relevant information in the Ashtanga text for your specific query. 

Could you please:
• Rephrase your question more specifically
• Ask about core Ayurvedic concepts like doshas, treatments, or diagnostic methods
• Use Sanskrit or English terms related to Ayurveda

For example: "What is Panchakarma?" or "Explain Vata dosha characteristics"

Type "what can you help with" to see all available topics.''',
            'sources': [],
            'processing_time': round(time.time() - start, 2)
        }

    # Enhanced system prompt
    system_prompt = """You are an expert Ayurvedic knowledge assistant specializing in the Ashtanga text. 

IMPORTANT GUIDELINES:
1. Answer based ONLY on the provided book content
2. Be specific and direct - avoid generic phrases like "according to the text" or "regarding your inquiry"
3. When referencing concepts, mention the specific page numbers in your response
4. If the context contains relevant information, provide detailed explanations
5. If asked for clarification, build upon the previous conversation
6. Use clear, educational language suitable for Ayurvedic students
7. Always cite page numbers when making specific claims
8. If the query seems too basic or unrelated to Ayurveda, politely guide them to ask about Ayurvedic topics

CONTEXT FORMAT: Each source is marked as [Source X, Page Y] followed by the relevant text."""

    # Build the prompt with conversation context
    user_prompt = f"""Previous conversation context:
{conversation_history}

Current reference material from the Ashtanga text:
{retrieved_text}

Current question: {query}

Please provide a comprehensive answer based on the reference material above. Include specific page references where appropriate."""

    gpt_start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=800,
        temperature=0.7
    )
    
    gpt_elapsed = time.time() - gpt_start
    total_elapsed = time.time() - start
    
    try:
        answer = response.choices[0].message.content
        
        # Store in conversation memory
        session_data['messages'].extend([
            {"role": "user", "content": query, "timestamp": datetime.now()},
            {"role": "assistant", "content": answer, "timestamp": datetime.now()}
        ])
        
        # Keep only last 20 messages to manage memory
        if len(session_data['messages']) > 20:
            session_data['messages'] = session_data['messages'][-20:]
        
        print(f"[DONE] Answer generated in {total_elapsed:.2f} seconds (GPT time: {gpt_elapsed:.2f}s).")
        
        # Return structured response
        return {
            'answer': answer,
            'sources': list(set(source_info)),  # Unique page numbers
            'processing_time': round(total_elapsed, 2)
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to process response: {e}")
        print("Full response object:", response)
        return {
            'answer': "I apologize, but I encountered an error processing your question.",
            'sources': [],
            'processing_time': round(total_elapsed, 2)
        }

# Streaming version for real-time responses
def ask_question_streaming(query, session_id, top_k=5):
    print(f"[INFO] Processing streaming query for session {session_id}: '{query}'")
    start = time.time()

    # Get or create conversation session
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = {
            'messages': [],
            'created_at': datetime.now()
        }
    
    session_data = conversation_sessions[session_id]
    
    # Check for predefined responses first
    predefined_response = get_predefined_response(query, session_data)
    if predefined_response:
        # Still store in conversation memory
        session_data['messages'].extend([
            {"role": "user", "content": query, "timestamp": datetime.now()},
            {"role": "assistant", "content": predefined_response['answer'], "timestamp": datetime.now()}
        ])
        yield f"data: {json.dumps(predefined_response)}\n\n"
        return

    # Search ChromaDB for actual content queries
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    retrieved_docs = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]
    
    # Build context with source information
    context_parts = []
    source_info = []
    
    for i, (doc, metadata) in enumerate(zip(retrieved_docs, retrieved_metadata)):
        page_num = metadata.get('page_number', 'Unknown')
        context_parts.append(f"[Source {i+1}, Page {page_num}]: {doc}")
        source_info.append(f"Page {page_num}")
    
    retrieved_text = "\n\n".join(context_parts)
    
    # Build conversation history for context
    conversation_history = ""
    if session_data['messages']:
        recent_messages = session_data['messages'][-6:]  # Last 3 exchanges
        for msg in recent_messages:
            conversation_history += f"{msg['role'].title()}: {msg['content']}\n"
    
    # Check if retrieved content is relevant
    if not any(len(doc.strip()) > 50 for doc in retrieved_docs):
        error_response = {
            'answer': '''I apologize, but I couldn't find relevant information in the Ashtanga text for your specific query. 

Could you please:
• Rephrase your question more specifically
• Ask about core Ayurvedic concepts like doshas, treatments, or diagnostic methods
• Use Sanskrit or English terms related to Ayurveda

For example: "What is Panchakarma?" or "Explain Vata dosha characteristics"

Type "what can you help with" to see all available topics.''',
            'sources': [],
            'processing_time': round(time.time() - start, 2),
            'error': True
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        return

    # Enhanced system prompt
    system_prompt = """You are an expert Ayurvedic knowledge assistant specializing in the Ashtanga text. 

IMPORTANT GUIDELINES:
1. Answer based ONLY on the provided book content
2. Be specific and direct - avoid generic phrases like "according to the text" or "regarding your inquiry"
3. When referencing concepts, mention the specific page numbers in your response
4. If the context contains relevant information, provide detailed explanations
5. If asked for clarification, build upon the previous conversation
6. Use clear, educational language suitable for Ayurvedic students
7. Always cite page numbers when making specific claims
8. If the query seems too basic or unrelated to Ayurveda, politely guide them to ask about Ayurvedic topics

CONTEXT FORMAT: Each source is marked as [Source X, Page Y] followed by the relevant text."""

    # Build the prompt with conversation context
    user_prompt = f"""Previous conversation context:
{conversation_history}

Current reference material from the Ashtanga text:
{retrieved_text}

Current question: {query}

Please provide a comprehensive answer based on the reference material above. Include specific page references where appropriate."""

    try:
        # Create streaming response
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800,
            temperature=0.7,
            stream=True
        )
        
        complete_answer = ""
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                chunk_text = chunk.choices[0].delta.content
                complete_answer += chunk_text
                
                yield f"data: {json.dumps({'chunk': chunk_text})}\n\n"
        
        # Store in conversation memory
        session_data['messages'].extend([
            {"role": "user", "content": query, "timestamp": datetime.now()},
            {"role": "assistant", "content": complete_answer, "timestamp": datetime.now()}
        ])
        
        # Keep only last 20 messages to manage memory
        if len(session_data['messages']) > 20:
            session_data['messages'] = session_data['messages'][-20:]
        
        # Send completion signal
        total_elapsed = time.time() - start
        completion_data = {
            'complete': True,
            'sources': list(set(source_info)),
            'processing_time': round(total_elapsed, 2),
            'formatted_text': complete_answer
        }
        yield f"data: {json.dumps(completion_data)}\n\n"
        
    except Exception as e:
        print(f"[ERROR] Streaming error: {e}")
        error_data = {
            'error': True,
            'answer': "I apologize, but I encountered an error processing your question."
        }
        yield f"data: {json.dumps(error_data)}\n\n"

@app.route('/')
def index():
    # Generate session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    # Serve the external index.html file
    try:
        with open('index.html', 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return jsonify({'error': 'index.html file not found. Please ensure it exists in the same directory as the Flask app.'}), 404

@app.route('/api/ask', methods=['POST'])
def api_ask():
    # Ensure proper content type
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 415
    
    data = request.get_json()
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Get session ID
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    
    try:
        result = ask_question_with_context(query, session_id)
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask_stream', methods=['POST'])
def api_ask_stream():
    # Ensure proper content type
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 415
    
    data = request.get_json()
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Get session ID
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    
    def generate():
        try:
            for chunk in ask_question_streaming(query, session_id):
                yield chunk
        except Exception as e:
            print(f"[ERROR] Streaming API error: {e}")
            error_data = {
                'error': True,
                'answer': f"Server error: {str(e)}"
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type'
        }
    )

@app.route('/api/clear_session', methods=['POST'])
def clear_session():
    session_id = session.get('session_id')
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
    return jsonify({'message': 'Session cleared'})

if __name__ == '__main__':
    file_path = "Ashtang.pdf"
    
    # Only build chunks if PDF exists
    if os.path.exists(file_path):
        load_or_build_chunks(file_path)
    else:
        print(f"[WARNING] PDF file '{file_path}' not found. Please make sure it exists or train new documents using train.py")
    
    print("\n" + "="*50)
    print("🕉️  ASHTANGA AYURVEDA CHATBOT READY")
    print("="*50)
    print("Features:")
    print("✓ Conversation memory")
    print("✓ Page number references") 
    print("✓ Context-aware responses")
    print("✓ Source attribution")
    print("✓ Streaming responses")
    print("="*50 + "\n")
    
    app.run(debug=True)