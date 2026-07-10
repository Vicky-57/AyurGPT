import fitz  # PyMuPDF
import numpy as np
from dotenv import load_dotenv
import os
import time
import chromadb
from chromadb.utils import embedding_functions
import argparse
from pathlib import Path

class PDFTrainer:
    def __init__(self, db_path="chroma_db", collection_name="ashtang_collection"):
        load_dotenv()
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002"
            )
        )
        
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF with page tracking"""
        print(f"[INFO] Extracting text from '{file_path}'...")
        start = time.time()
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        doc = fitz.open(file_path)
        pages_text = []
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text.strip():  # Only add non-empty pages
                pages_text.append({
                    'page_number': page_num + 1,
                    'text': page_text.strip()
                })
        
        doc.close()
        elapsed = time.time() - start
        print(f"[DONE] Text extraction complete in {elapsed:.2f} seconds. Pages processed: {len(pages_text)}")
        return pages_text

    def chunk_text_with_pages(self, pages_text, chunk_size=1000, overlap=200):
        """Split text into chunks with page references"""
        print("[INFO] Splitting text into chunks with page tracking...")
        start = time.time()
        chunks = []
        chunk_id = 0
        
        for page_data in pages_text:
            page_num = page_data['page_number']
            text = page_data['text']
            
            # Skip very short pages
            if len(text) < 100:
                continue
                
            start_idx = 0
            while start_idx < len(text):
                end_idx = min(start_idx + chunk_size, len(text))
                chunk_text = text[start_idx:end_idx].strip()
                
                # Only add meaningful chunks
                if len(chunk_text) > 50:
                    chunks.append({
                        'id': f"chunk_{chunk_id}",
                        'text': chunk_text,
                        'page_number': page_num,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'source_file': os.path.basename(page_data.get('source_file', 'unknown'))
                    })
                    chunk_id += 1
                
                start_idx += chunk_size - overlap
                
                # Break if remaining text is too small
                if len(text) - start_idx < chunk_size // 2:
                    break
        
        elapsed = time.time() - start
        print(f"[DONE] Created {len(chunks)} meaningful chunks in {elapsed:.2f} seconds.")
        return chunks

    def add_pdf_to_database(self, file_path, clear_existing=False):
        """Add a PDF to the database"""
        if clear_existing:
            print("[INFO] Clearing existing database...")
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name="text-embedding-ada-002"
                )
            )
        
        # Extract and chunk text
        pages_text = self.extract_text_from_pdf(file_path)
        for page in pages_text:
            page['source_file'] = file_path
            
        chunks = self.chunk_text_with_pages(pages_text)
        
        if not chunks:
            print("[WARNING] No meaningful chunks created from PDF")
            return False
        
        # Add to ChromaDB
        print(f"[INFO] Adding {len(chunks)} chunks to ChromaDB...")
        start = time.time()
        
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            ids = [chunk['id'] for chunk in batch]
            documents = [chunk['text'] for chunk in batch]
            metadatas = [{
                "source_file": chunk['source_file'],
                "page_number": chunk['page_number'],
                "start_idx": chunk['start_idx'],
                "end_idx": chunk['end_idx']
            } for chunk in batch]
            
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            print(f"  - Stored batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        
        elapsed = time.time() - start
        print(f"[DONE] Successfully added {len(chunks)} chunks in {elapsed:.2f} seconds.")
        return True

    def get_database_info(self):
        """Get information about the current database"""
        count = self.collection.count()
        print(f"\n📊 Database Information:")
        print(f"Collection: {self.collection_name}")
        print(f"Total chunks: {count}")
        
        if count > 0:
            # Get sample to show source files
            sample = self.collection.get(limit=5, include=['metadatas'])
            sources = set()
            for metadata in sample['metadatas']:
                sources.add(metadata.get('source_file', 'unknown'))
            print(f"Source files: {', '.join(sources)}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Train PDF documents for Ashtanga Ayurveda Chatbot')
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('--clear', action='store_true', help='Clear existing database before adding')
    parser.add_argument('--db-path', default='chroma_db', help='ChromaDB storage path')
    parser.add_argument('--collection', default='ashtang_collection', help='Collection name')
    
    args = parser.parse_args()
    
    # Check if PDF exists
    if not os.path.exists(args.pdf_path):
        print(f"❌ Error: PDF file '{args.pdf_path}' not found!")
        return
    
    # Initialize trainer
    trainer = PDFTrainer(db_path=args.db_path, collection_name=args.collection)
    
    print("="*60)
    print("🕉️  ASHTANGA AYURVEDA PDF TRAINER")
    print("="*60)
    
    # Show current database info
    trainer.get_database_info()
    
    try:
        # Add PDF to database
        success = trainer.add_pdf_to_database(args.pdf_path, clear_existing=args.clear)
        
        if success:
            print("\n✅ PDF training completed successfully!")
            trainer.get_database_info()
            print("🚀 You can now run the chatbot with: python chatbot.py")
        else:
            print("\n❌ PDF training failed!")
            
    except Exception as e:
        print(f"\n❌ Error during training: {e}")

if __name__ == '__main__':
    main()