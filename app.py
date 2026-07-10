import fitz  # PyMuPDF
import numpy as np
from dotenv import load_dotenv
import os
import time
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Load and extract text from PDF
def extract_text_from_pdf(file_path):
    print(f"[INFO] Extracting text from '{file_path}'...")
    start = time.time()
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    elapsed = time.time() - start
    print(f"[DONE] Text extraction complete in {elapsed:.2f} seconds.")
    return text

# Step 2: Split text into chunks
def chunk_text(text, chunk_size=1000, overlap=200):
    print("[INFO] Splitting text into chunks...")
    start = time.time()
    chunks = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = min(start_idx + chunk_size, len(text))
        chunks.append(text[start_idx:end_idx])
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

# Step 4: Load or insert chunks
def load_or_build_chunks(file_path):
    if collection.count() > 0:
        print("[INFO] Using existing ChromaDB collection.")
        return

    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)

    print("[INFO] Creating embeddings and storing in ChromaDB...")
    start = time.time()
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"chunk_{i}"],
            documents=[chunk],
            metadatas=[{"source": file_path, "chunk_index": i}]
        )
        if (i+1) % 10 == 0:
            print(f"  - Stored {i+1}/{len(chunks)} chunks...")
    elapsed = time.time() - start
    print(f"[DONE] Stored {len(chunks)} chunks in {elapsed:.2f} seconds.")

# Step 5: Ask questions
def ask_question(query, top_k=5):
    print(f"[INFO] Searching ChromaDB for: '{query}'")
    start = time.time()

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    retrieved_docs = results["documents"][0]
    retrieved_text = "\n\n".join(retrieved_docs)

    gpt_start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer using only the book content."},
            {"role": "user", "content": f"Use the following content:\n{retrieved_text}\n\nQuestion: {query}"}
        ],
        max_tokens=800
    )
    gpt_elapsed = time.time() - gpt_start
    total_elapsed = time.time() - start
    print(f"[DONE] Answer generated in {total_elapsed:.2f} seconds (GPT time: {gpt_elapsed:.2f}s).")

    # Access content safely
    try:
        return response.choices[0].message.content
    except Exception as e:
        # Debug print so you can inspect unexpected response shapes
        print("[WARN] Couldn't read response. Full response object:")
        print(response)
        print("Error:", e)
        # Fallback: try to coerce to string
        return str(response)

@app.route('/')
def index():
    # Serve the HTML file
    with open('index.html', 'r', encoding='utf-8') as file:
        return file.read()


@app.route('/api/ask', methods=['POST'])
def api_ask():
    data = request.get_json()
    query = data.get('query')
    
    try:
        answer = ask_question(query)  # Your existing function
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    file_path = "Ashtang.pdf"
    load_or_build_chunks(file_path)  # Your existing function
    app.run(debug=True)