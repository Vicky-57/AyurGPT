import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    # 1. Initialize local embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Initializing local embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    
    # 2. Connect to QdrantDB
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = "ashtang_collection"
    
    if qdrant_url and qdrant_api_key:
        print(f"Connecting to Qdrant Cloud at '{qdrant_url}'...")
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        qdrant_path = "qdrant_db"
        print(f"Connecting to local QdrantDB at '{qdrant_path}'...")
        qdrant_client = QdrantClient(path=qdrant_path)
    
    # 3. Perform a query
    query = "What is Ashtanga in Ayurveda?"
    print(f"\nQuerying: '{query}'")
    
    try:
        # Check collection exists
        info = qdrant_client.get_collection(collection_name)
        print(f"Collection '{collection_name}' has {info.points_count} items.")
    except Exception as e:
        print(f"Error checking collection: {e}. Has the migration script finished running?")
        return
        
    query_vector = model.encode([query])[0].tolist()
    
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3
    )
    
    print("\n--- RETRIEVED CHUNKS ---")
    retrieved_docs = []
    for i, hit in enumerate(results.points):
        payload = hit.payload
        doc = payload.get("document", "")
        page_num = payload.get("page_number", "Unknown")
        source_file = payload.get("source_file", "Unknown")
        retrieved_docs.append(doc)
        print(f"\nResult {i+1} [Source: {source_file}, Page: {page_num}, Score: {hit.score:.4f}]:")
        print(doc[:300] + "...")
        
    # 4. Integrate Sarvam LLM
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        print("\nError: SARVAM_API_KEY environment variable not set in .env")
        return
        
    print(f"\nInitializing Sarvam API client...")
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.sarvam.ai/v1"
    )
    
    context = "\n\n".join(retrieved_docs)
    
    system_prompt = "You are an expert Ayurvedic scholar. Answer the query based on the context provided."
    user_prompt = f"Context:\n{context}\n\nQuery: {query}\n\nAnswer in clear plain text, referencing page numbers where possible."
    
    print("\nGenerating response from Sarvam 105B (non-streaming)...")
    try:
        response = client.chat.completions.create(
            model="sarvam-105b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        print("\n--- SARVAM 105B RESPONSE ---")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error calling Sarvam API: {e}")

if __name__ == "__main__":
    main()
