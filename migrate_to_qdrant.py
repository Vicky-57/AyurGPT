import os
import chromadb
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import uuid

def main():
    load_dotenv()
    
    # 1. Connect to ChromaDB
    chroma_db_path = "chroma_db"
    collection_name = "ashtang_collection"
    
    print(f"Connecting to ChromaDB at '{chroma_db_path}'...")
    if not os.path.exists(chroma_db_path):
        print(f"Error: ChromaDB path '{chroma_db_path}' does not exist.")
        return
        
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)
    try:
        chroma_collection = chroma_client.get_collection(name=collection_name)
        count = chroma_collection.count()
        print(f"Connected to Chroma collection '{collection_name}' with {count} items.")
    except Exception as e:
        print(f"Error getting Chroma collection: {e}")
        return
        
    # 2. Extract documents and metadatas from Chroma
    print("Extracting all data from ChromaDB...")
    chroma_data = chroma_collection.get()
    ids = chroma_data.get("ids", [])
    documents = chroma_data.get("documents", [])
    metadatas = chroma_data.get("metadatas", [])
    
    print(f"Extracted {len(documents)} chunks.")
    if not documents:
        print("No documents found in ChromaDB collection.")
        return

    # 3. Initialize sentence-transformers model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Initializing local embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    vector_size = 384 # Dimension of all-MiniLM-L6-v2 embeddings
    
    # 4. Generate local embeddings
    print("Generating local embeddings for chunks (this runs on CPU)...")
    embeddings = model.encode(documents, show_progress_bar=True, batch_size=32)
    print(f"Generated {len(embeddings)} embeddings.")

    # 5. Initialize Qdrant Client (Cloud or Local storage depending on env vars)
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if qdrant_url and qdrant_api_key:
        print(f"Connecting to Qdrant Cloud at '{qdrant_url}'...")
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        qdrant_path = "qdrant_db"
        print(f"Connecting to local QdrantDB at '{qdrant_path}'...")
        qdrant_client = QdrantClient(path=qdrant_path)
    
    # Create or recreate collection in Qdrant
    qdrant_collection_name = "ashtang_collection"
    
    print(f"Recreating Qdrant collection '{qdrant_collection_name}'...")
    try:
        if qdrant_client.collection_exists(collection_name=qdrant_collection_name):
            print(f"Collection '{qdrant_collection_name}' already exists. Deleting it...")
            qdrant_client.delete_collection(collection_name=qdrant_collection_name)
    except Exception as e:
        print(f"Warning checking collection existence: {e}")

    qdrant_client.create_collection(
        collection_name=qdrant_collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    
    # 6. Upload points to Qdrant
    print("Preparing and uploading points to Qdrant...")
    points = []
    for i in range(len(documents)):
        meta = metadatas[i] if (metadatas and metadatas[i]) else {}
        chroma_id = ids[i]
        
        # Generate a stable UUID based on the Chroma ID (which is a string)
        qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chroma_id))
        
        payload = {
            "document": documents[i],
            "chroma_id": chroma_id
        }
        # Copy all custom metadata fields
        for k, v in meta.items():
            payload[k] = v
            
        vector = embeddings[i].tolist()
        
        points.append(
            PointStruct(
                id=qdrant_id,
                vector=vector,
                payload=payload
            )
        )
        
    # Batch upload points
    batch_size = 100
    for j in range(0, len(points), batch_size):
        chunk_points = points[j:j+batch_size]
        qdrant_client.upsert(
            collection_name=qdrant_collection_name,
            points=chunk_points
        )
        print(f"Uploaded points {j} to {min(j+batch_size, len(points))}")
        
    # Retrieve total points count from Qdrant
    info = qdrant_client.get_collection(qdrant_collection_name)
    print(f"\nMigration completed successfully! Total points in Qdrant collection: {info.points_count}")

if __name__ == "__main__":
    main()
