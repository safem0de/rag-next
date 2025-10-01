from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import uuid
import logging

logging.basicConfig(level=logging.INFO)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Connect Qdrant
qdrant = QdrantClient(url="http://localhost:6333")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

COLLECTION_NAME = "pdf_docs"
USE_OPENAI = False

def init_collection(vector_size: int, collection_name="pdf_docs"):
    try:
        # ถ้า collection มีอยู่แล้ว จะไม่ error
        qdrant.get_collection(collection_name=collection_name)
    except Exception:
        # ถ้าไม่มี ให้สร้างใหม่
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            ),
        )

def embed_and_store(chunks, payloads=None, collection="pdf_docs"):
    if USE_OPENAI:
        vectors = embeddings.embed_documents([c.page_content for c in chunks])
    else:
        vectors = st_model.encode(
            [c.page_content for c in chunks],
            convert_to_numpy=True,
            show_progress_bar=True
        )

    init_collection(len(vectors[0]), collection)

    qdrant.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i],
                payload={
                    "text": chunks[i].page_content,
                    **(payloads[i] if payloads else {}),
                },
            )
            for i in range(len(chunks))
        ],
    )
    logging.info(f"📦 Upserted {len(vectors)} points → {collection}")
    return {
        "stored": len(vectors),
        "collection": collection,
        "payload_samples": payloads[:2] if payloads else []
    }

def query_vector_db(query: str, top_k: int = 5):
    if USE_OPENAI:
        vector = embeddings.embed_query(query)
    else:
        vector = st_model.encode(query, convert_to_numpy=True)

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k
    )
    return [
        {
            "id": r.id,
            "score": r.score,
            "payload": r.payload
        }
        for r in results
    ]

