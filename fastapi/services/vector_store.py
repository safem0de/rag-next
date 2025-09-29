from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Connect Qdrant
qdrant = QdrantClient(url="http://localhost:6333")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

COLLECTION_NAME = "pdf_docs"


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
    vectors = embeddings.embed_documents([c.page_content for c in chunks])
    init_collection(len(vectors[0]), collection)

    qdrant.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=i,
                vector=vectors[i],
                payload={
                    "text": chunks[i].page_content,
                    **(payloads[i] if payloads else {}),
                },
            )
            for i in range(len(chunks))
        ],
    )
    return len(vectors)

def query_vector_db(query: str, top_k: int = 5):
    """
    ดึงข้อมูลใกล้เคียงจาก Qdrant
    """
    vector = embeddings.embed_query(query)
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
