from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# connect Qdrant
qdrant = QdrantClient(url="http://localhost:6333")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

def embed_and_store(chunks, collection="pdf_docs"):
    vectors = embeddings.embed_documents([c.page_content for c in chunks])

    # สร้าง collection ถ้ายังไม่มี
    qdrant.recreate_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=len(vectors[0]), distance=models.Distance.COSINE
        ),
    )

    qdrant.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=i,
                vector=vectors[i],
                payload={"text": chunks[i].page_content},
            )
            for i in range(len(chunks))
        ],
    )
    return len(vectors)
