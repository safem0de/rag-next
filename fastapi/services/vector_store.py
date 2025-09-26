from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

qdrant = QdrantClient(url="http://localhost:6333")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def store_in_qdrant(chunks, collection="pdf_docs"):
    vectors = []
    payloads = []
    for chunk in chunks:
        if chunk["type"] == "text":  # embed เฉพาะ text
            vec = embeddings.embed_query(chunk["content"])
            vectors.append(vec)
            payloads.append({"page": chunk["page"], "text": chunk["content"]})

    qdrant.upsert(
        collection_name=collection,
        points=[
            {"id": i, "vector": vectors[i], "payload": payloads[i]}
            for i in range(len(vectors))
        ]
    )
    return len(vectors)
