from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from dotenv import load_dotenv
import os
import uuid
import logging

logging.basicConfig(level=logging.INFO)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

COLLECTION_NAME = "pdf_docs"
USE_OPENAI = True

# Environment setup
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Connect Qdrant, load embeddings
qdrant = QdrantClient(url="http://localhost:6333")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# load reranker
reranker_model_name = "BAAI/bge-reranker-base" #"BAAI/bge-reranker-large"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(
    reranker_model_name,
    torch_dtype=dtype
).to(device)

# --- collection management ---
def init_collection(vector_size: int, collection_name="pdf_docs"):
    try:
        qdrant.get_collection(collection_name=collection_name)
    except Exception:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            ),
        )

# --- ingest ---
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

    # 🔧 merge payload ปลอดภัย
    qdrant.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i],
                payload={
                    "text": chunks[i].page_content,
                    **(payloads[i] if payloads and i < len(payloads) else {})
                }
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

# --- search (เก่า) ---
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

# --- search (ใหม่) ---
def hybrid_search(query: str, top_k: int = 30, keyword: str = None, collection=COLLECTION_NAME):
    if USE_OPENAI:
        vector = embeddings.embed_query(query)
    else:
        vector = st_model.encode(query, convert_to_numpy=True)

    query_filter = None
    if keyword:
        # 🔧 Qdrant ไม่มี MatchText ต้องใช้ MatchValue
        query_filter = models.Filter(
            must=[models.FieldCondition(key="text", match=models.MatchValue(value=keyword))]
        )

    results = qdrant.search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k,
        query_filter=query_filter
    )

    return results

# --- rerank ---
def rerank_results(query: str, results, top_k: int = 5):
    if not results:
        logging.warning("⚠️ No candidates to rerank")
        return []

    # 🔧 ใช้ .payload ไม่ต้อง check `"payload" in r`
    pairs = []
    for r in results:
        if not r.payload:
            continue
        doc_text = r.payload.get("text") or r.payload.get("raw")
        if doc_text:
            pairs.append((query, doc_text))

    if not pairs:
        logging.warning("⚠️ No valid text in candidates to rerank")
        return []

    inputs = reranker_tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["input_ids"] = inputs["input_ids"].long()

    with torch.no_grad():
        logits = reranker_model(**inputs).logits

    scores = logits.squeeze(-1).tolist()

    reranked = []
    for r, s in zip(results, scores):
        reranked.append({
            "id": r.id,
            "payload": r.payload,
            "vector_score": r.score,
            "rerank_score": float(s)
        })

    reranked = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]

# --- hybrid + rerank ---
def query_hybrid_rerank(query: str, keyword: str = None, top_k: int = 5, collection=COLLECTION_NAME):
    candidates = hybrid_search(query, top_k=30, keyword=keyword, collection=collection)

    if not candidates:
        logging.warning(f"⚠️ No candidates found for query='{query}'")
        return []

    reranked = rerank_results(query, candidates, top_k=top_k)
    return reranked
