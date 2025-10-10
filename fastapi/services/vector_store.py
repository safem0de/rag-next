from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import torch
from dotenv import load_dotenv
import os
import uuid
import logging
from functools import lru_cache
from tiktoken import get_encoding

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

# load reranker
reranker_model_name = "BAAI/bge-reranker-base" #"BAAI/bge-reranker-large"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(
    reranker_model_name,
    dtype=dtype
).to(device)

st_model = None

def get_or_load_st_model():
    global st_model
    if st_model is None:
        logging.info("🧠 Loading SentenceTransformer model...")
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return st_model

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
    """Embed chunks + store in Qdrant + extract metadata"""
    enc = get_encoding("cl100k_base")
    safe_chunks = []
    max_token_limit = 8000  # สำหรับ text-embedding-3-small/large
    max_total_tokens = 250000

    # --- 1. ตรวจสอบและ split text ยาวเกิน limit ---
    for i, c in enumerate(chunks):
        text = c.page_content.strip()
        tokens = len(enc.encode(text))

        if tokens > max_token_limit:
            logging.warning(f"⚠️ Chunk {i} ยาวเกิน limit ({tokens} tokens) → แบ่งย่อย")
            # แบ่ง text ย่อยเป็นส่วนละประมาณ 7000 tokens
            words = text.split()
            sub_chunk = ""
            sub_tokens = 0
            for w in words:
                t_len = len(enc.encode(w))
                if sub_tokens + t_len > 7000:
                    safe_chunks.append(c.__class__(page_content=sub_chunk.strip()))
                    sub_chunk = ""
                    sub_tokens = 0
                sub_chunk += " " + w
                sub_tokens += t_len
            if sub_chunk.strip():
                safe_chunks.append(c.__class__(page_content=sub_chunk.strip()))
        else:
            safe_chunks.append(c)

    logging.info(f"✅ รวมทั้งหมด {len(safe_chunks)} chunks หลังจากตรวจ limit")

    # --- 2. embed ---
    vectors = []
    if USE_OPENAI:
        logging.info("🚀 เริ่มสร้าง embeddings ด้วย OpenAI แบบ batch...")

        current_batch = []
        current_tokens = 0

        for i, c in enumerate(safe_chunks):
            text = c.page_content.strip()
            tokens = len(enc.encode(text))

            # ถ้าเกิน limit รวม → ส่ง batch นี้ก่อน
            if current_tokens + tokens > max_total_tokens:
                logging.info(f"📦 ส่ง batch (รวม {len(current_batch)} chunks, {current_tokens} tokens)")
                batch_vectors = embeddings.embed_documents(current_batch)
                vectors.extend(batch_vectors)
                current_batch = []
                current_tokens = 0

            current_batch.append(text)
            current_tokens += tokens

        # ส่ง batch สุดท้าย
        if current_batch:
            logging.info(f"📦 ส่ง batch สุดท้าย (รวม {len(current_batch)} chunks, {current_tokens} tokens)")
            batch_vectors = embeddings.embed_documents(current_batch)
            vectors.extend(batch_vectors)

        logging.info(f"✅ สร้าง embeddings เสร็จทั้งหมด {len(vectors)} vectors")

    else:
        # ถ้าใช้ SentenceTransformer
        st = get_or_load_st_model()
        vectors = st.encode(
            [c.page_content for c in safe_chunks],
            convert_to_numpy=True,
            show_progress_bar=True
        )

    # --- 3. สร้าง structured metadata ---
    schemas = [
        ResponseSchema(name="person", description="ชื่อบุคคลหรือองค์กร"),
        ResponseSchema(name="wealth", description="มูลค่าทรัพย์สินหรือรายได้"),
        ResponseSchema(name="industry", description="ประเภทอุตสาหกรรมหรือธุรกิจ"),
        ResponseSchema(name="rank", description="ลำดับของบุคคล เช่น อันดับ 1, อันดับ 2"),
    ]
    parser = StructuredOutputParser.from_response_schemas(schemas)
    format_instructions = parser.get_format_instructions()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt_template = ChatPromptTemplate.from_template(
        "Extract person, wealth, and industry from the following text.\n"
        "{format_instructions}\n\nText:\n{input_text}"
    )

    auto_metadata = []
    for i, chunk in enumerate(safe_chunks):
        prompt = prompt_template.format_messages(
            input_text=chunk.page_content,
            format_instructions=format_instructions
        )
        try:
            response = llm.invoke(prompt)
            # parsed = parser.parse(response.content)
            # logging.info(f"📄 Parsed metadata chunk {i}: {parsed}")
            raw_output = response.content.strip()
            logging.debug(f"🧩 Raw LLM output chunk {i}: {raw_output}")
            parsed = parser.parse(raw_output)
        except Exception as e:
            logging.warning(f"⚠️ Metadata extraction error ที่ chunk {i}: {e}")
            parsed = {}
        auto_metadata.append(parsed)

    # --- 4. upsert Qdrant ---
    qdrant.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i],
                payload={
                    "text": safe_chunks[i].page_content,
                    **(payloads[i] if payloads and i < len(payloads) else {}),
                    **(auto_metadata[i] if i < len(auto_metadata) else {}),
                }
            )
            for i in range(len(safe_chunks))
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
        vector = get_or_load_st_model().encode(query, convert_to_numpy=True)

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
        vector = get_or_load_st_model().encode(query, convert_to_numpy=True)

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
