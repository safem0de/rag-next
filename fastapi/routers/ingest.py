from fastapi import APIRouter, UploadFile, File
import os
import aiofiles
from services.pdf_extractor import (
    extract_pdf_content,
    preprocess_text,
    summarize_text,
    chunk_text,
)
from services.vector_store import embed_and_store

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()

@router.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        temp_path = os.path.join(UPLOAD_DIR, file.filename)

        async with aiofiles.open(temp_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        # 1. Extract PDF
        chunks = extract_pdf_content(temp_path)

        # 2. Preprocess + Summarize เฉพาะข้อความ
        texts = [c for c in chunks if c["type"] == "text"]

        summaries = []
        for t in texts:
            clean_text = preprocess_text(t["content"])
            summary = summarize_text(clean_text)
            summaries.append(summary)

        # 3. Chunk จาก summary
        splitted = chunk_text(summaries)

        # 4. Embed + Store in Qdrant
        stored = embed_and_store(splitted)

        return {
            "status": "ok",
            "file": file.filename,
            "pages": len(chunks),
            "summaries": summaries,
            "stored_vectors": stored,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
