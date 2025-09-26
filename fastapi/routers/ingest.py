from fastapi import APIRouter, UploadFile, File
import os
import aiofiles
from services.pdf_extractor import extract_pdf_content, preprocess_text, chunk_text
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

        # Extract text + images
        chunks = extract_pdf_content(temp_path)
        
        # Preprocess + Chunk (เฉพาะข้อความ)
        texts = [c for c in chunks if c["type"] == "text"]
        preprocessed = [preprocess_text(t["content"]) for t in texts]
        splitted = chunk_text(preprocessed)
        
        # Embed + Store in Qdrant
        stored = embed_and_store(splitted)

        return {
            "status": "ok",
            "file": file.filename,
            "pages": len(chunks),
            "stored_vectors": stored,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
