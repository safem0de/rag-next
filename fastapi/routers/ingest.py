from fastapi import APIRouter, UploadFile, File
import os
import aiofiles
from services.pdf_extractor import extract_pdf_content

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
        content_chunks = extract_pdf_content(temp_path)

        return {
            "status": "ingested",
            "file": file.filename,
            "chunks": content_chunks
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
