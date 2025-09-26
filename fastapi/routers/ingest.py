from fastapi import APIRouter, UploadFile, File
import os
from services.pdf_extractor import extract_pdf_content

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()

@router.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        # Save PDF
        temp_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Extract text + images
        content_chunks = extract_pdf_content(temp_path)

        return {
            "status": "ingested",
            "file": file.filename,
            "chunks": content_chunks
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
