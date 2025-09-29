from fastapi import APIRouter, UploadFile, File
import os
import aiofiles
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from services.pdf_extractor import extract_pdf_content, preprocess_text
from services.vector_store import embed_and_store

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()


@router.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        temp_path = os.path.join(UPLOAD_DIR, file.filename)

        # save file
        async with aiofiles.open(temp_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        # 1. Extract PDF
        chunks = extract_pdf_content(temp_path)

        # 2. Preprocess เฉพาะข้อความ
        texts = [c for c in chunks if c["type"] == "text"]

        documents = []
        payloads = []

        for t in texts:
            clean_text = preprocess_text(t["content"])

            # 3. Chunk ด้วย LangChain splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len
            )
            splits = splitter.split_text(clean_text)

            for s in splits:
                documents.append(Document(page_content=s))
                payloads.append({
                    "raw": clean_text,
                    "page": t.get("page", None),
                    "source": file.filename
                })

        # 4. Embed + Store
        stored = embed_and_store(documents, payloads=payloads)

        return {
            "status": "ok",
            "file": file.filename,
            "pages": len(chunks),
            "stored_vectors": stored,
            "examples": payloads[:3],
        }

    except Exception as e:
        return {"status": "error", "detail": str(e)}
