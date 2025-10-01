from fastapi import APIRouter, UploadFile, File
import os
import aiofiles
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from services.pdf_extractor import extract_pdf_content, preprocess_text
from services.vector_store import embed_and_store

import logging

logging.basicConfig(level=logging.INFO)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()


@router.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        temp_path = os.path.join(UPLOAD_DIR, file.filename)
        logging.info(f"📄 Extracted {len(chunks)} chunks from file={file.filename}")

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
        page_chunks = {}
        for t in texts:
            clean_text = preprocess_text(t["content"])

            # 3. Chunk ด้วย LangChain splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len
            )
            splits = splitter.split_text(clean_text)
            page_num = t.get("page", None)
            page_chunks[page_num] = len(splits)
            logging.info(f"🔹 Page {page_num}: {len(splits)} chunks")

            for s in splits:
                documents.append(Document(page_content=s))
                payloads.append({
                    "raw": clean_text,
                    "page": t.get("page", None),
                    "source": file.filename
                })

        # 4. Embed + Store
        result = embed_and_store(documents, payloads=payloads)
        logging.info(f"✅ Stored {result['stored']} vectors into collection={result['collection']}")

        return {
            "status": "ok",
            "file": file.filename,
            "pages": len(chunks),
            "stored_vectors": result["stored"],
            "collection": result["collection"],
            "examples": result["payload_samples"],
        }

    except Exception as e:
        return {"status": "error", "detail": str(e)}
