from fastapi import APIRouter, UploadFile, File, Depends
import os
import aiofiles
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from services.pdf_extractor import extract_pdf_content, preprocess_text, summarize_text
from services.vector_store import embed_and_store
from services.auth import verify_token

import logging

logging.basicConfig(level=logging.INFO)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter(dependencies=[Depends(verify_token)])


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
        page_chunks = {}
        for t in texts:
            clean_text = preprocess_text(t["content"])
            
            summary = summarize_text(clean_text[:1000]) if len(clean_text) > 500 else ""

            # 3. Chunk ด้วย LangChain splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=150,
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
                    "summary": summary,
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
