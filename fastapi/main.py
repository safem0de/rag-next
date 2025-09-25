import os
from fastapi import FastAPI, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # ถ้าไม่มี dir ให้สร้าง

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        # สร้าง path ชั่วคราว
        temp_path = os.path.join(UPLOAD_DIR, file.filename)

        # Save PDF ลงดิสก์
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # ใช้ PyPDFLoader โหลด
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        return {"status": "ingested", "pages": len(docs)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
