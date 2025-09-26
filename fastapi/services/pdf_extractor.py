import fitz
import os
from pythainlp.tokenize import word_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

IMG_DIR = "uploads/images"
os.makedirs(IMG_DIR, exist_ok=True)

def extract_pdf_content(pdf_path: str):
    doc = fitz.open(pdf_path)
    content_chunks = []

    for page_num, page in enumerate(doc, start=1):
        # ดึงข้อความ
        text = page.get_text("text").strip()
        if text:
            content_chunks.append({
                "type": "text",
                "page": page_num,
                "content": text
            })

        # ดึงรูปภาพ
        for img_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            img_filename = f"page{page_num}_img{img_index}.{ext}"
            img_path = os.path.join(IMG_DIR, img_filename)

            with open(img_path, "wb") as f:
                f.write(image_bytes)

            content_chunks.append({
                "type": "image",
                "page": page_num,
                "content": img_path
            })

    return content_chunks

def preprocess_text(text: str) -> str:
    return " ".join(word_tokenize(text, engine="newmm"))

def chunk_text(texts: list[str]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    documents = [Document(page_content=t) for t in texts]
    return splitter.split_documents(documents)