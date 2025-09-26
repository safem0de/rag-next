import fitz
import os
from pythainlp.tokenize import word_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

IMG_DIR = "uploads/images"
os.makedirs(IMG_DIR, exist_ok=True)

# โหลด mT5 Summarization model
model_name = "csebuetnlp/mT5_multilingual_XLSum"  # รองรับไทย
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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

def summarize_text(text: str, max_len: int = 150) -> str:
    """สรุปข้อความด้วย mT5"""
    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        length_penalty=1.0,
        max_length=max_len,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def chunk_text(texts: list[str]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    documents = [Document(page_content=t) for t in texts]
    return splitter.split_documents(documents)