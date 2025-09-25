### Concept Flow
1. อัปโหลด PDF → รับไฟล์เข้ามา
2. Extract content → ใช้ PyMuPDF (pymupdf) แยกข้อความและรูปภาพ
3. Preprocess → ถ้าเป็นภาษาไทย ใช้ pythainlp ตัดคำ
4. Summarize → ใช้ mt5 summarization model ทำสรุป
5. Embed → ใช้ SentenceTransformer ทำ embedding (vector) ของ text
6. Store → บันทึก vector + metadata ลง Qdrant (ซึ่งเป็น VectorDB)
7. Query → เวลาผู้ใช้ถาม จะ embed คำถาม → ค้นหาจาก vectorDB → ส่งบริบทไป LLM (Gemini Vercel AI)
8. ตอบกลับ → LLM สร้างคำตอบ พร้อมดึงรูปภาพที่เกี่ยวข้องมาโชว์

### fast api
```bash
python -m venv .venv
cd fastapi
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
http://localhost:8000/docs
```
### nextjs
```bash
npm install
npm run dev -- -p 3001
http://localhost:3001/upload
```
