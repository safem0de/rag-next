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
.venv\Scripts\activate
cd fastapi
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
http://localhost:8000/docs
pip freeze > requirements.txt
```
### nextjs
```bash
npm install
สร้าง .env.local
npm run dev -- -p 3001
http://localhost:3001/upload
```

### .env.local
INGEST_API_URL=http://localhost:8000
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

### ใช้ OpenAI
ใน Python จะถูกอ่านอัตโนมัติ (ผ่าน langchain_openai.OpenAIEmbeddings)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

### ใช้ Gemini
pip install google-generativeai sentence-transformers
GEMINI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx


OpenAI (text-embedding-3-small) → 1536 dimensions
Google models/embedding-001 → 768 dimensions
1. Ingest → Qdrant ด้วย OpenAI Embedding
2. User ถาม → encode question ด้วย OpenAI Embedding → ค้นหา context จาก Qdrant
3. ส่ง context + question เข้า Gemini 1.5 Flash → ให้มัน generate คำตอบ
embedding model กับ LLM model แยกกัน

git checkout -b feature/auto-ingest-structured
📄 PDF 
- 🧩 unstructured (แยก element)
- 🧠 structured parser (extract field เช่น ชื่อ / มูลค่า)
- 🪣 Qdrant (เก็บ 2 แบบ: vector + structured JSON)
- 🔍 hybrid search (ถ้า query เป็น natural language)
- 🧠 structured query (ถ้า query ระบุ field ชัดเจน)
- 📈 reranker (เฉพาะกรณี text retrieval)


docker build -t rag-fastapi .
docker run --env-file .env -p 8000:8000 rag-fastapi