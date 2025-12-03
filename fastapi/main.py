from fastapi import FastAPI
from routers import ingest, retrieve, auth
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # หรือจะกำหนดเป็น ["http://localhost:3000"] ถ้า Next.js
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include router
app.include_router(auth.router)
app.include_router(ingest.router, prefix="/api", tags=["ingest"])
app.include_router(retrieve.router, prefix="/api", tags=["retrieve"])

@app.get("/")
async def root():
    return {"message": "FastAPI RAG backend is running 🚀"}
