from fastapi import FastAPI
from routers import ingest

app = FastAPI()

# include router
app.include_router(ingest.router, prefix="/api", tags=["ingest"])

@app.get("/")
async def root():
    return {"message": "FastAPI RAG backend is running 🚀"}
