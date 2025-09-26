from fastapi import FastAPI
from routers import ingest, retrieve

app = FastAPI()

# include router
app.include_router(ingest.router, prefix="/api", tags=["ingest"])
app.include_router(retrieve.router, prefix="/api", tags=["retrieve"])

@app.get("/")
async def root():
    return {"message": "FastAPI RAG backend is running 🚀"}
