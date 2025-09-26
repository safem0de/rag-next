from fastapi import APIRouter, Query
from services.vector_store import query_vector_db

router = APIRouter()

@router.get("/retrieve")
async def retrieve(query: str, top_k: int = 3):
    results = query_vector_db(query, top_k=top_k)
    return {"results": results}
