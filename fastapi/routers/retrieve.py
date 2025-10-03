from fastapi import APIRouter, Query
from services.vector_store import query_hybrid_rerank

router = APIRouter()

@router.get("/retrieve")
async def retrieve_api(query: str = Query(...), top_k: int = Query(5), keyword: str = Query(None)):
    results = query_hybrid_rerank(query, keyword=keyword, top_k=top_k)

    context = "\n\n".join([r["payload"]["text"] for r in results if "payload" in r])

    return {
        "query": query,
        "context": context,
        "results": results
    }
