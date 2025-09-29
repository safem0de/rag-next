from fastapi import APIRouter, Query
from services.vector_store import query_vector_db

router = APIRouter()


@router.get("/retrieve")
async def retrieve_api(query: str = Query(...), top_k: int = Query(5)):
    results = query_vector_db(query, top_k=top_k)

    # รวม context text
    context = "\n\n".join([r["payload"]["text"] for r in results if "payload" in r])

    return {
        "query": query,
        "context": context,
        "results": results
    }
