from fastapi import FastAPI
from routers import ingest

app = FastAPI()

# include router
app.include_router(ingest.router)