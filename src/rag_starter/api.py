from fastapi import FastAPI
from pydantic import BaseModel, Field

from .ingest import ingest_path
from .rag import retrieve, build_prompt

app = FastAPI(title="RAG Starter", version="0.1.0")


class IngestRequest(BaseModel):
    path: str = Field(..., description="File or folder path containing .txt/.md")


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None
    return_prompt: bool = True


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ingest")
def ingest(req: IngestRequest):
    return ingest_path(req.path)


@app.post("/query")
def query(req: QueryRequest):
    result = retrieve(req.query, top_k=req.top_k)
    if req.return_prompt:
        result["prompt"] = build_prompt(req.query, result["hits"])
    return result
