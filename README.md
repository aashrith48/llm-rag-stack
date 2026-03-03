# RAG Starter (FastAPI + ChromaDB)

A minimal RAG project you can extend:

- Ingest `.txt` / `.md`
- Chunk + embed with SentenceTransformers
- Store + retrieve with Chroma
- Query via FastAPI or CLI

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

cp .env.example .env
pip install -e .
```

## Ingest

```bash
rag ingest data/raw
# or
rag ingest path/to/file.md
```

## Query

```bash
rag query "What is this repo about?" --top-k 5
```

## Run API

```bash
uvicorn rag_starter.api:app --reload
```

Then:
- POST `/ingest` with `{ "path": "data/raw" }`
- POST `/query` with `{ "query": "..." }`

## Next steps (recommended)

- Add PDF parsing (pymupdf) + HTML loaders
- Add re-ranking (bge-reranker / cross-encoder)
- Add LLM generator (OpenAI / Ollama)
- Add evals (RAGAS) + tracing (OpenTelemetry)
