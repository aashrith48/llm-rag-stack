from __future__ import annotations

from .config import settings
from .embeddings import embed_texts
from .vectorstore import get_collection


def retrieve(query: str, top_k: int | None = None) -> dict:
    top_k = top_k or settings.top_k
    col = get_collection()
    q_emb = embed_texts([query])[0]

    res = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    hits: list[dict] = []
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append(
            {
                "text": doc,
                "source": meta.get("source"),
                "chunk_index": meta.get("chunk_index"),
                "distance": float(dist),
            }
        )
    return {"query": query, "top_k": top_k, "hits": hits}


def build_context(hits: list[dict], max_chars: int = 6000) -> str:
    out: list[str] = []
    total = 0
    for h in hits:
        block = f"[source={h['source']} chunk={h['chunk_index']}]\n{h['text']}\n"
        if total + len(block) > max_chars:
            break
        out.append(block)
        total += len(block)
    return "\n---\n".join(out)


def build_prompt(question: str, hits: list[dict]) -> str:
    context = build_context(hits)
    return f"""You are a helpful assistant. Use ONLY the context below to answer.
If the answer is not in the context, say you don't know.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
