from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import settings
from .embeddings import embed_texts
from .vectorstore import get_collection


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.replace("\r\n", "\n")
    if chunk_size <= 0:
        return [text]
    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def ingest_path(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    files: list[Path] = []
    if p.is_dir():
        for ext in ("*.txt", "*.md"):
            files.extend(p.rglob(ext))
    else:
        files = [p]

    col = get_collection()

    total_chunks = 0
    upserted_files = 0

    for f in files:
        text = load_text_file(f)
        parts = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        chunks = [
            Chunk(
                id=str(uuid.uuid4()),
                text=part,
                source=os.path.relpath(str(f), start=str(Path.cwd())),
                chunk_index=i,
            )
            for i, part in enumerate(parts)
        ]
        if not chunks:
            continue

        embeddings = embed_texts([c.text for c in chunks])

        col.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
        )

        total_chunks += len(chunks)
        upserted_files += 1

    return {
        "files_found": len(files),
        "files_upserted": upserted_files,
        "chunks_added": total_chunks,
        "collection": settings.chroma_collection,
        "chroma_dir": settings.chroma_dir,
    }
