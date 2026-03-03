from rag_starter.ingest import chunk_text


def test_chunking():
    t = "a" * 2000
    chunks = chunk_text(t, 800, 120)
    assert len(chunks) >= 2
