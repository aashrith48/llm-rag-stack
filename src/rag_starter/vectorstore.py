import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import settings


def get_client():
    return chromadb.PersistentClient(
        path=settings.chroma_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection():
    client = get_client()
    return client.get_or_create_collection(name=settings.chroma_collection)
