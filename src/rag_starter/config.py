from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chroma_dir: str = ".chroma"
    chroma_collection: str = "docs"

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chunk_size: int = 800
    chunk_overlap: int = 120

    top_k: int = 5


settings = Settings()
