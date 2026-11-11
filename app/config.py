"""Centralized application configuration.

Values are read from environment variables (or from a `.env` file).
Using pydantic-settings means the config is validated and typed: if you set
MAX_UPLOAD_SIZE_MB="abc" the app refuses to start instead of breaking at runtime.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Upload limits
    max_upload_size_mb: int = 25
    upload_dir: str = "uploads"
    environment: str = "development"

    # Chunking parameters (Week 2).
    # chunk_size = roughly how many characters per block.
    # chunk_overlap = how many characters are repeated between consecutive blocks,
    #   so a sentence sitting across two chunks is not lost during retrieval.
    chunk_size_chars: int = 1000
    chunk_overlap_chars: int = 200

    # Vector database and embeddings (Week 3).
    # Connection URL for Postgres+pgvector (see docker-compose.yml).
    database_url: str = "postgresql://postgres:ragpass@localhost:5433/ragdb"
    # Embedding model. all-MiniLM-L6-v2: small, fast, 384 dimensions.
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    # Default number of chunks returned by a search.
    search_top_k: int = 5

    # Whitelist of allowed types. We do not trust the file extension:
    # we check both the declared content-type and the real "magic bytes".
    allowed_content_types: tuple[str, ...] = ("application/pdf",)

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Return a single (cached) Settings instance.

    Used as a FastAPI dependency, so tests can override it.
    """
    return Settings()
