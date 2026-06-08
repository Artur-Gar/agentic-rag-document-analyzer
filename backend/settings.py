from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    max_file_size_mb: int = 50
    max_total_size_mb: int = 200
    cache_expire_days: int = 7
    vector_search_k: int = 10
    max_research_iterations: int = 2
    hybrid_retriever_weights: Annotated[tuple[float, float], NoDecode] = (0.4, 0.6)
    allowed_extensions: Annotated[tuple[str, ...], NoDecode] = (".txt", ".pdf", ".docx", ".md")

    chroma_db_path: str = "backend_data/chroma_db"
    cache_dir: str = "backend_data/document_cache"
    upload_dir: str = "backend_data/uploads"

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    embedding_model_id: str = "text-embedding-3-small"
    research_model_id: str = "gpt-4o-mini"
    verification_model_id: str = "gpt-4o-mini"
    relevance_model_id: str = "gpt-4o-mini"

    @field_validator("hybrid_retriever_weights", mode="before")
    @classmethod
    def parse_hybrid_weights(cls, value: object) -> object:
        """Convert comma-separated retriever weights into a tuple of floats."""
        if isinstance(value, str):
            return tuple(float(item.strip()) for item in value.split(",") if item.strip())
        return value

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def parse_allowed_extensions(cls, value: object) -> object:
        """Convert comma-separated allowed extensions into a tuple of strings."""
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    def resolve_path(self, raw_path: str) -> Path:
        """Resolve a configured relative path against the project root."""
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return ROOT_DIR / path

    @property
    def chroma_directory(self) -> Path:
        """Return the resolved directory used for persisted Chroma data."""
        return self.resolve_path(self.chroma_db_path)

    @property
    def cache_directory(self) -> Path:
        """Return the resolved directory used for cached document chunks."""
        return self.resolve_path(self.cache_dir)

    @property
    def upload_directory(self) -> Path:
        """Return the resolved directory used for uploaded files."""
        return self.resolve_path(self.upload_dir)

    @property
    def max_file_size_bytes(self) -> int:
        """Return the per-file upload limit in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def max_total_size_bytes(self) -> int:
        """Return the combined upload size limit in bytes."""
        return self.max_total_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached backend settings instance."""
    return Settings()
