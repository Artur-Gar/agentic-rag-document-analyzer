from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]


class FrontendSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    frontend_host: str = "0.0.0.0"
    frontend_port: int = 7860
    frontend_backend_url: str = "http://127.0.0.1:8000"


@lru_cache(maxsize=1)
def get_frontend_settings() -> FrontendSettings:
    """Return a cached frontend settings instance."""
    return FrontendSettings()
