"""Application settings, loaded once from the environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- App ---------------------------------------------------------------
    project_name: str = "RAG Platform"
    api_v1_prefix: str = "/api/v1"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173"
    rate_limit_per_minute: int = 60

    # --- Security ----------------------------------------------------------
    secret_key: str = "dev-only-insecure-key-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 14
    cookie_secure: bool = False
    refresh_cookie_name: str = "rag_refresh"

    # --- Database ----------------------------------------------------------
    postgres_user: str = "rag"
    postgres_password: str = "rag"
    postgres_db: str = "rag"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str | None = None
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False

    # --- Model provider ----------------------------------------------------
    llm_provider: Literal["gemini", "fake"] = "gemini"
    google_api_key: str = ""
    google_api_base: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_chat_model: str = "gemini-3.6-flash"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    embedding_dim: int = 768
    llm_timeout_seconds: float = 60.0
    llm_max_output_tokens: int = 2048
    llm_temperature: float = 0.2

    # --- Retrieval / ingestion --------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.0
    embed_batch_size: int = 64
    max_upload_bytes: int = 10 * 1024 * 1024
    max_history_messages: int = 10
    storage_dir: Path = Field(default=Path("data/uploads"))

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_fits(cls, v: int, info: object) -> int:
        if v < 0:
            raise ValueError("chunk_overlap must be >= 0")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sqlalchemy_url(self) -> str:
        """Async SQLAlchemy URL, derived from parts unless DATABASE_URL is set."""
        if self.database_url:
            return _as_asyncpg(self.database_url)
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


def _as_asyncpg(url: str) -> str:
    """Normalise a libpq-style URL (as handed out by Azure/AWS) to asyncpg."""
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://") :]
    return url


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
