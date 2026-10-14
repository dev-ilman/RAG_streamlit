"""Builds the configured provider. One instance is shared per process."""

from __future__ import annotations

from app.core.config import Settings, settings
from app.providers.base import ModelProvider
from app.providers.fake import FakeProvider
from app.providers.gemini import GeminiProvider


def build_provider(config: Settings | None = None) -> ModelProvider:
    config = config or settings
    if config.llm_provider == "fake":
        return FakeProvider(embedding_dim=config.embedding_dim)
    return GeminiProvider(
        api_key=config.google_api_key,
        api_base=config.google_api_base,
        chat_model=config.gemini_chat_model,
        embedding_model=config.gemini_embedding_model,
        embedding_dim=config.embedding_dim,
        timeout=config.llm_timeout_seconds,
        max_output_tokens=config.llm_max_output_tokens,
        temperature=config.llm_temperature,
    )
