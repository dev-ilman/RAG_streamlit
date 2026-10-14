"""Provider-agnostic interface for embeddings and chat completion.

The notebook called LangChain's ``GoogleGenerativeAIEmbeddings`` and
``GoogleGenerativeAI`` directly. Here the two capabilities the application
actually needs are pinned behind a narrow protocol, which keeps the service
layer free of vendor types and makes the whole pipeline testable offline via
:mod:`app.providers.fake`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """Upstream model call failed. Mapped to HTTP 502 by the API layer."""


@dataclass(slots=True)
class ChatTurn:
    role: str  # "user" | "assistant"
    content: str


@dataclass(slots=True)
class CompletionResult:
    text: str
    model: str
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class ModelProvider(Protocol):
    """Everything the RAG pipeline needs from a model backend."""

    name: str
    embedding_model: str
    chat_model: str
    embedding_dim: int

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages for storage. Order matches ``texts``."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query. Split from ``embed_documents`` because several
        providers expect a different task type for retrieval queries."""
        ...

    async def complete(
        self,
        *,
        system_prompt: str,
        history: Sequence[ChatTurn],
        user_message: str,
    ) -> CompletionResult: ...

    def stream(
        self,
        *,
        system_prompt: str,
        history: Sequence[ChatTurn],
        user_message: str,
    ) -> AsyncIterator[str]:
        """Yield answer text incrementally."""
        ...

    async def aclose(self) -> None: ...
