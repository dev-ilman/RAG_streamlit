"""Deterministic offline provider.

Used by the test suite and by CI so that every push exercises the real
ingestion, storage, and retrieval paths without network access or an API key.
Embeddings are derived from a hash of the token set, which gives stable
vectors where lexical overlap produces genuine cosine similarity - enough for
retrieval assertions to be meaningful rather than vacuous.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import AsyncIterator, Sequence

from app.providers.base import ChatTurn, CompletionResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class FakeProvider:
    name = "fake"

    def __init__(self, *, embedding_dim: int = 768, chat_model: str = "fake-chat") -> None:
        self.embedding_dim = embedding_dim
        self.chat_model = chat_model
        self.embedding_model = "fake-embedding"

    # -- embeddings --------------------------------------------------------
    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.embedding_dim
        tokens = _TOKEN_RE.findall(text.lower()) or ["\x00empty"]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.embedding_dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    # -- generation --------------------------------------------------------
    def _answer(self, system_prompt: str, user_message: str) -> str:
        # Echo a slice of the grounding context so tests can assert that
        # retrieval actually reached the prompt.
        context = ""
        if "<context>" in system_prompt:
            context = system_prompt.split("<context>", 1)[1].split("</context>", 1)[0].strip()
        snippet = " ".join(context.split())[:240]
        return f"[fake-answer] {user_message.strip()} || context: {snippet}"

    async def complete(
        self,
        *,
        system_prompt: str,
        history: Sequence[ChatTurn],
        user_message: str,
    ) -> CompletionResult:
        text = self._answer(system_prompt, user_message)
        return CompletionResult(
            text=text,
            model=self.chat_model,
            finish_reason="stop",
            usage={"prompt_chars": len(system_prompt) + len(user_message)},
        )

    async def stream(
        self,
        *,
        system_prompt: str,
        history: Sequence[ChatTurn],
        user_message: str,
    ) -> AsyncIterator[str]:
        for word in self._answer(system_prompt, user_message).split(" "):
            yield word + " "

    async def aclose(self) -> None:
        return None
