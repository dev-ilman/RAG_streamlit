"""Google Generative Language (Gemini) provider over the REST API.

Talking to the documented REST surface with :mod:`httpx` rather than pulling in
LangChain keeps the dependency tree small, the request/response shapes
auditable, and retry/timeout behaviour under our own control - all of which
matter more in a long-running service than in a notebook.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from app.providers.base import ChatTurn, CompletionResult, ProviderError

logger = logging.getLogger(__name__)

_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 3


def _qualify(model: str) -> str:
    """Gemini resource names are ``models/<id>``; accept either spelling."""
    return model if model.startswith("models/") else f"models/{model}"


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        api_base: str,
        chat_model: str,
        embedding_model: str,
        embedding_dim: int,
        timeout: float = 60.0,
        max_output_tokens: int = 2048,
        temperature: float = 0.2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError("GOOGLE_API_KEY is not set. Set it, or run with LLM_PROVIDER=fake.")
        self._base = api_base.rstrip("/")
        self.chat_model = _qualify(chat_model)
        self.embedding_model = _qualify(embedding_model)
        self.embedding_dim = embedding_dim
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={"x-goog-api-key": api_key, "content-type": "application/json"},
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    # -- transport ---------------------------------------------------------
    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base}/{path.lstrip('/')}"
        last_error = "unknown error"
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = await self._client.post(url, json=payload)
            except httpx.HTTPError as exc:
                last_error = f"transport error: {exc}"
            else:
                if response.status_code < 400:
                    return dict(response.json())
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code not in _RETRY_STATUS:
                    break
            if attempt < _MAX_ATTEMPTS:
                delay = 0.5 * 2 ** (attempt - 1)
                logger.warning(
                    "gemini call failed, retrying",
                    extra={"attempt": attempt, "delay_s": delay, "error": last_error},
                )
                await asyncio.sleep(delay)
        raise ProviderError(f"Gemini request to {path} failed: {last_error}")

    # -- embeddings --------------------------------------------------------
    def _check_dim(self, vector: list[float]) -> None:
        if len(vector) != self.embedding_dim:
            raise ProviderError(
                f"embedding model returned {len(vector)} dimensions but the database "
                f"column is {self.embedding_dim}. Set EMBEDDING_DIM to match and run "
                "a migration (see docs/architecture.md)."
            )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "requests": [
                {
                    "model": self.embedding_model,
                    "content": {"parts": [{"text": text}]},
                    "taskType": "RETRIEVAL_DOCUMENT",
                    "outputDimensionality": self.embedding_dim,
                }
                for text in texts
            ]
        }
        data = await self._post(f"{self.embedding_model}:batchEmbedContents", payload)
        vectors = [list(item.get("values", [])) for item in data.get("embeddings", [])]
        if len(vectors) != len(texts):
            raise ProviderError(f"expected {len(texts)} embeddings, got {len(vectors)}")
        for vector in vectors:
            self._check_dim(vector)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        payload = {
            "model": self.embedding_model,
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_QUERY",
            "outputDimensionality": self.embedding_dim,
        }
        data = await self._post(f"{self.embedding_model}:embedContent", payload)
        vector = list(data.get("embedding", {}).get("values", []))
        self._check_dim(vector)
        return vector

    # -- generation --------------------------------------------------------
    def _body(
        self, system_prompt: str, history: Sequence[ChatTurn], user_message: str
    ) -> dict[str, Any]:
        contents: list[dict[str, Any]] = [
            {
                "role": "model" if turn.role == "assistant" else "user",
                "parts": [{"text": turn.content}],
            }
            for turn in history
        ]
        contents.append({"role": "user", "parts": [{"text": user_message}]})
        return {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "temperature": self._temperature,
                "maxOutputTokens": self._max_output_tokens,
            },
        }

    @staticmethod
    def _extract_text(candidate: dict[str, Any]) -> str:
        parts = candidate.get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts)

    async def complete(
        self,
        *,
        system_prompt: str,
        history: Sequence[ChatTurn],
        user_message: str,
    ) -> CompletionResult:
        data = await self._post(
            f"{self.chat_model}:generateContent",
            self._body(system_prompt, history, user_message),
        )
        candidates = data.get("candidates") or []
        if not candidates:
            blocked = data.get("promptFeedback", {}).get("blockReason")
            raise ProviderError(f"model returned no candidates (blockReason={blocked})")
        usage_raw = data.get("usageMetadata", {})
        return CompletionResult(
            text=self._extract_text(candidates[0]).strip(),
            model=self.chat_model,
            finish_reason=candidates[0].get("finishReason"),
            usage={k: v for k, v in usage_raw.items() if isinstance(v, int)},
        )

    async def stream(
        self,
        *,
        system_prompt: str,
        history: Sequence[ChatTurn],
        user_message: str,
    ) -> AsyncIterator[str]:
        url = f"{self._base}/{self.chat_model}:streamGenerateContent"
        body = self._body(system_prompt, history, user_message)
        try:
            async with self._client.stream(
                "POST", url, json=body, params={"alt": "sse"}
            ) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", "replace")
                    raise ProviderError(f"HTTP {response.status_code}: {detail[:300]}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    for candidate in chunk.get("candidates", []):
                        text = self._extract_text(candidate)
                        if text:
                            yield text
        except httpx.HTTPError as exc:
            raise ProviderError(f"Gemini stream failed: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()
