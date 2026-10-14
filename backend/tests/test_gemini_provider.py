"""Gemini provider tests against a mocked transport.

The provider hand-builds the Google Generative Language REST calls, so the
request shapes, retry policy, dimension guard, and SSE parsing are exactly the
parts worth pinning down. ``httpx.MockTransport`` keeps all of it offline.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.providers.base import ProviderError
from app.providers.gemini import GeminiProvider

API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def build(
    handler: Any, *, embedding_dim: int = 4, chat_model: str = "gemini-2.5-flash"
) -> GeminiProvider:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"x-goog-api-key": "test-key"},
    )
    return GeminiProvider(
        api_key="test-key",
        api_base=API_BASE,
        chat_model=chat_model,
        embedding_model="models/text-embedding-004",
        embedding_dim=embedding_dim,
        client=client,
    )


def test_missing_api_key_is_rejected_up_front() -> None:
    with pytest.raises(ProviderError, match="GOOGLE_API_KEY"):
        GeminiProvider(
            api_key="",
            api_base=API_BASE,
            chat_model="gemini-2.5-flash",
            embedding_model="models/text-embedding-004",
            embedding_dim=768,
        )


def test_bare_model_names_are_qualified() -> None:
    provider = build(lambda _r: httpx.Response(200, json={}), chat_model="gemini-2.5-flash")
    assert provider.chat_model == "models/gemini-2.5-flash"
    assert provider.embedding_model == "models/text-embedding-004"


async def test_embed_documents_batches_and_tags_the_task_type() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "embeddings": [{"values": [1.0, 0.0, 0.0, 0.0]}, {"values": [0.0, 1.0, 0.0, 0.0]}]
            },
        )

    provider = build(handler)
    vectors = await provider.embed_documents(["first", "second"])

    assert vectors == [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    assert seen["url"].endswith("models/text-embedding-004:batchEmbedContents")
    assert [r["taskType"] for r in seen["body"]["requests"]] == [
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_DOCUMENT",
    ]
    assert seen["body"]["requests"][0]["content"]["parts"][0]["text"] == "first"
    await provider.aclose()


async def test_embed_documents_short_circuits_on_empty_input() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("no request should be sent for an empty batch")

    provider = build(handler)
    assert await provider.embed_documents([]) == []
    await provider.aclose()


async def test_embed_query_uses_the_query_task_type() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embedding": {"values": [0.5, 0.5, 0.5, 0.5]}})

    provider = build(handler)
    assert await provider.embed_query("why?") == [0.5, 0.5, 0.5, 0.5]
    assert seen["url"].endswith(":embedContent")
    assert seen["body"]["taskType"] == "RETRIEVAL_QUERY"
    await provider.aclose()


async def test_a_dimension_mismatch_fails_loudly() -> None:
    # Writing a 3-wide vector into a 4-wide column would fail far from the cause.
    provider = build(
        lambda _r: httpx.Response(200, json={"embedding": {"values": [1.0, 2.0, 3.0]}})
    )
    with pytest.raises(ProviderError, match="3 dimensions"):
        await provider.embed_query("why?")
    await provider.aclose()


async def test_a_short_batch_response_is_rejected() -> None:
    provider = build(
        lambda _r: httpx.Response(200, json={"embeddings": [{"values": [1.0, 0, 0, 0]}]})
    )
    with pytest.raises(ProviderError, match="expected 2 embeddings"):
        await provider.embed_documents(["a", "b"])
    await provider.aclose()


async def test_complete_joins_the_candidate_parts() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Solid "}, {"text": "electrolyte."}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 8},
            },
        )

    provider = build(handler)
    result = await provider.complete(
        system_prompt="<context>passage</context>", history=[], user_message="why?"
    )

    assert result.text == "Solid electrolyte."
    assert result.finish_reason == "STOP"
    assert result.usage["promptTokenCount"] == 120
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "<context>passage</context>"
    assert seen["body"]["contents"][-1] == {"role": "user", "parts": [{"text": "why?"}]}
    await provider.aclose()


async def test_history_maps_assistant_turns_to_the_model_role() -> None:
    from app.providers.base import ChatTurn

    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    provider = build(handler)
    await provider.complete(
        system_prompt="s",
        history=[ChatTurn(role="user", content="q1"), ChatTurn(role="assistant", content="a1")],
        user_message="q2",
    )

    assert [c["role"] for c in seen["body"]["contents"]] == ["user", "model", "user"]
    await provider.aclose()


async def test_a_blocked_prompt_surfaces_the_reason() -> None:
    provider = build(
        lambda _r: httpx.Response(
            200, json={"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}
        )
    )
    with pytest.raises(ProviderError, match="SAFETY"):
        await provider.complete(system_prompt="s", history=[], user_message="q")
    await provider.aclose()


async def test_transient_failures_are_retried() -> None:
    attempts = {"n": 0}

    def handler(_r: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json={"embedding": {"values": [1.0, 0.0, 0.0, 0.0]}})

    provider = build(handler)
    assert await provider.embed_query("why?") == [1.0, 0.0, 0.0, 0.0]
    assert attempts["n"] == 2
    await provider.aclose()


async def test_a_client_error_is_not_retried() -> None:
    attempts = {"n": 0}

    def handler(_r: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(400, text="bad request")

    provider = build(handler)
    with pytest.raises(ProviderError, match="HTTP 400"):
        await provider.embed_query("why?")
    assert attempts["n"] == 1, "a 400 will not become a 200 on a second try"
    await provider.aclose()


async def test_stream_parses_sse_frames_and_skips_junk() -> None:
    frames = (
        'data: {"candidates":[{"content":{"parts":[{"text":"Lithium "}]}}]}\n\n'
        ": keep-alive comment\n\n"
        "data: not-json\n\n"
        'data: {"candidates":[{"content":{"parts":[{"text":"plating."}]}}]}\n\n'
        "data: [DONE]\n\n"
    )
    provider = build(lambda _r: httpx.Response(200, text=frames))

    chunks = [
        chunk async for chunk in provider.stream(system_prompt="s", history=[], user_message="q")
    ]
    assert "".join(chunks) == "Lithium plating."
    await provider.aclose()


async def test_stream_raises_on_an_error_status() -> None:
    provider = build(lambda _r: httpx.Response(500, text="upstream exploded"))
    with pytest.raises(ProviderError, match="HTTP 500"):
        async for _ in provider.stream(system_prompt="s", history=[], user_message="q"):
            pass
    await provider.aclose()
