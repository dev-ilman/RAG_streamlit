from __future__ import annotations

from httpx import AsyncClient

from app.services import rag
from tests.test_documents import BATTERY_TEXT, ingest_text, wait_until_ready

CHAT = "/api/v1/chat"


async def make_session(client: AsyncClient) -> str:
    response = await client.post(CHAT + "/sessions", json={})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def test_chat_requires_authentication(client: AsyncClient) -> None:
    assert (await client.get(CHAT + "/sessions")).status_code == 401


async def test_create_and_list_sessions(auth_client: AsyncClient) -> None:
    session_id = await make_session(auth_client)
    sessions = (await auth_client.get(CHAT + "/sessions")).json()
    assert [s["id"] for s in sessions] == [session_id]


async def test_ask_without_documents_declines_to_answer(auth_client: AsyncClient) -> None:
    session_id = await make_session(auth_client)
    response = await auth_client.post(
        f"{CHAT}/sessions/{session_id}/ask", json={"message": "What causes capacity fade?"}
    )
    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["content"] == rag.NO_CONTEXT_ANSWER
    assert assistant["citations"] == []


async def test_ask_grounds_the_answer_and_attaches_citations(auth_client: AsyncClient) -> None:
    document = await ingest_text(auth_client, "Battery ageing", BATTERY_TEXT * 6)
    await wait_until_ready(auth_client, document["id"])

    session_id = await make_session(auth_client)
    response = await auth_client.post(
        f"{CHAT}/sessions/{session_id}/ask",
        json={"message": "What consumes cyclable lithium?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["user_message"]["role"] == "user"
    assistant = body["assistant_message"]
    assert assistant["role"] == "assistant"
    assert assistant["citations"], "an answer from documents must carry citations"

    citation = assistant["citations"][0]
    assert citation["document_id"] == document["id"]
    assert citation["document_title"] == "Battery ageing"
    assert 0.0 <= citation["score"] <= 1.0
    # The fake provider echoes its context, proving retrieval reached the prompt.
    assert "interphase" in assistant["content"]


async def test_both_turns_are_persisted_and_the_title_is_derived(
    auth_client: AsyncClient,
) -> None:
    document = await ingest_text(auth_client, "Battery ageing", BATTERY_TEXT * 4)
    await wait_until_ready(auth_client, document["id"])
    session_id = await make_session(auth_client)

    await auth_client.post(
        f"{CHAT}/sessions/{session_id}/ask", json={"message": "Explain SEI growth"}
    )
    detail = (await auth_client.get(f"{CHAT}/sessions/{session_id}")).json()

    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["title"] == "Explain SEI growth"


async def test_history_is_retained_across_turns(auth_client: AsyncClient) -> None:
    session_id = await make_session(auth_client)
    for question in ("first question", "second question"):
        await auth_client.post(f"{CHAT}/sessions/{session_id}/ask", json={"message": question})

    detail = (await auth_client.get(f"{CHAT}/sessions/{session_id}")).json()
    assert len(detail["messages"]) == 4
    assert detail["title"] == "first question"


async def test_rename_and_delete_a_session(auth_client: AsyncClient) -> None:
    session_id = await make_session(auth_client)
    renamed = await auth_client.patch(
        f"{CHAT}/sessions/{session_id}", json={"title": "Battery questions"}
    )
    assert renamed.json()["title"] == "Battery questions"

    assert (await auth_client.delete(f"{CHAT}/sessions/{session_id}")).status_code == 204
    assert (await auth_client.get(f"{CHAT}/sessions/{session_id}")).status_code == 404


async def test_a_session_belonging_to_someone_else_is_not_found(client: AsyncClient) -> None:
    from tests.conftest import unique_email

    first = await client.post(
        "/api/v1/auth/register",
        json={"email": unique_email("owner"), "password": "correct-horse-battery"},
    )
    client.headers["Authorization"] = f"Bearer {first.json()['access_token']}"
    session_id = await make_session(client)

    second = await client.post(
        "/api/v1/auth/register",
        json={"email": unique_email("other"), "password": "correct-horse-battery"},
    )
    client.headers["Authorization"] = f"Bearer {second.json()['access_token']}"

    assert (await client.get(f"{CHAT}/sessions/{session_id}")).status_code == 404
    denied = await client.post(f"{CHAT}/sessions/{session_id}/ask", json={"message": "hi"})
    assert denied.status_code == 404


async def test_streaming_emits_meta_tokens_and_done(auth_client: AsyncClient) -> None:
    document = await ingest_text(auth_client, "Battery ageing", BATTERY_TEXT * 4)
    await wait_until_ready(auth_client, document["id"])
    session_id = await make_session(auth_client)

    events: list[str] = []
    async with auth_client.stream(
        "POST",
        f"{CHAT}/sessions/{session_id}/ask/stream",
        json={"message": "What is lithium plating?"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                events.append(line.removeprefix("event: "))

    assert events[0] == "meta"
    assert "token" in events
    assert events[-1] == "done"

    detail = (await auth_client.get(f"{CHAT}/sessions/{session_id}")).json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["citations"]


async def test_empty_message_is_rejected(auth_client: AsyncClient) -> None:
    session_id = await make_session(auth_client)
    response = await auth_client.post(f"{CHAT}/sessions/{session_id}/ask", json={"message": ""})
    assert response.status_code == 422
