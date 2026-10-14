from __future__ import annotations

import asyncio

from httpx import AsyncClient

from tests.conftest import unique_email

DOCS = "/api/v1/documents"

BATTERY_TEXT = (
    "Solid electrolyte interphase growth consumes cyclable lithium and is the "
    "dominant calendar ageing mechanism in graphite anodes. "
    "Lithium plating occurs at low temperature and high charge rates. "
)
COOKING_TEXT = (
    "Bring a large pot of salted water to a rolling boil before adding pasta. "
    "Reserve a cup of starchy cooking water to emulsify the sauce. "
)


async def ingest_text(client: AsyncClient, title: str, content: str) -> dict:
    response = await client.post(DOCS + "/text", json={"title": title, "content": content})
    assert response.status_code == 202, response.text
    return response.json()


async def wait_until_ready(client: AsyncClient, document_id: str, tries: int = 40) -> dict:
    """Poll the document until ingestion reaches a terminal state."""
    for _ in range(tries):
        response = await client.get(f"{DOCS}/{document_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"ready", "failed"}:
            return body
        await asyncio.sleep(0.05)
    raise AssertionError("document never reached a terminal state")


async def test_documents_require_authentication(client: AsyncClient) -> None:
    assert (await client.get(DOCS)).status_code == 401


async def test_list_is_empty_for_a_new_user(auth_client: AsyncClient) -> None:
    body = (await auth_client.get(DOCS)).json()
    assert body["total"] == 0
    assert body["items"] == []


async def test_text_ingestion_produces_ready_chunks(auth_client: AsyncClient) -> None:
    created = await ingest_text(auth_client, "Battery ageing", BATTERY_TEXT * 8)
    assert created["status"] == "pending"

    document = await wait_until_ready(auth_client, created["id"])
    assert document["status"] == "ready", document.get("error")
    assert document["chunk_count"] > 0
    assert document["error"] is None


async def test_identical_content_is_rejected_as_a_duplicate(auth_client: AsyncClient) -> None:
    await ingest_text(auth_client, "Battery ageing", BATTERY_TEXT)
    duplicate = await auth_client.post(
        DOCS + "/text", json={"title": "Different title", "content": BATTERY_TEXT}
    )
    assert duplicate.status_code == 409


async def test_upload_rejects_an_unsupported_file_type(auth_client: AsyncClient) -> None:
    response = await auth_client.post(
        DOCS + "/upload",
        files={"file": ("payload.zip", b"PK\x03\x04binary", "application/zip")},
    )
    assert response.status_code == 415


async def test_upload_rejects_an_empty_file(auth_client: AsyncClient) -> None:
    response = await auth_client.post(
        DOCS + "/upload", files={"file": ("empty.txt", b"", "text/plain")}
    )
    assert response.status_code == 400


async def test_upload_accepts_a_text_file_and_indexes_it(auth_client: AsyncClient) -> None:
    response = await auth_client.post(
        DOCS + "/upload",
        files={"file": ("notes.txt", (BATTERY_TEXT * 6).encode(), "text/plain")},
        data={"title": "Uploaded notes"},
    )
    assert response.status_code == 202, response.text
    document = await wait_until_ready(auth_client, response.json()["id"])
    assert document["status"] == "ready", document.get("error")
    assert document["title"] == "Uploaded notes"


async def test_search_ranks_the_relevant_document_first(auth_client: AsyncClient) -> None:
    battery = await ingest_text(auth_client, "Battery ageing", BATTERY_TEXT * 4)
    await ingest_text(auth_client, "Pasta method", COOKING_TEXT * 4)
    await wait_until_ready(auth_client, battery["id"])

    response = await auth_client.post(
        DOCS + "/search", json={"query": "lithium plating at low temperature", "top_k": 3}
    )
    assert response.status_code == 200
    hits = response.json()["hits"]
    assert hits, "expected at least one hit"
    assert hits[0]["document_title"] == "Battery ageing"
    assert hits[0]["score"] >= hits[-1]["score"]


async def test_deleting_a_document_removes_it_from_search(auth_client: AsyncClient) -> None:
    created = await ingest_text(auth_client, "Battery ageing", BATTERY_TEXT * 4)
    await wait_until_ready(auth_client, created["id"])

    assert (await auth_client.delete(f"{DOCS}/{created['id']}")).status_code == 204
    assert (await auth_client.get(f"{DOCS}/{created['id']}")).status_code == 404

    hits = (await auth_client.post(DOCS + "/search", json={"query": "lithium plating"})).json()[
        "hits"
    ]
    assert hits == []


async def test_one_user_cannot_see_another_users_document(client: AsyncClient) -> None:
    first = await client.post(
        "/api/v1/auth/register",
        json={"email": unique_email("a"), "password": "correct-horse-battery"},
    )
    client.headers["Authorization"] = f"Bearer {first.json()['access_token']}"
    created = await ingest_text(client, "Private notes", BATTERY_TEXT * 3)
    await wait_until_ready(client, created["id"])

    second = await client.post(
        "/api/v1/auth/register",
        json={"email": unique_email("b"), "password": "correct-horse-battery"},
    )
    client.headers["Authorization"] = f"Bearer {second.json()['access_token']}"

    assert (await client.get(f"{DOCS}/{created['id']}")).status_code == 404
    assert (await client.get(DOCS)).json()["total"] == 0
    hits = (await client.post(DOCS + "/search", json={"query": "lithium plating"})).json()
    assert hits["hits"] == []
