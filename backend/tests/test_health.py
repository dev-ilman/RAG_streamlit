from __future__ import annotations

from httpx import AsyncClient


async def test_live_needs_no_dependencies(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health/live")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["provider"] == "fake"


async def test_ready_reports_the_database(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health/ready")
    assert response.status_code == 200
    assert response.json()["database"] == "ok"


async def test_root_points_at_the_docs(client: AsyncClient) -> None:
    assert (await client.get("/")).json()["docs"] == "/docs"


async def test_openapi_schema_is_served(client: AsyncClient) -> None:
    schema = (await client.get("/openapi.json")).json()
    assert "/api/v1/chat/sessions/{session_id}/ask" in schema["paths"]


async def test_every_response_carries_a_request_id(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health/live")
    assert response.headers["x-request-id"]


async def test_unknown_route_uses_the_error_envelope(client: AsyncClient) -> None:
    response = await client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "http_404"
