from __future__ import annotations

from httpx import AsyncClient

from app.core.config import settings
from app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from tests.conftest import unique_email

REGISTER = "/api/v1/auth/register"
LOGIN = "/api/v1/auth/login"
ME = "/api/v1/auth/me"


def test_password_hash_roundtrip() -> None:
    hashed = hash_password("correct-horse-battery")
    assert hashed != "correct-horse-battery"
    assert verify_password("correct-horse-battery", hashed)
    assert not verify_password("wrong-password-entirely", hashed)


def test_verify_rejects_a_malformed_hash() -> None:
    assert not verify_password("anything", "not-a-bcrypt-hash")


async def test_register_returns_token_and_sets_refresh_cookie(client: AsyncClient) -> None:
    email = unique_email()
    response = await client.post(
        REGISTER, json={"email": email, "password": "correct-horse-battery"}
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["user"]["email"] == email
    assert body["expires_in"] == settings.access_token_expire_minutes * 60
    assert settings.refresh_cookie_name in response.cookies


async def test_register_rejects_a_duplicate_email(client: AsyncClient) -> None:
    email = unique_email()
    payload = {"email": email, "password": "correct-horse-battery"}
    assert (await client.post(REGISTER, json=payload)).status_code == 201
    conflict = await client.post(REGISTER, json=payload)
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "http_409"


async def test_register_rejects_a_short_password(client: AsyncClient) -> None:
    response = await client.post(REGISTER, json={"email": unique_email(), "password": "short"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_login_succeeds_then_rejects_a_wrong_password(client: AsyncClient) -> None:
    email = unique_email()
    await client.post(REGISTER, json={"email": email, "password": "correct-horse-battery"})

    ok = await client.post(LOGIN, json={"email": email, "password": "correct-horse-battery"})
    assert ok.status_code == 200
    assert ok.json()["access_token"]

    bad = await client.post(LOGIN, json={"email": email, "password": "not-the-password"})
    assert bad.status_code == 401


async def test_login_for_an_unknown_email_is_401_not_404(client: AsyncClient) -> None:
    # Must not reveal whether the address is registered.
    response = await client.post(
        LOGIN, json={"email": unique_email("ghost"), "password": "correct-horse-battery"}
    )
    assert response.status_code == 401


async def test_me_requires_a_token(client: AsyncClient) -> None:
    assert (await client.get(ME)).status_code == 401


async def test_me_rejects_a_forged_token(client: AsyncClient) -> None:
    response = await client.get(ME, headers={"Authorization": "Bearer not.a.jwt"})
    assert response.status_code == 401


async def test_me_rejects_a_token_for_a_deleted_user(client: AsyncClient) -> None:
    import uuid

    token = create_access_token(uuid.uuid4())
    response = await client.get(ME, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


async def test_me_returns_the_authenticated_user(auth_client: AsyncClient) -> None:
    response = await auth_client.get(ME)
    assert response.status_code == 200
    assert "@example.com" in response.json()["email"]


async def test_refresh_rotates_the_cookie_and_old_token_is_dead(client: AsyncClient) -> None:
    email = unique_email()
    await client.post(REGISTER, json={"email": email, "password": "correct-horse-battery"})
    original = client.cookies.get(settings.refresh_cookie_name)
    assert original

    refreshed = await client.post("/api/v1/auth/refresh")
    assert refreshed.status_code == 200
    assert client.cookies.get(settings.refresh_cookie_name) != original

    # Replaying the consumed token must fail.
    replay = await client.post(
        "/api/v1/auth/refresh", cookies={settings.refresh_cookie_name: original}
    )
    assert replay.status_code == 401


async def test_refresh_without_a_cookie_is_401(client: AsyncClient) -> None:
    assert (await client.post("/api/v1/auth/refresh")).status_code == 401


async def test_logout_revokes_the_refresh_token(client: AsyncClient) -> None:
    await client.post(REGISTER, json={"email": unique_email(), "password": "correct-horse-battery"})
    assert (await client.post("/api/v1/auth/logout")).status_code == 204
    assert (await client.post("/api/v1/auth/refresh")).status_code == 401
