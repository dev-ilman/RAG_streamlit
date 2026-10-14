"""Test fixtures.

Tests run against a real PostgreSQL with pgvector - the same engine used in
production - because the parts most worth testing (the unique constraints, the
cascade deletes, and the ANN query itself) do not exist in SQLite. The model
provider is swapped for the deterministic fake so no network or API key is
needed, which is what lets CI run the full suite on every push.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator

os.environ.setdefault("LLM_PROVIDER", "fake")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-used-anywhere-else")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "0")

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.base import Base
from app.db.session import SessionLocal, engine
from app.main import create_app
from app.providers.fake import FakeProvider


@pytest.fixture(scope="session")
async def _schema() -> AsyncIterator[None]:
    """Build the schema once per session from the ORM metadata.

    Creating tables straight from the metadata keeps the suite fast and
    independent of migration history; CI separately runs `alembic upgrade
    head` and `alembic check` against an empty database.
    """
    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def clean_tables(_schema: None) -> AsyncIterator[None]:
    """Truncate before handing over, so each test starts from a known state.

    Deliberately not autouse: the pure-function tests (chunking, extraction)
    must run without a database at all.
    """
    async with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            await conn.exec_driver_sql(f'DELETE FROM "{table.name}"')
    yield


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider(embedding_dim=settings.embedding_dim)


@pytest.fixture
async def db(clean_tables: None) -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


@pytest.fixture
async def client(clean_tables: None, provider: FakeProvider) -> AsyncIterator[AsyncClient]:
    app = create_app()
    # The lifespan context is entered explicitly so app.state.provider exists,
    # then overridden with the deterministic fake.
    async with (
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
            follow_redirects=True,
        ) as http,
        app.router.lifespan_context(app),
    ):
        app.state.provider = provider
        yield http


def unique_email(prefix: str = "user") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}@example.com"


@pytest.fixture
async def auth_client(client: AsyncClient) -> AsyncIterator[AsyncClient]:
    """A client already registered and carrying a valid access token."""
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": unique_email(), "password": "correct-horse-battery", "full_name": "T"},
    )
    assert response.status_code == 201, response.text
    client.headers["Authorization"] = f"Bearer {response.json()['access_token']}"
    yield client
