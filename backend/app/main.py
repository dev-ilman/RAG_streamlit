"""Application entrypoint: wiring, lifespan, and OpenAPI metadata."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RateLimitMiddleware, RequestContextMiddleware
from app.db.session import dispose_engine
from app.providers import build_provider

logger = logging.getLogger(__name__)

DESCRIPTION = """
Retrieval-augmented generation over your own documents.

* **Auth** - register/login issue a short-lived JWT access token plus a
  rotating refresh token held in an httpOnly cookie.
* **Documents** - upload a file, ingest a URL, or paste text. Extraction,
  chunking, and embedding run in the background; poll the document until its
  status is `ready`.
* **Chat** - ask a question in a session and get an answer grounded in your
  documents, with the source passages attached to every reply.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(settings.log_level, json_output=settings.is_production)
    await asyncio.to_thread(Path(settings.storage_dir).mkdir, parents=True, exist_ok=True)

    app.state.provider = build_provider()
    logger.info(
        "application started",
        extra={
            "provider": app.state.provider.name,
            "environment": settings.environment,
            "embedding_dim": settings.embedding_dim,
        },
    )
    # Catch both the code default and the placeholder shipped in .env.example:
    # copying the example and flipping ENVIRONMENT would otherwise slip through.
    if settings.is_production and (
        settings.secret_key.startswith("dev-only")
        or settings.secret_key.startswith("change-me")
        or len(settings.secret_key) < 32
    ):
        raise RuntimeError("SECRET_KEY must be set to a real secret in production")

    try:
        yield
    finally:
        await app.state.provider.aclose()
        await dispose_engine()
        logger.info("application stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.project_name,
        description=DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Order matters: the outermost middleware is registered last.
    # No GZip here on purpose. zlib buffers small writes, which stalls the
    # SSE answer stream; nginx compresses the static assets in front of us.
    app.add_middleware(
        RateLimitMiddleware,
        limit_per_minute=settings.rate_limit_per_minute,
        exempt_paths={
            f"{settings.api_v1_prefix}/health/live",
            f"{settings.api_v1_prefix}/health/ready",
        },
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["x-request-id"],
    )

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "name": settings.project_name,
            "docs": "/docs",
            "health": f"{settings.api_v1_prefix}/health/ready",
        }

    return app


app = create_app()
