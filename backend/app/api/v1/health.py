"""Liveness and readiness probes.

Split deliberately: orchestrators must not restart a healthy container just
because the database is briefly unreachable, so ``/live`` touches nothing and
``/ready`` is the one that reports dependency state.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status
from sqlalchemy import text

from app.core.config import settings
from app.core.deps import DbSession
from app.schemas.common import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])

VERSION = "1.0.0"


@router.get("/live", response_model=HealthResponse)
async def live(request: Request) -> HealthResponse:
    provider = getattr(request.app.state, "provider", None)
    return HealthResponse(
        status="ok",
        version=VERSION,
        environment=settings.environment,
        provider=getattr(provider, "name", "uninitialised"),
    )


@router.get("/ready", response_model=HealthResponse)
async def ready(request: Request, response: Response, db: DbSession) -> HealthResponse:
    try:
        await db.execute(text("SELECT 1"))
        database = "ok"
    except Exception as exc:  # the probe reports failure, it must never raise
        database = f"error: {type(exc).__name__}"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    provider = getattr(request.app.state, "provider", None)
    return HealthResponse(
        status="ok" if database == "ok" else "degraded",
        version=VERSION,
        environment=settings.environment,
        provider=getattr(provider, "name", "uninitialised"),
        database=database,
    )
