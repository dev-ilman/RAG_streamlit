"""Shared FastAPI dependencies: authentication and the model provider."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import TokenError, decode_access_token
from app.db.session import get_db
from app.models import User
from app.providers.base import ModelProvider

bearer_scheme = HTTPBearer(auto_error=False, description="JWT access token")

DbSession = Annotated[AsyncSession, Depends(get_db)]
_Credentials = Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)]


def _unauthorised(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(credentials: _Credentials, db: DbSession) -> User:
    if credentials is None or not credentials.credentials:
        raise _unauthorised("Not authenticated")
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = uuid.UUID(str(payload["sub"]))
    except (TokenError, ValueError, KeyError) as exc:
        raise _unauthorised("Could not validate credentials") from exc

    user = await db.get(User, user_id)
    if user is None:
        raise _unauthorised("Could not validate credentials")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Account is disabled")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def get_provider(request: Request) -> ModelProvider:
    """The process-wide provider built during application startup."""
    provider: ModelProvider | None = getattr(request.app.state, "provider", None)
    if provider is None:  # pragma: no cover - only if lifespan did not run
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Model provider is not initialised"
        )
    return provider


Provider = Annotated[ModelProvider, Depends(get_provider)]
