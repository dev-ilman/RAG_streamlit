"""Registration, login, refresh-token rotation, and logout."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy import select, update

from app.core.config import settings
from app.core.deps import CurrentUser, DbSession
from app.core.security import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    refresh_token_expiry,
    verify_password,
)
from app.models import RefreshToken, User
from app.schemas.auth import TokenResponse, UserCreate, UserLogin, UserRead

router = APIRouter(prefix="/auth", tags=["auth"])


def _set_refresh_cookie(response: Response, token: str) -> None:
    """Deliver the refresh token as an httpOnly cookie.

    Keeping it out of JavaScript means an XSS bug cannot exfiltrate a
    long-lived credential; the access token stays in memory only. SameSite=lax
    is safe here because the SPA is served from the same origin as the API.
    """
    response.set_cookie(
        key=settings.refresh_cookie_name,
        value=token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        max_age=settings.refresh_token_expire_days * 24 * 3600,
        path=f"{settings.api_v1_prefix}/auth",
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(
        key=settings.refresh_cookie_name,
        path=f"{settings.api_v1_prefix}/auth",
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
    )


async def _issue_session(
    db: DbSession, response: Response, user: User, user_agent: str | None
) -> TokenResponse:
    plaintext, token_hash = generate_refresh_token()
    db.add(
        RefreshToken(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=refresh_token_expiry(),
            user_agent=(user_agent or "")[:300] or None,
        )
    )
    await db.commit()
    _set_refresh_cookie(response, plaintext)
    return TokenResponse(
        access_token=create_access_token(user.id, extra_claims={"email": user.email}),
        expires_in=settings.access_token_expire_minutes * 60,
        user=UserRead.model_validate(user),
    )


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    payload: UserCreate, request: Request, response: Response, db: DbSession
) -> TokenResponse:
    email = payload.email.lower()
    if await db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with that email exists")

    user = User(
        email=email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return await _issue_session(db, response, user, request.headers.get("user-agent"))


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: UserLogin, request: Request, response: Response, db: DbSession
) -> TokenResponse:
    user = await db.scalar(select(User).where(User.email == payload.email.lower()))
    # Verify unconditionally against a dummy hash when the user is missing, so
    # response time does not reveal whether the address is registered.
    stored = user.hashed_password if user else "$2b$12$" + "." * 53
    if not verify_password(payload.password, stored) or user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Account is disabled")
    return await _issue_session(db, response, user, request.headers.get("user-agent"))


@router.post("/refresh", response_model=TokenResponse)
async def refresh(request: Request, response: Response, db: DbSession) -> TokenResponse:
    """Exchange the refresh cookie for a new access token, rotating the cookie."""
    cookie = request.cookies.get(settings.refresh_cookie_name)
    if not cookie:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing refresh token")

    record = await db.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(cookie))
    )
    now = datetime.now(UTC)
    if record is None or record.revoked_at is not None or record.expires_at <= now:
        _clear_refresh_cookie(response)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired refresh token")

    user = await db.get(User, record.user_id)
    if user is None or not user.is_active:
        _clear_refresh_cookie(response)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid refresh token")

    record.revoked_at = now  # single use: rotation makes replay detectable
    return await _issue_session(db, response, user, request.headers.get("user-agent"))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(request: Request, response: Response, db: DbSession) -> None:
    cookie = request.cookies.get(settings.refresh_cookie_name)
    if cookie:
        await db.execute(
            update(RefreshToken)
            .where(
                RefreshToken.token_hash == hash_refresh_token(cookie),
                RefreshToken.revoked_at.is_(None),
            )
            .values(revoked_at=datetime.now(UTC))
        )
        await db.commit()
    _clear_refresh_cookie(response)


@router.get("/me", response_model=UserRead)
async def me(user: CurrentUser) -> UserRead:
    return UserRead.model_validate(user)
