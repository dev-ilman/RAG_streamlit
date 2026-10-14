"""Password hashing and JWT / refresh-token primitives.

Token design
------------
* **Access token** - short-lived, stateless JWT (HS256). Sent as
  ``Authorization: Bearer <token>``. Never stored by the browser beyond memory.
* **Refresh token** - long-lived *opaque* random string delivered in an
  httpOnly cookie. Only its SHA-256 hash is persisted, so a database leak does
  not yield usable tokens, and every refresh rotates the value. Rotation lets
  us detect replay and revoke a whole session server-side, which a stateless
  refresh JWT cannot do.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import bcrypt
import jwt

from app.core.config import settings

BCRYPT_MAX_BYTES: Final = 72
TOKEN_TYPE_ACCESS: Final = "access"


class TokenError(Exception):
    """Raised when a token is malformed, expired, or of the wrong type."""


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #
def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt."""
    raw = password.encode("utf-8")
    if len(raw) > BCRYPT_MAX_BYTES:
        raise ValueError(f"password must be at most {BCRYPT_MAX_BYTES} bytes")
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time check of a plaintext password against a stored hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # Malformed hash in the database, or an over-long candidate password.
        return False


# --------------------------------------------------------------------------- #
# Access tokens (JWT)
# --------------------------------------------------------------------------- #
def create_access_token(
    subject: str | uuid.UUID,
    *,
    expires_delta: timedelta | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    now = datetime.now(UTC)
    expire = now + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": TOKEN_TYPE_ACCESS,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "jti": uuid.uuid4().hex,
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate an access token, or raise :class:`TokenError`."""
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "sub", "type"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("token is invalid") from exc

    if payload.get("type") != TOKEN_TYPE_ACCESS:
        raise TokenError("wrong token type")
    return payload


# --------------------------------------------------------------------------- #
# Refresh tokens (opaque + hashed at rest)
# --------------------------------------------------------------------------- #
def generate_refresh_token() -> tuple[str, str]:
    """Return ``(plaintext, sha256_hash)``. Only the hash is ever persisted."""
    plaintext = secrets.token_urlsafe(48)
    return plaintext, hash_refresh_token(plaintext)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def refresh_token_expiry() -> datetime:
    return datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days)
