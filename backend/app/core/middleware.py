"""Request correlation and a small in-process rate limiter.

Both are written as plain ASGI middleware rather than subclasses of Starlette's
``BaseHTTPMiddleware``. That base class runs the downstream application in a
separate anyio task and streams the response back through a memory object
stream, which has two consequences we cannot live with here:

* On cancellation, the async teardown of a dependency with ``yield`` can itself
  be cancelled. A request-scoped :class:`AsyncSession` then never returns its
  connection to the pool, leaving a transaction open indefinitely.
* The extra stream layer interferes with server-sent events, which need each
  frame forwarded the moment it is produced.

Plain ASGI middleware has neither problem: it wraps ``send`` and otherwise
stays out of the way.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict, deque

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.errors import error_body
from app.core.logging import request_id_var

logger = logging.getLogger("app.access")


class RequestContextMiddleware:
    """Attach a request id, time the request, and emit one access log line."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        request_id = headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)["x-request-id"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            logger.info(
                "request",
                extra={
                    "method": scope.get("method", "-"),
                    "path": scope.get("path", "-"),
                    "status": status_code,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    "request_id": request_id,
                },
            )
            request_id_var.reset(token)


class RateLimitMiddleware:
    """Fixed-window limiter keyed by client IP.

    Deliberately in-process: it costs nothing and blunts casual abuse of a
    single instance. Behind more than one replica, move this to Redis or to the
    platform's own gateway limits.
    """

    def __init__(self, app: ASGIApp, *, limit_per_minute: int, exempt_paths: set[str]) -> None:
        self.app = app
        self._limit = limit_per_minute
        self._exempt = exempt_paths
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _over_limit(self, client: str) -> int | None:
        """Return the Retry-After value when the caller is over budget."""
        now = time.monotonic()
        window = self._hits[client]
        while window and now - window[0] > 60.0:
            window.popleft()

        if len(window) >= self._limit:
            return max(1, int(60.0 - (now - window[0])))

        window.append(now)
        # Bound memory: drop idle clients once the table gets large.
        if len(self._hits) > 10_000:
            for key in [k for k, v in self._hits.items() if not v]:
                del self._hits[key]
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self._limit <= 0 or scope.get("path") in self._exempt:
            await self.app(scope, receive, send)
            return

        client_tuple = scope.get("client")
        client = client_tuple[0] if client_tuple else "unknown"
        retry_after = self._over_limit(client)
        if retry_after is None:
            await self.app(scope, receive, send)
            return

        body = json.dumps(error_body("rate_limited", "Too many requests. Slow down.")).encode(
            "utf-8"
        )
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("latin-1")),
                    (b"retry-after", str(retry_after).encode("latin-1")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
