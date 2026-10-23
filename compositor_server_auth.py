"""Server-to-server gate for compositor routes.

Public paths stay open. Every other route requires
``VIBODE_COMPOSITOR_AUTH_SECRET`` and a matching
``X-Vibode-Compositor-Auth`` header before the route runs.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse

AUTH_HEADER = "x-vibode-compositor-auth"
SECRET_ENV = "VIBODE_COMPOSITOR_AUTH_SECRET"
PUBLIC_PATHS = frozenset({"/", "/health"})

def _unauthorized() -> JSONResponse:
    return JSONResponse(status_code=401, content={"detail": "Unauthorized"})


def configured_secret() -> str:
    return os.environ.get(SECRET_ENV, "").strip()


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def credentials_match(presented: str, secret: str) -> bool:
    if secret == "":
        return False
    return hmac.compare_digest(_digest(presented), _digest(secret))


def authorize(path: str, presented: str, secret: str) -> bool:
    if path in PUBLIC_PATHS:
        return True
    return credentials_match(presented, secret)


async def compositor_server_auth_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable],
):
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)
    presented = request.headers.get(AUTH_HEADER, "")
    if not credentials_match(presented, configured_secret()):
        return _unauthorized()
    return await call_next(request)
