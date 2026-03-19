"""API middleware — auth, request ID, timing, CORS helpers."""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from core.audit import set_request_context
from core.config import settings

logger = structlog.get_logger(__name__)

# Paths that skip API-key authentication
_PUBLIC_PATHS: set[str] = {
    "/api/v1/health",
    "/api/v1/health/ready",
    "/docs",
    "/redoc",
    "/openapi.json",
}


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request ID into every request + structlog context."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        set_request_context(request_id=request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Log request duration."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = int((time.monotonic() - start) * 1000)
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
        )
        return response


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Validate X-API-Key header. Skip for public paths."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path.rstrip("/")

        # Skip auth for public paths, WebSocket upgrades, and CORS preflight
        if (
            path in _PUBLIC_PATHS
            or request.scope.get("type") == "websocket"
            or request.method == "OPTIONS"
        ):
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"error": "Missing API key", "detail": "Set X-API-Key header"},
            )

        if api_key != settings.api_key:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid API key"},
            )

        return await call_next(request)
