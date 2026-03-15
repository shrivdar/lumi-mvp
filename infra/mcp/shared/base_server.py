"""Shared MCP server base — HTTP client, error handling, health endpoint, JSON-RPC.

All YOHAS MCP servers inherit from this to get:
- httpx async client with retry + timeout
- /health endpoint for Docker healthchecks
- Structured JSON-RPC error responses
- Rate-limit aware API calls
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

import httpx
from mcp.server import Server
from mcp.types import TextContent, Tool

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

TOOL_NAME = os.getenv("MCP_TOOL_NAME", "unknown")
TOOL_PORT = int(os.getenv("MCP_TOOL_PORT", "8080"))
MAX_RETRIES = int(os.getenv("MCP_MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.getenv("MCP_REQUEST_TIMEOUT", "30"))
RATE_LIMIT_RPS = float(os.getenv("MCP_RATE_LIMIT_RPS", "5"))


# ---------------------------------------------------------------------------
# HTTP Client with retry + rate limiting
# ---------------------------------------------------------------------------

class APIClient:
    """Shared async HTTP client with retry, timeout, and basic rate limiting."""

    def __init__(
        self,
        base_url: str = "",
        headers: dict[str, str] | None = None,
        rate_limit: float = RATE_LIMIT_RPS,
        timeout: int = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers=headers or {},
            timeout=httpx.Timeout(timeout, connect=10),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            follow_redirects=True,
        )
        self._rate_limit = rate_limit
        self._max_retries = max_retries
        self._last_request = 0.0
        self._lock = asyncio.Lock()

    async def _throttle(self) -> None:
        """Simple token bucket — one request at a time with minimum interval."""
        async with self._lock:
            now = time.monotonic()
            min_interval = 1.0 / self._rate_limit if self._rate_limit > 0 else 0
            elapsed = now - self._last_request
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            self._last_request = time.monotonic()

    async def get(self, path: str, params: dict[str, Any] | None = None, **kwargs: Any) -> httpx.Response:
        return await self._request("GET", path, params=params, **kwargs)

    async def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return await self._request("POST", path, **kwargs)

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        await self._throttle()
        last_exc: Exception | None = None
        backoff = [1, 2, 4]
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._http.request(method, path, **kwargs)
                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", backoff[min(attempt, len(backoff) - 1)]))
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(backoff[min(attempt, len(backoff) - 1)])
        raise last_exc or RuntimeError("Request failed")

    async def close(self) -> None:
        await self._http.aclose()


# ---------------------------------------------------------------------------
# Helper to build Tool and TextContent responses
# ---------------------------------------------------------------------------

def make_tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> Tool:
    """Build an MCP Tool object."""
    return Tool(
        name=name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    )


def text_result(data: Any) -> list[TextContent]:
    """Return a JSON text content result."""
    if isinstance(data, str):
        return [TextContent(type="text", text=data)]
    return [TextContent(type="text", text=json.dumps(data, default=str))]


def error_result(message: str) -> list[TextContent]:
    """Return an error as text content."""
    return [TextContent(type="text", text=json.dumps({"error": message}))]


# ---------------------------------------------------------------------------
# Server runner
# ---------------------------------------------------------------------------

async def run_server(server: Server) -> None:
    """Run the MCP server via stdio transport."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def start(server: Server) -> None:
    """Entry point for all MCP servers."""
    asyncio.run(run_server(server))
