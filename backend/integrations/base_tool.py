"""BaseTool framework — caching (Redis), token-bucket rate limiting, retry, error normalization.

Every external API call in YOHAS goes through a BaseTool subclass.
Tools auto-register with the ToolRegistry on instantiation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

import httpx
import redis.asyncio as aioredis
import structlog

from core.audit import AuditLogger, Timer
from core.constants import (
    CACHE_TTL_DEFAULT,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_RETRIES,
    RATE_LIMIT_DEFAULT,
    RETRY_BACKOFF_SECONDS,
)
from core.exceptions import ToolError
from core.interfaces import BaseTool as BaseToolABC
from core.models import ToolRegistryEntry, ToolSourceType

logger = structlog.get_logger(__name__)
audit = AuditLogger("integrations")


# ---------------------------------------------------------------------------
# Redis-backed token-bucket rate limiter
# ---------------------------------------------------------------------------

class TokenBucketRateLimiter:
    """Async token-bucket rate limiter backed by Redis for cross-process correctness."""

    def __init__(self, redis: aioredis.Redis, key: str, rate: float, burst: int | None = None) -> None:
        self._redis = redis
        self._key = f"yohas:ratelimit:{key}"
        self._rate = rate  # tokens per second
        self._burst = burst or max(int(rate * 2), 1)

    async def acquire(self, timeout: float = 30.0) -> None:
        """Block until a token is available, or raise after *timeout* seconds."""
        deadline = time.monotonic() + timeout
        while True:
            allowed = await self._try_acquire()
            if allowed:
                return
            if time.monotonic() >= deadline:
                raise ToolError(
                    "Rate limit timeout — could not acquire token",
                    error_code="RATE_LIMIT_TIMEOUT",
                )
            await asyncio.sleep(1.0 / self._rate)

    async def _try_acquire(self) -> bool:
        """Lua-based atomic token-bucket check-and-decrement."""
        lua = """
        local key = KEYS[1]
        local rate = tonumber(ARGV[1])
        local burst = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])

        local bucket = redis.call('HMGET', key, 'tokens', 'last')
        local tokens = tonumber(bucket[1])
        local last = tonumber(bucket[2])

        if tokens == nil then
            tokens = burst
            last = now
        end

        local elapsed = now - last
        tokens = math.min(burst, tokens + elapsed * rate)

        if tokens >= 1 then
            tokens = tokens - 1
            redis.call('HMSET', key, 'tokens', tokens, 'last', now)
            redis.call('EXPIRE', key, 300)
            return 1
        end
        redis.call('HMSET', key, 'tokens', tokens, 'last', now)
        redis.call('EXPIRE', key, 300)
        return 0
        """
        result = await self._redis.eval(lua, 1, self._key, self._rate, self._burst, time.time())  # type: ignore[arg-type]
        return result == 1


# ---------------------------------------------------------------------------
# BaseTool concrete implementation
# ---------------------------------------------------------------------------

class BaseTool(BaseToolABC):
    """Concrete base for all external API tools.

    Provides:
    - Redis caching (key = hash of tool_id + sorted query params)
    - Token-bucket rate limiting (Redis-backed)
    - Retry with configurable exponential back-off
    - Structured audit logging
    - Timeout handling via httpx
    - Error normalization to ``ToolError``
    - Auto-registration with ToolRegistry

    Subclasses implement ``_execute(**kwargs)`` with the actual API call.
    """

    tool_id: str = ""
    name: str = ""
    description: str = ""
    category: str = ""
    rate_limit: float = RATE_LIMIT_DEFAULT
    cache_ttl: int = CACHE_TTL_DEFAULT
    max_retries: int = MAX_RETRIES
    retry_backoff: list[float] = RETRY_BACKOFF_SECONDS  # type: ignore[assignment]
    timeout: int = DEFAULT_TIMEOUT_SECONDS

    def __init__(
        self,
        redis: aioredis.Redis | None = None,
        http_client: httpx.AsyncClient | None = None,
        registry: Any | None = None,
    ) -> None:
        self._redis = redis
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=10),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            follow_redirects=True,
        )
        self._owns_http = http_client is None
        self._rate_limiter: TokenBucketRateLimiter | None = None
        if self._redis is not None:
            self._rate_limiter = TokenBucketRateLimiter(
                self._redis, self.tool_id, self.rate_limit
            )
        # Auto-register with the registry if provided
        if registry is not None:
            self._register(registry)

    def _register(self, registry: Any) -> None:
        entry = ToolRegistryEntry(
            name=self.name,
            description=self.description,
            source_type=ToolSourceType.NATIVE,
            category=self.category,
            cache_ttl=self.cache_ttl,
            rate_limit_rps=self.rate_limit,
        )
        registry.register(entry)

    # ------------------------------------------------------------------
    # Public execute (template method)
    # ------------------------------------------------------------------

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Cache → rate-limit → retry → _execute → audit."""
        cache_key = self._cache_key(**kwargs)

        # 1. Cache check
        cached = await self._cache_get(cache_key)
        if cached is not None:
            audit.log("tool_cache_hit", tool=self.name)
            return cached  # type: ignore[return-value]

        # 2. Rate-limit
        await self._rate_limit_acquire()

        # 3. Retry loop
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with Timer() as t:
                    result = await self._execute(**kwargs)
                audit.log(
                    "tool_call_success",
                    tool=self.name,
                    attempt=attempt + 1,
                    duration_ms=t.elapsed_ms,
                )
                # 4. Cache store
                await self._cache_set(cache_key, result, self.cache_ttl)
                return result
            except ToolError:
                raise
            except httpx.HTTPStatusError as exc:
                last_err = exc
                if exc.response.status_code in (429, 500, 502, 503, 504):
                    backoff = self.retry_backoff[min(attempt, len(self.retry_backoff) - 1)]
                    audit.log(
                        "tool_retry",
                        tool=self.name,
                        attempt=attempt + 1,
                        status=exc.response.status_code,
                        backoff_s=backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue
                # Non-retryable HTTP error
                raise ToolError(
                    f"{self.name}: HTTP {exc.response.status_code}",
                    error_code="HTTP_ERROR",
                    details={"status": exc.response.status_code, "body": exc.response.text[:500]},
                ) from exc
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_err = exc
                backoff = self.retry_backoff[min(attempt, len(self.retry_backoff) - 1)]
                audit.log("tool_retry", tool=self.name, attempt=attempt + 1, error=str(exc), backoff_s=backoff)
                await asyncio.sleep(backoff)
            except Exception as exc:
                raise ToolError(
                    f"{self.name}: unexpected error — {exc}",
                    error_code="TOOL_UNEXPECTED",
                    details={"error": str(exc)},
                ) from exc

        raise ToolError(
            f"{self.name}: all {self.max_retries + 1} attempts failed",
            error_code="TOOL_MAX_RETRIES",
            details={"last_error": str(last_err)},
        )

    # ------------------------------------------------------------------
    # Subclass hook — override this
    # ------------------------------------------------------------------

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        """Subclasses implement the actual API call here."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, **kwargs: Any) -> str:
        raw = json.dumps({"tool": self.tool_id, **kwargs}, sort_keys=True, default=str)
        return f"yohas:cache:{hashlib.sha256(raw.encode()).hexdigest()}"

    async def _cache_get(self, key: str) -> Any | None:
        if self._redis is None:
            return None
        try:
            data = await self._redis.get(key)
            if data is not None:
                return json.loads(data)
        except Exception:
            logger.debug("cache_get_error", key=key)
        return None

    async def _cache_set(self, key: str, value: Any, ttl: int | None = None) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.set(key, json.dumps(value, default=str), ex=ttl or self.cache_ttl)
        except Exception:
            logger.debug("cache_set_error", key=key)

    async def _rate_limit_acquire(self) -> None:
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        if self._owns_http:
            await self._http.aclose()
