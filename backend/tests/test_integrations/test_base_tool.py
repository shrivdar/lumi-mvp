"""Tests for the BaseTool framework — caching, rate-limiting, retry, error normalization."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from core.exceptions import ToolError
from integrations.base_tool import BaseTool

# ---------------------------------------------------------------------------
# Concrete test tool
# ---------------------------------------------------------------------------

class DummyTool(BaseTool):
    tool_id = "dummy"
    name = "dummy_tool"
    description = "A tool for testing"
    category = "test"
    rate_limit = 100.0
    cache_ttl = 60
    max_retries = 2
    retry_backoff = [0.01, 0.02, 0.04]


class SuccessTool(DummyTool):
    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        return {"result": "ok", "query": kwargs.get("query", "")}


class FailTool(DummyTool):
    """Raises on every call."""

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        raise httpx.TimeoutException("timed out")


class FailOnceTool(DummyTool):
    """Fails first call then succeeds."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._call_count = 0

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        self._call_count += 1
        if self._call_count == 1:
            raise httpx.TimeoutException("timed out")
        return {"result": "recovered", "attempts": self._call_count}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaseToolExecution:
    @pytest.mark.asyncio
    async def test_successful_execution(self) -> None:
        tool = SuccessTool()
        result = await tool.execute(query="test")
        assert result["result"] == "ok"
        assert result["query"] == "test"
        await tool.close()

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self) -> None:
        tool = FailOnceTool()
        result = await tool.execute(query="retry")
        assert result["result"] == "recovered"
        assert result["attempts"] == 2
        await tool.close()

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self) -> None:
        tool = FailTool()
        with pytest.raises(ToolError, match="all 3 attempts failed"):
            await tool.execute(query="fail")
        await tool.close()

    @pytest.mark.asyncio
    async def test_non_retryable_http_error(self) -> None:
        class NotFoundTool(DummyTool):
            async def _execute(self, **kwargs: Any) -> dict[str, Any]:
                resp = httpx.Response(404, request=httpx.Request("GET", "http://test"))
                raise httpx.HTTPStatusError("Not found", request=resp.request, response=resp)

        tool = NotFoundTool()
        with pytest.raises(ToolError, match="HTTP 404"):
            await tool.execute()
        await tool.close()

    @pytest.mark.asyncio
    async def test_retryable_http_500(self) -> None:
        call_count = 0

        class Flaky500Tool(DummyTool):
            async def _execute(self, **kwargs: Any) -> dict[str, Any]:
                nonlocal call_count
                call_count += 1
                if call_count <= 1:
                    resp = httpx.Response(500, request=httpx.Request("GET", "http://test"))
                    raise httpx.HTTPStatusError("Server error", request=resp.request, response=resp)
                return {"result": "ok"}

        tool = Flaky500Tool()
        result = await tool.execute()
        assert result["result"] == "ok"
        assert call_count == 2
        await tool.close()


class TestBaseToolCaching:
    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached(self) -> None:
        tool = SuccessTool()
        # Manually inject a cache hit
        tool._cache_get = AsyncMock(return_value={"cached": True})  # type: ignore[method-assign]
        result = await tool.execute(query="cached")
        assert result == {"cached": True}
        await tool.close()

    @pytest.mark.asyncio
    async def test_cache_miss_calls_execute(self) -> None:
        tool = SuccessTool()
        tool._cache_get = AsyncMock(return_value=None)  # type: ignore[method-assign]
        tool._cache_set = AsyncMock()  # type: ignore[method-assign]
        result = await tool.execute(query="miss")
        assert result["result"] == "ok"
        tool._cache_set.assert_called_once()
        await tool.close()


class TestBaseToolCacheKey:
    def test_cache_key_deterministic(self) -> None:
        tool = SuccessTool()
        k1 = tool._cache_key(query="test", limit=5)
        k2 = tool._cache_key(query="test", limit=5)
        assert k1 == k2

    def test_cache_key_different_params(self) -> None:
        tool = SuccessTool()
        k1 = tool._cache_key(query="a")
        k2 = tool._cache_key(query="b")
        assert k1 != k2


class TestBaseToolRegistration:
    def test_auto_register(self) -> None:
        from core.tool_registry import InMemoryToolRegistry

        reg = InMemoryToolRegistry()
        SuccessTool(registry=reg)
        entry = reg.get_tool("dummy_tool")
        assert entry is not None
        assert entry.name == "dummy_tool"
        assert entry.category == "test"


class TestToolErrorNormalization:
    @pytest.mark.asyncio
    async def test_unexpected_error_wrapped(self) -> None:
        class BrokenTool(DummyTool):
            async def _execute(self, **kwargs: Any) -> dict[str, Any]:
                raise RuntimeError("kaboom")

        tool = BrokenTool()
        with pytest.raises(ToolError, match="unexpected error"):
            await tool.execute()
        await tool.close()
