"""Tests for async LLM client — verifies true concurrency and token tracking."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class _FakeContentBlock:
    text: str


@dataclass
class _FakeResponse:
    usage: _FakeUsage
    content: list[_FakeContentBlock]


def _make_fake_response(
    text: str = "Hello",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> _FakeResponse:
    return _FakeResponse(
        usage=_FakeUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        content=[_FakeContentBlock(text=text)],
    )


@pytest.fixture()
def llm_client():
    """Create an LLMClient with a mocked AsyncAnthropic client."""
    with patch("core.llm.anthropic.AsyncAnthropic") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        mock_instance.messages.create = AsyncMock(
            return_value=_make_fake_response()
        )

        from core.llm import LLMClient

        client = LLMClient()
        # Replace the real client with our mock
        client._client = mock_instance
        yield client


class TestAsyncLLMClient:
    """Verify the LLM client uses async calls and tracks tokens correctly."""

    async def test_query_returns_text(self, llm_client):
        """Basic query returns the mocked response text."""
        result = await llm_client.query("What is BRCA1?")
        assert result.text == "Hello"

    async def test_query_awaits_async_create(self, llm_client):
        """Verify that messages.create is awaited (async), not called synchronously."""
        await llm_client.query("test prompt")
        llm_client._client.messages.create.assert_awaited_once()

    async def test_token_tracking_single_call(self, llm_client):
        """Token counters update after a single call."""
        await llm_client.query("test")
        assert llm_client.total_input_tokens == 10
        assert llm_client.total_output_tokens == 5
        assert llm_client.call_count == 1

    async def test_token_tracking_multiple_sequential_calls(self, llm_client):
        """Token counters accumulate across sequential calls."""
        for _ in range(3):
            await llm_client.query("test")
        assert llm_client.total_input_tokens == 30
        assert llm_client.total_output_tokens == 15
        assert llm_client.call_count == 3

    async def test_concurrent_queries_run_in_parallel(self, llm_client):
        """Multiple concurrent query() calls actually overlap — not serialized.

        We simulate each LLM call taking 0.1s. If 5 calls run truly in
        parallel, total wall-clock time should be ~0.1s, not ~0.5s.
        """
        call_count = 5
        concurrency_tracker: list[int] = []
        active = 0
        lock = asyncio.Lock()

        original_create = llm_client._client.messages.create

        async def _slow_create(**kwargs):
            nonlocal active
            async with lock:
                active += 1
                concurrency_tracker.append(active)
            await asyncio.sleep(0.1)
            async with lock:
                active -= 1
            return _make_fake_response()

        llm_client._client.messages.create = AsyncMock(side_effect=_slow_create)

        start = asyncio.get_event_loop().time()
        results = await asyncio.gather(
            *[llm_client.query(f"prompt-{i}") for i in range(call_count)]
        )
        elapsed = asyncio.get_event_loop().time() - start

        # All calls should complete
        assert len(results) == call_count
        assert all(r.text == "Hello" for r in results)

        # Wall-clock time should be ~0.1s (parallel), not ~0.5s (serial)
        assert elapsed < 0.3, f"Calls appear serialized: {elapsed:.2f}s elapsed"

        # At least 2 calls should have been active simultaneously
        max_concurrent = max(concurrency_tracker)
        assert max_concurrent >= 2, (
            f"Expected concurrent execution, but max concurrency was {max_concurrent}"
        )

    async def test_token_tracking_concurrent_calls(self, llm_client):
        """Token counters are correct even with concurrent calls (lock safety)."""
        call_count = 20

        async def _fast_create(**kwargs):
            await asyncio.sleep(0.01)
            return _make_fake_response(input_tokens=100, output_tokens=50)

        llm_client._client.messages.create = AsyncMock(side_effect=_fast_create)

        await asyncio.gather(
            *[llm_client.query(f"prompt-{i}") for i in range(call_count)]
        )

        assert llm_client.total_input_tokens == 100 * call_count
        assert llm_client.total_output_tokens == 50 * call_count
        assert llm_client.call_count == call_count

        summary = llm_client.token_summary
        assert summary["total_tokens"] == 150 * call_count
        assert summary["calls"] == call_count

    async def test_per_model_usage_tracking(self, llm_client):
        """Per-model usage is tracked correctly across concurrent calls."""
        async def _create(**kwargs):
            await asyncio.sleep(0.01)
            return _make_fake_response(input_tokens=10, output_tokens=5)

        llm_client._client.messages.create = AsyncMock(side_effect=_create)

        await asyncio.gather(
            llm_client.query("p1", model="claude-sonnet-4-20250514"),
            llm_client.query("p2", model="claude-sonnet-4-20250514"),
            llm_client.query("p3", model="claude-haiku-4-20250414"),
        )

        summary = llm_client.token_summary
        per_model = summary["per_model"]

        assert per_model["claude-sonnet-4-20250514"]["calls"] == 2
        assert per_model["claude-sonnet-4-20250514"]["input_tokens"] == 20
        assert per_model["claude-haiku-4-20250414"]["calls"] == 1
        assert per_model["claude-haiku-4-20250414"]["input_tokens"] == 10

    async def test_client_is_async_anthropic(self):
        """Verify the client instantiates AsyncAnthropic, not sync Anthropic."""
        with patch("core.llm.anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value = AsyncMock()
            from core.llm import LLMClient

            client = LLMClient()
            mock_cls.assert_called_once()
