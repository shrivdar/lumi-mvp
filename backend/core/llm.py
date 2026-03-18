"""Anthropic SDK wrapper — single LLM entry-point for the entire system."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import defaultdict
from typing import Any, NamedTuple

import anthropic
import structlog

from core.config import settings

logger = structlog.get_logger(__name__)


class LLMResponse(NamedTuple):
    """Return type for LLMClient.query() — text plus per-call token usage."""
    text: str
    call_tokens: int  # input_tokens + output_tokens for this single call


class LLMClient:
    """Thin wrapper around the Anthropic SDK with KG context injection,
    token tracking, audit logging, and structured output parsing."""

    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.call_count: int = 0
        self._per_model_usage: dict[str, dict[str, int]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        )
        self._token_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Main query
    # ------------------------------------------------------------------

    async def query(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        kg_context: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        research_id: str = "",
        agent_id: str = "",
    ) -> LLMResponse:
        """Send a prompt to Claude and return text + per-call token count.

        Returns an ``LLMResponse`` named-tuple.  Because it inherits from
        ``tuple`` and the first element is ``text``, existing callers that
        treat the return value as ``str`` will see a deprecation-free
        migration path — but the *recommended* usage is::

            resp = await llm.query(...)
            text = resp.text          # the generated text
            tokens = resp.call_tokens # input + output tokens for this call
        """
        resolved_model = model or settings.llm_model
        max_tokens = max_tokens or settings.llm_max_tokens

        system = self._build_system_prompt(system_prompt, kg_context)

        log = logger.bind(research_id=research_id, agent_id=agent_id)
        log.info("llm.call_start", model=resolved_model, prompt_len=len(prompt))

        retryable = (
            anthropic.InternalServerError,   # 500
            anthropic.APIConnectionError,    # network failures
            anthropic.RateLimitError,        # 429
        )
        backoff_delays = (1.0, 2.0, 4.0)    # 3 retries with exponential backoff

        start = time.monotonic()
        last_exc: Exception | None = None
        for attempt in range(1 + len(backoff_delays)):
            try:
                response = await self._client.messages.create(
                    model=resolved_model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                break  # success
            except retryable as exc:
                last_exc = exc
                # Also catch 529 (overloaded) which arrives as APIStatusError
                if attempt < len(backoff_delays):
                    delay = backoff_delays[attempt]
                    log.warning(
                        "llm.call_retry",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)
                    continue
                # Exhausted retries — fall through to raise
            except anthropic.APIStatusError as exc:
                # Retry on 529 (overloaded), but not on 400/401/other client errors
                if exc.status_code == 529 and attempt < len(backoff_delays):
                    last_exc = exc
                    delay = backoff_delays[attempt]
                    log.warning(
                        "llm.call_retry",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)
                    continue
                last_exc = exc
                break  # non-retryable status — stop immediately
            except anthropic.APIError as exc:
                last_exc = exc
                break  # non-retryable — stop immediately
        else:
            # All retries exhausted
            from core.exceptions import LLMError

            log.error("llm.call_error", error=str(last_exc), retries_exhausted=True)
            raise LLMError(
                str(last_exc),
                error_code="LLM_API_ERROR",
                details={"model": resolved_model, "retries": len(backoff_delays)},
            ) from last_exc

        if last_exc is not None:
            # Non-retryable error on first attempt
            from core.exceptions import LLMError

            log.error("llm.call_error", error=str(last_exc))
            raise LLMError(
                str(last_exc),
                error_code="LLM_API_ERROR",
                details={"model": resolved_model},
            ) from last_exc

        duration_ms = int((time.monotonic() - start) * 1000)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        async with self._token_lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.call_count += 1

            model_stats = self._per_model_usage[resolved_model]
            model_stats["input_tokens"] += input_tokens
            model_stats["output_tokens"] += output_tokens
            model_stats["calls"] += 1

        text = response.content[0].text if response.content else ""
        call_tokens = input_tokens + output_tokens

        log.info(
            "llm.call_end",
            model=resolved_model,
            duration_ms=duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return LLMResponse(text=text, call_tokens=call_tokens)

    # ------------------------------------------------------------------
    # Structured output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_json(text: str) -> Any:
        """Extract the first JSON object or array from *text*.

        Handles common patterns: ```json ... ```, bare JSON, etc.
        """
        # Try fenced code block first
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())

        # Fallback: find first { or [
        for i, ch in enumerate(text):
            if ch in ("{", "["):
                try:
                    return json.loads(text[i:])
                except json.JSONDecodeError:
                    continue

        raise ValueError("No valid JSON found in LLM response")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        base_prompt: str,
        kg_context: dict[str, Any] | None,
    ) -> str:
        parts: list[str] = []
        if base_prompt:
            parts.append(base_prompt)
        if kg_context:
            kg_json = json.dumps(kg_context, indent=2, default=str)
            # Cap KG context to prevent system prompt from consuming the token budget
            max_kg_chars = 4000
            if len(kg_json) > max_kg_chars:
                kg_json = kg_json[:max_kg_chars] + "\n... (truncated)"
            parts.append(
                "## Knowledge Graph Context (current state)\n" + kg_json
            )
        return "\n\n".join(parts) if parts else "You are a helpful biomedical research assistant."

    @property
    def token_summary(self) -> dict[str, Any]:
        return {
            "calls": self.call_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "per_model": dict(self._per_model_usage),
        }
