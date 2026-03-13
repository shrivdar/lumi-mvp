"""Structured audit logging via structlog — JSON format with context propagation."""

from __future__ import annotations

import contextvars
import logging
import time
from typing import Any

import structlog

# ---------------------------------------------------------------------------
# Context vars for request-scoped propagation
# ---------------------------------------------------------------------------
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")
_research_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("research_id", default="")
_agent_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("agent_id", default="")


def set_request_context(
    *,
    request_id: str = "",
    research_id: str = "",
    agent_id: str = "",
) -> None:
    """Set context vars for the current async task / request scope."""
    if request_id:
        _request_id_var.set(request_id)
    if research_id:
        _research_id_var.set(research_id)
    if agent_id:
        _agent_id_var.set(agent_id)


def get_request_context() -> dict[str, str]:
    """Return the current request context dict."""
    return {
        "request_id": _request_id_var.get(),
        "research_id": _research_id_var.get(),
        "agent_id": _agent_id_var.get(),
    }


# ---------------------------------------------------------------------------
# structlog configuration
# ---------------------------------------------------------------------------

def _inject_request_context(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    ctx = get_request_context()
    for key, value in ctx.items():
        if value and key not in event_dict:
            event_dict[key] = value
    return event_dict


def configure_audit_logging(log_level: str = "INFO", json_format: bool = True) -> None:
    """Configure structlog for structured JSON audit logging. Call once at startup."""
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _inject_request_context,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(log_level)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_audit_logger(component: str) -> structlog.stdlib.BoundLogger:
    """Get a bound logger for a specific component."""
    return structlog.get_logger(component=component)


# ---------------------------------------------------------------------------
# AuditLogger — convenience wrapper for common audit events
# ---------------------------------------------------------------------------

class AuditLogger:
    """Structured audit logger that auto-injects context fields."""

    def __init__(self, component: str) -> None:
        self._log = get_audit_logger(component)

    def log(self, event: str, *, duration_ms: int | None = None, **kwargs: Any) -> None:
        bound = self._log.bind(**kwargs)
        if duration_ms is not None:
            bound = bound.bind(duration_ms=duration_ms)
        bound.info(event)

    def tool_call(self, tool_name: str, agent_id: str, **kwargs: Any) -> None:
        self._log.info("tool_call", tool=tool_name, agent_id=agent_id, **kwargs)

    def tool_result(self, tool_name: str, agent_id: str, success: bool, **kwargs: Any) -> None:
        level = "info" if success else "warning"
        getattr(self._log, level)("tool_result", tool=tool_name, agent_id=agent_id, success=success, **kwargs)

    def llm_call(self, model: str, prompt_tokens: int, completion_tokens: int, **kwargs: Any) -> None:
        self._log.info(
            "llm_call", model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, **kwargs
        )

    def kg_mutation(self, operation: str, agent_id: str, hypothesis_branch: str, **kwargs: Any) -> None:
        self._log.info(
            "kg_mutation", operation=operation, agent_id=agent_id, hypothesis_branch=hypothesis_branch, **kwargs
        )

    def agent_event(self, agent_id: str, event: str, **kwargs: Any) -> None:
        self._log.info("agent_event", agent_id=agent_id, event=event, **kwargs)

    def research_event(self, session_id: str, event: str, **kwargs: Any) -> None:
        self._log.info("research_event", session_id=session_id, event=event, **kwargs)

    def falsification(self, agent_id: str, edge_id: str, result: str, **kwargs: Any) -> None:
        self._log.info("falsification", agent_id=agent_id, edge_id=edge_id, result=result, **kwargs)

    def error(self, error_type: str, **kwargs: Any) -> None:
        self._log.error("error", error_type=error_type, **kwargs)

    def warn(self, event: str, **kwargs: Any) -> None:
        self._log.warning(event, **kwargs)


# ---------------------------------------------------------------------------
# Timer context manager for measuring durations
# ---------------------------------------------------------------------------

class Timer:
    """Simple timer that records elapsed milliseconds."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: int = 0

    def __enter__(self) -> Timer:
        self._start = time.monotonic()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = int((time.monotonic() - self._start) * 1000)
