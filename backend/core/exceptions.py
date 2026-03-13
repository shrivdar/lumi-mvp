"""Custom exception hierarchy for YOHAS."""

from __future__ import annotations

from typing import Any


class YOHASError(Exception):
    """Base exception for all YOHAS errors."""

    def __init__(
        self,
        message: str = "",
        *,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details: dict[str, Any] = details or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": type(self).__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class ToolError(YOHASError):
    """Raised when an external tool/API call fails."""


class AgentError(YOHASError):
    """Raised when an agent encounters an unrecoverable error."""


class OrchestrationError(YOHASError):
    """Raised when the orchestrator/MCTS encounters an error."""


class GraphError(YOHASError):
    """Raised when a knowledge graph operation fails."""


class LLMError(YOHASError):
    """Raised when an LLM call fails."""


class MCPError(YOHASError):
    """Raised when an MCP server connection or tool call fails."""
