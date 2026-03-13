"""Custom exception hierarchy for YOHAS."""


class YOHASError(Exception):
    """Base exception for all YOHAS errors."""

    def __init__(self, message: str = "", detail: str | None = None):
        self.message = message
        self.detail = detail
        super().__init__(message)


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
