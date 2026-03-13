"""FastAPI dependency injection providers."""

from __future__ import annotations

from typing import Any

from core.config import Settings, settings
from world_model.knowledge_graph import InMemoryKnowledgeGraph

# ---------------------------------------------------------------------------
# In-memory stores (replaced by DB + Redis in production)
# ---------------------------------------------------------------------------

# Session store: research_id → ResearchSession
_sessions: dict[str, Any] = {}

# KG store: research_id → InMemoryKnowledgeGraph
_knowledge_graphs: dict[str, InMemoryKnowledgeGraph] = {}

# Orchestrator store: research_id → ResearchOrchestrator
_orchestrators: dict[str, Any] = {}

# Agent results store: research_id → list[AgentResult]
_agent_results: dict[str, list[Any]] = {}


def get_config() -> Settings:
    """Return the global Settings instance."""
    return settings


def get_sessions() -> dict[str, Any]:
    return _sessions


def get_knowledge_graphs() -> dict[str, InMemoryKnowledgeGraph]:
    return _knowledge_graphs


def get_orchestrators() -> dict[str, Any]:
    return _orchestrators


def get_agent_results() -> dict[str, list[Any]]:
    return _agent_results
