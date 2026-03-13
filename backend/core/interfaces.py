"""Core interfaces — ABCs and Protocols for YOHAS components."""

from __future__ import annotations

import abc
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from core.models import (
        AgentResult,
        AgentTask,
        AgentTemplate,
        EvidenceSource,
        FalsificationResult,
        KGEdge,
        KGNode,
        MCPServerConfig,
        MCPToolManifest,
        NodeType,
        ToolRegistryEntry,
        UncertaintyVector,
    )


# ---------------------------------------------------------------------------
# KnowledgeGraph Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class KnowledgeGraph(Protocol):
    """Protocol defining the knowledge graph interface."""

    # -- CRUD --
    def add_node(self, node: KGNode) -> str: ...
    def add_edge(self, edge: KGEdge) -> str: ...
    def get_node(self, node_id: str) -> KGNode | None: ...
    def get_node_by_name(self, name: str, type: NodeType | None = None) -> KGNode | None: ...
    def get_edge(self, edge_id: str) -> KGEdge | None: ...
    def get_edges_from(self, source_id: str) -> list[KGEdge]: ...
    def get_edges_to(self, target_id: str) -> list[KGEdge]: ...
    def get_edges_between(self, source_id: str, target_id: str) -> list[KGEdge]: ...
    def update_node(self, node_id: str, updates: dict[str, Any]) -> None: ...
    def update_edge_confidence(self, edge_id: str, new_confidence: float, evidence: EvidenceSource) -> None: ...
    def mark_edge_falsified(self, edge_id: str, evidence: list[EvidenceSource]) -> None: ...

    # -- Queries --
    def get_subgraph(self, center_id: str, hops: int = 2) -> dict[str, Any]: ...
    def get_contradictions(self, edge: KGEdge) -> list[KGEdge]: ...
    def get_recent_edges(self, n: int = 20) -> list[KGEdge]: ...
    def get_edges_by_hypothesis(self, branch_id: str) -> list[KGEdge]: ...
    def get_weakest_edges(self, n: int = 10) -> list[KGEdge]: ...
    def get_orphan_nodes(self) -> list[KGNode]: ...
    def shortest_path(self, source_id: str, target_id: str) -> list[str] | None: ...
    def get_upstream(self, node_id: str, depth: int = 3) -> dict[str, Any]: ...
    def get_downstream(self, node_id: str, depth: int = 3) -> dict[str, Any]: ...

    # -- Stats --
    def node_count(self) -> int: ...
    def edge_count(self) -> int: ...
    def avg_confidence(self) -> float: ...
    def edges_added_since(self, timestamp: datetime) -> int: ...

    # -- Serialization --
    def to_cytoscape(self) -> dict[str, Any]: ...
    def to_json(self) -> dict[str, Any]: ...
    def to_markdown_summary(self) -> str: ...

    # -- Persistence --
    async def save(self, session_id: str) -> None: ...
    async def load(self, session_id: str) -> None: ...


# ---------------------------------------------------------------------------
# YamiInterface Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class YamiInterface(Protocol):
    """Protocol for the Yami/ESM protein modelling interface."""

    async def get_logits(self, sequence: str) -> Any: ...
    async def get_embeddings(self, sequence: str) -> np.ndarray: ...
    async def predict_structure(self, sequence: str) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# BaseTool ABC
# ---------------------------------------------------------------------------

class BaseTool(abc.ABC):
    """Abstract base for all external API tools (caching, rate-limit, retry)."""

    tool_id: str
    name: str
    description: str
    rate_limit: float  # requests per second
    cache_ttl: int  # seconds

    @abc.abstractmethod
    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool query. Subclasses implement specific API calls."""

    # Hooks for the base framework (implementations in integrations/base_tool.py)
    async def _cache_get(self, key: str) -> Any | None:
        return None

    async def _cache_set(self, key: str, value: Any, ttl: int | None = None) -> None:
        pass

    async def _rate_limit_acquire(self) -> None:
        pass


# ---------------------------------------------------------------------------
# MCPToolServer Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class MCPToolServer(Protocol):
    """Protocol for connecting to an MCP tool server."""

    async def connect(self, config: MCPServerConfig) -> None:
        """Establish connection to the MCP server."""
        ...

    async def list_tools(self) -> list[MCPToolManifest]:
        """List all tools available on this server."""
        ...

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        ...

    @property
    def is_connected(self) -> bool:
        """Whether the server connection is active."""
        ...


# ---------------------------------------------------------------------------
# ToolRegistry Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ToolRegistry(Protocol):
    """Protocol for the unified tool registry.

    Supports native tools, MCP-connected tools, and container-based tools.
    """

    def register(self, entry: ToolRegistryEntry) -> None:
        """Register a tool in the registry."""
        ...

    def unregister(self, tool_name: str) -> None:
        """Remove a tool from the registry."""
        ...

    def get_tool(self, tool_name: str) -> ToolRegistryEntry | None:
        """Get a tool entry by name."""
        ...

    def list_tools(self, category: str | None = None, enabled_only: bool = True) -> list[ToolRegistryEntry]:
        """List tools, optionally filtered by category."""
        ...

    def discover(self) -> list[ToolRegistryEntry]:
        """Discover available tools from MCP servers and containers."""
        ...


# ---------------------------------------------------------------------------
# BaseAgent ABC
# ---------------------------------------------------------------------------

class BaseAgent(abc.ABC):
    """Abstract base for all YOHAS research agents."""

    agent_id: str
    agent_type: str
    template: AgentTemplate
    kg: KnowledgeGraph
    tools: dict[str, BaseTool]

    @abc.abstractmethod
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute the assigned task and return results."""

    async def falsify(self, edges: list[KGEdge]) -> list[FalsificationResult]:
        """Default falsification: for each edge, ask LLM to find counter-evidence."""
        return []

    def get_uncertainty(self) -> UncertaintyVector:
        """Compute the uncertainty vector from agent state."""
        from core.models import UncertaintyVector

        return UncertaintyVector()

    async def query_llm(self, prompt: str, *, kg_context: dict[str, Any] | None = None) -> str:
        """Wraps LLM call with KG subgraph injection, audit logging, token tracking."""
        return ""

    async def query_yami(self, method: str, **kwargs: Any) -> dict[str, Any]:
        """Wraps Yami call with audit logging, error handling, caching."""
        return {}

    def write_node(self, node: KGNode) -> str:
        """Write node to KG with audit trail."""
        return self.kg.add_node(node)

    def write_edge(self, edge: KGEdge) -> str:
        """Write edge to KG with audit trail, auto-triggers contradiction check."""
        return self.kg.add_edge(edge)
