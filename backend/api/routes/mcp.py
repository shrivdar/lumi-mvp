"""MCP API routes — server health, tool discovery, tool invocation."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.deps import get_config
from core.tool_registry import InMemoryToolRegistry
from integrations.mcp_server_manager import MCPServerManager
from integrations.mcp_tool_retriever import MCPToolRetriever

router = APIRouter(prefix="/mcp", tags=["mcp"])

# ---------------------------------------------------------------------------
# Module-level singletons (initialized on first use)
# ---------------------------------------------------------------------------

_registry: InMemoryToolRegistry | None = None
_manager: MCPServerManager | None = None
_retriever: MCPToolRetriever | None = None


def _get_registry() -> InMemoryToolRegistry:
    global _registry
    if _registry is None:
        _registry = InMemoryToolRegistry()
    return _registry


def _get_manager() -> MCPServerManager:
    global _manager
    if _manager is None:
        _manager = MCPServerManager(registry=_get_registry())
    return _manager


def _get_retriever() -> MCPToolRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MCPToolRetriever(registry=_get_registry())
    return _retriever


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class MCPToolCallRequest(BaseModel):
    server_name: str
    tool_name: str
    arguments: dict[str, Any] = {}


class MCPToolCallByNameRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = {}


class MCPToolRetrieveRequest(BaseModel):
    query: str
    max_tools: int = 20
    agent_type: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
async def mcp_health() -> dict[str, Any]:
    """Health status of all MCP servers."""
    manager = _get_manager()
    status = await manager.health_status()
    return {
        "status": "ok" if manager.connected_servers else "degraded",
        "connected_count": len(manager.connected_servers),
        "total_count": manager.server_count,
        "servers": status,
    }


@router.post("/start")
async def mcp_start() -> dict[str, Any]:
    """Start MCP server connections (called on app startup)."""
    manager = _get_manager()
    config = get_config()
    configs = config.mcp_servers if config.mcp_servers else None
    results = await manager.start(configs, auto_discover=config.mcp_auto_discover)
    return {
        "started": True,
        "results": results,
        "connected": manager.connected_servers,
    }


@router.post("/shutdown")
async def mcp_shutdown() -> dict[str, str]:
    """Shutdown all MCP server connections."""
    manager = _get_manager()
    await manager.shutdown()
    return {"status": "shutdown"}


@router.get("/tools")
async def mcp_list_tools(
    category: str | None = None,
    server: str | None = None,
) -> dict[str, Any]:
    """List all discovered MCP tools."""
    registry = _get_registry()
    from core.models import ToolSourceType

    tools = registry.list_tools(category=category, source_type=ToolSourceType.MCP)
    if server:
        tools = [t for t in tools if t.mcp_server == server]

    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "server": t.mcp_server,
                "capabilities": t.capabilities,
            }
            for t in tools
        ],
        "count": len(tools),
    }


@router.post("/tools/call")
async def mcp_call_tool(req: MCPToolCallRequest) -> dict[str, Any]:
    """Call a tool on a specific MCP server."""
    manager = _get_manager()
    try:
        result = await manager.call_tool(req.server_name, req.tool_name, req.arguments)
        return {"result": result, "server": req.server_name, "tool": req.tool_name}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/tools/call-by-name")
async def mcp_call_tool_by_name(req: MCPToolCallByNameRequest) -> dict[str, Any]:
    """Call a tool by its registry name."""
    manager = _get_manager()
    try:
        result = await manager.call_tool_by_name(req.tool_name, req.arguments)
        return {"result": result, "tool": req.tool_name}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/tools/retrieve")
async def mcp_retrieve_tools(req: MCPToolRetrieveRequest) -> dict[str, Any]:
    """Retrieve relevant tools for a query or agent type."""
    retriever = _get_retriever()
    if req.agent_type:
        tools = retriever.retrieve_for_agent(req.agent_type, req.query, max_tools=req.max_tools)
    else:
        tools = retriever.retrieve_for_query(req.query, max_tools=req.max_tools)

    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "source_type": t.source_type.value,
                "server": t.mcp_server,
            }
            for t in tools
        ],
        "count": len(tools),
        "query": req.query,
    }


@router.post("/tools/refresh")
async def mcp_refresh_tools(server_name: str | None = None) -> dict[str, Any]:
    """Re-discover tools from MCP servers."""
    manager = _get_manager()
    count = await manager.refresh_tools(server_name)
    return {"refreshed": True, "tools_discovered": count}


@router.get("/stats")
async def mcp_stats() -> dict[str, Any]:
    """Registry statistics."""
    registry = _get_registry()
    return registry.stats()
