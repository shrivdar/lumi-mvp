"""MCP Server Manager — lifecycle management, health monitoring, auto-discovery.

Manages the pool of MCP server connections. Handles:
- Connecting to configured MCP servers on startup
- Health checking and automatic reconnection
- Dynamic tool discovery and registration with ToolRegistry
- Graceful shutdown
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from core.audit import AuditLogger
from core.exceptions import MCPError
from core.models import MCPServerConfig, MCPTransportType
from core.tool_registry import InMemoryToolRegistry
from integrations.mcp_client import MCPClient

logger = structlog.get_logger(__name__)
audit = AuditLogger("mcp_manager")

# ---------------------------------------------------------------------------
# Default MCP server configurations for Docker Compose deployment
# ---------------------------------------------------------------------------

DEFAULT_MCP_SERVERS: list[MCPServerConfig] = [
    MCPServerConfig(
        name="genomics",
        transport=MCPTransportType.STREAMABLE_HTTP,
        url="http://mcp-genomics:8080",
        timeout_seconds=30,
    ),
    MCPServerConfig(
        name="cancer-genomics",
        transport=MCPTransportType.STREAMABLE_HTTP,
        url="http://mcp-cancer-genomics:8080",
        timeout_seconds=30,
    ),
    MCPServerConfig(
        name="drug-discovery",
        transport=MCPTransportType.STREAMABLE_HTTP,
        url="http://mcp-drug-discovery:8080",
        timeout_seconds=30,
    ),
    MCPServerConfig(
        name="protein-structure",
        transport=MCPTransportType.STREAMABLE_HTTP,
        url="http://mcp-protein-structure:8080",
        timeout_seconds=30,
    ),
    MCPServerConfig(
        name="gene-expression",
        transport=MCPTransportType.STREAMABLE_HTTP,
        url="http://mcp-gene-expression:8080",
        timeout_seconds=30,
    ),
    MCPServerConfig(
        name="single-cell",
        transport=MCPTransportType.STREAMABLE_HTTP,
        url="http://mcp-single-cell:8080",
        timeout_seconds=30,
    ),
    MCPServerConfig(
        name="regulatory-epigenomics",
        transport=MCPTransportType.STREAMABLE_HTTP,
        url="http://mcp-regulatory-epigenomics:8080",
        timeout_seconds=30,
    ),
    MCPServerConfig(
        name="pathway-network",
        transport=MCPTransportType.STREAMABLE_HTTP,
        url="http://mcp-pathway-network:8080",
        timeout_seconds=30,
    ),
    MCPServerConfig(
        name="disease-phenotype",
        transport=MCPTransportType.STREAMABLE_HTTP,
        url="http://mcp-disease-phenotype:8080",
        timeout_seconds=30,
    ),
    MCPServerConfig(
        name="bioinformatics-tools",
        transport=MCPTransportType.STREAMABLE_HTTP,
        url="http://mcp-bioinformatics-tools:8080",
        timeout_seconds=60,
    ),
    MCPServerConfig(
        name="computational-analysis",
        transport=MCPTransportType.STREAMABLE_HTTP,
        url="http://mcp-computational-analysis:8080",
        timeout_seconds=60,
    ),
]

# Category mapping for MCP servers → tool catalog categories
SERVER_CATEGORY_MAP: dict[str, str] = {
    "genomics": "genomics",
    "cancer-genomics": "variant_analysis",
    "drug-discovery": "drug_discovery",
    "protein-structure": "structural_biology",
    "gene-expression": "gene_expression",
    "single-cell": "gene_expression",
    "regulatory-epigenomics": "epigenetics",
    "pathway-network": "pathway_analysis",
    "disease-phenotype": "ontology_annotation",
    "bioinformatics-tools": "computation",
    "computational-analysis": "computation",
}


class MCPServerManager:
    """Manages connections to all MCP tool servers.

    Usage::

        manager = MCPServerManager(registry=tool_registry)
        await manager.start(configs)           # connect to all servers
        result = await manager.call_tool("genomics", "ncbi_gene_search", {"query": "TP53"})
        await manager.shutdown()               # disconnect all
    """

    def __init__(self, registry: InMemoryToolRegistry | None = None) -> None:
        self._registry = registry or InMemoryToolRegistry()
        self._clients: dict[str, MCPClient] = {}
        self._configs: dict[str, MCPServerConfig] = {}
        self._health_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()

    @property
    def connected_servers(self) -> list[str]:
        return [name for name, client in self._clients.items() if client.is_connected]

    @property
    def server_count(self) -> int:
        return len(self._clients)

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def start(
        self,
        configs: list[MCPServerConfig] | None = None,
        *,
        auto_discover: bool = True,
        health_check_interval: int = 60,
    ) -> dict[str, bool]:
        """Connect to all configured MCP servers and discover their tools.

        Returns a dict of server_name → connected status.
        """
        configs = configs if configs is not None else DEFAULT_MCP_SERVERS
        results: dict[str, bool] = {}

        # Connect to all servers concurrently
        connect_tasks = []
        for cfg in configs:
            if not cfg.enabled:
                results[cfg.name] = False
                continue
            self._configs[cfg.name] = cfg
            connect_tasks.append(self._connect_server(cfg))

        outcomes = await asyncio.gather(*connect_tasks, return_exceptions=True)
        for cfg, outcome in zip([c for c in configs if c.enabled], outcomes):
            if isinstance(outcome, Exception):
                logger.warning("mcp_server_connect_failed", server=cfg.name, error=str(outcome))
                results[cfg.name] = False
            else:
                results[cfg.name] = outcome

        # Discover tools from connected servers
        if auto_discover:
            await self._discover_all_tools()

        # Start health check background task
        if health_check_interval > 0:
            self._health_task = asyncio.create_task(
                self._health_check_loop(health_check_interval)
            )

        total = len(results)
        connected = sum(1 for v in results.values() if v)
        audit.log("mcp_manager_started", total=total, connected=connected)
        logger.info("mcp_manager_started", total=total, connected=connected, servers=self.connected_servers)
        return results

    async def _connect_server(self, cfg: MCPServerConfig) -> bool:
        """Connect to a single MCP server with retry."""
        client = MCPClient(cfg)
        for attempt in range(cfg.max_reconnect_attempts):
            try:
                await client.connect()
                self._clients[cfg.name] = client
                self._registry.add_mcp_server(cfg)
                return True
            except MCPError:
                if attempt < cfg.max_reconnect_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
        return False

    # ------------------------------------------------------------------
    # Tool Discovery
    # ------------------------------------------------------------------

    async def _discover_all_tools(self) -> int:
        """Discover tools from all connected servers."""
        total = 0
        for name, client in self._clients.items():
            if not client.is_connected:
                continue
            try:
                manifests = await client.list_tools()
                count = self._registry.discover_mcp_tools(name, manifests)
                total += count
                logger.info("mcp_tools_discovered", server=name, count=count)
            except Exception as exc:
                logger.warning("mcp_discovery_failed", server=name, error=str(exc))
        audit.log("mcp_total_tools_discovered", count=total)
        return total

    async def refresh_tools(self, server_name: str | None = None) -> int:
        """Re-discover tools from one or all servers."""
        if server_name:
            client = self._clients.get(server_name)
            if not client or not client.is_connected:
                return 0
            manifests = await client.list_tools()
            return self._registry.discover_mcp_tools(server_name, manifests)
        return await self._discover_all_tools()

    # ------------------------------------------------------------------
    # Tool Calling
    # ------------------------------------------------------------------

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on a specific MCP server."""
        client = self._clients.get(server_name)
        if not client:
            raise MCPError(
                f"MCP server '{server_name}' not found",
                error_code="MCP_SERVER_NOT_FOUND",
                details={"server": server_name, "available": list(self._clients.keys())},
            )
        if not client.is_connected:
            # Try reconnect
            cfg = self._configs.get(server_name)
            if cfg:
                await self._connect_server(cfg)
                client = self._clients.get(server_name)
                if not client or not client.is_connected:
                    raise MCPError(
                        f"MCP server '{server_name}' is not connected and reconnect failed",
                        error_code="MCP_NOT_CONNECTED",
                    )

        return await client.call_tool(tool_name, arguments)

    async def call_tool_by_name(self, full_tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool by its registry name (e.g. 'mcp_genomics_ncbi_gene_search').

        Parses the server name from the registry entry's mcp_server field.
        """
        entry = self._registry.get_tool(full_tool_name)
        if not entry or not entry.mcp_server:
            raise MCPError(
                f"Tool '{full_tool_name}' not found in MCP registry",
                error_code="MCP_TOOL_NOT_FOUND",
            )
        # The actual tool name on the server (without the mcp_server_ prefix)
        actual_name = full_tool_name
        if entry.mcp_manifest:
            actual_name = entry.mcp_manifest.name
        return await self.call_tool(entry.mcp_server, actual_name, arguments)

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------

    async def _health_check_loop(self, interval: int) -> None:
        """Periodically check server health and reconnect if needed."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=interval)
                break  # shutdown was signaled
            except TimeoutError:
                pass  # timeout = time to check health

            for name, client in list(self._clients.items()):
                if not client.is_connected:
                    cfg = self._configs.get(name)
                    if cfg:
                        logger.info("mcp_reconnecting", server=name)
                        try:
                            await self._connect_server(cfg)
                            # Re-discover tools after reconnect
                            if client.is_connected:
                                manifests = await client.list_tools()
                                self._registry.discover_mcp_tools(name, manifests)
                        except Exception as exc:
                            logger.warning("mcp_reconnect_failed", server=name, error=str(exc))

    async def health_status(self) -> dict[str, dict[str, Any]]:
        """Return health status of all configured servers."""
        status: dict[str, dict[str, Any]] = {}
        for name, client in self._clients.items():
            tool_count = len([
                t for t in self._registry.list_tools()
                if t.mcp_server == name
            ])
            status[name] = {
                "connected": client.is_connected,
                "tool_count": tool_count,
                "url": self._configs.get(name, MCPServerConfig(name=name)).url or "",
            }
        return status

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Disconnect all MCP servers and stop health checks."""
        self._shutdown_event.set()
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        disconnect_tasks = [client.disconnect() for client in self._clients.values()]
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        self._clients.clear()
        audit.log("mcp_manager_shutdown")
        logger.info("mcp_manager_shutdown")
