"""ToolRegistry implementation — unified interface for native, MCP, and container tools."""

from __future__ import annotations

import structlog

from core.models import (
    MCPServerConfig,
    MCPToolManifest,
    ToolRegistryEntry,
    ToolSourceType,
)

logger = structlog.get_logger(__name__)


class InMemoryToolRegistry:
    """In-memory tool registry supporting native tools, MCP servers, and container tools.

    Provides a unified lookup interface regardless of tool source. MCP tool discovery
    and container tool scanning are triggered via ``discover()``.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolRegistryEntry] = {}
        self._mcp_configs: list[MCPServerConfig] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, entry: ToolRegistryEntry) -> None:
        """Register a tool. Overwrites any existing entry with the same name."""
        self._tools[entry.name] = entry
        logger.info("tool_registered", tool=entry.name, source=entry.source_type)

    def unregister(self, tool_name: str) -> None:
        removed = self._tools.pop(tool_name, None)
        if removed:
            logger.info("tool_unregistered", tool=tool_name)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_tool(self, tool_name: str) -> ToolRegistryEntry | None:
        return self._tools.get(tool_name)

    def list_tools(
        self,
        category: str | None = None,
        source_type: ToolSourceType | None = None,
        enabled_only: bool = True,
    ) -> list[ToolRegistryEntry]:
        entries = self._tools.values()
        if enabled_only:
            entries = [e for e in entries if e.enabled]
        if category:
            entries = [e for e in entries if e.category == category]
        if source_type:
            entries = [e for e in entries if e.source_type == source_type]
        return sorted(entries, key=lambda e: e.name)

    # ------------------------------------------------------------------
    # MCP Discovery
    # ------------------------------------------------------------------

    def add_mcp_server(self, config: MCPServerConfig) -> None:
        """Register an MCP server config for later discovery."""
        self._mcp_configs.append(config)

    def discover_mcp_tools(self, server_name: str, manifests: list[MCPToolManifest]) -> int:
        """Register tools discovered from an MCP server.

        Called after connecting to a server and receiving its tool list.
        Returns the number of tools registered.
        """
        count = 0
        for manifest in manifests:
            entry = ToolRegistryEntry(
                name=f"mcp_{server_name}_{manifest.name}",
                description=manifest.description,
                source_type=ToolSourceType.MCP,
                category="mcp",
                mcp_server=server_name,
                mcp_manifest=manifest,
            )
            self.register(entry)
            count += 1
        logger.info("mcp_discovery_complete", server=server_name, tools_found=count)
        return count

    # ------------------------------------------------------------------
    # Container Discovery
    # ------------------------------------------------------------------

    def register_container_tool(self, entry: ToolRegistryEntry) -> None:
        """Register a container-based tool."""
        if entry.source_type != ToolSourceType.CONTAINER:
            entry.source_type = ToolSourceType.CONTAINER
        self.register(entry)

    # ------------------------------------------------------------------
    # Bulk discovery
    # ------------------------------------------------------------------

    def discover(self) -> list[ToolRegistryEntry]:
        """Return all currently registered tools.

        In production, this would also trigger MCP server connections
        and container image scanning. For now it returns what's registered.
        """
        return self.list_tools(enabled_only=False)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    @property
    def mcp_server_count(self) -> int:
        return len(self._mcp_configs)

    def stats(self) -> dict[str, int]:
        tools = list(self._tools.values())
        return {
            "total": len(tools),
            "native": sum(1 for t in tools if t.source_type == ToolSourceType.NATIVE),
            "mcp": sum(1 for t in tools if t.source_type == ToolSourceType.MCP),
            "container": sum(1 for t in tools if t.source_type == ToolSourceType.CONTAINER),
            "enabled": sum(1 for t in tools if t.enabled),
            "mcp_servers": len(self._mcp_configs),
        }
