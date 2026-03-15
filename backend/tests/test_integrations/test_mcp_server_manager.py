"""Tests for MCPServerManager — lifecycle, discovery, tool calling."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from core.exceptions import MCPError
from core.models import MCPServerConfig, MCPTransportType
from core.tool_registry import InMemoryToolRegistry
from integrations.mcp_server_manager import MCPServerManager


def _test_configs() -> list[MCPServerConfig]:
    return [
        MCPServerConfig(
            name="test-genomics",
            transport=MCPTransportType.STREAMABLE_HTTP,
            url="https://mcp-genomics.test.local",
            timeout_seconds=5,
            max_reconnect_attempts=1,
        ),
        MCPServerConfig(
            name="test-drugs",
            transport=MCPTransportType.STREAMABLE_HTTP,
            url="https://mcp-drugs.test.local",
            timeout_seconds=5,
            max_reconnect_attempts=1,
        ),
    ]


def _mock_rpc_tools(tools: list[dict]) -> httpx.Response:
    return httpx.Response(200, json={
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"tools": tools},
    })


def _mock_rpc_call(result: dict) -> httpx.Response:
    return httpx.Response(200, json={
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": json.dumps(result)}]},
    })


class TestMCPServerManagerStartup:
    @respx.mock
    @pytest.mark.asyncio
    async def test_start_connects_to_servers(self) -> None:
        # Mock both servers
        respx.get("https://mcp-genomics.test.local/health").mock(return_value=httpx.Response(200))
        respx.get("https://mcp-drugs.test.local/health").mock(return_value=httpx.Response(200))

        # Mock tool discovery for both
        respx.post("https://mcp-genomics.test.local/rpc").mock(
            return_value=_mock_rpc_tools([
                {"name": "ncbi_gene_search", "description": "Search genes", "inputSchema": {
                    "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
                }},
            ])
        )
        respx.post("https://mcp-drugs.test.local/rpc").mock(
            return_value=_mock_rpc_tools([
                {"name": "pubchem_search", "description": "Search PubChem", "inputSchema": {
                    "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
                }},
            ])
        )

        registry = InMemoryToolRegistry()
        manager = MCPServerManager(registry=registry)
        results = await manager.start(_test_configs(), health_check_interval=0)

        assert results["test-genomics"] is True
        assert results["test-drugs"] is True
        assert len(manager.connected_servers) == 2
        assert registry.tool_count == 2

        await manager.shutdown()

    @respx.mock
    @pytest.mark.asyncio
    async def test_start_tolerates_unavailable_server(self) -> None:
        # First server works
        respx.get("https://mcp-genomics.test.local/health").mock(return_value=httpx.Response(200))
        respx.post("https://mcp-genomics.test.local/rpc").mock(
            return_value=_mock_rpc_tools([{"name": "gene_search", "description": "Search", "inputSchema": {}}])
        )
        # Second server: health check fails but connection still succeeds
        # (health check is optional in MCPClient), but RPC discovery fails
        respx.get("https://mcp-drugs.test.local/health").mock(side_effect=httpx.ConnectError("refused"))
        respx.post("https://mcp-drugs.test.local/rpc").mock(side_effect=httpx.ConnectError("refused"))

        configs = [
            MCPServerConfig(
                name="test-genomics",
                transport=MCPTransportType.STREAMABLE_HTTP,
                url="https://mcp-genomics.test.local",
                timeout_seconds=5,
                max_reconnect_attempts=1,
            ),
            MCPServerConfig(
                name="test-drugs",
                transport=MCPTransportType.STREAMABLE_HTTP,
                url="https://mcp-drugs.test.local",
                timeout_seconds=2,
                max_reconnect_attempts=1,
            ),
        ]

        registry = InMemoryToolRegistry()
        manager = MCPServerManager(registry=registry)
        results = await manager.start(configs, health_check_interval=0)

        # Both connections succeed (health check is optional), but only
        # genomics discovers tools. The manager tolerates discovery failure.
        assert results["test-genomics"] is True
        assert results["test-drugs"] is True  # Connection succeeded
        # But only genomics tools are in the registry (drugs discovery failed)
        assert registry.tool_count == 1  # Only gene_search from genomics

        await manager.shutdown()

    @respx.mock
    @pytest.mark.asyncio
    async def test_disabled_server_skipped(self) -> None:
        configs = [
            MCPServerConfig(
                name="disabled-server",
                transport=MCPTransportType.STREAMABLE_HTTP,
                url="https://disabled.test.local",
                enabled=False,
            ),
        ]
        manager = MCPServerManager()
        results = await manager.start(configs, health_check_interval=0)
        assert results["disabled-server"] is False
        assert len(manager.connected_servers) == 0
        await manager.shutdown()


class TestMCPServerManagerToolCalling:
    @respx.mock
    @pytest.mark.asyncio
    async def test_call_tool(self) -> None:
        respx.get("https://mcp-genomics.test.local/health").mock(return_value=httpx.Response(200))

        # First call = tool discovery, second call = actual tool call
        route = respx.post("https://mcp-genomics.test.local/rpc")
        route.side_effect = [
            _mock_rpc_tools([{"name": "ncbi_gene_search", "description": "Search", "inputSchema": {}}]),
            _mock_rpc_call({"genes": [{"symbol": "TP53", "gene_id": "7157"}]}),
        ]

        configs = [MCPServerConfig(
            name="test-genomics",
            transport=MCPTransportType.STREAMABLE_HTTP,
            url="https://mcp-genomics.test.local",
            timeout_seconds=5,
            max_reconnect_attempts=1,
        )]

        registry = InMemoryToolRegistry()
        manager = MCPServerManager(registry=registry)
        await manager.start(configs, health_check_interval=0)

        result = await manager.call_tool("test-genomics", "ncbi_gene_search", {"query": "TP53"})
        assert result["genes"][0]["symbol"] == "TP53"

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_call_tool_unknown_server(self) -> None:
        manager = MCPServerManager()
        # Pass empty list explicitly to avoid default servers
        results = await manager.start([], health_check_interval=0)
        assert len(results) == 0
        with pytest.raises(MCPError, match="not found"):
            await manager.call_tool("nonexistent", "tool", {})
        await manager.shutdown()


class TestMCPServerManagerHealth:
    @respx.mock
    @pytest.mark.asyncio
    async def test_health_status(self) -> None:
        respx.get("https://mcp-genomics.test.local/health").mock(return_value=httpx.Response(200))
        respx.post("https://mcp-genomics.test.local/rpc").mock(
            return_value=_mock_rpc_tools([])
        )

        configs = [MCPServerConfig(
            name="test-genomics",
            transport=MCPTransportType.STREAMABLE_HTTP,
            url="https://mcp-genomics.test.local",
            timeout_seconds=5,
            max_reconnect_attempts=1,
        )]

        manager = MCPServerManager()
        await manager.start(configs, health_check_interval=0)

        status = await manager.health_status()
        assert "test-genomics" in status
        assert status["test-genomics"]["connected"] is True

        await manager.shutdown()
