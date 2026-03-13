"""Tests for MCP client — connection, tool discovery, tool calling."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from core.exceptions import MCPError
from core.models import MCPServerConfig, MCPTransportType
from integrations.mcp_client import MCPClient


@pytest.fixture
def http_config() -> MCPServerConfig:
    return MCPServerConfig(
        name="test-server",
        transport=MCPTransportType.STREAMABLE_HTTP,
        url="https://mcp.test.local",
        timeout_seconds=5,
    )


class TestMCPClientHTTP:
    @respx.mock
    @pytest.mark.asyncio
    async def test_connect_and_list_tools(self, http_config: MCPServerConfig) -> None:
        # Mock health endpoint
        respx.get("https://mcp.test.local/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        # Mock tools/list RPC
        respx.post("https://mcp.test.local/rpc").mock(
            return_value=httpx.Response(200, json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "tools": [
                        {
                            "name": "blast_search",
                            "description": "Run BLAST sequence search",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "sequence": {"type": "string", "description": "Input sequence"},
                                    "database": {"type": "string", "description": "BLAST database"},
                                },
                                "required": ["sequence"],
                            },
                        }
                    ]
                },
            })
        )

        client = MCPClient(http_config)
        await client.connect()
        assert client.is_connected

        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "blast_search"
        assert len(tools[0].parameters) == 2
        assert tools[0].parameters[0].required is True  # "sequence"
        assert tools[0].parameters[1].required is False  # "database"

        await client.disconnect()
        assert not client.is_connected

    @respx.mock
    @pytest.mark.asyncio
    async def test_call_tool(self, http_config: MCPServerConfig) -> None:
        respx.get("https://mcp.test.local/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        respx.post("https://mcp.test.local/rpc").mock(
            return_value=httpx.Response(200, json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"hits": 5})}]
                },
            })
        )

        client = MCPClient(http_config)
        await client.connect()
        result = await client.call_tool("blast_search", {"sequence": "MALWMR"})
        assert result == {"hits": 5}
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self, http_config: MCPServerConfig) -> None:
        client = MCPClient(http_config)
        with pytest.raises(MCPError, match="Not connected"):
            await client.call_tool("blast_search", {})


class TestMCPClientStdio:
    @pytest.mark.asyncio
    async def test_stdio_requires_command(self) -> None:
        config = MCPServerConfig(
            name="bad",
            transport=MCPTransportType.STDIO,
            command=None,
        )
        client = MCPClient(config)
        with pytest.raises(MCPError, match="Failed to connect"):
            await client.connect()


class TestMCPClientRPCError:
    @respx.mock
    @pytest.mark.asyncio
    async def test_rpc_error_raises(self, http_config: MCPServerConfig) -> None:
        respx.get("https://mcp.test.local/health").mock(
            return_value=httpx.Response(200)
        )
        respx.post("https://mcp.test.local/rpc").mock(
            return_value=httpx.Response(200, json={
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32601, "message": "Method not found"},
            })
        )

        client = MCPClient(http_config)
        await client.connect()
        with pytest.raises(MCPError, match="Method not found"):
            await client.list_tools()
        await client.disconnect()
