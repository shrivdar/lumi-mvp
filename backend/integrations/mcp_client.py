"""MCP protocol client — connect to MCP servers, discover tools, call tools.

Supports stdio and SSE transports. Auto-discovers running MCP servers when enabled.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import structlog

from core.audit import AuditLogger
from core.exceptions import MCPError
from core.models import MCPServerConfig, MCPToolManifest, MCPToolParameter, MCPTransportType

logger = structlog.get_logger(__name__)
audit = AuditLogger("mcp_client")


class MCPClient:
    """Client that speaks the MCP protocol to external tool servers.

    Supports:
    - stdio transport: launch subprocess, communicate via stdin/stdout JSON-RPC
    - SSE transport: connect to HTTP endpoint, stream events
    - Streamable HTTP: standard HTTP POST for tool calls
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._connected = False
        self._tools: list[MCPToolManifest] = []
        self._process: asyncio.subprocess.Process | None = None
        self._http: httpx.AsyncClient | None = None
        self._request_id = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def server_name(self) -> str:
        return self._config.name

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish connection based on transport type."""
        try:
            if self._config.transport == MCPTransportType.STDIO:
                await self._connect_stdio()
            elif self._config.transport in (MCPTransportType.SSE, MCPTransportType.STREAMABLE_HTTP):
                await self._connect_http()
            self._connected = True
            audit.log("mcp_connected", server=self._config.name, transport=self._config.transport)
        except Exception as exc:
            raise MCPError(
                f"Failed to connect to MCP server '{self._config.name}'",
                error_code="MCP_CONNECT_FAILED",
                details={"server": self._config.name, "error": str(exc)},
            ) from exc

    async def _connect_stdio(self) -> None:
        if not self._config.command:
            raise MCPError("stdio transport requires a command", error_code="MCP_CONFIG_ERROR")
        env = {**self._config.env} if self._config.env else None
        self._process = await asyncio.create_subprocess_exec(
            self._config.command,
            *self._config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

    async def _connect_http(self) -> None:
        if not self._config.url:
            raise MCPError("SSE/HTTP transport requires a url", error_code="MCP_CONFIG_ERROR")
        self._http = httpx.AsyncClient(
            base_url=self._config.url,
            headers=self._config.headers,
            timeout=httpx.Timeout(self._config.timeout_seconds),
        )
        # Ping health endpoint
        try:
            resp = await self._http.get("/health")
            resp.raise_for_status()
        except Exception:
            # Health check is optional, log warning but proceed
            logger.warning("mcp_health_check_failed", server=self._config.name)

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[MCPToolManifest]:
        """Discover tools available on this MCP server."""
        if not self._connected:
            raise MCPError("Not connected", error_code="MCP_NOT_CONNECTED")

        raw = await self._rpc("tools/list", {})
        tools: list[MCPToolManifest] = []
        for item in raw.get("tools", []):
            params = []
            input_schema = item.get("inputSchema", {})
            properties = input_schema.get("properties", {})
            required = set(input_schema.get("required", []))
            for pname, pschema in properties.items():
                params.append(MCPToolParameter(
                    name=pname,
                    type=pschema.get("type", "string"),
                    description=pschema.get("description", ""),
                    required=pname in required,
                    default=pschema.get("default"),
                ))
            manifest = MCPToolManifest(
                name=item["name"],
                description=item.get("description", ""),
                server_name=self._config.name,
                parameters=params,
                input_schema=input_schema,
            )
            tools.append(manifest)

        self._tools = tools
        audit.log("mcp_tools_discovered", server=self._config.name, count=len(tools))
        return tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        if not self._connected:
            raise MCPError("Not connected", error_code="MCP_NOT_CONNECTED")

        result = await self._rpc("tools/call", {"name": tool_name, "arguments": arguments})
        content = result.get("content", [])
        # Extract text content
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        if len(texts) == 1:
            # Try JSON parse
            try:
                return json.loads(texts[0])
            except (json.JSONDecodeError, TypeError):
                return texts[0]
        return content

    # ------------------------------------------------------------------
    # JSON-RPC transport
    # ------------------------------------------------------------------

    async def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._request_id += 1
        message = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        if self._process is not None:
            return await self._rpc_stdio(message)
        elif self._http is not None:
            return await self._rpc_http(message)
        raise MCPError("No transport available", error_code="MCP_NO_TRANSPORT")

    async def _rpc_stdio(self, message: dict[str, Any]) -> dict[str, Any]:
        assert self._process and self._process.stdin and self._process.stdout
        payload = json.dumps(message) + "\n"
        self._process.stdin.write(payload.encode())
        await self._process.stdin.drain()

        line = await asyncio.wait_for(
            self._process.stdout.readline(),
            timeout=self._config.timeout_seconds,
        )
        if not line:
            raise MCPError("Empty response from MCP server", error_code="MCP_EMPTY_RESPONSE")
        resp = json.loads(line)
        if "error" in resp:
            err = resp["error"]
            raise MCPError(
                f"MCP error: {err.get('message', 'unknown')}",
                error_code="MCP_RPC_ERROR",
                details=err,
            )
        return resp.get("result", {})

    async def _rpc_http(self, message: dict[str, Any]) -> dict[str, Any]:
        assert self._http is not None
        resp = await self._http.post("/rpc", json=message)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            err = data["error"]
            raise MCPError(
                f"MCP error: {err.get('message', 'unknown')}",
                error_code="MCP_RPC_ERROR",
                details=err,
            )
        return data.get("result", {})

    # ------------------------------------------------------------------
    # Disconnect
    # ------------------------------------------------------------------

    async def disconnect(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except TimeoutError:
                self._process.kill()
            self._process = None
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        self._connected = False
        audit.log("mcp_disconnected", server=self._config.name)


# ---------------------------------------------------------------------------
# Auto-discovery helper
# ---------------------------------------------------------------------------

async def discover_mcp_servers(
    configs: list[MCPServerConfig],
) -> dict[str, MCPClient]:
    """Connect to all configured MCP servers and return connected clients."""
    clients: dict[str, MCPClient] = {}
    for cfg in configs:
        if not cfg.enabled:
            continue
        client = MCPClient(cfg)
        try:
            await client.connect()
            clients[cfg.name] = client
        except MCPError as exc:
            logger.warning("mcp_server_unavailable", server=cfg.name, error=str(exc))
    return clients
