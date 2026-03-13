"""Container tool manager — launch and manage Docker containers as tool servers."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from core.audit import AuditLogger
from core.config import settings
from core.exceptions import ToolError
from core.models import ContainerToolConfig

logger = structlog.get_logger(__name__)
audit = AuditLogger("container_tool")


class ContainerTool:
    """Manages a Docker/Podman container that exposes a tool API.

    Features:
    - Starts container on demand
    - Health checking with restart on failure
    - Resource limits (memory, CPU)
    - Network isolation (default: none)
    - Automatic cleanup on close
    """

    def __init__(self, config: ContainerToolConfig) -> None:
        self._config = config
        self._container_id: str | None = None
        self._runtime = settings.container_runtime
        self._healthy = False

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def is_running(self) -> bool:
        return self._container_id is not None and self._healthy

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> str:
        """Pull image if needed and start the container. Returns container ID."""
        cmd = [
            self._runtime, "run", "-d",
            "--name", f"yohas-tool-{self._config.name}",
            "--memory", self._config.memory_limit,
            "--cpus", self._config.cpu_limit,
            "--network", self._config.network_mode,
        ]
        for key, value in self._config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        for host_path, container_path in self._config.volumes.items():
            cmd.extend(["-v", f"{host_path}:{container_path}:ro"])
        cmd.append(self._config.image)
        cmd.extend(self._config.command)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise ToolError(
                f"Failed to start container '{self._config.name}': {stderr.decode().strip()}",
                error_code="CONTAINER_START_FAILED",
            )
        self._container_id = stdout.decode().strip()[:12]
        audit.log("container_started", tool=self._config.name, container_id=self._container_id)
        return self._container_id

    async def stop(self) -> None:
        if self._container_id is None:
            return
        proc = await asyncio.create_subprocess_exec(
            self._runtime, "rm", "-f", self._container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        audit.log("container_stopped", tool=self._config.name, container_id=self._container_id)
        self._container_id = None
        self._healthy = False

    async def health_check(self) -> bool:
        """Check if the container is running and healthy."""
        if self._container_id is None:
            self._healthy = False
            return False
        proc = await asyncio.create_subprocess_exec(
            self._runtime, "inspect", "-f", "{{.State.Running}}", self._container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        self._healthy = stdout.decode().strip().lower() == "true"
        return self._healthy

    async def restart(self) -> None:
        """Stop and re-start the container."""
        await self.stop()
        await self.start()

    # ------------------------------------------------------------------
    # Execute a tool call via docker exec
    # ------------------------------------------------------------------

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run a tool inside the container by invoking its entrypoint with JSON args."""
        if not self.is_running:
            await self.start()

        payload = json.dumps({"tool": tool_name, "arguments": arguments})
        proc = await asyncio.create_subprocess_exec(
            self._runtime, "exec", self._container_id or "",
            "python", "-c",
            "import sys,json;from server import call_tool;print(json.dumps(call_tool(**json.loads(sys.argv[1]))))",
            payload,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._config.timeout_seconds,
            )
        except TimeoutError:
            proc.kill()
            raise ToolError(
                f"Container tool '{self._config.name}' timed out after {self._config.timeout_seconds}s",
                error_code="CONTAINER_TIMEOUT",
            )

        if proc.returncode != 0:
            raise ToolError(
                f"Container tool '{self._config.name}' failed: {stderr.decode().strip()[:500]}",
                error_code="CONTAINER_EXEC_FAILED",
            )

        try:
            return json.loads(stdout.decode())
        except json.JSONDecodeError as exc:
            raise ToolError(
                f"Container tool '{self._config.name}' returned invalid JSON",
                error_code="CONTAINER_INVALID_RESPONSE",
            ) from exc


class ContainerToolManager:
    """Manages multiple container tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ContainerTool] = {}

    def register(self, config: ContainerToolConfig) -> ContainerTool:
        tool = ContainerTool(config)
        self._tools[config.name] = tool
        return tool

    def get(self, name: str) -> ContainerTool | None:
        return self._tools.get(name)

    async def health_check_all(self) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for name, tool in self._tools.items():
            results[name] = await tool.health_check()
            if not results[name] and tool._container_id is not None:
                logger.warning("container_unhealthy_restarting", tool=name)
                try:
                    await tool.restart()
                    results[name] = True
                except ToolError:
                    logger.error("container_restart_failed", tool=name)
        return results

    async def stop_all(self) -> None:
        for tool in self._tools.values():
            await tool.stop()
