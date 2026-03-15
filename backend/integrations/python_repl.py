"""PythonREPLTool — persistent sandboxed Python execution environment for YOHAS agents.

Each agent session gets a dedicated Docker container with a persistent Python namespace.
Variables set in one call are available in subsequent calls within the same session.

Security:
- No network access (--network none)
- Read-only data mount
- Resource limits (memory, CPU)
- Timeout enforcement
- Output truncation
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import structlog

from core.audit import AuditLogger
from core.config import settings
from core.exceptions import ToolError
from integrations.base_tool import BaseTool

logger = structlog.get_logger(__name__)
audit = AuditLogger("python_repl")

# The Python wrapper script injected into the container.
# It maintains a persistent namespace dict in /tmp/ns.json (pickle would be
# richer but JSON is safer for the sandbox boundary).  For objects that aren't
# JSON-serialisable (DataFrames, arrays, etc.) we keep them only in the
# in-process globals dict — persistence across `docker exec` invocations is
# achieved by keeping the container alive and using a long-running helper.
_RUNNER_SCRIPT = r'''
import sys, json, io, traceback, signal

# ── timeout handler ──────────────────────────────────────────────────
def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

timeout = int(sys.argv[1])
signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(timeout)

code = sys.argv[2]
max_chars = int(sys.argv[3])
ns_path = sys.argv[4]

# ── load namespace ───────────────────────────────────────────────────
import pickle, os
ns = {}
if os.path.exists(ns_path):
    try:
        with open(ns_path, "rb") as f:
            ns = pickle.load(f)
    except Exception:
        pass

# ── capture stdout/stderr ────────────────────────────────────────────
old_stdout, old_stderr = sys.stdout, sys.stderr
capture_out = io.StringIO()
capture_err = io.StringIO()
sys.stdout = capture_out
sys.stderr = capture_err

error = None
try:
    compiled = compile(code, "<repl>", "exec")
    exec(compiled, ns)
except Exception:
    error = traceback.format_exc()
finally:
    signal.alarm(0)
    sys.stdout = old_stdout
    sys.stderr = old_stderr

# ── persist namespace (skip unpicklable) ─────────────────────────────
import types
save_ns = {k: v for k, v in ns.items()
           if not k.startswith("_") and not isinstance(v, types.ModuleType)}
try:
    with open(ns_path, "wb") as f:
        pickle.dump(save_ns, f)
except Exception:
    pass

stdout_text = capture_out.getvalue()
stderr_text = capture_err.getvalue()
if len(stdout_text) > max_chars:
    stdout_text = stdout_text[:max_chars] + "\n... [truncated]"
if len(stderr_text) > max_chars:
    stderr_text = stderr_text[:max_chars] + "\n... [truncated]"

result = {"stdout": stdout_text, "stderr": stderr_text, "error": error}
print(json.dumps(result))
'''


class PythonREPLTool(BaseTool):
    """Sandboxed Python REPL with persistent namespace per agent session.

    Usage::

        repl = PythonREPLTool()
        session = await repl.create_session()
        result = await repl.execute(session_id=session, code="x = 42")
        result = await repl.execute(session_id=session, code="print(x)")
        await repl.destroy_session(session)
    """

    tool_id: str = "python_repl"
    name: str = "python_repl"
    description: str = "Execute Python code in a sandboxed Docker container with persistent namespace"
    category: str = "compute"
    rate_limit: float = 50.0
    cache_ttl: int = 0  # no caching — code execution is side-effectful
    max_retries: int = 0  # no retries — code execution is not idempotent
    timeout: int = settings.repl_timeout_seconds

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sessions: dict[str, str] = {}  # session_id -> container_id
        self._runtime = settings.container_runtime

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def create_session(self, session_id: str | None = None) -> str:
        """Start a new REPL container and return the session ID."""
        sid = session_id or uuid.uuid4().hex[:12]
        container_name = f"yohas-repl-{sid}"

        cmd = [
            self._runtime, "run", "-d",
            "--name", container_name,
            "--memory", settings.repl_memory_limit,
            "--cpus", settings.repl_cpu_limit,
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp:size=256m",
            "--tmpfs", "/home/repl:size=64m",
        ]

        # Mount data lake read-only if the path exists
        data_path = settings.data_lake_path
        if data_path:
            cmd.extend(["-v", f"{data_path}:/data:ro"])

        cmd.append(settings.repl_image)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise ToolError(
                f"Failed to start REPL container: {stderr.decode().strip()[:500]}",
                error_code="REPL_START_FAILED",
            )

        container_id = stdout.decode().strip()[:12]
        self._sessions[sid] = container_id
        audit.log("repl_session_created", session_id=sid, container_id=container_id)
        return sid

    async def destroy_session(self, session_id: str) -> None:
        """Stop and remove the REPL container for a session."""
        container_id = self._sessions.pop(session_id, None)
        if container_id is None:
            return
        proc = await asyncio.create_subprocess_exec(
            self._runtime, "rm", "-f", container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        audit.log("repl_session_destroyed", session_id=session_id)

    async def destroy_all_sessions(self) -> None:
        """Clean up all active sessions."""
        for sid in list(self._sessions):
            await self.destroy_session(sid)

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        """Run Python code in the session's container.

        Required kwargs:
            session_id: str — the session to execute in
            code: str — Python code to execute

        Returns dict with keys: stdout, stderr, error, success
        """
        session_id: str = kwargs["session_id"]
        code: str = kwargs["code"]
        timeout = kwargs.get("timeout", settings.repl_timeout_seconds)
        max_output = kwargs.get("max_output", settings.repl_max_output_chars)

        container_id = self._sessions.get(session_id)
        if container_id is None:
            raise ToolError(
                f"REPL session '{session_id}' not found — create one first",
                error_code="REPL_SESSION_NOT_FOUND",
            )

        ns_path = f"/tmp/ns_{session_id}.pkl"

        proc = await asyncio.create_subprocess_exec(
            self._runtime, "exec", container_id,
            "python", "-c", _RUNNER_SCRIPT,
            str(timeout), code, str(max_output), ns_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout + 5,  # grace period beyond in-container SIGALRM
            )
        except TimeoutError:
            proc.kill()
            audit.log("repl_timeout", session_id=session_id, timeout=timeout)
            return {
                "stdout": "",
                "stderr": "",
                "error": f"Execution timed out after {timeout}s",
                "success": False,
            }

        if proc.returncode != 0:
            err_text = stderr.decode().strip()[:500]
            return {
                "stdout": "",
                "stderr": err_text,
                "error": f"Container execution failed (exit {proc.returncode}): {err_text}",
                "success": False,
            }

        try:
            result = json.loads(stdout.decode())
        except json.JSONDecodeError:
            raw = stdout.decode()[:500]
            return {
                "stdout": raw,
                "stderr": stderr.decode()[:500],
                "error": "Failed to parse runner output as JSON",
                "success": False,
            }

        result["success"] = result.get("error") is None
        audit.log(
            "repl_exec",
            session_id=session_id,
            success=result["success"],
            stdout_len=len(result.get("stdout", "")),
        )
        return result

    # ------------------------------------------------------------------
    # Override caching — disabled for REPL
    # ------------------------------------------------------------------

    async def _cache_get(self, key: str) -> Any | None:
        return None  # never cache code execution

    async def _cache_set(self, key: str, value: Any, ttl: int | None = None) -> None:
        pass  # never cache code execution

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await self.destroy_all_sessions()
        await super().close()
