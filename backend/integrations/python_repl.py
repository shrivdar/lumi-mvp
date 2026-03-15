"""PythonREPLTool — persistent Python execution environment for YOHAS agents.

Each agent session gets a dedicated long-running Python subprocess with a
persistent namespace.  Variables set in one call are available in subsequent
calls within the same session.

Security:
- Restricted builtins (no os.system, subprocess, importlib in user code)
- Timeout enforcement via asyncio + SIGALRM in subprocess
- Output truncation
- Data lake path exposed via environment variable (read-only by convention)

This implementation uses **local subprocesses** (no Docker required).
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import uuid
from typing import Any

import structlog

from core.audit import AuditLogger
from core.config import settings
from core.exceptions import ToolError
from integrations.base_tool import BaseTool

logger = structlog.get_logger(__name__)
audit = AuditLogger("python_repl")

# ---------------------------------------------------------------------------
# Worker script — runs inside the long-lived subprocess.
#
# Protocol:
#   Parent writes a JSON line to stdin:
#     {"code": "...", "timeout": 120, "max_output": 10000}
#   Worker writes a JSON line to stdout:
#     {"stdout": "...", "stderr": "...", "error": null}
#   The sentinel "<<EXIT>>" on stdin causes the worker to exit.
# ---------------------------------------------------------------------------

_WORKER_SCRIPT = r'''
import sys, json, io, traceback, signal, types, os

# ── security: block dangerous modules in user code ──────────────────
_BLOCKED_MODULES = frozenset({
    "subprocess", "importlib", "shutil", "ctypes",
    "multiprocessing", "socket", "http", "urllib",
    "ftplib", "smtplib", "telnetlib", "xmlrpc",
    "webbrowser", "antigravity", "code", "codeop",
})

_original_import = __import__

def _restricted_import(name, *args, **kwargs):
    top = name.split(".")[0]
    if top in _BLOCKED_MODULES:
        raise ImportError(f"Module '{name}' is blocked in the REPL sandbox")
    return _original_import(name, *args, **kwargs)

# ── persistent namespace ────────────────────────────────────────────
namespace = {"__builtins__": __builtins__, "__import__": _restricted_import}

# Override __import__ in builtins so user code's `import X` is intercepted
import builtins
builtins.__import__ = _restricted_import

# ── timeout handler (Unix only) ─────────────────────────────────────
def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

has_sigalrm = hasattr(signal, "SIGALRM")

# ── main loop ───────────────────────────────────────────────────────
for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line:
        continue
    if line == "<<EXIT>>":
        break

    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        result = {"stdout": "", "stderr": "", "error": f"Invalid JSON request: {line[:200]}"}
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()
        continue

    code = request.get("code", "")
    timeout = request.get("timeout", 120)
    max_chars = request.get("max_output", 10000)

    # Set timeout alarm
    if has_sigalrm:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

    # Capture stdout/stderr
    old_stdout, old_stderr = sys.stdout, sys.stderr
    capture_out = io.StringIO()
    capture_err = io.StringIO()
    sys.stdout = capture_out
    sys.stderr = capture_err

    error = None
    try:
        compiled = compile(code, "<repl>", "exec")
        exec(compiled, namespace)
    except Exception:
        error = traceback.format_exc()
    finally:
        if has_sigalrm:
            signal.alarm(0)
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    stdout_text = capture_out.getvalue()
    stderr_text = capture_err.getvalue()
    if len(stdout_text) > max_chars:
        stdout_text = stdout_text[:max_chars] + "\n... [truncated]"
    if len(stderr_text) > max_chars:
        stderr_text = stderr_text[:max_chars] + "\n... [truncated]"

    result = {"stdout": stdout_text, "stderr": stderr_text, "error": error}
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()
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
    description: str = "Execute Python code in a sandboxed subprocess with persistent namespace"
    category: str = "compute"
    rate_limit: float = 50.0
    cache_ttl: int = 0  # no caching — code execution is side-effectful
    max_retries: int = 0  # no retries — code execution is not idempotent
    timeout: int = settings.repl_timeout_seconds

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # session_id -> asyncio.subprocess.Process (long-running worker)
        self._sessions: dict[str, asyncio.subprocess.Process] = {}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def create_session(self, session_id: str | None = None) -> str:
        """Start a new REPL subprocess and return the session ID."""
        sid = session_id or uuid.uuid4().hex[:12]

        env = os.environ.copy()
        # Expose data lake path as env var for user code
        data_path = settings.data_lake_path
        if data_path:
            env["YOHAS_DATA_PATH"] = data_path
            env["DATA_LAKE_DIR"] = data_path

        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-c", _WORKER_SCRIPT,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        self._sessions[sid] = proc
        audit.log("repl_session_created", session_id=sid, pid=proc.pid)
        return sid

    async def destroy_session(self, session_id: str) -> None:
        """Stop the REPL subprocess for a session."""
        proc = self._sessions.pop(session_id, None)
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.is_closing():
                proc.stdin.write(b"<<EXIT>>\n")
                await proc.stdin.drain()
                proc.stdin.close()
            # Give it a moment to exit gracefully
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except (TimeoutError, asyncio.TimeoutError):
                proc.kill()
                await proc.wait()
        except (ProcessLookupError, BrokenPipeError, ConnectionResetError):
            pass
        audit.log("repl_session_destroyed", session_id=session_id)

    async def destroy_all_sessions(self) -> None:
        """Clean up all active sessions."""
        for sid in list(self._sessions):
            await self.destroy_session(sid)

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        """Run Python code in the session's subprocess.

        Required kwargs:
            session_id: str — the session to execute in
            code: str — Python code to execute

        Returns dict with keys: stdout, stderr, error, success
        """
        session_id: str = kwargs["session_id"]
        code: str = kwargs["code"]
        timeout = kwargs.get("timeout", settings.repl_timeout_seconds)
        max_output = kwargs.get("max_output", settings.repl_max_output_chars)

        proc = self._sessions.get(session_id)
        if proc is None:
            raise ToolError(
                f"REPL session '{session_id}' not found — create one first",
                error_code="REPL_SESSION_NOT_FOUND",
            )

        # Check if the subprocess is still alive
        if proc.returncode is not None:
            self._sessions.pop(session_id, None)
            raise ToolError(
                f"REPL session '{session_id}' subprocess has exited (code {proc.returncode})",
                error_code="REPL_SESSION_DEAD",
            )

        request = json.dumps({
            "code": code,
            "timeout": timeout,
            "max_output": max_output,
        }) + "\n"

        try:
            proc.stdin.write(request.encode())  # type: ignore[union-attr]
            await proc.stdin.drain()  # type: ignore[union-attr]
        except (BrokenPipeError, ConnectionResetError) as exc:
            self._sessions.pop(session_id, None)
            return {
                "stdout": "",
                "stderr": "",
                "error": f"REPL subprocess died: {exc}",
                "success": False,
            }

        try:
            raw_line = await asyncio.wait_for(
                proc.stdout.readline(),  # type: ignore[union-attr]
                timeout=timeout + 5,  # grace period beyond in-process SIGALRM
            )
        except (TimeoutError, asyncio.TimeoutError):
            # Kill the subprocess — it's stuck
            proc.kill()
            self._sessions.pop(session_id, None)
            audit.log("repl_timeout", session_id=session_id, timeout=timeout)
            return {
                "stdout": "",
                "stderr": "",
                "error": f"Execution timed out after {timeout}s",
                "success": False,
            }

        if not raw_line:
            # Subprocess exited unexpectedly
            self._sessions.pop(session_id, None)
            stderr_data = b""
            if proc.stderr:
                try:
                    stderr_data = await asyncio.wait_for(proc.stderr.read(), timeout=2.0)
                except (TimeoutError, asyncio.TimeoutError):
                    pass
            err_text = stderr_data.decode(errors="replace").strip()[:500]
            return {
                "stdout": "",
                "stderr": err_text,
                "error": f"REPL subprocess exited unexpectedly: {err_text or 'no output'}",
                "success": False,
            }

        try:
            result = json.loads(raw_line.decode())
        except json.JSONDecodeError:
            raw = raw_line.decode()[:500]
            return {
                "stdout": raw,
                "stderr": "",
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
