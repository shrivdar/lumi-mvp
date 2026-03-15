"""Tests for PythonREPLTool — subprocess-based Python execution with persistent namespace."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from core.exceptions import ToolError
from integrations.python_repl import _WORKER_SCRIPT, PythonREPLTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session_proc(
    *,
    returncode: int | None = None,
    readline_data: bytes = b"",
    pid: int = 12345,
) -> AsyncMock:
    """Create a mock long-running subprocess for session tests.

    The mock simulates the persistent worker subprocess with stdin/stdout/stderr.
    """
    proc = AsyncMock()
    # returncode: None means still running, int means exited
    type(proc).returncode = PropertyMock(return_value=returncode)
    proc.pid = pid

    # stdin mock
    proc.stdin = AsyncMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdin.is_closing = MagicMock(return_value=False)
    proc.stdin.close = MagicMock()

    # stdout mock — readline returns the given data
    proc.stdout = AsyncMock()
    proc.stdout.readline = AsyncMock(return_value=readline_data)

    # stderr mock
    proc.stderr = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=b"")

    proc.kill = MagicMock()
    proc.wait = AsyncMock()

    return proc


# ---------------------------------------------------------------------------
# Session lifecycle tests
# ---------------------------------------------------------------------------

class TestSessionLifecycle:
    @pytest.mark.asyncio
    async def test_create_session_success(self) -> None:
        tool = PythonREPLTool()
        proc = _make_session_proc()

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            sid = await tool.create_session()
            assert sid in tool._sessions
            assert tool._sessions[sid] is proc
            tool._sessions.clear()
        await tool.close()

    @pytest.mark.asyncio
    async def test_create_session_with_explicit_id(self) -> None:
        tool = PythonREPLTool()
        proc = _make_session_proc()

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            sid = await tool.create_session(session_id="my-session")
            assert sid == "my-session"
            assert "my-session" in tool._sessions
            tool._sessions.clear()
        await tool.close()

    @pytest.mark.asyncio
    async def test_destroy_session(self) -> None:
        tool = PythonREPLTool()
        proc = _make_session_proc()
        tool._sessions["sess1"] = proc

        await tool.destroy_session("sess1")
        assert "sess1" not in tool._sessions
        await tool.close()

    @pytest.mark.asyncio
    async def test_destroy_nonexistent_session(self) -> None:
        tool = PythonREPLTool()
        await tool.destroy_session("nonexistent")  # should not raise
        await tool.close()

    @pytest.mark.asyncio
    async def test_destroy_all_sessions(self) -> None:
        tool = PythonREPLTool()
        proc1 = _make_session_proc()
        proc2 = _make_session_proc()
        tool._sessions["s1"] = proc1
        tool._sessions["s2"] = proc2

        await tool.destroy_all_sessions()
        assert len(tool._sessions) == 0
        await tool.close()


# ---------------------------------------------------------------------------
# Code execution tests
# ---------------------------------------------------------------------------

class TestCodeExecution:
    @pytest.mark.asyncio
    async def test_execute_success(self) -> None:
        tool = PythonREPLTool()
        result_payload = json.dumps({
            "stdout": "(3, 1)\n",
            "stderr": "",
            "error": None,
        }).encode() + b"\n"
        proc = _make_session_proc(readline_data=result_payload)
        tool._sessions["sess1"] = proc

        result = await tool.execute(
            session_id="sess1",
            code="import pandas as pd; print(pd.DataFrame({'a': [1,2,3]}).shape)",
        )
        assert result["success"] is True
        assert "(3, 1)" in result["stdout"]
        assert result["error"] is None
        tool._sessions.clear()
        await tool.close()

    @pytest.mark.asyncio
    async def test_execute_with_error(self) -> None:
        tool = PythonREPLTool()
        result_payload = json.dumps({
            "stdout": "",
            "stderr": "",
            "error": "Traceback ...\nNameError: name 'undefined_var' is not defined",
        }).encode() + b"\n"
        proc = _make_session_proc(readline_data=result_payload)
        tool._sessions["sess1"] = proc

        result = await tool.execute(session_id="sess1", code="print(undefined_var)")
        assert result["success"] is False
        assert "NameError" in result["error"]
        tool._sessions.clear()
        await tool.close()

    @pytest.mark.asyncio
    async def test_execute_session_not_found(self) -> None:
        tool = PythonREPLTool()

        with pytest.raises(ToolError, match="session.*not found"):
            await tool.execute(session_id="nonexistent", code="print(1)")
        await tool.close()

    @pytest.mark.asyncio
    async def test_execute_subprocess_dead(self) -> None:
        """When the subprocess has already exited, raise ToolError."""
        tool = PythonREPLTool()
        proc = _make_session_proc(returncode=137)
        tool._sessions["sess1"] = proc

        with pytest.raises(ToolError, match="subprocess has exited"):
            await tool.execute(session_id="sess1", code="x = 1")
        await tool.close()

    @pytest.mark.asyncio
    async def test_execute_empty_readline(self) -> None:
        """When subprocess exits mid-execution (empty readline), return error."""
        tool = PythonREPLTool()
        proc = _make_session_proc(readline_data=b"")
        tool._sessions["sess1"] = proc

        result = await tool.execute(session_id="sess1", code="x = 1")
        assert result["success"] is False
        assert "exited unexpectedly" in result["error"]
        await tool.close()

    @pytest.mark.asyncio
    async def test_execute_invalid_json_output(self) -> None:
        tool = PythonREPLTool()
        proc = _make_session_proc(readline_data=b"not json at all\n")
        tool._sessions["sess1"] = proc

        result = await tool.execute(session_id="sess1", code="x = 1")
        assert result["success"] is False
        assert "Failed to parse" in result["error"]
        tool._sessions.clear()
        await tool.close()

    @pytest.mark.asyncio
    async def test_execute_timeout(self) -> None:
        tool = PythonREPLTool()
        proc = _make_session_proc()
        # Make readline raise TimeoutError
        proc.stdout.readline = AsyncMock(side_effect=TimeoutError)
        tool._sessions["sess1"] = proc

        with patch("asyncio.wait_for", side_effect=TimeoutError):
            result = await tool.execute(session_id="sess1", code="import time; time.sleep(200)")
            assert result["success"] is False
            assert "timed out" in result["error"]
        await tool.close()

    @pytest.mark.asyncio
    async def test_execute_broken_pipe(self) -> None:
        """When stdin write fails (broken pipe), return error gracefully."""
        tool = PythonREPLTool()
        proc = _make_session_proc()
        proc.stdin.write = MagicMock(side_effect=BrokenPipeError("broken"))
        tool._sessions["sess1"] = proc

        result = await tool.execute(session_id="sess1", code="x = 1")
        assert result["success"] is False
        assert "subprocess died" in result["error"]
        await tool.close()


# ---------------------------------------------------------------------------
# Namespace persistence tests (integration-style)
# ---------------------------------------------------------------------------

class TestNamespacePersistence:
    @pytest.mark.asyncio
    async def test_namespace_persists_across_calls(self) -> None:
        """Verify that the same subprocess is reused for the same session."""
        tool = PythonREPLTool()
        proc = _make_session_proc()
        tool._sessions["my-sess"] = proc

        # First call
        result1 = json.dumps({"stdout": "", "stderr": "", "error": None}).encode() + b"\n"
        proc.stdout.readline = AsyncMock(return_value=result1)
        await tool.execute(session_id="my-sess", code="x = 1")

        # Second call — same proc should be used
        result2 = json.dumps({"stdout": "1\n", "stderr": "", "error": None}).encode() + b"\n"
        proc.stdout.readline = AsyncMock(return_value=result2)
        r = await tool.execute(session_id="my-sess", code="print(x)")
        assert r["success"] is True

        # Verify stdin.write was called twice (once per execute)
        assert proc.stdin.write.call_count == 2
        tool._sessions.clear()
        await tool.close()


# ---------------------------------------------------------------------------
# Caching disabled tests
# ---------------------------------------------------------------------------

class TestCachingDisabled:
    @pytest.mark.asyncio
    async def test_cache_get_always_returns_none(self) -> None:
        tool = PythonREPLTool()
        assert await tool._cache_get("any-key") is None
        await tool.close()

    @pytest.mark.asyncio
    async def test_cache_set_is_noop(self) -> None:
        tool = PythonREPLTool()
        await tool._cache_set("key", {"data": 1})  # should not raise
        await tool.close()


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_tool_metadata(self) -> None:
        tool = PythonREPLTool()
        assert tool.tool_id == "python_repl"
        assert tool.name == "python_repl"
        assert tool.category == "compute"

    def test_auto_register(self) -> None:
        from core.tool_registry import InMemoryToolRegistry

        reg = InMemoryToolRegistry()
        PythonREPLTool(registry=reg)
        entry = reg.get_tool("python_repl")
        assert entry is not None
        assert entry.category == "compute"


# ---------------------------------------------------------------------------
# Worker script validation
# ---------------------------------------------------------------------------

class TestWorkerScript:
    def test_worker_script_is_valid_python(self) -> None:
        """The embedded worker script should compile without syntax errors."""
        compile(_WORKER_SCRIPT, "<worker>", "exec")

    def test_worker_script_contains_timeout_handling(self) -> None:
        assert "SIGALRM" in _WORKER_SCRIPT
        assert "TimeoutError" in _WORKER_SCRIPT

    def test_worker_script_contains_output_truncation(self) -> None:
        assert "truncated" in _WORKER_SCRIPT

    def test_worker_script_contains_security_restrictions(self) -> None:
        assert "BLOCKED_MODULES" in _WORKER_SCRIPT
        assert "subprocess" in _WORKER_SCRIPT
        assert "_restricted_import" in _WORKER_SCRIPT

    def test_worker_script_uses_persistent_namespace(self) -> None:
        assert "namespace" in _WORKER_SCRIPT
        assert "exec(compiled, namespace)" in _WORKER_SCRIPT
