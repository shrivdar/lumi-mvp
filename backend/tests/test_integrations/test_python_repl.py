"""Tests for PythonREPLTool — sandboxed Python execution with persistent namespace."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.exceptions import ToolError
from integrations.python_repl import _RUNNER_SCRIPT, PythonREPLTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proc(returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> AsyncMock:
    """Create a mock subprocess that returns the given output."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.kill = MagicMock()
    return proc


# ---------------------------------------------------------------------------
# Session lifecycle tests
# ---------------------------------------------------------------------------

class TestSessionLifecycle:
    @pytest.mark.asyncio
    async def test_create_session_success(self) -> None:
        tool = PythonREPLTool()
        container_id = "abc123def456"
        proc = _make_proc(returncode=0, stdout=container_id.encode())

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            sid = await tool.create_session()
            assert sid in tool._sessions
            assert tool._sessions[sid] == container_id[:12]
            await tool.close()

    @pytest.mark.asyncio
    async def test_create_session_with_explicit_id(self) -> None:
        tool = PythonREPLTool()
        proc = _make_proc(returncode=0, stdout=b"container123")

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            sid = await tool.create_session(session_id="my-session")
            assert sid == "my-session"
            assert "my-session" in tool._sessions
            await tool.close()

    @pytest.mark.asyncio
    async def test_create_session_failure_raises(self) -> None:
        tool = PythonREPLTool()
        proc = _make_proc(returncode=1, stderr=b"image not found")

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            with pytest.raises(ToolError, match="Failed to start REPL container"):
                await tool.create_session()
            await tool.close()

    @pytest.mark.asyncio
    async def test_destroy_session(self) -> None:
        tool = PythonREPLTool()
        tool._sessions["test-sess"] = "container123"
        proc = _make_proc(returncode=0)

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            await tool.destroy_session("test-sess")
            assert "test-sess" not in tool._sessions
            await tool.close()

    @pytest.mark.asyncio
    async def test_destroy_nonexistent_session_noop(self) -> None:
        tool = PythonREPLTool()
        await tool.destroy_session("does-not-exist")  # should not raise
        await tool.close()

    @pytest.mark.asyncio
    async def test_destroy_all_sessions(self) -> None:
        tool = PythonREPLTool()
        tool._sessions = {"a": "c1", "b": "c2", "c": "c3"}
        proc = _make_proc(returncode=0)

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            await tool.destroy_all_sessions()
            assert len(tool._sessions) == 0
            await tool.close()


# ---------------------------------------------------------------------------
# Code execution tests
# ---------------------------------------------------------------------------

class TestCodeExecution:
    @pytest.mark.asyncio
    async def test_execute_simple_code(self) -> None:
        tool = PythonREPLTool()
        tool._sessions["sess1"] = "container123"

        result_payload = json.dumps({
            "stdout": "(3, 1)\n",
            "stderr": "",
            "error": None,
        }).encode()
        proc = _make_proc(returncode=0, stdout=result_payload)

        with patch("asyncio.create_subprocess_exec", return_value=proc):
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
        tool._sessions["sess1"] = "container123"

        result_payload = json.dumps({
            "stdout": "",
            "stderr": "",
            "error": "Traceback ...\nNameError: name 'undefined_var' is not defined",
        }).encode()
        proc = _make_proc(returncode=0, stdout=result_payload)

        with patch("asyncio.create_subprocess_exec", return_value=proc):
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
    async def test_execute_container_crash(self) -> None:
        tool = PythonREPLTool()
        tool._sessions["sess1"] = "container123"

        proc = _make_proc(returncode=137, stderr=b"Killed")

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await tool.execute(session_id="sess1", code="x = 1")
            assert result["success"] is False
            assert "exit 137" in result["error"]
            tool._sessions.clear()
        await tool.close()

    @pytest.mark.asyncio
    async def test_execute_invalid_json_output(self) -> None:
        tool = PythonREPLTool()
        tool._sessions["sess1"] = "container123"

        proc = _make_proc(returncode=0, stdout=b"not json at all")

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await tool.execute(session_id="sess1", code="x = 1")
            assert result["success"] is False
            assert "Failed to parse" in result["error"]
            tool._sessions.clear()
        await tool.close()

    @pytest.mark.asyncio
    async def test_execute_timeout(self) -> None:
        tool = PythonREPLTool()
        tool._sessions["sess1"] = "container123"

        proc = AsyncMock()
        proc.communicate = AsyncMock(side_effect=TimeoutError)
        proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            with patch("asyncio.wait_for", side_effect=TimeoutError):
                result = await tool.execute(session_id="sess1", code="import time; time.sleep(200)")
                assert result["success"] is False
                assert "timed out" in result["error"]
                tool._sessions.clear()
        await tool.close()


# ---------------------------------------------------------------------------
# Namespace persistence tests (logic validation)
# ---------------------------------------------------------------------------

class TestNamespacePersistence:
    @pytest.mark.asyncio
    async def test_namespace_path_includes_session_id(self) -> None:
        """Verify the namespace file path is session-specific."""
        tool = PythonREPLTool()
        tool._sessions["my-sess"] = "container123"

        calls: list[tuple[Any, ...]] = []

        async def capture_exec(*args: Any, **kwargs: Any) -> Any:
            calls.append(args)
            result_payload = json.dumps({"stdout": "", "stderr": "", "error": None}).encode()
            return _make_proc(returncode=0, stdout=result_payload)

        with patch("asyncio.create_subprocess_exec", side_effect=capture_exec):
            await tool.execute(session_id="my-sess", code="x = 1")
            # The namespace path should contain the session ID
            exec_args = calls[0]
            ns_path_arg = exec_args[-1]  # last positional arg is ns_path
            assert "my-sess" in ns_path_arg
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
# Runner script validation
# ---------------------------------------------------------------------------

class TestRunnerScript:
    def test_runner_script_is_valid_python(self) -> None:
        """The embedded runner script should compile without syntax errors."""
        compile(_RUNNER_SCRIPT, "<runner>", "exec")

    def test_runner_script_contains_timeout_handling(self) -> None:
        assert "SIGALRM" in _RUNNER_SCRIPT
        assert "TimeoutError" in _RUNNER_SCRIPT

    def test_runner_script_contains_output_truncation(self) -> None:
        assert "truncated" in _RUNNER_SCRIPT

    def test_runner_script_persists_namespace(self) -> None:
        assert "pickle" in _RUNNER_SCRIPT
