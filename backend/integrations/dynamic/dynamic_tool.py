"""DynamicTool — runtime wrapper for agent-generated tool code.

A DynamicTool executes auto-generated wrapper code inside a sandboxed
PythonREPLTool session.  The wrapper code is validated and tested before
the tool is registered, and all executions run in the same Docker-sandboxed
REPL with ``--network none`` (no arbitrary network access from generated code).

Lifecycle:
    1. ToolCreatorAgent writes wrapper code + test code
    2. Tests run in sandboxed REPL
    3. If tests pass → DynamicTool is created and registered
    4. Other agents call ``execute()`` which delegates to the REPL
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import structlog

from core.models import DynamicToolSpec, ToolRegistryEntry, ToolSourceType
from integrations.base_tool import BaseTool

logger = structlog.get_logger(__name__)

# Directory where generated wrapper .py files are persisted
DYNAMIC_TOOL_DIR = Path(__file__).parent


class DynamicTool(BaseTool):
    """A tool whose ``_execute`` method delegates to agent-generated Python code
    running in a sandboxed REPL session.

    The wrapper code must define a function named ``run(**kwargs) -> dict``
    that the DynamicTool calls via ``exec`` in the REPL namespace.
    """

    tool_id: str = ""
    name: str = ""
    description: str = ""
    category: str = "dynamic"
    rate_limit: float = 2.0
    cache_ttl: int = 3600
    max_retries: int = 1
    timeout: int = 60

    def __init__(
        self,
        spec: DynamicToolSpec,
        repl_tool: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.spec = spec
        self.tool_id = f"dynamic_{spec.name}"
        self.name = spec.name
        self.description = spec.description
        self.category = spec.category
        self._repl = repl_tool
        self._repl_session_id: str | None = None
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Registration override — source_type = DYNAMIC
    # ------------------------------------------------------------------

    def _register(self, registry: Any) -> None:
        entry = ToolRegistryEntry(
            name=self.name,
            description=self.description,
            source_type=ToolSourceType.DYNAMIC,
            category=self.category,
            capabilities=self.spec.capabilities,
            cache_ttl=self.cache_ttl,
            rate_limit_rps=self.rate_limit,
        )
        registry.register(entry)

    # ------------------------------------------------------------------
    # Execute via REPL
    # ------------------------------------------------------------------

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        """Run the generated wrapper code in the sandboxed REPL."""
        if self._repl is None:
            return {"error": "No REPL available for dynamic tool execution"}

        # Ensure a session exists
        if self._repl_session_id is None:
            self._repl_session_id = await self._repl.execute(
                action="create_session",
            )
            if isinstance(self._repl_session_id, dict):
                self._repl_session_id = self._repl_session_id.get("session_id", "dynamic")

        # Build the execution code: load wrapper, call run()
        call_kwargs = json.dumps(kwargs, default=str)
        code = textwrap.dedent(f"""\
            import json
            {self.spec.wrapper_code}

            _result = run(**json.loads('''{call_kwargs}'''))
            print(json.dumps(_result, default=str))
        """)

        result = await self._repl.execute(
            action="execute",
            session_id=self._repl_session_id,
            code=code,
        )

        if isinstance(result, dict):
            if result.get("error"):
                return {"error": result["error"], "stderr": result.get("stderr", "")}
            stdout = result.get("stdout", "")
            try:
                return json.loads(stdout)
            except (json.JSONDecodeError, TypeError):
                return {"raw_output": stdout}

        return {"raw_output": str(result)}

    # ------------------------------------------------------------------
    # Persistence — save wrapper code to disk
    # ------------------------------------------------------------------

    def save_wrapper(self) -> Path:
        """Persist the generated wrapper code to ``integrations/dynamic/<name>.py``."""
        path = DYNAMIC_TOOL_DIR / f"{self.spec.name}.py"
        header = f'"""Auto-generated tool wrapper: {self.spec.name}\n\n{self.spec.description}\n"""\n\n'
        path.write_text(header + self.spec.wrapper_code, encoding="utf-8")
        logger.info("dynamic_tool_saved", tool=self.spec.name, path=str(path))
        return path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        if self._repl and self._repl_session_id:
            try:
                await self._repl.execute(
                    action="destroy_session",
                    session_id=self._repl_session_id,
                )
            except Exception:
                pass
        await super().close()
