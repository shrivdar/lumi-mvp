"""Tests for ToolCreatorAgent — tool discovery, wrapper generation, testing, registration."""

from __future__ import annotations

import json
import textwrap
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.templates import get_template
from agents.tool_creator import KNOWN_API_CATALOG, ToolCreatorAgent
from core.models import AgentTask, AgentType, DynamicToolSpec, DynamicToolStatus
from core.tool_registry import InMemoryToolRegistry
from tests.test_agents.conftest import MockLLMClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tool_creator_task() -> AgentTask:
    """Task that drives tool creation for STRING-db integration."""
    return AgentTask(
        task_id="task-tc-001",
        research_id="research-001",
        agent_type=AgentType.TOOL_CREATOR,
        agent_id="agent-tc-001",
        hypothesis_branch="h-protein-interactions",
        instruction=(
            "We need to find protein-protein interaction data for B7-H3 in NSCLC. "
            "The current tool registry lacks a protein interaction database. "
            "Discover and create a wrapper for a suitable API."
        ),
        context={"protein": "B7-H3", "disease": "NSCLC"},
    )


@pytest.fixture()
def mock_repl_tool() -> MagicMock:
    """Mock PythonREPLTool that simulates REPL responses."""
    tool = MagicMock()
    tool.tool_id = "python_repl"
    tool.name = "python_repl"
    tool.description = "Sandboxed Python REPL"

    async def mock_execute(**kwargs):
        action = kwargs.get("action", "execute")
        if action == "create_session":
            return {"session_id": "test-session-1"}
        if action == "destroy_session":
            return {"success": True}
        # For code execution, check what code is being run
        code = kwargs.get("code", "")
        if "ast.parse" in code:
            return {"stdout": json.dumps({"success": True, "message": "Syntax valid"}), "stderr": "", "success": True}
        if "callable(run)" in code:
            return {"stdout": json.dumps({"success": True, "has_run": True}), "stderr": "", "success": True}
        return {"stdout": "", "stderr": "", "success": True}

    tool.execute = AsyncMock(side_effect=mock_execute)
    return tool


STRING_DB_WRAPPER = textwrap.dedent("""\
    import urllib.request
    import json

    def run(*, query: str = "", species: int = 9606, limit: int = 10, **kwargs) -> dict:
        \"\"\"Query STRING-db for protein-protein interactions.\"\"\"
        base = "https://string-db.org/api/json"
        params = f"identifiers={query}&species={species}&limit={limit}"
        url = f"{base}/network?{params}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            interactions = []
            for item in data:
                interactions.append({
                    "protein_a": item.get("preferredName_A", ""),
                    "protein_b": item.get("preferredName_B", ""),
                    "score": item.get("score", 0),
                })
            return {"query": query, "interactions": interactions}
        except Exception as e:
            return {"error": str(e), "query": query, "interactions": []}
""")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolCreatorAgent:
    """Unit tests for the ToolCreatorAgent."""

    @pytest.mark.asyncio
    async def test_investigates_and_produces_tool_spec(self, agent_kg, mock_tools, tool_creator_task):
        """ToolCreator should use multi-turn to discover and specify a new tool."""
        llm = MockLLMClient(responses=[
            # Turn 0: plan
            "<think>1. Check API catalog for protein interaction APIs\n"
            "2. STRING-db looks relevant\n"
            "3. Write wrapper code\n"
            "4. Test in REPL\n"
            "5. Output tool spec</think>",
            # Turn 1: search pubmed for API docs
            '<tool>pubmed:{"action": "search", "query": "STRING protein interaction API"'
            ', "max_results": 3}</tool>',
            # Turn 2: write code in REPL
            '<execute>import ast\ncode = """def run(): pass"""\nast.parse(code)\nprint("valid")</execute>',
            # Turn 3: answer with tool spec
            '<answer>' + json.dumps({
                "entities": [
                    {
                        "name": "string_db",
                        "type": "MECHANISM",
                        "description": "STRING database protein-protein interaction tool",
                        "properties": {
                            "wrapper_code": STRING_DB_WRAPPER,
                            "test_code": "result = run(query='TP53')",
                            "api_base_url": "https://string-db.org/api",
                            "category": "protein",
                            "capabilities": ["protein-protein interaction network retrieval"],
                            "example_tasks": ["Find interaction partners for B7-H3"],
                        },
                    }
                ],
                "relationships": [],
                "summary": "Created STRING-db tool wrapper for protein-protein interactions.",
            }) + '</answer>',
            # Falsification response
            '{"search_query": "STRING-db API issues"}',
        ])

        template = get_template(AgentType.TOOL_CREATOR)
        agent = ToolCreatorAgent(
            template=template, llm=llm, kg=agent_kg, tools=mock_tools,
        )

        result = await agent.execute(tool_creator_task)

        assert result.success is True
        assert len(result.turns) >= 3

    @pytest.mark.asyncio
    async def test_post_investigation_extracts_specs(self, agent_kg, tool_creator_task):
        """_post_investigation should parse tool specs from investigation result."""
        llm = MockLLMClient()
        template = get_template(AgentType.TOOL_CREATOR)
        agent = ToolCreatorAgent(template=template, llm=llm, kg=agent_kg, tools={})

        investigation = {
            "entities": [
                {
                    "name": "string_db",
                    "description": "STRING protein interactions",
                    "properties": {
                        "wrapper_code": STRING_DB_WRAPPER,
                        "api_base_url": "https://string-db.org/api",
                        "category": "protein",
                        "capabilities": ["PPI retrieval"],
                    },
                },
                {
                    "name": "no_code_entity",
                    "description": "Entity without wrapper code",
                    "properties": {},
                },
            ],
        }

        specs = await agent._post_investigation(tool_creator_task, investigation)

        assert len(specs) == 1
        assert specs[0].name == "string_db"
        assert specs[0].status == DynamicToolStatus.DRAFT
        assert "def run(" in specs[0].wrapper_code
        assert specs[0].category == "protein"

    @pytest.mark.asyncio
    async def test_create_and_register_tools(self, agent_kg, mock_tools, mock_repl_tool, tool_creator_task):
        """Full lifecycle: investigate → extract → test → register."""
        llm = MockLLMClient(responses=[
            # Plan
            "<think>Create STRING-db wrapper</think>",
            # Answer with wrapper
            '<answer>' + json.dumps({
                "entities": [
                    {
                        "name": "string_db",
                        "type": "MECHANISM",
                        "description": "STRING database tool",
                        "properties": {
                            "wrapper_code": STRING_DB_WRAPPER,
                            "api_base_url": "https://string-db.org/api",
                            "category": "protein",
                            "capabilities": ["PPI networks"],
                        },
                    }
                ],
                "relationships": [],
                "summary": "Created STRING-db wrapper.",
            }) + '</answer>',
            # Falsification
            '{}',
        ])

        template = get_template(AgentType.TOOL_CREATOR)
        agent = ToolCreatorAgent(
            template=template, llm=llm, kg=agent_kg, tools=mock_tools,
        )

        registry = InMemoryToolRegistry()
        specs = await agent.create_and_register_tools(
            tool_creator_task, registry, repl_tool=mock_repl_tool,
        )

        assert len(specs) >= 1
        assert specs[0].status == DynamicToolStatus.REGISTERED
        assert specs[0].name == "string_db"

        # Tool should be in registry
        entry = registry.get_tool("string_db")
        assert entry is not None
        assert entry.source_type.value == "DYNAMIC"

    @pytest.mark.asyncio
    async def test_rejects_wrapper_without_run_function(self, agent_kg, mock_tools, mock_repl_tool, tool_creator_task):
        """Wrappers missing run() should be rejected."""
        bad_wrapper = "def fetch_data(): return {}"

        llm = MockLLMClient(responses=[
            "<think>Write tool</think>",
            '<answer>' + json.dumps({
                "entities": [
                    {
                        "name": "bad_tool",
                        "type": "MECHANISM",
                        "description": "A bad tool",
                        "properties": {
                            "wrapper_code": bad_wrapper,
                            "category": "protein",
                        },
                    }
                ],
                "relationships": [],
                "summary": "Created tool.",
            }) + '</answer>',
            '{}',
        ])

        template = get_template(AgentType.TOOL_CREATOR)
        agent = ToolCreatorAgent(
            template=template, llm=llm, kg=agent_kg, tools=mock_tools,
        )

        registry = InMemoryToolRegistry()
        specs = await agent.create_and_register_tools(
            tool_creator_task, registry, repl_tool=mock_repl_tool,
        )

        # Bad tool should not be registered
        assert len(specs) == 0
        assert registry.get_tool("bad_tool") is None

    @pytest.mark.asyncio
    async def test_repl_syntax_failure_rejects_tool(self, agent_kg, mock_tools, tool_creator_task):
        """If REPL reports syntax error, tool should be rejected."""
        repl = MagicMock()

        async def failing_execute(**kwargs):
            code = kwargs.get("code", "")
            if "ast.parse" in code:
                return {"stdout": json.dumps({"success": False, "error": "SyntaxError"}), "stderr": ""}
            return {"stdout": "", "stderr": ""}

        repl.execute = AsyncMock(side_effect=failing_execute)

        llm = MockLLMClient(responses=[
            "<think>Write tool</think>",
            '<answer>' + json.dumps({
                "entities": [
                    {
                        "name": "syntax_error_tool",
                        "type": "MECHANISM",
                        "description": "Tool with syntax error",
                        "properties": {
                            "wrapper_code": "def run(**kw):\n  return {broken",
                            "category": "protein",
                        },
                    }
                ],
                "relationships": [],
                "summary": "test",
            }) + '</answer>',
            '{}',
        ])

        template = get_template(AgentType.TOOL_CREATOR)
        agent = ToolCreatorAgent(
            template=template, llm=llm, kg=agent_kg, tools=mock_tools,
        )

        registry = InMemoryToolRegistry()
        specs = await agent.create_and_register_tools(
            tool_creator_task, registry, repl_tool=repl,
        )

        assert len(specs) == 0

    def test_known_api_catalog_has_entries(self):
        """The known API catalog should contain discoverable APIs."""
        assert len(KNOWN_API_CATALOG) >= 3
        assert "string_db" in KNOWN_API_CATALOG

        for api_id, info in KNOWN_API_CATALOG.items():
            assert "name" in info
            assert "description" in info
            assert "api_base_url" in info
            assert "category" in info

    def test_agent_type_registered(self):
        """TOOL_CREATOR should be in AgentType enum and template registry."""
        assert AgentType.TOOL_CREATOR == "tool_creator"

        from agents.templates import AGENT_TEMPLATES
        assert AgentType.TOOL_CREATOR in AGENT_TEMPLATES

        from agents.factory import _AGENT_CLASS_MAP
        assert AgentType.TOOL_CREATOR in _AGENT_CLASS_MAP


class TestDynamicTool:
    """Tests for the DynamicTool wrapper."""

    @pytest.mark.asyncio
    async def test_dynamic_tool_executes_via_repl(self):
        """DynamicTool should delegate execution to the REPL."""
        from integrations.dynamic.dynamic_tool import DynamicTool

        spec = DynamicToolSpec(
            name="test_tool",
            description="A test dynamic tool",
            wrapper_code='def run(*, query="", **kw): return {"query": query, "result": "ok"}',
            status=DynamicToolStatus.VALIDATED,
        )

        mock_repl = MagicMock()

        async def mock_execute(**kwargs):
            action = kwargs.get("action", "execute")
            if action == "create_session":
                return {"session_id": "sess-1"}
            return {"stdout": json.dumps({"query": "test", "result": "ok"}), "stderr": "", "success": True}

        mock_repl.execute = AsyncMock(side_effect=mock_execute)

        tool = DynamicTool(spec=spec, repl_tool=mock_repl)
        result = await tool._execute(query="test")

        assert result["query"] == "test"
        assert result["result"] == "ok"

    @pytest.mark.asyncio
    async def test_dynamic_tool_handles_repl_error(self):
        """DynamicTool should return error dict if REPL fails."""
        from integrations.dynamic.dynamic_tool import DynamicTool

        spec = DynamicToolSpec(
            name="error_tool",
            description="Tool that errors",
            wrapper_code='def run(**kw): raise ValueError("boom")',
            status=DynamicToolStatus.VALIDATED,
        )

        mock_repl = MagicMock()

        async def mock_execute(**kwargs):
            action = kwargs.get("action", "execute")
            if action == "create_session":
                return {"session_id": "sess-1"}
            return {"stdout": "", "stderr": "ValueError: boom", "error": "ValueError: boom"}

        mock_repl.execute = AsyncMock(side_effect=mock_execute)

        tool = DynamicTool(spec=spec, repl_tool=mock_repl)
        result = await tool._execute(query="test")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_dynamic_tool_no_repl(self):
        """DynamicTool without REPL should return error."""
        from integrations.dynamic.dynamic_tool import DynamicTool

        spec = DynamicToolSpec(name="no_repl_tool", wrapper_code='def run(**kw): return {}')
        tool = DynamicTool(spec=spec, repl_tool=None)
        result = await tool._execute()

        assert "error" in result

    def test_dynamic_tool_registers_as_dynamic(self):
        """DynamicTool should register with source_type=DYNAMIC."""
        from integrations.dynamic.dynamic_tool import DynamicTool

        spec = DynamicToolSpec(
            name="reg_test_tool",
            description="Registration test",
            wrapper_code='def run(**kw): return {}',
            capabilities=["test"],
        )

        registry = InMemoryToolRegistry()
        DynamicTool(spec=spec, repl_tool=None, registry=registry)

        entry = registry.get_tool("reg_test_tool")
        assert entry is not None
        assert entry.source_type.value == "DYNAMIC"
        assert entry.capabilities == ["test"]


class TestDynamicToolSpec:
    """Tests for the DynamicToolSpec model."""

    def test_default_status_is_draft(self):
        """New specs should default to DRAFT status."""
        spec = DynamicToolSpec(name="test")
        assert spec.status == DynamicToolStatus.DRAFT

    def test_spec_fields(self):
        """All spec fields should serialize correctly."""
        spec = DynamicToolSpec(
            name="string_db",
            description="STRING database tool",
            api_base_url="https://string-db.org/api",
            wrapper_code="def run(**kw): return {}",
            category="protein",
            capabilities=["PPI lookup"],
            created_by="agent-tc-001",
        )
        data = spec.model_dump()
        assert data["name"] == "string_db"
        assert data["status"] == "DRAFT"
        assert data["capabilities"] == ["PPI lookup"]
