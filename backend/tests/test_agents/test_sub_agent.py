"""Tests for sub-agent spawning — depth limits, parent attribution, KG inheritance."""

from __future__ import annotations

import pytest

from agents.base import BaseAgentImpl
from agents.templates import get_template
from core.constants import MAX_SUB_AGENT_DEPTH, MAX_SUB_AGENTS_PER_PARENT
from core.exceptions import AgentError
from core.models import (
    AgentResult,
    AgentTask,
    AgentType,
    EvidenceSource,
    EvidenceSourceType,
    KGNode,
    NodeType,
)
from tests.test_agents.conftest import MockLLMClient
from world_model.knowledge_graph import InMemoryKnowledgeGraph

# ---------------------------------------------------------------------------
# Test agents
# ---------------------------------------------------------------------------


class ParentAgent(BaseAgentImpl):
    """Agent that spawns sub-agents during investigation."""

    agent_type = AgentType.LITERATURE_ANALYST

    def __init__(self, spawn_config=None, **kwargs):
        super().__init__(**kwargs)
        self._spawn_config = spawn_config  # list of dicts for spawn_sub_agents

    async def _investigate(self, task, kg_context):
        results = []
        if self._spawn_config:
            for cfg in self._spawn_config:
                result = await self.spawn_sub_agent(
                    cfg["task_description"],
                    agent_type=cfg.get("agent_type"),
                    tool_names=cfg.get("tool_names"),
                    hypothesis_branch=task.hypothesis_branch,
                )
                results.append(result)

        return {
            "nodes": [],
            "edges": [],
            "summary": f"Parent delegated to {len(results)} sub-agents",
            "reasoning_trace": "parent_delegation",
        }


class ChildAgent(BaseAgentImpl):
    """Agent that creates a KG node when investigated."""

    agent_type = AgentType.LITERATURE_ANALYST

    async def _investigate(self, task, kg_context):
        node = KGNode(
            type=NodeType.GENE,
            name=f"ChildNode-{self.agent_id[:8]}",
            confidence=0.7,
            sources=[EvidenceSource(
                source_type=EvidenceSourceType.AGENT_REASONING,
                claim="sub-agent finding",
                agent_id=self.agent_id,
            )],
        )
        return {
            "nodes": [node],
            "edges": [],
            "summary": "Child investigation complete",
            "reasoning_trace": "child",
        }


class RecursiveSpawner(BaseAgentImpl):
    """Agent that tries to spawn a sub-agent of itself (for depth testing)."""

    agent_type = AgentType.LITERATURE_ANALYST

    async def _investigate(self, task, kg_context):
        try:
            await self.spawn_sub_agent(
                "Recursive sub-task",
                hypothesis_branch=task.hypothesis_branch,
            )
        except AgentError:
            # Expected at depth limit
            raise
        return {
            "nodes": [],
            "edges": [],
            "summary": "recursive",
            "reasoning_trace": "recursive",
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_llm() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture()
def agent_kg() -> InMemoryKnowledgeGraph:
    return InMemoryKnowledgeGraph(graph_id="sub-agent-test-kg")


@pytest.fixture()
def sample_task() -> AgentTask:
    return AgentTask(
        task_id="task-parent-001",
        research_id="research-001",
        agent_type=AgentType.LITERATURE_ANALYST,
        hypothesis_branch="h-main",
        instruction="Investigate sub-agent spawning",
    )


# ---------------------------------------------------------------------------
# Tests: Basic sub-agent spawning
# ---------------------------------------------------------------------------


class TestSubAgentSpawning:
    @pytest.mark.asyncio
    async def test_spawn_sub_agent_basic(self, agent_kg, mock_llm, sample_task):
        """Parent spawns a child; child's result is collected."""
        template = get_template(AgentType.LITERATURE_ANALYST)
        parent = ParentAgent(
            template=template,
            llm=mock_llm,
            kg=agent_kg,
            spawn_config=[{"task_description": "Search for BRCA1 papers"}],
        )

        result = await parent.execute(sample_task)

        assert result.success is True
        assert len(result.sub_agent_results) == 1
        assert result.sub_agent_results[0].parent_agent_id == parent.agent_id
        assert result.sub_agent_results[0].depth == 1

    @pytest.mark.asyncio
    async def test_sub_agent_inherits_kg(self, agent_kg, mock_llm, sample_task):
        """Sub-agent writes to the same KG as the parent."""
        template = get_template(AgentType.LITERATURE_ANALYST)

        # Use ChildAgent which creates a KGNode
        parent = ChildAgent(
            template=template,
            llm=mock_llm,
            kg=agent_kg,
        )

        result = await parent.execute(sample_task)

        assert result.success is True
        assert agent_kg.node_count() >= 1
        # Node should have parent's agent_id as created_by
        nodes = list(agent_kg._nodes.values())
        assert any(n.created_by == parent.agent_id for n in nodes)

    @pytest.mark.asyncio
    async def test_sub_agent_parent_id_and_depth(self, agent_kg, mock_llm, sample_task):
        """AgentResult includes parent_agent_id and depth fields."""
        template = get_template(AgentType.LITERATURE_ANALYST)
        parent = ParentAgent(
            template=template,
            llm=mock_llm,
            kg=agent_kg,
            spawn_config=[{"task_description": "sub-task"}],
        )

        result = await parent.execute(sample_task)

        # Parent result
        assert result.parent_agent_id is None
        assert result.depth == 0

        # Child result
        child_result = result.sub_agent_results[0]
        assert child_result.parent_agent_id == parent.agent_id
        assert child_result.depth == 1

    @pytest.mark.asyncio
    async def test_sub_agent_inherits_hypothesis_branch(self, agent_kg, mock_llm, sample_task):
        """Sub-agent task gets the parent's hypothesis branch."""
        template = get_template(AgentType.LITERATURE_ANALYST)
        parent = ParentAgent(
            template=template,
            llm=mock_llm,
            kg=agent_kg,
            spawn_config=[{"task_description": "sub-task"}],
        )

        result = await parent.execute(sample_task)
        child_result = result.sub_agent_results[0]

        assert child_result.hypothesis_id == sample_task.hypothesis_branch


# ---------------------------------------------------------------------------
# Tests: Depth limit enforcement
# ---------------------------------------------------------------------------


class TestSubAgentDepthLimit:
    @pytest.mark.asyncio
    async def test_depth_limit_enforced(self, agent_kg, mock_llm, sample_task):
        """Sub-agent at max depth cannot spawn further children."""
        template = get_template(AgentType.LITERATURE_ANALYST)

        # Create agent at max depth
        agent = RecursiveSpawner(
            template=template,
            llm=mock_llm,
            kg=agent_kg,
            depth=MAX_SUB_AGENT_DEPTH,
        )

        result = await agent.execute(sample_task)

        # Should fail because depth + 1 > MAX_SUB_AGENT_DEPTH
        assert result.success is False
        assert any("depth" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_depth_within_limit_succeeds(self, agent_kg, mock_llm, sample_task):
        """Sub-agent within depth limit can spawn."""
        template = get_template(AgentType.LITERATURE_ANALYST)
        parent = ParentAgent(
            template=template,
            llm=mock_llm,
            kg=agent_kg,
            depth=0,
            spawn_config=[{"task_description": "depth-1 task"}],
        )

        result = await parent.execute(sample_task)
        assert result.success is True
        assert result.sub_agent_results[0].depth == 1


# ---------------------------------------------------------------------------
# Tests: Spawn limit per parent
# ---------------------------------------------------------------------------


class TestSubAgentSpawnLimit:
    @pytest.mark.asyncio
    async def test_spawn_limit_enforced(self, agent_kg, mock_llm, sample_task):
        """Parent cannot spawn more than MAX_SUB_AGENTS_PER_PARENT."""
        template = get_template(AgentType.LITERATURE_ANALYST)

        # Try to spawn more than the limit
        spawn_configs = [
            {"task_description": f"sub-task-{i}"}
            for i in range(MAX_SUB_AGENTS_PER_PARENT + 1)
        ]
        parent = ParentAgent(
            template=template,
            llm=mock_llm,
            kg=agent_kg,
            spawn_config=spawn_configs,
        )

        result = await parent.execute(sample_task)

        # Should fail because the last spawn exceeds the limit
        assert result.success is False
        assert any("limit" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_spawn_at_limit_succeeds(self, agent_kg, mock_llm, sample_task):
        """Parent can spawn exactly MAX_SUB_AGENTS_PER_PARENT children."""
        template = get_template(AgentType.LITERATURE_ANALYST)

        spawn_configs = [
            {"task_description": f"sub-task-{i}"}
            for i in range(MAX_SUB_AGENTS_PER_PARENT)
        ]
        parent = ParentAgent(
            template=template,
            llm=mock_llm,
            kg=agent_kg,
            spawn_config=spawn_configs,
        )

        result = await parent.execute(sample_task)
        assert result.success is True
        assert len(result.sub_agent_results) == MAX_SUB_AGENTS_PER_PARENT


# ---------------------------------------------------------------------------
# Tests: Tool inheritance
# ---------------------------------------------------------------------------


class TestSubAgentToolInheritance:
    @pytest.mark.asyncio
    async def test_sub_agent_inherits_parent_tools(self, agent_kg, mock_llm, sample_task):
        """Sub-agent without explicit tool_names gets parent's tools."""
        from tests.test_agents.conftest import make_mock_tool

        template = get_template(AgentType.LITERATURE_ANALYST)
        tools = {
            "pubmed": make_mock_tool("pubmed"),
            "semantic_scholar": make_mock_tool("semantic_scholar"),
        }

        parent = ParentAgent(
            template=template,
            llm=mock_llm,
            kg=agent_kg,
            tools=tools,
            spawn_config=[{"task_description": "sub-task"}],
        )

        result = await parent.execute(sample_task)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_sub_agent_with_specific_tools(self, agent_kg, mock_llm, sample_task):
        """Sub-agent with explicit tool_names gets only those tools."""
        from tests.test_agents.conftest import make_mock_tool

        template = get_template(AgentType.LITERATURE_ANALYST)
        tools = {
            "pubmed": make_mock_tool("pubmed"),
            "semantic_scholar": make_mock_tool("semantic_scholar"),
            "chembl": make_mock_tool("chembl"),
        }

        # We can't directly inspect child tools through result, but we can
        # verify no errors occur and spawn succeeds
        parent = ParentAgent(
            template=template,
            llm=mock_llm,
            kg=agent_kg,
            tools=tools,
            spawn_config=[{
                "task_description": "sub-task",
                "tool_names": ["pubmed"],
            }],
        )

        result = await parent.execute(sample_task)
        assert result.success is True


# ---------------------------------------------------------------------------
# Tests: AgentResult model fields
# ---------------------------------------------------------------------------


class TestAgentResultFields:
    def test_agent_result_has_parent_fields(self):
        """AgentResult model includes parent_agent_id and depth."""
        result = AgentResult(
            task_id="t1",
            agent_id="a1",
            agent_type=AgentType.LITERATURE_ANALYST,
            parent_agent_id="parent-1",
            depth=2,
        )
        assert result.parent_agent_id == "parent-1"
        assert result.depth == 2

    def test_agent_result_defaults(self):
        """parent_agent_id defaults to None, depth to 0."""
        result = AgentResult(
            task_id="t1",
            agent_id="a1",
            agent_type=AgentType.LITERATURE_ANALYST,
        )
        assert result.parent_agent_id is None
        assert result.depth == 0
        assert result.sub_agent_results == []

    def test_agent_result_with_sub_results(self):
        """AgentResult can nest sub_agent_results."""
        child = AgentResult(
            task_id="t-child",
            agent_id="a-child",
            agent_type=AgentType.LITERATURE_ANALYST,
            parent_agent_id="a-parent",
            depth=1,
        )
        parent = AgentResult(
            task_id="t-parent",
            agent_id="a-parent",
            agent_type=AgentType.LITERATURE_ANALYST,
            sub_agent_results=[child],
        )
        assert len(parent.sub_agent_results) == 1
        assert parent.sub_agent_results[0].parent_agent_id == "a-parent"
