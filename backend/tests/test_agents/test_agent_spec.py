"""Tests for AgentSpec — dynamic agent creation, spec-driven constraints, and factory."""

from __future__ import annotations

import pytest

from agents.base import BaseAgentImpl
from agents.factory import create_agent_from_spec
from agents.templates import get_template
from core.models import (
    AgentConstraints,
    AgentSpec,
    AgentTask,
    AgentType,
    EdgeRelationType,
    EvidenceSource,
    EvidenceSourceType,
    KGNode,
    NodeType,
)
from tests.test_agents.conftest import MockLLMClient
from world_model.knowledge_graph import InMemoryKnowledgeGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubSpecAgent(BaseAgentImpl):
    """Stub that returns canned results, for testing spec plumbing."""

    agent_type = AgentType.LITERATURE_ANALYST

    def __init__(self, nodes_to_return=None, **kwargs):
        super().__init__(**kwargs)
        self._stub_nodes = nodes_to_return or []

    async def _investigate(self, task, kg_context):
        return {
            "nodes": self._stub_nodes,
            "edges": [],
            "summary": "Stub spec investigation",
            "reasoning_trace": "stub_spec",
        }


@pytest.fixture()
def mock_llm() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture()
def agent_kg() -> InMemoryKnowledgeGraph:
    return InMemoryKnowledgeGraph(graph_id="spec-test-kg")


@pytest.fixture()
def sample_spec() -> AgentSpec:
    return AgentSpec(
        role="Literature searcher for BRCA1",
        instructions="Search PubMed for BRCA1 papers and extract key findings.",
        tools=["pubmed", "semantic_scholar"],
        constraints=AgentConstraints(
            max_turns=10,
            token_budget=25_000,
            timeout_seconds=120,
            max_llm_calls=15,
        ),
        hypothesis_branch="h-brca1",
        agent_type_hint=AgentType.LITERATURE_ANALYST,
        system_prompt="You are a focused literature searcher.",
        kg_write_permissions=[NodeType.GENE, NodeType.PROTEIN],
        kg_edge_permissions=[EdgeRelationType.ASSOCIATED_WITH],
        falsification_protocol="Search for contradicting publications.",
    )


@pytest.fixture()
def sample_task() -> AgentTask:
    return AgentTask(
        task_id="task-spec-001",
        research_id="research-001",
        agent_type=AgentType.LITERATURE_ANALYST,
        hypothesis_branch="h-brca1",
        instruction="Find BRCA1 literature",
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestAgentSpecModel:
    def test_default_constraints(self):
        spec = AgentSpec(role="test", instructions="test")
        assert spec.constraints.max_turns == 200
        assert spec.constraints.token_budget == 50_000
        assert spec.constraints.timeout_seconds == 300
        assert spec.constraints.max_llm_calls == 20

    def test_custom_constraints(self, sample_spec):
        assert sample_spec.constraints.max_turns == 10
        assert sample_spec.constraints.token_budget == 25_000

    def test_agent_type_hint_optional(self):
        spec = AgentSpec(role="custom", instructions="do stuff")
        assert spec.agent_type_hint is None

    def test_spec_with_permissions(self, sample_spec):
        assert NodeType.GENE in sample_spec.kg_write_permissions
        assert EdgeRelationType.ASSOCIATED_WITH in sample_spec.kg_edge_permissions


# ---------------------------------------------------------------------------
# BaseAgentImpl with spec
# ---------------------------------------------------------------------------


class TestBaseAgentWithSpec:
    def test_init_with_spec_only(self, mock_llm, agent_kg, sample_spec):
        agent = BaseAgentImpl(spec=sample_spec, llm=mock_llm, kg=agent_kg)
        assert agent.spec is sample_spec
        assert agent.template is None
        assert agent.agent_type == AgentType.LITERATURE_ANALYST

    def test_init_with_both_template_and_spec(self, mock_llm, agent_kg, sample_spec):
        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = BaseAgentImpl(template=template, spec=sample_spec, llm=mock_llm, kg=agent_kg)
        assert agent.spec is sample_spec
        assert agent.template is template
        # Spec hint takes precedence
        assert agent.agent_type == AgentType.LITERATURE_ANALYST

    def test_init_requires_template_or_spec(self, mock_llm, agent_kg):
        with pytest.raises(ValueError, match="Either template or spec"):
            BaseAgentImpl(llm=mock_llm, kg=agent_kg)

    def test_effective_system_prompt_from_spec(self, mock_llm, agent_kg, sample_spec):
        agent = BaseAgentImpl(spec=sample_spec, llm=mock_llm, kg=agent_kg)
        assert agent.effective_system_prompt == "You are a focused literature searcher."

    def test_effective_system_prompt_from_template(self, mock_llm, agent_kg):
        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = BaseAgentImpl(template=template, llm=mock_llm, kg=agent_kg)
        assert "biomedical literature analyst" in agent.effective_system_prompt

    def test_effective_system_prompt_spec_overrides_template(self, mock_llm, agent_kg, sample_spec):
        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = BaseAgentImpl(template=template, spec=sample_spec, llm=mock_llm, kg=agent_kg)
        assert agent.effective_system_prompt == "You are a focused literature searcher."

    def test_effective_kg_permissions_from_spec(self, mock_llm, agent_kg, sample_spec):
        agent = BaseAgentImpl(spec=sample_spec, llm=mock_llm, kg=agent_kg)
        assert agent.effective_kg_write_permissions == [NodeType.GENE, NodeType.PROTEIN]
        assert agent.effective_kg_edge_permissions == [EdgeRelationType.ASSOCIATED_WITH]

    def test_effective_constraints_from_spec(self, mock_llm, agent_kg, sample_spec):
        agent = BaseAgentImpl(spec=sample_spec, llm=mock_llm, kg=agent_kg)
        c = agent.effective_constraints
        assert c.max_turns == 10
        assert c.token_budget == 25_000

    def test_effective_constraints_from_template(self, mock_llm, agent_kg):
        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = BaseAgentImpl(template=template, llm=mock_llm, kg=agent_kg)
        c = agent.effective_constraints
        assert c.max_turns == template.max_iterations * 2
        assert c.timeout_seconds == template.timeout_seconds

    def test_effective_falsification_from_spec(self, mock_llm, agent_kg, sample_spec):
        agent = BaseAgentImpl(spec=sample_spec, llm=mock_llm, kg=agent_kg)
        assert "contradicting" in agent.effective_falsification_protocol

    def test_parent_agent_id_from_spec(self, mock_llm, agent_kg):
        spec = AgentSpec(
            role="child", instructions="sub-task",
            parent_agent_id="parent-abc",
        )
        agent = BaseAgentImpl(spec=spec, llm=mock_llm, kg=agent_kg)
        assert agent.parent_agent_id == "parent-abc"


class TestBaseAgentSpecExecute:
    @pytest.mark.asyncio
    async def test_execute_with_spec_returns_result(self, mock_llm, agent_kg, sample_task):
        """A spec-only agent (no subclass) should run _investigate via multi-turn loop."""
        # Provide enough mock responses for the planning turn + answer turn
        llm = MockLLMClient(responses=[
            "<think>Plan: search PubMed for BRCA1</think>",
            '<answer>{"entities": [], "relationships": [], '
            '"summary": "No results", "reasoning_trace": "searched"}</answer>',
        ])
        spec = AgentSpec(
            role="Lit searcher",
            instructions="Search PubMed",
            constraints=AgentConstraints(max_turns=5, token_budget=100_000),
        )
        agent = BaseAgentImpl(spec=spec, llm=llm, kg=agent_kg)
        result = await agent.execute(sample_task)

        assert result.success is True
        assert result.summary  # should have something

    @pytest.mark.asyncio
    async def test_execute_with_template_still_works(self, mock_llm, agent_kg, sample_task):
        """Template-based StubAgent should still work fine."""
        template = get_template(AgentType.LITERATURE_ANALYST)
        node = KGNode(
            type=NodeType.GENE, name="TestGene", confidence=0.8,
            sources=[EvidenceSource(source_type=EvidenceSourceType.PUBMED, agent_id="t")],
        )
        agent = StubSpecAgent(template=template, llm=mock_llm, kg=agent_kg, nodes_to_return=[node])
        result = await agent.execute(sample_task)

        assert result.success is True
        assert len(result.nodes_added) == 1


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestCreateAgentFromSpec:
    def test_create_with_type_hint(self, mock_llm, agent_kg):
        spec = AgentSpec(
            role="Literature search",
            instructions="Search papers",
            agent_type_hint=AgentType.LITERATURE_ANALYST,
        )
        agent = create_agent_from_spec(spec, llm=mock_llm, kg=agent_kg)
        # Should be the specialised subclass
        from agents.literature_analyst import LiteratureAnalystAgent
        assert isinstance(agent, LiteratureAnalystAgent)
        assert agent.spec is spec
        assert agent.template is not None  # should also have the template

    def test_create_without_type_hint(self, mock_llm, agent_kg):
        spec = AgentSpec(
            role="Custom investigator",
            instructions="Do custom investigation",
        )
        agent = create_agent_from_spec(spec, llm=mock_llm, kg=agent_kg)
        assert type(agent) is BaseAgentImpl
        assert agent.spec is spec
        assert agent.template is None

    def test_create_with_unknown_type_hint(self, mock_llm, agent_kg):
        """If the type hint isn't in the class map, fall back to base."""
        spec = AgentSpec(
            role="Weird agent",
            instructions="Do weird stuff",
            agent_type_hint=AgentType.LITERATURE_ANALYST,
        )
        # Normally this would use the subclass, but let's test the generic path
        agent = create_agent_from_spec(spec, llm=mock_llm, kg=agent_kg)
        assert agent.spec is spec

    def test_create_passes_tools(self, mock_llm, agent_kg):
        from tests.test_agents.conftest import make_mock_tool

        tools = {"pubmed": make_mock_tool("pubmed")}
        spec = AgentSpec(role="test", instructions="test", tools=["pubmed"])
        agent = create_agent_from_spec(spec, llm=mock_llm, kg=agent_kg, tools=tools)
        assert "pubmed" in agent.tools


# ---------------------------------------------------------------------------
# Spec-driven spawn_sub_agent
# ---------------------------------------------------------------------------


class TestSpawnSubAgentWithSpec:
    @pytest.mark.asyncio
    async def test_spawn_with_spec(self, mock_llm, agent_kg, sample_task):
        """Parent can spawn a sub-agent using an AgentSpec."""
        # Parent agent (template-based)
        template = get_template(AgentType.LITERATURE_ANALYST)
        llm = MockLLMClient(responses=[
            # Parent's investigation (stub, not used because we override _investigate)
            # Sub-agent planning turn
            "<think>Plan: do sub-task</think>",
            # Sub-agent answer turn
            '<answer>{"entities": [], "relationships": [], '
            '"summary": "Sub done", "reasoning_trace": "sub"}</answer>',
        ])

        class SpawnWithSpecAgent(BaseAgentImpl):
            agent_type = AgentType.LITERATURE_ANALYST

            async def _investigate(self, task, kg_context):
                child_spec = AgentSpec(
                    role="Sub-investigator",
                    instructions="Investigate sub-topic",
                    constraints=AgentConstraints(max_turns=3),
                )
                result = await self.spawn_sub_agent(
                    "Sub-task description",
                    spec=child_spec,
                    hypothesis_branch=task.hypothesis_branch,
                )
                return {
                    "nodes": [],
                    "edges": [],
                    "summary": f"Spawned sub: {result.summary}",
                    "reasoning_trace": "parent",
                }

        parent = SpawnWithSpecAgent(template=template, llm=llm, kg=agent_kg)
        result = await parent.execute(sample_task)

        assert result.success is True
        assert len(result.sub_agent_results) == 1
        child = result.sub_agent_results[0]
        assert child.parent_agent_id == parent.agent_id
        assert child.depth == 1


# ---------------------------------------------------------------------------
# Spec with no KG permissions defaults
# ---------------------------------------------------------------------------


class TestSpecDefaults:
    def test_spec_without_permissions_gets_all_types(self, mock_llm, agent_kg):
        """Spec with empty permissions → all node/edge types allowed."""
        spec = AgentSpec(role="test", instructions="test")
        agent = BaseAgentImpl(spec=spec, llm=mock_llm, kg=agent_kg)
        assert len(agent.effective_kg_write_permissions) == len(NodeType)
        assert len(agent.effective_kg_edge_permissions) == len(EdgeRelationType)

    def test_spec_with_empty_system_prompt_uses_empty(self, mock_llm, agent_kg):
        spec = AgentSpec(role="test", instructions="test", system_prompt="")
        agent = BaseAgentImpl(spec=spec, llm=mock_llm, kg=agent_kg)
        assert agent.effective_system_prompt == ""

    def test_agent_type_defaults_to_class_level(self, mock_llm, agent_kg):
        """Spec without agent_type_hint → agent keeps class default."""
        spec = AgentSpec(role="test", instructions="test")
        agent = BaseAgentImpl(spec=spec, llm=mock_llm, kg=agent_kg)
        assert agent.agent_type == AgentType.LITERATURE_ANALYST  # class default
