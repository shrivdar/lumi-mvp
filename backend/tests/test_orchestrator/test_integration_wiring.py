"""Integration tests verifying the full stack wiring.

Checks that agents get tools, trajectory collector runs,
living document attaches, and biosecurity screening fires.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.models import (
    AgentResult,
    AgentType,
    EdgeConfidence,
    EdgeRelationType,
    EvidenceSource,
    EvidenceSourceType,
    KGEdge,
    KGNode,
    NodeType,
    ResearchConfig,
    ScreeningTier,
    SessionStatus,
    ToolRegistryEntry,
    ToolSourceType,
)
from orchestrator.research_loop import ResearchOrchestrator
from world_model.knowledge_graph import InMemoryKnowledgeGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_result(
    agent_type: AgentType = AgentType.LITERATURE_ANALYST,
) -> AgentResult:
    return AgentResult(
        task_id="t1",
        agent_id="a1",
        agent_type=agent_type,
        hypothesis_id="h1",
        nodes_added=[
            KGNode(type=NodeType.GENE, name="BRCA1", created_by="a1"),
        ],
        edges_added=[
            KGEdge(
                source_id="n1",
                target_id="n2",
                relation=EdgeRelationType.ASSOCIATED_WITH,
                confidence=EdgeConfidence(overall=0.7, evidence_quality=0.8),
                evidence=[
                    EvidenceSource(
                        source_type=EvidenceSourceType.PUBMED,
                        source_id="PMID:12345",
                        title="Test paper",
                        claim="BRCA1 is associated",
                    ),
                ],
                created_by="a1",
            ),
        ],
        summary="Found relevant publications",
        success=True,
    )


class MockToolAgent:
    """Mock agent that records which tools were passed to it."""

    def __init__(self, agent_type=None, llm=None, kg=None, yami=None, tools=None, spec=None, **kwargs):
        self.agent_id = "mock-agent-001"
        self.agent_type = agent_type or (spec.agent_type_hint if spec else AgentType.LITERATURE_ANALYST)
        self._tools = tools or {}
        self._result = _make_agent_result(self.agent_type)

    async def execute(self, task):
        task.status = "completed"
        return self._result


def _make_mock_llm() -> MagicMock:
    llm = MagicMock()

    async def mock_query(prompt, system_prompt="", **kwargs):
        if "hypotheses" in prompt.lower() or "competing" in prompt.lower():
            return """[
                {"hypothesis": "BRCA1 drives NSCLC", "rationale": "Tumor suppressor"},
                {"hypothesis": "EGFR pathway involved", "rationale": "Common in NSCLC"},
                {"hypothesis": "Immune evasion via B7-H3", "rationale": "Immune checkpoint"}
            ]"""
        if "composing an agent swarm" in system_prompt.lower() or "agent spec" in prompt.lower():
            return __import__("json").dumps([
                {
                    "role": "Literature analyst for hypothesis investigation",
                    "instructions": "Search PubMed for relevant literature on this hypothesis.",
                    "tools": ["pubmed", "semantic_scholar"],
                    "agent_type_hint": "literature_analyst",
                },
                {
                    "role": "Genomics mapper for pathway analysis",
                    "instructions": "Map genes to pathways relevant to the hypothesis.",
                    "tools": ["pubmed"],
                    "agent_type_hint": "genomics_mapper",
                },
            ])
        if "instruction" in prompt.lower() or "investigation" in prompt.lower():
            return '{"literature_analyst": "Search for BRCA1 in NSCLC", "genomics_mapper": "Map gene pathways", "scientific_critic": "Verify claims"}'
        if "select" in prompt.lower() and "tool" in prompt.lower():
            return '["pubmed", "semantic_scholar"]'
        return '{"summary": "mock response"}'

    llm.query = AsyncMock(side_effect=mock_query)
    llm.parse_json = MagicMock(side_effect=lambda text: __import__("json").loads(text))
    llm.token_summary = {"calls": 0, "total_tokens": 0}
    return llm


def _make_tool_entries() -> list[ToolRegistryEntry]:
    """Create mock tool registry entries."""
    return [
        ToolRegistryEntry(
            name="pubmed",
            description="Search PubMed biomedical literature",
            category="literature_search",
            source_type=ToolSourceType.NATIVE,
            enabled=True,
        ),
        ToolRegistryEntry(
            name="semantic_scholar",
            description="Search Semantic Scholar",
            category="literature_search",
            source_type=ToolSourceType.NATIVE,
            enabled=True,
        ),
        ToolRegistryEntry(
            name="python_repl",
            description="Execute Python code in sandboxed environment",
            category="code_execution",
            source_type=ToolSourceType.NATIVE,
            enabled=True,
        ),
    ]


def _make_tool_instances() -> dict:
    """Create mock tool instances."""
    pubmed = MagicMock()
    pubmed.name = "pubmed"
    pubmed.description = "Search PubMed"

    s2 = MagicMock()
    s2.name = "semantic_scholar"
    s2.description = "Search Semantic Scholar"

    repl = MagicMock()
    repl.name = "python_repl"
    repl.description = "Execute Python code"

    return {"pubmed": pubmed, "semantic_scholar": s2, "python_repl": repl}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolWiring:
    """Verify agents receive dynamically-selected tools."""

    @pytest.mark.asyncio
    async def test_agents_receive_tools(self) -> None:
        """Spec factory is called with tools dict from dynamic selection."""
        llm = _make_mock_llm()
        kg = InMemoryKnowledgeGraph(graph_id="test-wiring")
        tool_instances = _make_tool_instances()
        tools_received: list[dict] = []

        def tracking_spec_factory(spec, llm, kg, yami=None, tools=None, **kwargs):
            tools_received.append(dict(tools) if tools else {})
            return MockToolAgent(spec=spec, llm=llm, kg=kg, tools=tools)

        orch = ResearchOrchestrator(
            llm=llm,
            kg=kg,
            spec_factory=tracking_spec_factory,
            tool_entries=_make_tool_entries(),
            tool_instances=tool_instances,
        )

        config = ResearchConfig(max_mcts_iterations=1, max_hypothesis_depth=1)
        session = await orch.run("B7-H3 in NSCLC", config)

        assert session.status == SessionStatus.COMPLETED
        # At least one agent should have been passed tools
        assert len(tools_received) > 0
        # Every agent should have python_repl (always injected)
        for tr in tools_received:
            assert "python_repl" in tr, f"python_repl missing from tools: {list(tr.keys())}"

    @pytest.mark.asyncio
    async def test_agents_get_python_repl_even_without_selection(self) -> None:
        """python_repl is always injected even if tool selection fails."""
        llm = _make_mock_llm()
        kg = InMemoryKnowledgeGraph(graph_id="test-repl-inject")

        # Only provide python_repl, no tool entries (so selection will have nothing)
        repl = MagicMock()
        repl.name = "python_repl"
        tool_instances = {"python_repl": repl}
        tools_received: list[dict] = []

        def tracking_spec_factory(spec, llm, kg, yami=None, tools=None, **kwargs):
            tools_received.append(dict(tools) if tools else {})
            return MockToolAgent(spec=spec, llm=llm, kg=kg, tools=tools)

        orch = ResearchOrchestrator(
            llm=llm,
            kg=kg,
            spec_factory=tracking_spec_factory,
            tool_instances=tool_instances,
        )

        config = ResearchConfig(max_mcts_iterations=1, max_hypothesis_depth=1)
        await orch.run("Test query", config)

        assert len(tools_received) > 0
        for tr in tools_received:
            assert "python_repl" in tr


class TestTrajectoryCollection:
    """Verify trajectory collector hooks into agent execution."""

    @pytest.mark.asyncio
    async def test_trajectories_collected(self) -> None:
        """TrajectoryCollector.collect() is called for each agent result."""
        llm = _make_mock_llm()
        kg = InMemoryKnowledgeGraph(graph_id="test-traj")

        def factory(agent_type, llm, kg, yami=None, **kwargs):
            return MockToolAgent(agent_type, llm, kg)

        orch = ResearchOrchestrator(llm=llm, kg=kg, agent_factory=factory)

        config = ResearchConfig(max_mcts_iterations=1, max_hypothesis_depth=1)

        with patch("orchestrator.research_loop.TrajectoryCollector") as MockTC:
            mock_collector = MagicMock()
            MockTC.return_value = mock_collector

            session = await orch.run("B7-H3 in NSCLC", config)

            assert session.status == SessionStatus.COMPLETED
            # collect() should have been called for each agent execution
            assert mock_collector.collect.call_count > 0
            # flush() should have been called in finally block
            assert mock_collector.flush.call_count == 1


class TestLivingDocument:
    """Verify living document attaches to KG and produces output."""

    @pytest.mark.asyncio
    async def test_living_doc_attached_and_rendered(self) -> None:
        """LivingDocument is created, attached to KG, and rendered."""
        llm = _make_mock_llm()
        kg = InMemoryKnowledgeGraph(graph_id="test-living")

        def factory(agent_type, llm, kg, yami=None, **kwargs):
            return MockToolAgent(agent_type, llm, kg)

        orch = ResearchOrchestrator(llm=llm, kg=kg, agent_factory=factory)

        config = ResearchConfig(max_mcts_iterations=1, max_hypothesis_depth=1)

        with patch("orchestrator.research_loop.LivingDocument") as MockLD:
            mock_doc = MagicMock()
            mock_doc.render.return_value = "# Research Report\n\nLive findings..."
            MockLD.return_value = mock_doc

            session = await orch.run("B7-H3 in NSCLC", config)

            assert session.status == SessionStatus.COMPLETED
            # attach() should have been called with the KG
            mock_doc.attach.assert_called_once_with(kg)
            # render() should have been called to snapshot the document
            mock_doc.render.assert_called()
            # Result should contain the living document
            assert session.result is not None
            assert session.result.living_document == "# Research Report\n\nLive findings..."
            # detach() should have been called in finally
            mock_doc.detach.assert_called()


class TestBiosecurityIntegration:
    """Verify biosecurity screening runs on compiled results."""

    @pytest.mark.asyncio
    async def test_screening_runs_and_result_attached(self) -> None:
        """Biosecurity screening result is attached to research result."""
        llm = _make_mock_llm()
        kg = InMemoryKnowledgeGraph(graph_id="test-bio")

        def factory(agent_type, llm, kg, yami=None, **kwargs):
            return MockToolAgent(agent_type, llm, kg)

        orch = ResearchOrchestrator(llm=llm, kg=kg, agent_factory=factory)

        config = ResearchConfig(max_mcts_iterations=1, max_hypothesis_depth=1)
        session = await orch.run("B7-H3 in NSCLC", config)

        assert session.status == SessionStatus.COMPLETED
        assert session.result is not None
        # Screening result should be attached
        assert session.result.screening is not None
        assert session.result.screening.tier in (
            ScreeningTier.CLEAR,
            ScreeningTier.WARNING,
            ScreeningTier.BLOCKED,
        )


class TestFullStackIntegration:
    """End-to-end test: all components wired together."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_tools_and_collection(self) -> None:
        """Full pipeline: tools → agents → KG → trajectory → living doc → screening."""
        llm = _make_mock_llm()
        kg = InMemoryKnowledgeGraph(graph_id="test-full")
        tool_instances = _make_tool_instances()
        agents_created: list[MockToolAgent] = []

        def tracking_spec_factory(spec, llm, kg, yami=None, tools=None, **kwargs):
            agent = MockToolAgent(spec=spec, llm=llm, kg=kg, tools=tools)
            agents_created.append(agent)
            return agent

        orch = ResearchOrchestrator(
            llm=llm,
            kg=kg,
            spec_factory=tracking_spec_factory,
            tool_entries=_make_tool_entries(),
            tool_instances=tool_instances,
        )

        config = ResearchConfig(max_mcts_iterations=2, max_hypothesis_depth=2)
        session = await orch.run("B7-H3 in NSCLC", config)

        # Session completes
        assert session.status == SessionStatus.COMPLETED
        assert session.result is not None

        # Agents were created with tools
        assert len(agents_created) > 0
        for agent in agents_created:
            assert "python_repl" in agent._tools

        # Result has all integration outputs
        assert session.result.screening is not None
        assert session.result.living_document  # non-empty string

        # Events were emitted
        events = orch.drain_events()
        event_types = {e.event_type for e in events}
        assert "session_created" in event_types
        assert "agent_started" in event_types
        assert "agent_completed" in event_types
        assert "biosecurity_screening_completed" in event_types
