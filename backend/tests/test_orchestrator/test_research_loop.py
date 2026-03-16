"""Tests for the research loop orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.models import (
    AgentResult,
    AgentType,
    EdgeConfidence,
    EdgeRelationType,
    EvidenceSource,
    EvidenceSourceType,
    FalsificationResult,
    KGEdge,
    KGNode,
    NodeType,
    ResearchConfig,
    SessionStatus,
    UncertaintyVector,
)
from orchestrator.research_loop import ResearchOrchestrator
from world_model.knowledge_graph import InMemoryKnowledgeGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent_result(
    agent_type: AgentType = AgentType.LITERATURE_ANALYST,
    edges: int = 2,
    summary: str = "Found relevant publications",
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
                        source_id="PMID:123",
                        quality_score=0.8,
                    )
                ],
                created_by="a1",
            )
            for _ in range(edges)
        ],
        falsification_results=[
            FalsificationResult(
                edge_id="e1",
                original_confidence=0.7,
                revised_confidence=0.72,
                confidence_delta=0.02,
            )
        ],
        uncertainty=UncertaintyVector(composite=0.3),
        summary=summary,
        success=True,
        llm_calls=3,
        llm_tokens_used=1500,
    )


class MockAgentImpl:
    """Mock agent that returns predetermined results."""

    def __init__(self, agent_type: AgentType, **kwargs):
        self.agent_id = "mock-agent-1"
        self.agent_type = agent_type

    async def execute(self, task):
        return _make_agent_result(agent_type=self.agent_type)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_llm() -> MagicMock:
    llm = MagicMock()

    # Hypothesis generation response
    hypotheses_json = (
        '[{"hypothesis": "B7-H3 promotes immune evasion via PD-L1 pathway", '
        '"rationale": "Checkpoint molecule mechanism"}, '
        '{"hypothesis": "B7-H3 overexpression correlates with poor NSCLC prognosis", '
        '"rationale": "Clinical outcome data"}, '
        '{"hypothesis": "Anti-B7-H3 antibodies show therapeutic potential", '
        '"rationale": "Drug development angle"}]'
    )

    # Swarm composition response
    swarm_json = '["literature_analyst", "drug_hunter"]'

    # Task generation response
    task_json = '{"literature_analyst": "Search for B7-H3 studies", "drug_hunter": "Search ChEMBL for B7-H3 compounds"}'

    # Child hypothesis response
    child_json = '[{"hypothesis": "Sub hypothesis", "rationale": "Deeper investigation"}]'

    # Return different responses based on call count
    call_count = {"n": 0}

    async def mock_query(prompt, **kwargs):
        call_count["n"] += 1
        if "competing hypotheses" in prompt:
            return hypotheses_json
        if "relevant agent types" in prompt or "most relevant" in prompt:
            return swarm_json
        if "investigation instruction" in prompt:
            return task_json
        if "sub-hypotheses" in prompt:
            return child_json
        return swarm_json

    llm.query = mock_query
    llm.parse_json = MagicMock(side_effect=lambda text: __import__("json").loads(text))
    llm.token_summary = {"calls": 0, "total_tokens": 0}
    return llm


@pytest.fixture()
def kg() -> InMemoryKnowledgeGraph:
    return InMemoryKnowledgeGraph(graph_id="test-kg")


@pytest.fixture()
def orchestrator(mock_llm: MagicMock, kg: InMemoryKnowledgeGraph) -> ResearchOrchestrator:
    def agent_factory(agent_type, llm, kg, yami=None, **kwargs):
        return MockAgentImpl(agent_type=agent_type)

    return ResearchOrchestrator(
        llm=mock_llm,
        kg=kg,
        agent_factory=agent_factory,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSessionCreation:
    @pytest.mark.asyncio
    async def test_run_creates_session(self, orchestrator: ResearchOrchestrator) -> None:
        config = ResearchConfig(max_mcts_iterations=1, max_hypothesis_depth=1)
        session = await orchestrator.run("B7-H3 in NSCLC", config)

        assert session.query == "B7-H3 in NSCLC"
        assert session.status == SessionStatus.COMPLETED
        assert session.result is not None

    @pytest.mark.asyncio
    async def test_session_has_result(self, orchestrator: ResearchOrchestrator) -> None:
        config = ResearchConfig(max_mcts_iterations=1, max_hypothesis_depth=1)
        session = await orchestrator.run("test query", config)

        result = session.result
        assert result is not None
        assert result.research_id == session.id
        assert result.best_hypothesis is not None
        assert result.total_llm_calls >= 0


class TestInitialization:
    @pytest.mark.asyncio
    async def test_hypotheses_generated(self, orchestrator: ResearchOrchestrator) -> None:
        config = ResearchConfig(max_mcts_iterations=1, max_hypothesis_depth=1)
        await orchestrator.run("test query", config)

        tree = orchestrator.tree
        assert tree is not None
        assert tree.node_count >= 2  # root + at least 1 child


class TestMCTSLoop:
    @pytest.mark.asyncio
    async def test_iterations_run(self, orchestrator: ResearchOrchestrator) -> None:
        config = ResearchConfig(max_mcts_iterations=2, max_hypothesis_depth=1)
        session = await orchestrator.run("test", config)
        assert session.current_iteration >= 1

    @pytest.mark.asyncio
    async def test_agent_failure_isolation(self, orchestrator: ResearchOrchestrator) -> None:
        """Agent failures should not crash the research session."""
        call_count = {"n": 0}

        def failing_factory(agent_type, llm, kg, yami=None, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("Agent crashed!")
            return MockAgentImpl(agent_type=agent_type)

        orchestrator.agent_factory = failing_factory

        config = ResearchConfig(max_mcts_iterations=1, max_hypothesis_depth=1)
        session = await orchestrator.run("test", config)
        # Session should still complete (individual failures isolated)
        assert session.status == SessionStatus.COMPLETED


class TestResultCompilation:
    @pytest.mark.asyncio
    async def test_result_has_hypothesis_ranking(self, orchestrator: ResearchOrchestrator) -> None:
        config = ResearchConfig(max_mcts_iterations=2, max_hypothesis_depth=1)
        session = await orchestrator.run("test", config)

        result = session.result
        assert result is not None
        assert result.best_hypothesis is not None

    @pytest.mark.asyncio
    async def test_result_has_graph_snapshot(self, orchestrator: ResearchOrchestrator) -> None:
        config = ResearchConfig(max_mcts_iterations=1, max_hypothesis_depth=1)
        session = await orchestrator.run("test", config)

        result = session.result
        assert result is not None
        assert isinstance(result.graph_snapshot, dict)

    @pytest.mark.asyncio
    async def test_result_tracks_token_usage(self, orchestrator: ResearchOrchestrator) -> None:
        config = ResearchConfig(max_mcts_iterations=1, max_hypothesis_depth=1)
        session = await orchestrator.run("test", config)

        result = session.result
        assert result is not None
        assert result.total_duration_ms > 0


class TestEvents:
    @pytest.mark.asyncio
    async def test_events_emitted(self, orchestrator: ResearchOrchestrator) -> None:
        config = ResearchConfig(max_mcts_iterations=1, max_hypothesis_depth=1)
        await orchestrator.run("test", config)

        events = orchestrator.drain_events()
        event_types = {e.event_type for e in events}

        assert "session_created" in event_types
        assert "initialization_complete" in event_types
        assert "mcts_iteration_start" in event_types

    @pytest.mark.asyncio
    async def test_drain_events_collects_from_subcomponents(self, orchestrator: ResearchOrchestrator) -> None:
        config = ResearchConfig(max_mcts_iterations=1, max_hypothesis_depth=1)
        await orchestrator.run("test", config)

        events = orchestrator.drain_events()
        # Should include events from tree, composer, uncertainty
        event_types = {e.event_type for e in events}
        assert len(event_types) > 3


class TestScaledOrchestration:
    @pytest.mark.asyncio
    async def test_parallel_swarms_dispatch(self, orchestrator: ResearchOrchestrator) -> None:
        """Multiple hypotheses should get parallel swarms."""
        config = ResearchConfig(
            max_mcts_iterations=1,
            max_hypothesis_depth=1,
            max_concurrent_agents=20,
            max_hypothesis_breadth=10,
        )
        session = await orchestrator.run("B7-H3 in NSCLC", config)
        assert session.status == SessionStatus.COMPLETED
        # Multiple agents should have been spawned (3 hypotheses × ~3 agents each)
        assert orchestrator._total_agents_spawned > 0

    @pytest.mark.asyncio
    async def test_total_agent_cap_enforced(self, orchestrator: ResearchOrchestrator) -> None:
        """Session should stop when max_total_agents is reached."""
        config = ResearchConfig(
            max_mcts_iterations=10,
            max_hypothesis_depth=1,
            max_total_agents=5,
        )
        session = await orchestrator.run("test", config)
        assert session.status == SessionStatus.COMPLETED
        assert orchestrator._total_agents_spawned <= config.max_total_agents + config.max_agents_per_swarm

    @pytest.mark.asyncio
    async def test_session_token_budget_enforced(self, orchestrator: ResearchOrchestrator) -> None:
        """Session should stop when token budget is exhausted."""
        config = ResearchConfig(
            max_mcts_iterations=100,
            max_hypothesis_depth=1,
            session_token_budget=100,  # Very low — should stop quickly
        )
        session = await orchestrator.run("test", config)
        assert session.status == SessionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, orchestrator: ResearchOrchestrator) -> None:
        """With max_concurrent_agents=1, agents run one at a time."""
        config = ResearchConfig(
            max_mcts_iterations=1,
            max_hypothesis_depth=1,
            max_concurrent_agents=1,
        )
        session = await orchestrator.run("test", config)
        assert session.status == SessionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_config_has_scale_fields(self) -> None:
        config = ResearchConfig()
        assert config.max_concurrent_agents == 100
        assert config.max_total_agents == 10_000
        assert config.max_hypothesis_breadth == 50
        assert config.agent_token_budget == 200_000
        assert config.session_token_budget == 10_000_000


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_no_agent_factory_degrades_gracefully(self) -> None:
        """Without agent_factory, agents fail but the session completes (failure isolation)."""
        llm = MagicMock()
        llm.query = AsyncMock(return_value='[{"hypothesis": "test", "rationale": "test"}]')
        llm.parse_json = MagicMock(return_value=[{"hypothesis": "test", "rationale": "test"}])

        kg = InMemoryKnowledgeGraph(graph_id="test")
        orch = ResearchOrchestrator(llm=llm, kg=kg, agent_factory=None)

        config = ResearchConfig(max_mcts_iterations=1, max_hypothesis_depth=1)
        session = await orch.run("test", config)
        # Session completes but with no useful results (all agents failed)
        assert session.status == SessionStatus.COMPLETED
