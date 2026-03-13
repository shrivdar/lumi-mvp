"""Tests for BaseAgentImpl — execute flow, write_node/write_edge, falsification."""

from __future__ import annotations

import pytest

from agents.base import BaseAgentImpl
from agents.templates import get_template
from core.models import (
    AgentTask,
    AgentType,
    EdgeConfidence,
    EdgeRelationType,
    EvidenceSource,
    EvidenceSourceType,
    KGEdge,
    KGNode,
    NodeType,
)
from tests.test_agents.conftest import MockLLMClient
from world_model.knowledge_graph import InMemoryKnowledgeGraph


# ---------------------------------------------------------------------------
# Concrete subclass for testing
# ---------------------------------------------------------------------------


class StubAgent(BaseAgentImpl):
    """Minimal agent for testing the base execute flow."""

    agent_type = AgentType.LITERATURE_ANALYST

    def __init__(self, nodes_to_return=None, edges_to_return=None, **kwargs):
        super().__init__(**kwargs)
        self._stub_nodes = nodes_to_return or []
        self._stub_edges = edges_to_return or []

    async def _investigate(self, task, kg_context):
        return {
            "nodes": self._stub_nodes,
            "edges": self._stub_edges,
            "summary": "Stub investigation complete",
            "reasoning_trace": "stub",
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaseAgentExecute:
    """Test the execute() template method."""

    @pytest.mark.asyncio
    async def test_execute_returns_agent_result(self, agent_kg, mock_llm, sample_task):
        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = StubAgent(template=template, llm=mock_llm, kg=agent_kg)

        result = await agent.execute(sample_task)

        assert result.success is True
        assert result.task_id == sample_task.task_id
        assert result.agent_type == AgentType.LITERATURE_ANALYST
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_writes_nodes_to_kg(self, agent_kg, mock_llm, sample_task):
        template = get_template(AgentType.LITERATURE_ANALYST)
        test_node = KGNode(
            type=NodeType.GENE,
            name="TestGene",
            confidence=0.8,
            sources=[
                EvidenceSource(
                    source_type=EvidenceSourceType.PUBMED,
                    claim="test",
                    agent_id="test",
                )
            ],
        )
        agent = StubAgent(
            template=template, llm=mock_llm, kg=agent_kg,
            nodes_to_return=[test_node],
        )

        result = await agent.execute(sample_task)

        assert result.success is True
        assert len(result.nodes_added) == 1
        assert result.nodes_added[0].name == "TestGene"
        # Verify node was written to KG
        assert agent_kg.node_count() == 1
        # Verify agent_id is stamped
        kg_node = agent_kg.get_node(test_node.id)
        assert kg_node is not None
        assert kg_node.created_by == agent.agent_id

    @pytest.mark.asyncio
    async def test_execute_writes_edges_with_hypothesis_branch(self, agent_kg, mock_llm, sample_task):
        template = get_template(AgentType.LITERATURE_ANALYST)
        # Create two nodes first
        n1 = KGNode(type=NodeType.GENE, name="G1", confidence=0.8,
                     sources=[EvidenceSource(source_type=EvidenceSourceType.PUBMED, agent_id="t")])
        n2 = KGNode(type=NodeType.DISEASE, name="D1", confidence=0.8,
                     sources=[EvidenceSource(source_type=EvidenceSourceType.PUBMED, agent_id="t")])
        test_edge = KGEdge(
            source_id=n1.id, target_id=n2.id,
            relation=EdgeRelationType.ASSOCIATED_WITH,
            confidence=EdgeConfidence(overall=0.7),
            evidence=[EvidenceSource(source_type=EvidenceSourceType.PUBMED, claim="test", agent_id="t")],
        )
        agent = StubAgent(
            template=template, llm=mock_llm, kg=agent_kg,
            nodes_to_return=[n1, n2],
            edges_to_return=[test_edge],
        )

        result = await agent.execute(sample_task)

        assert result.success is True
        assert len(result.edges_added) == 1
        kg_edge = agent_kg.get_edge(test_edge.id)
        assert kg_edge is not None
        assert kg_edge.created_by == agent.agent_id
        assert kg_edge.hypothesis_branch == sample_task.hypothesis_branch

    @pytest.mark.asyncio
    async def test_execute_handles_investigation_error(self, agent_kg, mock_llm, sample_task):
        """Agent should return failure result if _investigate raises."""
        template = get_template(AgentType.LITERATURE_ANALYST)

        class FailingAgent(BaseAgentImpl):
            async def _investigate(self, task, kg_context):
                raise RuntimeError("Investigation failed")

        agent = FailingAgent(template=template, llm=mock_llm, kg=agent_kg)
        result = await agent.execute(sample_task)

        assert result.success is False
        assert "Investigation failed" in result.errors[0]


class TestBaseAgentFalsification:
    """Test the falsify() method."""

    @pytest.mark.asyncio
    async def test_falsify_with_counter_evidence_lowers_confidence(self, seeded_kg, mock_tools):
        """When counter-evidence is found, confidence should decrease."""
        template = get_template(AgentType.LITERATURE_ANALYST)
        llm = MockLLMClient(responses=[
            '{"disproof_criteria": "Find papers showing BRCA1 is not linked to breast cancer", "search_query": "BRCA1 NOT breast cancer"}',
        ])
        agent = StubAgent(template=template, llm=llm, kg=seeded_kg, tools=mock_tools)

        weak_edge = seeded_kg.get_edge("e-brca1-bc")
        assert weak_edge is not None
        original_confidence = weak_edge.confidence.overall

        results = await agent.falsify([weak_edge])

        assert len(results) == 1
        assert results[0].counter_evidence_found is True
        assert results[0].revised_confidence < original_confidence

    @pytest.mark.asyncio
    async def test_falsify_without_counter_evidence_boosts_confidence(self, seeded_kg):
        """When no counter-evidence is found, confidence should slightly increase."""
        template = get_template(AgentType.LITERATURE_ANALYST)
        llm = MockLLMClient(responses=[
            '{"disproof_criteria": "test", "search_query": "test query"}',
        ])
        # No tools = no search results
        agent = StubAgent(template=template, llm=llm, kg=seeded_kg, tools={})

        edge = seeded_kg.get_edge("e-tp53-bc")
        assert edge is not None
        original = edge.confidence.overall

        results = await agent.falsify([edge])

        assert len(results) == 1
        assert results[0].counter_evidence_found is False
        assert results[0].revised_confidence > original


class TestBaseAgentUncertainty:
    """Test get_uncertainty()."""

    @pytest.mark.asyncio
    async def test_uncertainty_vector_computed(self, agent_kg, mock_llm, sample_task):
        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = StubAgent(template=template, llm=mock_llm, kg=agent_kg)

        result = await agent.execute(sample_task)

        assert result.uncertainty is not None
        assert 0.0 <= result.uncertainty.composite <= 1.0


class TestBaseAgentQueryLLM:
    """Test the query_llm helper."""

    @pytest.mark.asyncio
    async def test_query_llm_tracks_calls(self, agent_kg):
        llm = MockLLMClient(responses=["response 1", "response 2"])
        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = StubAgent(template=template, llm=llm, kg=agent_kg)

        r1 = await agent.query_llm("prompt 1")
        r2 = await agent.query_llm("prompt 2")

        assert r1 == "response 1"
        assert r2 == "response 2"
        assert agent._llm_calls == 2
