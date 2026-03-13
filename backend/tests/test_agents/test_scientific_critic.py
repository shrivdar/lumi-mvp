"""Tests for ScientificCriticAgent — seed KG with weak edges, verify adjustments."""

from __future__ import annotations

import json

import pytest

from agents.scientific_critic import ScientificCriticAgent
from agents.templates import get_template
from core.models import AgentTask, AgentType
from tests.test_agents.conftest import MockLLMClient


class TestScientificCritic:
    """Test the critic agent's falsification behavior."""

    @pytest.mark.asyncio
    async def test_critic_lowers_weak_edge_confidence(self, seeded_kg, mock_tools):
        """Critic should lower confidence when counter-evidence is found."""
        llm = MockLLMClient(responses=[
            # Disproof for edge e-brca1-bc (weak, 0.4 confidence)
            json.dumps({
                "disproof_criteria": "Show BRCA1 is not linked to breast cancer",
                "search_queries": ["BRCA1 NOT associated breast cancer"],
                "prior_wrong_probability": 0.3,
            }),
            # Assessment of counter-evidence
            json.dumps({
                "refutes": True,
                "strength": "moderate",
                "reasoning": "Papers suggest weak association",
                "confidence_adjustment": -0.1,
            }),
            # Disproof for edge e-tp53-bc (strong, 0.9 confidence)
            json.dumps({
                "disproof_criteria": "Show TP53 is not a tumor suppressor",
                "search_queries": ["TP53 NOT tumor suppressor"],
                "prior_wrong_probability": 0.05,
            }),
            # Assessment
            json.dumps({
                "refutes": False,
                "strength": "irrelevant",
                "reasoning": "Counter-evidence is about different context",
                "confidence_adjustment": 0.01,
            }),
        ])

        template = get_template(AgentType.SCIENTIFIC_CRITIC)
        task = AgentTask(
            task_id="task-critic-001",
            research_id="research-001",
            agent_type=AgentType.SCIENTIFIC_CRITIC,
            hypothesis_branch="h-main",
            instruction="Critically evaluate all recent edges in the knowledge graph.",
        )

        agent = ScientificCriticAgent(
            template=template, llm=llm, kg=seeded_kg, tools=mock_tools,
        )

        # Record original confidences
        weak_edge = seeded_kg.get_edge("e-brca1-bc")
        strong_edge = seeded_kg.get_edge("e-tp53-bc")
        assert weak_edge is not None
        assert strong_edge is not None
        original_weak = weak_edge.confidence.overall
        original_strong = strong_edge.confidence.overall

        result = await agent.execute(task)

        assert result.success is True
        # Critic should have evaluated edges
        assert len(result.edges_updated) > 0

        # Weak edge confidence should have changed
        updated_weak = seeded_kg.get_edge("e-brca1-bc")
        assert updated_weak is not None
        assert updated_weak.confidence.overall != original_weak

    @pytest.mark.asyncio
    async def test_critic_only_adds_evidence_against_edges(self, seeded_kg, mock_tools):
        """Critic must not add new biological claims — only EVIDENCE_AGAINST."""
        llm = MockLLMClient(responses=[
            json.dumps({
                "disproof_criteria": "test",
                "search_queries": ["test query"],
                "prior_wrong_probability": 0.5,
            }),
            json.dumps({
                "refutes": True,
                "strength": "strong",
                "reasoning": "Strong refutation",
                "confidence_adjustment": -0.2,
            }),
            json.dumps({
                "disproof_criteria": "test2",
                "search_queries": ["test query 2"],
                "prior_wrong_probability": 0.1,
            }),
            json.dumps({
                "refutes": False,
                "strength": "irrelevant",
                "reasoning": "Not relevant",
                "confidence_adjustment": 0.01,
            }),
        ])

        template = get_template(AgentType.SCIENTIFIC_CRITIC)
        task = AgentTask(
            task_id="task-critic-002",
            research_id="research-001",
            agent_type=AgentType.SCIENTIFIC_CRITIC,
            hypothesis_branch="h-main",
            instruction="Evaluate edges.",
        )

        agent = ScientificCriticAgent(
            template=template, llm=llm, kg=seeded_kg, tools=mock_tools,
        )

        result = await agent.execute(task)

        # Critic can only add PUBLICATION nodes and EVIDENCE_AGAINST edges
        from core.models import EdgeRelationType, NodeType
        for node in result.nodes_added:
            assert node.type == NodeType.PUBLICATION, f"Critic should only add PUBLICATION nodes, got {node.type}"
        for edge in result.edges_added:
            assert edge.relation in (EdgeRelationType.EVIDENCE_AGAINST, EdgeRelationType.CONTRADICTS), \
                f"Critic should only add EVIDENCE_AGAINST/CONTRADICTS edges, got {edge.relation}"

    @pytest.mark.asyncio
    async def test_critic_handles_empty_kg(self, agent_kg, mock_tools):
        """Critic should handle empty KG gracefully."""
        llm = MockLLMClient()
        template = get_template(AgentType.SCIENTIFIC_CRITIC)
        task = AgentTask(
            task_id="task-critic-003",
            research_id="research-001",
            agent_type=AgentType.SCIENTIFIC_CRITIC,
            hypothesis_branch="h-main",
            instruction="Evaluate edges.",
        )

        agent = ScientificCriticAgent(
            template=template, llm=llm, kg=agent_kg, tools=mock_tools,
        )

        result = await agent.execute(task)

        assert result.success is True
        assert "empty" in result.summary.lower() or "no edges" in result.summary.lower()
