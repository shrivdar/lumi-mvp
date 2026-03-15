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
            # Plan
            "<think>1. Get recent edges\n2. Get weak edges\n"
            "3. Search for counter-evidence\n4. Update confidence</think>",
            # Get recent edges
            '<tool>kg_get_recent_edges:{"n": 20}</tool>',
            # Get weakest edges
            '<tool>kg_get_weakest_edges:{"n": 10}</tool>',
            # Search for counter-evidence for weak edge
            '<tool>pubmed:{"action": "search", "query": "BRCA1 NOT breast cancer"}</tool>',
            # Update confidence for the weak edge
            '<tool>kg_update_edge_confidence:'
            '{"edge_id": "e-brca1-bc", "confidence": 0.3, '
            '"reason": "Counter-evidence found"}</tool>',
            # Answer
            '<answer>' + json.dumps({
                "entities": [],
                "relationships": [],
                "summary": "Evaluated edges. Lowered confidence on BRCA1-BC edge.",
            }) + '</answer>',
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
        assert weak_edge is not None
        original_weak = weak_edge.confidence.overall

        result = await agent.execute(task)

        assert result.success is True
        # Critic should have updated edges via kg_update_edge_confidence tool
        assert len(result.edges_updated) > 0

        # Weak edge confidence should have changed
        updated_weak = seeded_kg.get_edge("e-brca1-bc")
        assert updated_weak is not None
        assert updated_weak.confidence.overall != original_weak

    @pytest.mark.asyncio
    async def test_critic_only_adds_evidence_against_edges(self, seeded_kg, mock_tools):
        """Critic must not add new biological claims — only EVIDENCE_AGAINST."""
        llm = MockLLMClient(responses=[
            # Plan
            "<think>Evaluate edges for validity</think>",
            # Get edges
            '<tool>kg_get_recent_edges:{"n": 20}</tool>',
            # Search counter-evidence
            '<tool>pubmed:{"action": "search", "query": "BRCA1 disprove"}</tool>',
            # Update confidence
            '<tool>kg_update_edge_confidence:'
            '{"edge_id": "e-brca1-bc", "confidence": 0.2, '
            '"reason": "Strong refutation"}</tool>',
            # Answer with PUBLICATION node and EVIDENCE_AGAINST edge
            '<answer>' + json.dumps({
                "entities": [
                    {"name": "Counter-evidence: BRCA1 reassessment", "type": "PUBLICATION",
                     "description": "Paper reassessing BRCA1 association",
                     "evidence_source": "PUBMED", "evidence_id": "PMID:99999999"},
                ],
                "relationships": [
                    {"source": "Counter-evidence: BRCA1 reassessment", "target": "Breast Cancer",
                     "relation": "EVIDENCE_AGAINST", "confidence": 0.6,
                     "claim": "Weak BRCA1 association", "evidence_source": "PUBMED"},
                ],
                "summary": "Evaluated edges. Found counter-evidence for BRCA1-BC link.",
            }) + '</answer>',
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
        llm = MockLLMClient(responses=[
            # Plan
            "<think>Get edges to evaluate</think>",
            # Get recent edges — will return empty
            '<tool>kg_get_recent_edges:{"n": 20}</tool>',
            # Get weakest edges — will return empty
            '<tool>kg_get_weakest_edges:{"n": 10}</tool>',
            # Answer with empty findings
            '<answer>' + json.dumps({
                "entities": [],
                "relationships": [],
                "summary": "No edges to evaluate — knowledge graph is empty.",
            }) + '</answer>',
        ])

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
