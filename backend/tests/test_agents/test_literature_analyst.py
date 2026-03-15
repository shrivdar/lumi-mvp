"""Tests for LiteratureAnalystAgent — mock tools, verify KG mutations."""

from __future__ import annotations

import json

import pytest

from agents.literature_analyst import LiteratureAnalystAgent
from agents.templates import get_template
from core.models import AgentType, NodeType
from tests.test_agents.conftest import MockLLMClient


class TestLiteratureAnalyst:
    """End-to-end test with mocked tools and LLM."""

    @pytest.mark.asyncio
    async def test_produces_nodes_and_edges(self, agent_kg, mock_tools, sample_task):
        """Literature analyst should produce KG nodes and edges from mock papers."""
        llm = MockLLMClient(responses=[
            # Turn 0 (plan)
            "<think>1. Search PubMed for BRCA1 breast cancer\n2. Extract entities\n3. Build answer</think>",
            # Turn 1: tool call for pubmed
            '<tool>pubmed:{"action": "search", "query": "BRCA1 breast cancer", "max_results": 5}</tool>',
            # Turn 2: final answer with entities and relationships
            '<answer>' + json.dumps({
                "entities": [
                    {"name": "BRCA1", "type": "GENE", "description": "Breast cancer gene 1"},
                    {"name": "Breast Cancer", "type": "DISEASE", "description": "Malignant breast neoplasm"},
                ],
                "relationships": [
                    {
                        "source": "BRCA1",
                        "target": "Breast Cancer",
                        "relation": "ASSOCIATED_WITH",
                        "evidence_source": "PUBMED",
                        "evidence_id": "PMID:12345678",
                        "claim": "BRCA1 mutations increase breast cancer risk",
                        "confidence": 0.85,
                    }
                ],
                "summary": "BRCA1 is associated with breast cancer.",
            }) + '</answer>',
            # Falsification response
            '{"disproof_criteria": "test", "search_query": "BRCA1 NOT breast cancer"}',
        ])

        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = LiteratureAnalystAgent(
            template=template, llm=llm, kg=agent_kg, tools=mock_tools,
        )

        result = await agent.execute(sample_task)

        assert result.success is True
        assert len(result.nodes_added) >= 2  # BRCA1 + Breast Cancer
        assert len(result.edges_added) >= 1  # ASSOCIATED_WITH

        # Verify nodes in KG
        assert agent_kg.node_count() >= 2
        brca1 = agent_kg.get_node_by_name("BRCA1")
        assert brca1 is not None
        assert brca1.type == NodeType.GENE
        assert brca1.created_by == agent.agent_id

        # Verify edge in KG
        assert agent_kg.edge_count() >= 1

        # Verify multi-turn metadata
        assert len(result.turns) >= 3

    @pytest.mark.asyncio
    async def test_handles_no_papers_found(self, agent_kg, sample_task):
        """Should gracefully handle when no papers are returned."""
        llm = MockLLMClient(responses=[
            # Plan
            "<think>1. Search for papers\n2. Report findings</think>",
            # Tool call
            '<tool>pubmed:{"action": "search", "query": "obscure query", "max_results": 5}</tool>',
            # Answer with no results
            '<answer>' + json.dumps({
                "entities": [],
                "relationships": [],
                "summary": "No papers found for the given queries.",
            }) + '</answer>',
        ])
        from unittest.mock import AsyncMock, MagicMock
        empty_tools = {
            "pubmed": MagicMock(),
            "semantic_scholar": MagicMock(),
        }
        for t in empty_tools.values():
            t.execute = AsyncMock(return_value={"results": []})
            t.description = "Mock tool"

        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = LiteratureAnalystAgent(
            template=template, llm=llm, kg=agent_kg, tools=empty_tools,
        )

        result = await agent.execute(sample_task)

        assert result.success is True
        assert "No papers found" in result.summary

    @pytest.mark.asyncio
    async def test_all_nodes_have_provenance(self, agent_kg, mock_tools, sample_task):
        """Every node must have agent_id and hypothesis_branch set."""
        llm = MockLLMClient(responses=[
            # Plan
            "<think>Search and extract</think>",
            # Tool call
            '<tool>pubmed:{"action": "search", "query": "BRCA1", "max_results": 5}</tool>',
            # Answer
            '<answer>' + json.dumps({
                "entities": [
                    {"name": "BRCA1", "type": "GENE", "description": "test"},
                ],
                "relationships": [],
                "summary": "test",
            }) + '</answer>',
        ])

        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = LiteratureAnalystAgent(
            template=template, llm=llm, kg=agent_kg, tools=mock_tools,
        )

        result = await agent.execute(sample_task)

        for node in result.nodes_added:
            assert node.created_by == agent.agent_id
            assert node.hypothesis_branch == sample_task.hypothesis_branch
