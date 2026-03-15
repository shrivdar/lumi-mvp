"""Integration tests for the falsification pipeline in BaseAgentImpl.

Tests verify:
1. PubMed counter-evidence (articles key) → confidence decreases
2. Semantic Scholar counter-evidence (papers key) → confidence decreases
3. Irrelevant results (LLM says not contradicting) → confidence unchanged (slight boost)
4. No search results → confidence unchanged (slight boost)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.base import BaseAgentImpl
from core.models import (
    AgentConstraints,
    AgentSpec,
    AgentType,
    EdgeConfidence,
    EdgeRelationType,
    EvidenceSource,
    EvidenceSourceType,
    KGEdge,
    KGNode,
    NodeType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    kg,
    tools: dict[str, Any] | None = None,
    llm_responses: list[str] | None = None,
) -> BaseAgentImpl:
    """Create a BaseAgentImpl with mocked LLM and optional tool overrides."""
    mock_llm = MagicMock()
    mock_llm.parse_json = MagicMock(side_effect=lambda text: json.loads(text))

    # Build a response iterator for query_llm
    if llm_responses is None:
        llm_responses = ['{"search_query": "test counter query"}']

    mock_llm.query = AsyncMock(side_effect=llm_responses)

    spec = AgentSpec(
        role="test_agent",
        instructions="test instructions",
        tools=list((tools or {}).keys()),
        constraints=AgentConstraints(max_turns=10, max_llm_calls=50),
        agent_type_hint=AgentType.LITERATURE_ANALYST,
        system_prompt="You are a test agent.",
    )

    agent = BaseAgentImpl(
        agent_id="agent-test-1",
        spec=spec,
        llm=mock_llm,
        kg=kg,
        tools=tools,
    )

    # Patch query_llm to bypass know-how retriever / data-lake / audit overhead
    async def patched_query_llm(self, prompt, **kwargs):
        resp = await self.llm.query(prompt)
        return resp

    agent.query_llm = patched_query_llm.__get__(agent, BaseAgentImpl)

    return agent


def _setup_kg_with_edge(kg) -> tuple[KGNode, KGNode, KGEdge]:
    """Add two nodes and a connecting edge to the KG, return them."""
    src = KGNode(
        id="n-b7h3",
        type=NodeType.PROTEIN,
        name="B7-H3",
        description="Immune checkpoint protein",
        created_by="agent-test-1",
        hypothesis_branch="h-test",
    )
    tgt = KGNode(
        id="n-nsclc",
        type=NodeType.DISEASE,
        name="NSCLC",
        description="Non-small cell lung cancer",
        created_by="agent-test-1",
        hypothesis_branch="h-test",
    )
    edge = KGEdge(
        id="e-b7h3-nsclc",
        source_id="n-b7h3",
        target_id="n-nsclc",
        relation=EdgeRelationType.OVEREXPRESSED_IN,
        confidence=EdgeConfidence(overall=0.8, evidence_quality=0.75, evidence_count=2),
        created_by="agent-test-1",
        hypothesis_branch="h-test",
        evidence=[
            EvidenceSource(
                source_type=EvidenceSourceType.PUBMED,
                source_id="PMID:99999999",
                title="B7-H3 in NSCLC",
                claim="B7-H3 is overexpressed in NSCLC tumors",
                quality_score=0.8,
                confidence=0.8,
                agent_id="agent-test-1",
            )
        ],
    )
    kg.add_node(src)
    kg.add_node(tgt)
    kg.add_edge(edge)
    return src, tgt, edge


def _make_pubmed_tool(articles: list[dict]) -> MagicMock:
    """Mock PubMed tool returning the given articles under the 'articles' key."""
    tool = MagicMock()
    tool.execute = AsyncMock(
        return_value={"articles": articles, "total_count": len(articles), "query": "test"}
    )
    return tool


def _make_s2_tool(papers: list[dict]) -> MagicMock:
    """Mock Semantic Scholar tool returning papers under the 'papers' key."""
    tool = MagicMock()
    tool.execute = AsyncMock(
        return_value={"papers": papers, "total": len(papers), "query": "test"}
    )
    return tool


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pubmed_counter_evidence_decreases_confidence(kg):
    """PubMed returns a paper whose abstract contradicts the claim → confidence drops."""
    _setup_kg_with_edge(kg)

    counter_article = {
        "pmid": "PMID:11111111",
        "title": "B7-H3 is NOT overexpressed in NSCLC",
        "abstract": "Our study found no significant B7-H3 overexpression in NSCLC tumors.",
        "doi": "10.1234/counter",
    }

    pubmed_tool = _make_pubmed_tool([counter_article])

    # LLM responses: 1) falsification query, 2) LLM eval says contradicts
    llm_responses = [
        '{"disproof_criteria": "No overexpression", "search_query": "B7-H3 NOT overexpressed NSCLC"}',
        '{"contradicts": true, "reasoning": "The paper explicitly states no overexpression."}',
    ]

    agent = _make_agent(kg, tools={"pubmed": pubmed_tool}, llm_responses=llm_responses)

    edge = kg.get_edge("e-b7h3-nsclc")
    assert edge is not None
    original = edge.confidence.overall

    results = await agent.falsify([edge])

    assert len(results) == 1
    r = results[0]
    assert r.counter_evidence_found is True
    assert r.revised_confidence < original
    assert r.confidence_delta < 0
    assert len(r.counter_evidence) == 1
    assert r.counter_evidence[0].source_id == "PMID:11111111"


@pytest.mark.asyncio
async def test_semantic_scholar_counter_evidence_decreases_confidence(kg):
    """Semantic Scholar returns a contradicting paper → confidence drops."""
    _setup_kg_with_edge(kg)

    counter_paper = {
        "paper_id": "abc123",
        "title": "Contradicting B7-H3 NSCLC overexpression",
        "abstract": "Meta-analysis shows B7-H3 is not significantly overexpressed in NSCLC.",
        "doi": "10.5678/counter",
    }

    s2_tool = _make_s2_tool([counter_paper])

    llm_responses = [
        '{"disproof_criteria": "No overexpression", "search_query": "B7-H3 expression NSCLC meta-analysis"}',
        '{"contradicts": true, "reasoning": "Meta-analysis contradicts overexpression claim."}',
    ]

    agent = _make_agent(kg, tools={"semantic_scholar": s2_tool}, llm_responses=llm_responses)

    edge = kg.get_edge("e-b7h3-nsclc")
    original = edge.confidence.overall

    results = await agent.falsify([edge])

    assert len(results) == 1
    r = results[0]
    assert r.counter_evidence_found is True
    assert r.revised_confidence < original
    assert r.counter_evidence[0].source_type == EvidenceSourceType.SEMANTIC_SCHOLAR
    assert r.counter_evidence[0].source_id == "abc123"


@pytest.mark.asyncio
async def test_irrelevant_results_confidence_unchanged(kg):
    """PubMed returns papers but LLM says they don't contradict → slight confidence boost."""
    _setup_kg_with_edge(kg)

    irrelevant_article = {
        "pmid": "PMID:22222222",
        "title": "B7-H3 structure analysis",
        "abstract": "We determined the crystal structure of B7-H3. No disease associations examined.",
        "doi": "10.1234/irrelevant",
    }

    pubmed_tool = _make_pubmed_tool([irrelevant_article])

    llm_responses = [
        '{"disproof_criteria": "No overexpression", "search_query": "B7-H3 NOT overexpressed NSCLC"}',
        '{"contradicts": false, "reasoning": "Paper is about structure, not expression levels."}',
    ]

    agent = _make_agent(kg, tools={"pubmed": pubmed_tool}, llm_responses=llm_responses)

    edge = kg.get_edge("e-b7h3-nsclc")
    original = edge.confidence.overall

    results = await agent.falsify([edge])

    assert len(results) == 1
    r = results[0]
    assert r.counter_evidence_found is False
    # Survived falsification → slight boost
    assert r.revised_confidence == pytest.approx(original + 0.02)
    assert r.confidence_delta == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_no_search_results_confidence_boosted(kg):
    """No results from any tool → survived falsification, slight confidence boost."""
    _setup_kg_with_edge(kg)

    empty_pubmed = _make_pubmed_tool([])

    llm_responses = [
        '{"disproof_criteria": "No overexpression", "search_query": "B7-H3 NOT overexpressed NSCLC"}',
    ]

    agent = _make_agent(kg, tools={"pubmed": empty_pubmed}, llm_responses=llm_responses)

    edge = kg.get_edge("e-b7h3-nsclc")
    original = edge.confidence.overall

    results = await agent.falsify([edge])

    assert len(results) == 1
    r = results[0]
    assert r.counter_evidence_found is False
    assert r.falsified is False
    assert r.revised_confidence == pytest.approx(original + 0.02)


@pytest.mark.asyncio
async def test_both_tools_with_mixed_results(kg):
    """PubMed has contradicting paper, S2 has irrelevant paper → only PubMed counted."""
    _setup_kg_with_edge(kg)

    pubmed_article = {
        "pmid": "PMID:33333333",
        "title": "B7-H3 low expression in NSCLC",
        "abstract": "We found B7-H3 was underexpressed in our NSCLC cohort.",
        "doi": "10.1234/low",
    }
    s2_paper = {
        "paper_id": "def456",
        "title": "Immune checkpoints review",
        "abstract": "A general review of immune checkpoint proteins.",
    }

    pubmed_tool = _make_pubmed_tool([pubmed_article])
    s2_tool = _make_s2_tool([s2_paper])

    llm_responses = [
        '{"disproof_criteria": "No overexpression", "search_query": "B7-H3 low expression NSCLC"}',
        '{"contradicts": true, "reasoning": "Found underexpression."}',   # PubMed paper
        '{"contradicts": false, "reasoning": "General review, no contradiction."}',  # S2 paper
    ]

    agent = _make_agent(
        kg,
        tools={"pubmed": pubmed_tool, "semantic_scholar": s2_tool},
        llm_responses=llm_responses,
    )

    edge = kg.get_edge("e-b7h3-nsclc")
    original = edge.confidence.overall

    results = await agent.falsify([edge])

    r = results[0]
    assert r.counter_evidence_found is True
    assert len(r.counter_evidence) == 1
    assert r.counter_evidence[0].source_type == EvidenceSourceType.PUBMED
    assert r.revised_confidence < original
