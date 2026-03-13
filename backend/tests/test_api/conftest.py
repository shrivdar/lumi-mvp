"""Test fixtures for API tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.deps import (
    _agent_results,
    _knowledge_graphs,
    _orchestrators,
    _sessions,
)
from api.main import create_app
from core.config import settings
from core.models import (
    EdgeConfidence,
    EdgeRelationType,
    EvidenceSource,
    EvidenceSourceType,
    HypothesisNode,
    KGEdge,
    KGNode,
    NodeType,
    ResearchConfig,
    ResearchResult,
    ResearchSession,
    SessionStatus,
)
from world_model.knowledge_graph import InMemoryKnowledgeGraph


@pytest.fixture(autouse=True)
def _clear_stores():
    """Clear in-memory stores between tests."""
    _sessions.clear()
    _knowledge_graphs.clear()
    _orchestrators.clear()
    _agent_results.clear()
    yield
    _sessions.clear()
    _knowledge_graphs.clear()
    _orchestrators.clear()
    _agent_results.clear()


@pytest.fixture()
def app():
    return create_app()


@pytest.fixture()
def client(app) -> TestClient:
    return TestClient(app)


@pytest.fixture()
def auth_headers() -> dict[str, str]:
    return {"X-API-Key": settings.api_key}


@pytest.fixture()
def seeded_session() -> ResearchSession:
    """A completed research session with result."""
    session = ResearchSession(
        id="test-session-1",
        query="Role of B7-H3 in NSCLC",
        status=SessionStatus.COMPLETED,
        config=ResearchConfig(),
        current_iteration=5,
        total_nodes=5,
        total_edges=4,
        total_hypotheses=3,
    )
    session.result = ResearchResult(
        research_id="test-session-1",
        best_hypothesis=HypothesisNode(
            id="h-best",
            hypothesis="B7-H3 overexpression drives immune evasion in NSCLC",
            confidence=0.85,
            visit_count=3,
        ),
        hypothesis_ranking=[
            HypothesisNode(
                id="h-best",
                hypothesis="B7-H3 overexpression drives immune evasion in NSCLC",
                confidence=0.85,
                visit_count=3,
            ),
        ],
        key_findings=[],
        total_duration_ms=15000,
        total_llm_calls=20,
        total_tokens=50000,
    )
    return session


@pytest.fixture()
def seeded_kg() -> InMemoryKnowledgeGraph:
    """KG with sample nodes and edges for testing."""
    kg = InMemoryKnowledgeGraph(graph_id="test-session-1")

    nodes = [
        KGNode(id="n-b7h3", type=NodeType.PROTEIN, name="B7-H3",
               created_by="agent-lit-1", hypothesis_branch="h-best",
               sources=[EvidenceSource(source_type=EvidenceSourceType.PUBMED, source_id="PMID:99999")]),
        KGNode(id="n-nsclc", type=NodeType.DISEASE, name="NSCLC",
               created_by="agent-lit-1", hypothesis_branch="h-best"),
        KGNode(id="n-pd1", type=NodeType.PROTEIN, name="PD-1",
               created_by="agent-lit-1", hypothesis_branch="h-best"),
    ]
    edges = [
        KGEdge(id="e-b7h3-nsclc", source_id="n-b7h3", target_id="n-nsclc",
               relation=EdgeRelationType.OVEREXPRESSED_IN,
               confidence=EdgeConfidence(overall=0.88, evidence_quality=0.85),
               created_by="agent-lit-1", hypothesis_branch="h-best",
               evidence=[EvidenceSource(source_type=EvidenceSourceType.PUBMED, source_id="PMID:99999")]),
        KGEdge(id="e-pd1-nsclc", source_id="n-pd1", target_id="n-nsclc",
               relation=EdgeRelationType.ASSOCIATED_WITH,
               confidence=EdgeConfidence(overall=0.75),
               created_by="agent-lit-1", hypothesis_branch="h-best"),
    ]

    for n in nodes:
        kg.add_node(n)
    for e in edges:
        kg.add_edge(e)

    return kg


@pytest.fixture()
def seeded_stores(seeded_session, seeded_kg):
    """Populate global stores with test data."""
    _sessions["test-session-1"] = seeded_session
    _knowledge_graphs["test-session-1"] = seeded_kg
    _agent_results["test-session-1"] = []
    return {
        "session": seeded_session,
        "kg": seeded_kg,
    }
