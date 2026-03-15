"""Tests for monitoring API endpoints."""

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
    AgentResult,
    AgentType,
    EdgeRelationType,
    KGEdge,
    KGNode,
    NodeType,
    ResearchConfig,
    ResearchSession,
    SessionStatus,
    UncertaintyVector,
)
from orchestrator.uncertainty import UncertaintyAggregator
from world_model.knowledge_graph import InMemoryKnowledgeGraph


@pytest.fixture(autouse=True)
def _clear_stores():
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
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture()
def auth_headers() -> dict[str, str]:
    return {"X-API-Key": settings.api_key}


class FakeOrchestrator:
    """Minimal orchestrator mock for monitoring tests."""

    def __init__(self) -> None:
        self._total_agents_spawned = 5
        self._session_tokens_used = 25000
        self._all_results: list[AgentResult] = [
            AgentResult(
                task_id="t1", agent_id="a1",
                agent_type=AgentType.LITERATURE_ANALYST,
                success=True,
                nodes_added=[KGNode(type=NodeType.GENE, name="BRCA1", created_by="a1")],
                edges_added=[KGEdge(
                    source_id="n1", target_id="n2",
                    relation=EdgeRelationType.ASSOCIATED_WITH,
                    created_by="a1",
                )],
                llm_tokens_used=5000,
            ),
            AgentResult(
                task_id="t2", agent_id="a2",
                agent_type=AgentType.SCIENTIFIC_CRITIC,
                success=True,
                llm_tokens_used=3000,
            ),
        ]
        self._uncertainty = UncertaintyAggregator(session_id="test-session")
        self._uncertainty._history = [
            UncertaintyVector(
                input_ambiguity=0.3, data_quality=0.4,
                reasoning_divergence=0.2, conflict_uncertainty=0.1,
                composite=0.25, is_critical=False,
            ),
        ]
        self.tree = None

    def drain_events(self):
        return []


@pytest.fixture()
def seeded_monitoring():
    """Seed stores with monitoring test data."""
    session = ResearchSession(
        id="test-session",
        query="B7-H3 NSCLC",
        status=SessionStatus.RUNNING,
        config=ResearchConfig(),
        current_iteration=3,
        total_nodes=5,
        total_edges=4,
    )
    kg = InMemoryKnowledgeGraph(graph_id="test-session")
    kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="BRCA1", created_by="a1"))
    kg.add_node(KGNode(id="n2", type=NodeType.DISEASE, name="NSCLC", created_by="a1"))

    orch = FakeOrchestrator()

    _sessions["test-session"] = session
    _knowledge_graphs["test-session"] = kg
    _orchestrators["test-session"] = orch

    return {"session": session, "kg": kg, "orch": orch}


class TestMonitoringOverview:
    def test_overview_empty(self, client, auth_headers):
        resp = client.get("/api/v1/monitoring/overview", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["sessions"]["total"] == 0

    def test_overview_with_sessions(self, client, auth_headers, seeded_monitoring):
        resp = client.get("/api/v1/monitoring/overview", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["sessions"]["total"] == 1
        assert data["sessions"]["active"] == 1
        assert data["knowledge_graph"]["total_nodes"] == 2
        assert data["agents"]["total_spawned"] == 5


class TestResearchStats:
    def test_stats_not_found(self, client, auth_headers):
        resp = client.get("/api/v1/monitoring/research/nonexistent/stats", headers=auth_headers)
        assert resp.status_code == 404

    def test_stats_for_running_session(self, client, auth_headers, seeded_monitoring):
        resp = client.get("/api/v1/monitoring/research/test-session/stats", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["research_id"] == "test-session"
        assert data["status"] == "RUNNING"
        assert data["agents"]["total_spawned"] == 5
        assert data["tokens"]["session_tokens_used"] == 25000
        assert data["tokens"]["budget_utilization"] > 0

    def test_agent_type_counts(self, client, auth_headers, seeded_monitoring):
        resp = client.get("/api/v1/monitoring/research/test-session/stats", headers=auth_headers)
        data = resp.json()
        counts = data["agents"]["type_counts"]
        assert "literature_analyst" in counts
        assert "scientific_critic" in counts


class TestAgentConstellation:
    def test_agents_not_found(self, client, auth_headers):
        resp = client.get("/api/v1/monitoring/research/nonexistent/agents", headers=auth_headers)
        assert resp.status_code == 404

    def test_agents_for_session(self, client, auth_headers, seeded_monitoring):
        resp = client.get("/api/v1/monitoring/research/test-session/agents", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_spawned"] == 2
        agents = data["agents"]
        assert len(agents) == 2
        # Check agent info structure
        agent = agents[0]
        assert "agent_id" in agent
        assert "agent_type" in agent
        assert "nodes_added" in agent
        assert "edges_added" in agent


class TestUncertaintyRadar:
    def test_uncertainty_not_found(self, client, auth_headers):
        resp = client.get("/api/v1/monitoring/research/nonexistent/uncertainty", headers=auth_headers)
        assert resp.status_code == 404

    def test_uncertainty_for_session(self, client, auth_headers, seeded_monitoring):
        resp = client.get("/api/v1/monitoring/research/test-session/uncertainty", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "current" in data
        assert "history" in data
        assert "trend" in data
        current = data["current"]
        assert "composite" in current
        assert "input_ambiguity" in current
