"""Tests for research CRUD endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestCreateResearch:
    def test_create_returns_201(self, client: TestClient, auth_headers: dict):
        resp = client.post(
            "/api/v1/research",
            json={"query": "Role of B7-H3 in NSCLC"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "research_id" in data
        assert data["status"] == "INITIALIZING"

    def test_create_with_config(self, client: TestClient, auth_headers: dict):
        resp = client.post(
            "/api/v1/research",
            json={
                "query": "BRCA1 signaling",
                "config": {"max_mcts_iterations": 5, "max_agents": 3},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201


class TestGetResearch:
    def test_get_existing_session(self, client: TestClient, auth_headers: dict, seeded_stores):
        resp = client.get("/api/v1/research/test-session-1", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "test-session-1"
        assert data["query"] == "Role of B7-H3 in NSCLC"
        assert data["status"] == "COMPLETED"

    def test_get_nonexistent_returns_404(self, client: TestClient, auth_headers: dict):
        resp = client.get("/api/v1/research/nonexistent", headers=auth_headers)
        assert resp.status_code == 404


class TestListResearch:
    def test_list_empty(self, client: TestClient, auth_headers: dict):
        resp = client.get("/api/v1/research", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_with_sessions(self, client: TestClient, auth_headers: dict, seeded_stores):
        resp = client.get("/api/v1/research", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1

    def test_list_filter_by_status(self, client: TestClient, auth_headers: dict, seeded_stores):
        resp = client.get("/api/v1/research?status=COMPLETED", headers=auth_headers)
        assert resp.json()["total"] == 1

        resp = client.get("/api/v1/research?status=RUNNING", headers=auth_headers)
        assert resp.json()["total"] == 0


class TestGetResult:
    def test_get_result_completed(self, client: TestClient, auth_headers: dict, seeded_stores):
        resp = client.get("/api/v1/research/test-session-1/result", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["research_id"] == "test-session-1"
        assert data["best_hypothesis"]["hypothesis"] == "B7-H3 overexpression drives immune evasion in NSCLC"

    def test_get_result_not_completed_returns_409(self, client: TestClient, auth_headers: dict):
        from api.deps import _sessions
        from core.models import ResearchSession, SessionStatus
        _sessions["running-1"] = ResearchSession(
            id="running-1", query="test", status=SessionStatus.RUNNING
        )
        resp = client.get("/api/v1/research/running-1/result", headers=auth_headers)
        assert resp.status_code == 409


class TestCancelResearch:
    def test_cancel_running_session(self, client: TestClient, auth_headers: dict):
        from api.deps import _sessions
        from core.models import ResearchSession, SessionStatus
        _sessions["running-1"] = ResearchSession(
            id="running-1", query="test", status=SessionStatus.RUNNING
        )
        resp = client.post("/api/v1/research/running-1/cancel", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "CANCELLED"

    def test_cancel_completed_returns_409(self, client: TestClient, auth_headers: dict, seeded_stores):
        resp = client.post("/api/v1/research/test-session-1/cancel", headers=auth_headers)
        assert resp.status_code == 409


class TestFeedback:
    def test_submit_feedback(self, client: TestClient, auth_headers: dict, seeded_stores):
        resp = client.post(
            "/api/v1/research/test-session-1/feedback",
            json={"edge_id": "e-b7h3-nsclc", "feedback": "agree", "confidence_override": 0.95},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["applied"] is True

    def test_feedback_missing_edge_returns_404(self, client: TestClient, auth_headers: dict, seeded_stores):
        resp = client.post(
            "/api/v1/research/test-session-1/feedback",
            json={"edge_id": "nonexistent", "feedback": "disagree"},
            headers=auth_headers,
        )
        assert resp.status_code == 404
