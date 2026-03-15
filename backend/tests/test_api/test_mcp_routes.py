"""Tests for MCP API routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app, headers={"X-API-Key": "dev-api-key-change-me"})


class TestMCPRoutes:
    def test_health_endpoint(self, client: TestClient) -> None:
        resp = client.get("/api/v1/mcp/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "connected_count" in data

    def test_tools_endpoint(self, client: TestClient) -> None:
        resp = client.get("/api/v1/mcp/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data
        assert "count" in data

    def test_stats_endpoint(self, client: TestClient) -> None:
        resp = client.get("/api/v1/mcp/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "mcp" in data

    def test_retrieve_tools(self, client: TestClient) -> None:
        resp = client.post("/api/v1/mcp/tools/retrieve", json={
            "query": "gene mutation cancer",
            "max_tools": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data
        assert "count" in data
