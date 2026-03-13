"""Tests for graph query endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestGraphEndpoints:
    def test_get_full_graph_cytoscape(self, client: TestClient, auth_headers: dict, seeded_stores):
        resp = client.get("/api/v1/research/test-session-1/graph?format=cytoscape", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["format"] == "cytoscape"

    def test_get_full_graph_json(self, client: TestClient, auth_headers: dict, seeded_stores):
        resp = client.get("/api/v1/research/test-session-1/graph?format=json", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["format"] == "json"
        assert "nodes" in data["data"]
        assert "edges" in data["data"]

    def test_get_full_graph_summary(self, client: TestClient, auth_headers: dict, seeded_stores):
        resp = client.get("/api/v1/research/test-session-1/graph?format=summary", headers=auth_headers)
        assert resp.status_code == 200

    def test_get_graph_404(self, client: TestClient, auth_headers: dict):
        resp = client.get("/api/v1/research/nonexistent/graph", headers=auth_headers)
        assert resp.status_code == 404

    def test_get_subgraph(self, client: TestClient, auth_headers: dict, seeded_stores):
        resp = client.get(
            "/api/v1/research/test-session-1/graph/subgraph?center=n-b7h3&hops=1",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["center"] == "n-b7h3"

    def test_get_subgraph_node_not_found(self, client: TestClient, auth_headers: dict, seeded_stores):
        resp = client.get(
            "/api/v1/research/test-session-1/graph/subgraph?center=nonexistent&hops=1",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_get_nodes(self, client: TestClient, auth_headers: dict, seeded_stores):
        resp = client.get("/api/v1/research/test-session-1/graph/nodes", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3

    def test_get_nodes_filtered_by_type(self, client: TestClient, auth_headers: dict, seeded_stores):
        resp = client.get(
            "/api/v1/research/test-session-1/graph/nodes?type=PROTEIN",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2  # B7-H3 and PD-1

    def test_get_edges(self, client: TestClient, auth_headers: dict, seeded_stores):
        resp = client.get("/api/v1/research/test-session-1/graph/edges", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    def test_get_edges_filtered(self, client: TestClient, auth_headers: dict, seeded_stores):
        resp = client.get(
            "/api/v1/research/test-session-1/graph/edges?source=n-b7h3",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_get_contradictions(self, client: TestClient, auth_headers: dict, seeded_stores):
        resp = client.get("/api/v1/research/test-session-1/graph/contradictions", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_get_stats(self, client: TestClient, auth_headers: dict, seeded_stores):
        resp = client.get("/api/v1/research/test-session-1/graph/stats", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_count"] == 3
        assert data["edge_count"] == 2
        assert "type_distribution" in data
