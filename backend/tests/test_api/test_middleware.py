"""Tests for API middleware — auth, request ID, timing."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestAPIKeyAuth:
    """Auth middleware tests."""

    def test_missing_api_key_returns_401(self, client: TestClient):
        resp = client.get("/api/v1/templates")
        assert resp.status_code == 401
        assert "Missing API key" in resp.json()["error"]

    def test_invalid_api_key_returns_401(self, client: TestClient):
        resp = client.get("/api/v1/templates", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401
        assert "Invalid API key" in resp.json()["error"]

    def test_valid_api_key_returns_200(self, client: TestClient, auth_headers: dict):
        resp = client.get("/api/v1/templates", headers=auth_headers)
        assert resp.status_code == 200

    def test_health_skips_auth(self, client: TestClient):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_ready_skips_auth(self, client: TestClient):
        resp = client.get("/api/v1/health/ready")
        assert resp.status_code == 200


class TestRequestID:
    """Request ID middleware tests."""

    def test_response_has_request_id_header(self, client: TestClient):
        resp = client.get("/api/v1/health")
        assert "X-Request-ID" in resp.headers

    def test_custom_request_id_is_preserved(self, client: TestClient):
        resp = client.get("/api/v1/health", headers={"X-Request-ID": "my-custom-id"})
        assert resp.headers["X-Request-ID"] == "my-custom-id"


class TestTiming:
    """Timing middleware tests."""

    def test_response_has_timing_header(self, client: TestClient):
        resp = client.get("/api/v1/health")
        assert "X-Response-Time-Ms" in resp.headers
        assert int(resp.headers["X-Response-Time-Ms"]) >= 0
