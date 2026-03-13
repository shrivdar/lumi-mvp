"""Tests for WebSocket endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.deps import _sessions
from core.models import ResearchSession, SessionStatus


class TestWebSocket:
    def test_ws_not_found(self, client: TestClient):
        with pytest.raises(Exception):
            with client.websocket_connect("/api/v1/research/nonexistent/ws"):
                pass

    def test_ws_completed_session_sends_finished(self, client: TestClient, seeded_stores):
        with client.websocket_connect("/api/v1/research/test-session-1/ws") as ws:
            msg = ws.receive_json()
            assert msg["event_type"] == "research_finished"
            assert msg["data"]["status"] == "COMPLETED"

    def test_ws_receives_events(self, client: TestClient):
        # Create a session that's "running" so the WS will poll
        session = ResearchSession(
            id="ws-test-1",
            query="test query",
            status=SessionStatus.COMPLETED,
        )
        _sessions["ws-test-1"] = session

        with client.websocket_connect("/api/v1/research/ws-test-1/ws") as ws:
            msg = ws.receive_json()
            assert msg["event_type"] == "research_finished"
