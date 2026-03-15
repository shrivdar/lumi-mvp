"""Tests for the Slack MCP server — tool listing and tool calls."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from core.models import (
    ResearchConfig,
    ResearchSession,
    SessionStatus,
)
from integrations.slack_mcp import SlackMCPServer
from world_model.knowledge_graph import InMemoryKnowledgeGraph


@pytest.fixture()
def session() -> ResearchSession:
    return ResearchSession(
        id="sess-1",
        query="B7-H3 in NSCLC",
        status=SessionStatus.RUNNING,
        config=ResearchConfig(),
        current_iteration=3,
    )


@pytest.fixture()
def server(
    populated_kg: InMemoryKnowledgeGraph, session: ResearchSession
) -> SlackMCPServer:
    return SlackMCPServer(
        session_store={"sess-1": session},
        orchestrator_store={},
        kg_store={"sess-1": populated_kg},
        living_doc_store={},
    )


class TestSlackMCPToolsList:
    @pytest.mark.asyncio
    async def test_tools_list(self, server: SlackMCPServer) -> None:
        resp = await server.handle_rpc({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        })

        assert resp["id"] == 1
        assert "error" not in resp
        tools = resp["result"]["tools"]
        tool_names = {t["name"] for t in tools}
        assert "yohas_research" in tool_names
        assert "yohas_query_kg" in tool_names
        assert "yohas_status" in tool_names
        assert "yohas_findings" in tool_names


class TestSlackMCPInitialize:
    @pytest.mark.asyncio
    async def test_initialize(self, server: SlackMCPServer) -> None:
        resp = await server.handle_rpc({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        })
        assert resp["result"]["serverInfo"]["name"] == "yohas-slack-mcp"


class TestSlackMCPStatus:
    @pytest.mark.asyncio
    async def test_status_existing_session(self, server: SlackMCPServer) -> None:
        resp = await server.handle_rpc({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "yohas_status",
                "arguments": {"session_id": "sess-1"},
            },
        })

        assert "error" not in resp
        content = resp["result"]["content"]
        result = json.loads(content[0]["text"])
        assert result["session_id"] == "sess-1"
        assert result["query"] == "B7-H3 in NSCLC"
        assert result["status"] == "RUNNING"
        assert result["knowledge_graph"]["nodes"] == 5
        assert result["knowledge_graph"]["edges"] == 4

    @pytest.mark.asyncio
    async def test_status_missing_session(self, server: SlackMCPServer) -> None:
        resp = await server.handle_rpc({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "yohas_status",
                "arguments": {"session_id": "nonexistent"},
            },
        })

        assert "error" in resp
        assert "Session not found" in resp["error"]["message"]

    @pytest.mark.asyncio
    async def test_status_missing_arg(self, server: SlackMCPServer) -> None:
        resp = await server.handle_rpc({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "yohas_status",
                "arguments": {},
            },
        })

        assert "error" in resp


class TestSlackMCPQueryKG:
    @pytest.mark.asyncio
    async def test_query_kg_finds_nodes(self, server: SlackMCPServer) -> None:
        resp = await server.handle_rpc({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "yohas_query_kg",
                "arguments": {"session_id": "sess-1", "question": "What about BRCA1?"},
            },
        })

        assert "error" not in resp
        result = json.loads(resp["result"]["content"][0]["text"])
        assert len(result["matching_nodes"]) >= 1
        assert any(n["name"] == "BRCA1" for n in result["matching_nodes"])
        assert len(result["relevant_edges"]) >= 1

    @pytest.mark.asyncio
    async def test_query_kg_no_match(self, server: SlackMCPServer) -> None:
        resp = await server.handle_rpc({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "yohas_query_kg",
                "arguments": {"session_id": "sess-1", "question": "zzz_no_match_zzz"},
            },
        })

        result = json.loads(resp["result"]["content"][0]["text"])
        assert len(result["matching_nodes"]) == 0


class TestSlackMCPFindings:
    @pytest.mark.asyncio
    async def test_findings_summary(self, server: SlackMCPServer) -> None:
        resp = await server.handle_rpc({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "yohas_findings",
                "arguments": {"session_id": "sess-1", "format": "summary"},
            },
        })

        result = json.loads(resp["result"]["content"][0]["text"])
        assert result["total_nodes"] == 5
        assert result["total_edges"] == 4
        assert "top_findings" in result

    @pytest.mark.asyncio
    async def test_findings_full_format(self, server: SlackMCPServer) -> None:
        resp = await server.handle_rpc({
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "yohas_findings",
                "arguments": {"session_id": "sess-1", "format": "full"},
            },
        })

        result = json.loads(resp["result"]["content"][0]["text"])
        assert "document" in result
        assert "Knowledge Graph Summary" in result["document"]


class TestSlackMCPResearch:
    @pytest.mark.asyncio
    async def test_research_creates_session(self, server: SlackMCPServer) -> None:
        # Mock the deps to avoid real session creation
        with (
            patch("integrations.slack_mcp.settings") as mock_settings,
            patch("integrations.slack_mcp.SlackMCPServer._tool_research") as mock_research,
        ):
            mock_settings.slack_bot_token = ""
            mock_research.return_value = {
                "session_id": "new-sess",
                "query": "test query",
                "status": "pending",
                "message": "Research session created.",
            }

            resp = await server.handle_rpc({
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {
                    "name": "yohas_research",
                    "arguments": {"query": "test query"},
                },
            })

            result = json.loads(resp["result"]["content"][0]["text"])
            assert result["session_id"] == "new-sess"
            assert result["status"] == "pending"


class TestSlackMCPErrors:
    @pytest.mark.asyncio
    async def test_unknown_method(self, server: SlackMCPServer) -> None:
        resp = await server.handle_rpc({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "unknown/method",
            "params": {},
        })

        assert "error" in resp
        assert resp["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_unknown_tool(self, server: SlackMCPServer) -> None:
        resp = await server.handle_rpc({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "nonexistent_tool",
                "arguments": {},
            },
        })

        assert "error" in resp
        assert "Unknown tool" in resp["error"]["message"]
