"""Slack MCP server — expose YOHAS research tools via MCP protocol for bidirectional Slack integration.

Provides four MCP tools:
- yohas_research(query): Trigger a new research session from Slack
- yohas_query_kg(question): Query the knowledge graph in natural language
- yohas_status(session_id): Get session status and progress
- yohas_findings(session_id): Get current findings for a session

Uses the existing MCP infrastructure (mcp_client.py) and Slack integration (slack.py).
Designed to run as a JSON-RPC server over stdio or HTTP.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from core.audit import AuditLogger
from core.config import settings
from core.exceptions import MCPError

logger = structlog.get_logger(__name__)
audit = AuditLogger("slack_mcp")


# ---------------------------------------------------------------------------
# Tool definitions (MCP manifest format)
# ---------------------------------------------------------------------------

YOHAS_MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "yohas_research",
        "description": "Start a new YOHAS research session. Triggers autonomous hypothesis-driven investigation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The research query to investigate "
                        "(e.g. 'Role of B7-H3 in NSCLC immunotherapy resistance')"
                    ),
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Maximum MCTS iterations (default: 15)",
                    "default": 15,
                },
                "notify_channel": {
                    "type": "string",
                    "description": "Slack channel to post updates to",
                    "default": "",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "yohas_query_kg",
        "description": "Query the knowledge graph for a running or completed research session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The research session ID to query",
                },
                "question": {
                    "type": "string",
                    "description": (
                        "Natural language question about the knowledge graph "
                        "(e.g. 'What proteins interact with BRCA1?')"
                    ),
                },
            },
            "required": ["session_id", "question"],
        },
    },
    {
        "name": "yohas_status",
        "description": "Get the current status and progress of a research session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The research session ID",
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "yohas_findings",
        "description": "Get key findings, contradictions, and uncertainties from a research session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The research session ID",
                },
                "format": {
                    "type": "string",
                    "description": "Output format: 'summary' (brief) or 'full' (detailed markdown)",
                    "enum": ["summary", "full"],
                    "default": "summary",
                },
            },
            "required": ["session_id"],
        },
    },
]


class SlackMCPServer:
    """MCP server that exposes YOHAS tools for bidirectional Slack integration.

    Handles JSON-RPC requests conforming to the MCP protocol, routing tool calls
    to the appropriate YOHAS backend functions.

    Usage::

        server = SlackMCPServer(session_store=sessions, orchestrator_store=orchestrators)
        # For stdio transport:
        response = await server.handle_rpc(request_dict)
        # For HTTP transport:
        app.post("/rpc")(server.handle_rpc)
    """

    def __init__(
        self,
        session_store: dict[str, Any] | None = None,
        orchestrator_store: dict[str, Any] | None = None,
        kg_store: dict[str, Any] | None = None,
        living_doc_store: dict[str, Any] | None = None,
    ) -> None:
        self._sessions = session_store or {}
        self._orchestrators = orchestrator_store or {}
        self._kgs = kg_store or {}
        self._living_docs = living_doc_store or {}

    # ── JSON-RPC dispatch ─────────────────────────────────────────────

    async def handle_rpc(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a single JSON-RPC request and return a response."""
        rpc_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        try:
            if method == "tools/list":
                result = self._handle_tools_list()
            elif method == "tools/call":
                result = await self._handle_tool_call(params)
            elif method == "initialize":
                result = self._handle_initialize()
            else:
                return self._rpc_error(rpc_id, -32601, f"Method not found: {method}")

            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": result,
            }
        except MCPError as exc:
            audit.error("slack_mcp_error", error_code=exc.error_code, message=exc.message)
            return self._rpc_error(rpc_id, -32000, exc.message)
        except Exception as exc:
            audit.error("slack_mcp_unexpected_error", message=str(exc))
            return self._rpc_error(rpc_id, -32603, f"Internal error: {exc}")

    @staticmethod
    def _rpc_error(rpc_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": code, "message": message},
        }

    # ── MCP protocol methods ──────────────────────────────────────────

    def _handle_initialize(self) -> dict[str, Any]:
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "yohas-slack-mcp", "version": "1.0.0"},
        }

    @staticmethod
    def _handle_tools_list() -> dict[str, Any]:
        return {"tools": YOHAS_MCP_TOOLS}

    async def _handle_tool_call(self, params: dict[str, Any]) -> dict[str, Any]:
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        audit.log("slack_mcp_tool_call", tool=tool_name)

        handler = {
            "yohas_research": self._tool_research,
            "yohas_query_kg": self._tool_query_kg,
            "yohas_status": self._tool_status,
            "yohas_findings": self._tool_findings,
        }.get(tool_name)

        if handler is None:
            raise MCPError(
                f"Unknown tool: {tool_name}",
                error_code="MCP_UNKNOWN_TOOL",
            )

        result = await handler(arguments)
        return {
            "content": [{"type": "text", "text": json.dumps(result, default=str)}],
        }

    # ── Tool implementations ──────────────────────────────────────────

    async def _tool_research(self, args: dict[str, Any]) -> dict[str, Any]:
        """Start a new research session."""
        query = args.get("query", "")
        if not query:
            raise MCPError("Query is required", error_code="MCP_INVALID_ARGS")

        max_iterations = args.get("max_iterations", settings.max_mcts_iterations)
        notify_channel = args.get("notify_channel", settings.slack_default_channel)

        # Create session via orchestrator (lazy import to avoid circular deps)
        from api.deps import get_sessions

        sessions = get_sessions()

        # Build a minimal session
        from core.models import ResearchConfig, ResearchSession, SessionStatus

        session = ResearchSession(
            query=query,
            status=SessionStatus.INITIALIZING,
            config=ResearchConfig(
                max_mcts_iterations=max_iterations,
            ),
        )
        sessions[session.id] = session

        audit.log(
            "slack_mcp_research_started",
            session_id=session.id,
            query=query,
            channel=notify_channel,
        )

        # Notify Slack about the new research session
        if notify_channel and settings.slack_bot_token:
            try:
                from integrations.slack import SlackTool

                slack = SlackTool()
                await slack.execute(
                    action="notify",
                    message=(
                        f"🔬 *YOHAS Research Started*\n*Query:* {query}\n"
                        f"*Session:* `{session.id}`\n*Max iterations:* {max_iterations}"
                    ),
                    channel=notify_channel,
                )
            except Exception as exc:
                logger.warning("slack_notify_failed", error=str(exc))

        return {
            "session_id": session.id,
            "query": query,
            "status": "pending",
            "message": f"Research session created. Use yohas_status('{session.id}') to track progress.",
        }

    async def _tool_query_kg(self, args: dict[str, Any]) -> dict[str, Any]:
        """Query the knowledge graph for a session."""
        session_id = args.get("session_id", "")
        question = args.get("question", "")
        if not session_id or not question:
            raise MCPError("session_id and question are required", error_code="MCP_INVALID_ARGS")

        kg = self._kgs.get(session_id)
        if kg is None:
            session = self._find_session(session_id)
            if session is None:
                raise MCPError(f"Session not found: {session_id}", error_code="MCP_SESSION_NOT_FOUND")
            return {
                "session_id": session_id,
                "question": question,
                "answer": "Knowledge graph not yet available for this session.",
                "nodes": 0,
                "edges": 0,
            }

        # Search nodes by name match (simple keyword matching)
        import re
        keywords = [w for w in re.findall(r'\w+', question.lower()) if len(w) > 2]
        matching_nodes = []
        for node in list(kg._nodes.values())[:100]:
            name_lower = node.name.lower()
            if any(kw in name_lower for kw in keywords):
                matching_nodes.append({
                    "id": node.id,
                    "name": node.name,
                    "type": node.type.value if hasattr(node.type, "value") else str(node.type),
                    "confidence": node.confidence,
                })

        # Find edges involving matched nodes
        matched_ids = {n["id"] for n in matching_nodes}
        relevant_edges = []
        for edge in list(kg._edges.values())[:200]:
            if edge.source_id in matched_ids or edge.target_id in matched_ids:
                src = kg._nodes.get(edge.source_id)
                tgt = kg._nodes.get(edge.target_id)
                relevant_edges.append({
                    "source": src.name if src else edge.source_id,
                    "target": tgt.name if tgt else edge.target_id,
                    "relation": edge.relation.value,
                    "confidence": edge.confidence.overall,
                    "falsified": edge.falsified,
                })

        return {
            "session_id": session_id,
            "question": question,
            "matching_nodes": matching_nodes[:20],
            "relevant_edges": relevant_edges[:20],
            "total_nodes": len(kg._nodes),
            "total_edges": len(kg._edges),
        }

    @staticmethod
    def _node_name(kg: Any, node_id: str) -> str:
        """Get node name from KG, falling back to the ID."""
        node = kg._nodes.get(node_id)
        return node.name if node is not None else node_id

    def _find_session(self, session_id: str) -> Any:
        """Look up a session from local store, falling back to api.deps."""
        session = self._sessions.get(session_id)
        if session is not None:
            return session
        try:
            from api.deps import get_sessions
            return get_sessions().get(session_id)
        except Exception:
            return None

    async def _tool_status(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get research session status."""
        session_id = args.get("session_id", "")
        if not session_id:
            raise MCPError("session_id is required", error_code="MCP_INVALID_ARGS")

        session = self._find_session(session_id)
        if session is None:
            raise MCPError(f"Session not found: {session_id}", error_code="MCP_SESSION_NOT_FOUND")

        kg = self._kgs.get(session_id)
        kg_stats = {}
        if kg is not None:
            kg_stats = {
                "nodes": len(kg._nodes),
                "edges": len(kg._edges),
                "avg_confidence": kg.avg_confidence() if hasattr(kg, "avg_confidence") else 0.0,
            }

        return {
            "session_id": session_id,
            "query": session.query,
            "status": session.status.value if hasattr(session.status, "value") else str(session.status),
            "current_iteration": session.current_iteration,
            "knowledge_graph": kg_stats,
        }

    async def _tool_findings(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get findings for a session."""
        session_id = args.get("session_id", "")
        fmt = args.get("format", "summary")
        if not session_id:
            raise MCPError("session_id is required", error_code="MCP_INVALID_ARGS")

        # Check for living document first
        living_doc = self._living_docs.get(session_id)
        if living_doc is not None and fmt == "full":
            return {
                "session_id": session_id,
                "format": "full",
                "document": living_doc.current_content,
                "version": living_doc.version_count,
            }

        # Fall back to KG markdown summary
        kg = self._kgs.get(session_id)
        if kg is None:
            session = self._find_session(session_id)
            if session is None:
                raise MCPError(f"Session not found: {session_id}", error_code="MCP_SESSION_NOT_FOUND")
            return {
                "session_id": session_id,
                "format": fmt,
                "findings": "No findings available yet.",
            }

        if fmt == "full":
            return {
                "session_id": session_id,
                "format": "full",
                "document": kg.to_markdown_summary(),
            }

        # Summary format — extract key stats
        contradictions = [e for e in kg._edges.values() if e.is_contradiction]
        falsified = [e for e in kg._edges.values() if e.falsified]
        high_conf = [
            e for e in kg._edges.values()
            if e.confidence.overall >= 0.7 and not e.falsified and not e.is_contradiction
        ]

        return {
            "session_id": session_id,
            "format": "summary",
            "total_nodes": len(kg._nodes),
            "total_edges": len(kg._edges),
            "high_confidence_findings": len(high_conf),
            "contradictions": len(contradictions),
            "falsified": len(falsified),
            "top_findings": [
                {
                    "source": self._node_name(kg, e.source_id),
                    "relation": e.relation.value,
                    "target": self._node_name(kg, e.target_id),
                    "confidence": e.confidence.overall,
                }
                for e in sorted(
                    high_conf, key=lambda x: x.confidence.overall, reverse=True
                )[:5]
            ],
        }
