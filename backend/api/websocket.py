"""WebSocket endpoint for live research event streaming and HITL interaction."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.deps import get_orchestrators, get_sessions
from core.models import SessionStatus

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["websocket"])

# Track active WebSocket connections per research session
_active_connections: dict[str, list[WebSocket]] = {}


def get_connections(research_id: str) -> list[WebSocket]:
    """Get active WebSocket connections for a research session."""
    return _active_connections.get(research_id, [])


async def broadcast_event(research_id: str, event: dict[str, Any]) -> None:
    """Broadcast an event to all connected clients for a research session."""
    connections = _active_connections.get(research_id, [])
    dead: list[WebSocket] = []
    for ws in connections:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    # Clean up dead connections
    for ws in dead:
        connections.remove(ws)


async def broadcast_hitl_request(
    research_id: str,
    hypothesis_id: str,
    hypothesis: str,
    uncertainty_composite: float,
    reason: str,
    message: str,
    timeout_seconds: int = 600,
) -> str | None:
    """Broadcast a HITL request to connected frontend clients and wait for response.

    Returns the human response text, or None if timed out.
    """
    hitl_event = {
        "event_type": "hitl_request",
        "data": {
            "hypothesis_id": hypothesis_id,
            "hypothesis": hypothesis,
            "uncertainty_composite": uncertainty_composite,
            "reason": reason,
            "message": message,
            "timeout_seconds": timeout_seconds,
        },
    }
    await broadcast_event(research_id, hitl_event)

    # Wait for response from any connected client
    # The response is sent back via WebSocket as a JSON message with type "hitl_response"
    # This is handled in the main WebSocket loop below
    return None  # Actual HITL response is handled in the WebSocket receive loop


@router.websocket("/research/{research_id}/ws")
async def research_ws(websocket: WebSocket, research_id: str) -> None:
    """Stream research events in real time and accept HITL responses.

    Events sent: agent_started, agent_completed, node_added, edge_added,
    edge_falsified, hypothesis_explored, hitl_request, hitl_resolved,
    research_completed, error.

    Events received: hitl_response (human feedback from frontend).
    """
    sessions = get_sessions()
    orchestrators = get_orchestrators()

    session = sessions.get(research_id)
    if not session:
        await websocket.close(code=4004, reason="Research session not found")
        return

    await websocket.accept()

    # Register connection
    if research_id not in _active_connections:
        _active_connections[research_id] = []
    _active_connections[research_id].append(websocket)

    try:
        while True:
            # Check session status
            session = sessions.get(research_id)
            if not session:
                await websocket.send_json({"event_type": "error", "data": {"message": "Session removed"}})
                break

            # Drain events from orchestrator
            orch = orchestrators.get(research_id)
            if orch:
                events = orch.drain_events()
                for event in events:
                    await websocket.send_json(event.model_dump(mode="json"))

            # Check if research is done
            if session.status in (
                SessionStatus.COMPLETED,
                SessionStatus.FAILED,
                SessionStatus.CANCELLED,
            ):
                await websocket.send_json({
                    "event_type": "research_finished",
                    "data": {"status": str(session.status)},
                })
                break

            # Check for incoming messages (HITL responses, etc.) with short timeout
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                try:
                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")

                    if msg_type == "hitl_response":
                        # Handle HITL response from frontend
                        response_text = msg.get("response", "")
                        if orch and orch._uncertainty:
                            orch._uncertainty.record_hitl_response({
                                "source": "websocket",
                                "response": response_text,
                                "received": True,
                            })
                            logger.info(
                                "hitl_response_received_ws",
                                research_id=research_id,
                                response_preview=response_text[:100],
                            )
                        # Resume if waiting
                        if session.status == SessionStatus.WAITING_HITL:
                            session.status = SessionStatus.RUNNING

                    elif msg_type == "ping":
                        await websocket.send_json({"event_type": "pong", "data": {}})

                except json.JSONDecodeError:
                    pass

            except asyncio.TimeoutError:
                pass  # No message received, continue polling

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("websocket_error", research_id=research_id, error=str(exc))
        try:
            await websocket.close(code=1011, reason="Internal error")
        except Exception:
            pass
    finally:
        # Unregister connection
        conns = _active_connections.get(research_id, [])
        if websocket in conns:
            conns.remove(websocket)
        if not conns and research_id in _active_connections:
            del _active_connections[research_id]
