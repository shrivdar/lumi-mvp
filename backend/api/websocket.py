"""WebSocket endpoint for live research event streaming."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.deps import get_orchestrators, get_sessions
from core.models import SessionStatus

router = APIRouter(tags=["websocket"])


@router.websocket("/research/{research_id}/ws")
async def research_ws(websocket: WebSocket, research_id: str) -> None:
    """Stream research events in real time.

    Events: agent_started, agent_completed, node_added, edge_added,
    edge_falsified, hypothesis_explored, hitl_triggered, hitl_resolved,
    research_completed, error.
    """
    sessions = get_sessions()
    orchestrators = get_orchestrators()

    session = sessions.get(research_id)
    if not session:
        await websocket.close(code=4004, reason="Research session not found")
        return

    await websocket.accept()

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

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close(code=1011, reason="Internal error")
        except Exception:
            pass
