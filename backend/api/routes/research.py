"""Research session CRUD endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.deps import (
    get_agent_results,
    get_knowledge_graphs,
    get_orchestrators,
    get_sessions,
)
from core.models import (
    ResearchConfig,
    ResearchSession,
    SessionStatus,
)
from world_model.knowledge_graph import InMemoryKnowledgeGraph

router = APIRouter(prefix="/research", tags=["research"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class CreateResearchRequest(BaseModel):
    query: str
    config: ResearchConfig = Field(default_factory=ResearchConfig)


class CreateResearchResponse(BaseModel):
    research_id: str
    status: str


class FeedbackRequest(BaseModel):
    edge_id: str
    feedback: str  # "agree" | "disagree" | free text
    confidence_override: float | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", status_code=201)
async def create_research(
    body: CreateResearchRequest,
    sessions: dict = Depends(get_sessions),
    kgs: dict = Depends(get_knowledge_graphs),
    orchestrators: dict = Depends(get_orchestrators),
    agent_results: dict = Depends(get_agent_results),
) -> CreateResearchResponse:
    """Create a new research session and dispatch to background execution."""
    from core.llm import LLMClient
    from orchestrator.research_loop import ResearchOrchestrator

    session = ResearchSession(query=body.query, config=body.config)
    kg = InMemoryKnowledgeGraph(graph_id=session.id)

    llm = LLMClient()
    orchestrator = ResearchOrchestrator(llm=llm, kg=kg)

    sessions[session.id] = session
    kgs[session.id] = kg
    orchestrators[session.id] = orchestrator
    agent_results[session.id] = []

    # Dispatch to Celery in production; for now run in background task
    import asyncio

    async def _run() -> None:
        try:
            result_session = await orchestrator.run(body.query, body.config)
            sessions[session.id] = result_session
        except Exception:
            session.status = SessionStatus.FAILED
            sessions[session.id] = session

    asyncio.create_task(_run())

    return CreateResearchResponse(
        research_id=session.id,
        status=str(session.status),
    )


@router.get("")
async def list_research(
    status: str | None = None,
    offset: int = 0,
    limit: int = 20,
    sessions: dict = Depends(get_sessions),
) -> dict:
    """List research sessions with optional status filter and pagination."""
    items: list[ResearchSession] = list(sessions.values())
    if status:
        items = [s for s in items if str(s.status) == status]

    total = len(items)
    items = items[offset : offset + limit]

    return {
        "items": [s.model_dump(mode="json") for s in items],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@router.get("/{research_id}")
async def get_research(
    research_id: str,
    sessions: dict = Depends(get_sessions),
) -> dict:
    """Return full ResearchSession."""
    session = sessions.get(research_id)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")
    return session.model_dump(mode="json")


@router.get("/{research_id}/result")
async def get_research_result(
    research_id: str,
    sessions: dict = Depends(get_sessions),
) -> dict:
    """Return ResearchResult (only if completed)."""
    session: ResearchSession | None = sessions.get(research_id)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")
    if session.status != SessionStatus.COMPLETED or session.result is None:
        raise HTTPException(status_code=409, detail="Research not yet completed")
    return session.result.model_dump(mode="json")


@router.post("/{research_id}/cancel", status_code=200)
async def cancel_research(
    research_id: str,
    sessions: dict = Depends(get_sessions),
) -> dict:
    """Cancel a running research session."""
    session: ResearchSession | None = sessions.get(research_id)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")
    if session.status not in (SessionStatus.RUNNING, SessionStatus.INITIALIZING, SessionStatus.WAITING_HITL):
        raise HTTPException(status_code=409, detail=f"Cannot cancel session in {session.status} state")
    session.status = SessionStatus.CANCELLED
    return {"research_id": research_id, "status": str(session.status)}


@router.post("/{research_id}/feedback", status_code=200)
async def submit_feedback(
    research_id: str,
    body: FeedbackRequest,
    sessions: dict = Depends(get_sessions),
    kgs: dict = Depends(get_knowledge_graphs),
) -> dict:
    """Submit human feedback on a specific edge."""
    session: ResearchSession | None = sessions.get(research_id)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    kg: InMemoryKnowledgeGraph | None = kgs.get(research_id)
    if not kg:
        raise HTTPException(status_code=404, detail="Knowledge graph not found")

    edge = kg.get_edge(body.edge_id)
    if not edge:
        raise HTTPException(status_code=404, detail="Edge not found")

    if body.confidence_override is not None:
        from core.models import EvidenceSource, EvidenceSourceType

        evidence = EvidenceSource(
            source_type=EvidenceSourceType.HUMAN_INPUT,
            claim=body.feedback,
            quality_score=1.0,
            confidence=body.confidence_override,
        )
        kg.update_edge_confidence(body.edge_id, body.confidence_override, evidence)

    return {
        "research_id": research_id,
        "edge_id": body.edge_id,
        "feedback": body.feedback,
        "applied": True,
    }
