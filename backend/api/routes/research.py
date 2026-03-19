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
    """Create a new research session and dispatch to background execution.

    Production mode: dispatches to Celery for distributed agent execution.
    Development mode: runs as an asyncio background task.
    """
    from agents.factory import create_agent
    from core.config import settings
    from core.llm import LLMClient
    from orchestrator.research_loop import ResearchOrchestrator

    session = ResearchSession(query=body.query, config=body.config)
    kg = InMemoryKnowledgeGraph(graph_id=session.id)

    llm = LLMClient()

    # Build full tool registry + instances (same as E2E script)
    tool_registry, tool_instances = _build_full_tool_stack()
    tool_entries = tool_registry.list_tools()

    orchestrator = ResearchOrchestrator(
        llm=llm,
        kg=kg,
        agent_factory=create_agent,
        tool_entries=tool_entries,
        tool_instances=tool_instances,
    )

    sessions[session.id] = session
    kgs[session.id] = kg
    orchestrators[session.id] = orchestrator
    agent_results[session.id] = []

    config_dict = body.config.model_dump(mode="json")

    if settings.environment == "production":
        # Production: dispatch to Celery
        from workers.tasks import run_research
        celery_result = run_research.delay(session.id, body.query, config_dict)
        return CreateResearchResponse(
            research_id=session.id,
            status=str(session.status),
        )
    else:
        # Development: run as asyncio background task
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


_tool_cache: tuple | None = None


def _build_full_tool_stack() -> tuple:
    """Build the full tool registry + instances (native + MCP catalog).

    Returns (InMemoryToolRegistry, dict[str, BaseTool]).
    Caches after first call so tools are reused across requests.
    """
    global _tool_cache
    if _tool_cache is not None:
        return _tool_cache

    from core.models import ToolRegistryEntry, ToolSourceType
    from core.tool_registry import InMemoryToolRegistry

    registry = InMemoryToolRegistry()
    tool_instances: dict = {}

    # Import and instantiate all native tools
    tool_classes = []
    try:
        from integrations.pubmed import PubMedTool
        tool_classes.append(PubMedTool)
    except Exception:
        pass
    try:
        from integrations.semantic_scholar import SemanticScholarTool
        tool_classes.append(SemanticScholarTool)
    except Exception:
        pass
    try:
        from integrations.uniprot import UniProtTool
        tool_classes.append(UniProtTool)
    except Exception:
        pass
    try:
        from integrations.kegg import KEGGTool
        tool_classes.append(KEGGTool)
    except Exception:
        pass
    try:
        from integrations.reactome import ReactomeTool
        tool_classes.append(ReactomeTool)
    except Exception:
        pass
    try:
        from integrations.mygene import MyGeneTool
        tool_classes.append(MyGeneTool)
    except Exception:
        pass
    try:
        from integrations.chembl import ChEMBLTool
        tool_classes.append(ChEMBLTool)
    except Exception:
        pass
    try:
        from integrations.clinicaltrials import ClinicalTrialsTool
        tool_classes.append(ClinicalTrialsTool)
    except Exception:
        pass
    try:
        from integrations.python_repl import PythonREPLTool
        tool_classes.append(PythonREPLTool)
    except Exception:
        pass
    try:
        from integrations.opentargets import OpenTargetsTool
        tool_classes.append(OpenTargetsTool)
    except Exception:
        pass
    try:
        from integrations.clinvar import ClinVarTool
        tool_classes.append(ClinVarTool)
    except Exception:
        pass
    try:
        from integrations.gtex import GTExTool
        tool_classes.append(GTExTool)
    except Exception:
        pass
    try:
        from integrations.gnomad import GnomADTool
        tool_classes.append(GnomADTool)
    except Exception:
        pass
    try:
        from integrations.hpo import HPOTool
        tool_classes.append(HPOTool)
    except Exception:
        pass
    try:
        from integrations.omim import OMIMTool
        tool_classes.append(OMIMTool)
    except Exception:
        pass
    try:
        from integrations.biogrid import BioGRIDTool
        tool_classes.append(BioGRIDTool)
    except Exception:
        pass
    try:
        from integrations.depmap import DepMapTool
        tool_classes.append(DepMapTool)
    except Exception:
        pass
    try:
        from integrations.cellxgene import CellxGeneTool
        tool_classes.append(CellxGeneTool)
    except Exception:
        pass
    try:
        from integrations.string_db import StringDBTool
        tool_classes.append(StringDBTool)
    except Exception:
        pass

    for cls in tool_classes:
        try:
            instance = cls(registry=registry)
            tool_instances[instance.name] = instance
        except Exception:
            pass

    # Register MCP + container catalog entries (metadata only)
    try:
        from integrations.tool_catalog import get_catalog
        for entry in get_catalog():
            if entry.source_type in (ToolSourceType.MCP, ToolSourceType.CONTAINER):
                registry.register(entry)
    except Exception:
        pass

    _tool_cache = (registry, tool_instances)
    return _tool_cache


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
