"""Agent status and audit trail endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_agent_results, get_orchestrators

router = APIRouter(prefix="/research/{research_id}/agents", tags=["agents"])


@router.get("")
async def list_agents(
    research_id: str,
    results_store: dict = Depends(get_agent_results),
) -> dict:
    """List agents that participated in a research session."""
    results = results_store.get(research_id)
    if results is None:
        raise HTTPException(status_code=404, detail="Research session not found")

    # Also pull from orchestrator's live results
    from api.deps import get_orchestrators
    orchestrators = get_orchestrators()
    orch = orchestrators.get(research_id)
    all_results = list(results)
    if orch and hasattr(orch, "_all_results"):
        all_results = orch._all_results

    agents = []
    for r in all_results:
        agents.append({
            "agent_id": r.agent_id,
            "agent_type": str(r.agent_type),
            "status": "COMPLETED" if r.success else "FAILED",
            "hypothesis_branch": r.hypothesis_id or None,
            "task_count": 1,
            "nodes_added": len(r.nodes_added),
            "edges_added": len(r.edges_added),
            "task_id": r.task_id,
            "success": r.success,
            "duration_ms": r.duration_ms,
            "summary": r.summary,
        })

    return {"agents": agents, "count": len(agents)}


@router.get("/{agent_id}/log")
async def get_agent_log(
    research_id: str,
    agent_id: str,
    orchestrators: dict = Depends(get_orchestrators),
) -> dict:
    """Full audit trail for an agent."""
    orch = orchestrators.get(research_id)
    if not orch:
        raise HTTPException(status_code=404, detail="Research session not found")

    all_results = orch._all_results if hasattr(orch, "_all_results") else []
    agent_result = next((r for r in all_results if r.agent_id == agent_id), None)

    if not agent_result:
        raise HTTPException(status_code=404, detail="Agent not found")

    return {
        "agent_id": agent_id,
        "agent_type": str(agent_result.agent_type),
        "reasoning_trace": agent_result.reasoning_trace,
        "falsification_results": [f.model_dump(mode="json") for f in agent_result.falsification_results],
        "errors": agent_result.errors,
        "token_usage": agent_result.token_usage,
        "duration_ms": agent_result.duration_ms,
    }


@router.get("/{agent_id}/result")
async def get_agent_result(
    research_id: str,
    agent_id: str,
    orchestrators: dict = Depends(get_orchestrators),
) -> dict:
    """Full AgentResult for an agent."""
    orch = orchestrators.get(research_id)
    if not orch:
        raise HTTPException(status_code=404, detail="Research session not found")

    all_results = orch._all_results if hasattr(orch, "_all_results") else []
    agent_result = next((r for r in all_results if r.agent_id == agent_id), None)

    if not agent_result:
        raise HTTPException(status_code=404, detail="Agent not found")

    return agent_result.model_dump(mode="json")
