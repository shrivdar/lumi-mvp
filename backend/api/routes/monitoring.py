"""Monitoring endpoints — real-time stats for active agents, token usage, KG size, etc."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_knowledge_graphs, get_orchestrators, get_sessions
from core.models import SessionStatus

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/overview")
async def get_overview(
    sessions: dict = Depends(get_sessions),
    kgs: dict = Depends(get_knowledge_graphs),
    orchestrators: dict = Depends(get_orchestrators),
) -> dict:
    """Global monitoring overview: active sessions, total KG size, token usage."""
    all_sessions = list(sessions.values())
    active_statuses = (SessionStatus.RUNNING, SessionStatus.INITIALIZING, SessionStatus.WAITING_HITL)
    active = [s for s in all_sessions if s.status in active_statuses]
    completed = [s for s in all_sessions if s.status == SessionStatus.COMPLETED]
    failed = [s for s in all_sessions if s.status == SessionStatus.FAILED]

    total_nodes = sum(kg.node_count() for kg in kgs.values())
    total_edges = sum(kg.edge_count() for kg in kgs.values())
    total_tokens = sum(s.total_tokens_used for s in all_sessions)

    total_agents_spawned = 0
    for orch in orchestrators.values():
        total_agents_spawned += getattr(orch, "_total_agents_spawned", 0)

    return {
        "sessions": {
            "total": len(all_sessions),
            "active": len(active),
            "completed": len(completed),
            "failed": len(failed),
        },
        "knowledge_graph": {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
        },
        "tokens": {
            "total_used": total_tokens,
        },
        "agents": {
            "total_spawned": total_agents_spawned,
        },
    }


@router.get("/research/{research_id}/stats")
async def get_research_stats(
    research_id: str,
    sessions: dict = Depends(get_sessions),
    kgs: dict = Depends(get_knowledge_graphs),
    orchestrators: dict = Depends(get_orchestrators),
) -> dict:
    """Real-time stats for a specific research session."""
    session = sessions.get(research_id)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    kg = kgs.get(research_id)
    orch = orchestrators.get(research_id)

    # Agent stats
    agent_type_counts: dict[str, int] = {}
    if orch:
        for result in orch._all_results:
            agent_type = str(result.agent_type)
            agent_type_counts[agent_type] = agent_type_counts.get(agent_type, 0) + 1

    # Hypothesis tree stats
    tree_stats: dict[str, Any] = {}
    if orch and orch.tree:
        tree = orch.tree
        tree_stats = {
            "node_count": tree.node_count,
            "total_visits": tree.total_visits,
            "max_depth": max((n.depth for n in tree.all_nodes), default=0),
            "confirmed_count": sum(
                1 for n in tree.all_nodes if str(n.status) == "CONFIRMED"
            ),
            "refuted_count": sum(
                1 for n in tree.all_nodes if str(n.status) == "REFUTED"
            ),
            "exploring_count": sum(
                1 for n in tree.all_nodes if str(n.status) == "EXPLORING"
            ),
            "unexplored_count": sum(
                1 for n in tree.all_nodes if str(n.status) == "UNEXPLORED"
            ),
        }
        best = tree.get_best_hypothesis()
        if best:
            tree_stats["best_hypothesis"] = {
                "id": best.id,
                "hypothesis": best.hypothesis,
                "confidence": best.confidence,
                "avg_info_gain": best.avg_info_gain,
            }

    # KG stats
    kg_stats: dict[str, Any] = {}
    if kg:
        kg_stats = {
            "node_count": kg.node_count(),
            "edge_count": kg.edge_count(),
            "avg_confidence": kg.avg_confidence(),
        }

    # Uncertainty trend
    uncertainty_trend: dict[str, Any] = {}
    if orch and orch._uncertainty:
        uncertainty_trend = orch._uncertainty.get_trend()

    # Token usage
    token_stats = {
        "session_tokens_used": getattr(orch, "_session_tokens_used", 0) if orch else 0,
        "session_token_budget": session.config.session_token_budget,
        "budget_utilization": (
            getattr(orch, "_session_tokens_used", 0) / max(session.config.session_token_budget, 1)
        ) if orch else 0,
    }

    return {
        "research_id": research_id,
        "status": str(session.status),
        "current_iteration": session.current_iteration,
        "agents": {
            "total_spawned": getattr(orch, "_total_agents_spawned", 0) if orch else 0,
            "max_total": session.config.max_total_agents,
            "type_counts": agent_type_counts,
        },
        "hypothesis_tree": tree_stats,
        "knowledge_graph": kg_stats,
        "uncertainty": uncertainty_trend,
        "tokens": token_stats,
    }


@router.get("/research/{research_id}/agents")
async def get_active_agents(
    research_id: str,
    sessions: dict = Depends(get_sessions),
    orchestrators: dict = Depends(get_orchestrators),
) -> dict:
    """Get agent constellation data for a research session.

    Returns data shaped for the agent-constellation.tsx component.
    """
    session = sessions.get(research_id)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    orch = orchestrators.get(research_id)
    agents: list[dict[str, Any]] = []

    if orch:
        # Aggregate agent info from results
        agent_map: dict[str, dict[str, Any]] = {}
        for result in orch._all_results:
            agent_id = result.agent_id
            if agent_id not in agent_map:
                agent_map[agent_id] = {
                    "agent_id": agent_id,
                    "agent_type": str(result.agent_type),
                    "status": "COMPLETED" if result.success else "FAILED",
                    "hypothesis_branch": result.hypothesis_id,
                    "task_count": 0,
                    "nodes_added": 0,
                    "edges_added": 0,
                    "tokens_used": 0,
                }
            info = agent_map[agent_id]
            info["task_count"] += 1
            info["nodes_added"] += len(result.nodes_added)
            info["edges_added"] += len(result.edges_added)
            info["tokens_used"] += result.llm_tokens_used

        agents = list(agent_map.values())

    return {
        "research_id": research_id,
        "agents": agents,
        "total_spawned": len(agents),
    }


@router.get("/research/{research_id}/uncertainty")
async def get_uncertainty_data(
    research_id: str,
    sessions: dict = Depends(get_sessions),
    orchestrators: dict = Depends(get_orchestrators),
) -> dict:
    """Get uncertainty radar data for a research session.

    Returns data shaped for the uncertainty-radar.tsx component.
    """
    session = sessions.get(research_id)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    orch = orchestrators.get(research_id)

    if not orch or not orch._uncertainty:
        return {
            "research_id": research_id,
            "current": {
                "input_ambiguity": 0,
                "data_quality": 0,
                "reasoning_divergence": 0,
                "model_disagreement": 0,
                "conflict_uncertainty": 0,
                "novelty_uncertainty": 0,
                "composite": 0,
                "is_critical": False,
            },
            "history": [],
            "trend": "no_data",
        }

    history = orch._uncertainty._history
    trend_data = orch._uncertainty.get_trend()

    current = history[-1] if history else None
    current_dict = current.model_dump(mode="json") if current else {
        "input_ambiguity": 0,
        "data_quality": 0,
        "reasoning_divergence": 0,
        "model_disagreement": 0,
        "conflict_uncertainty": 0,
        "novelty_uncertainty": 0,
        "composite": 0,
        "is_critical": False,
    }

    history_dicts = [u.model_dump(mode="json") for u in history]

    return {
        "research_id": research_id,
        "current": current_dict,
        "history": history_dicts,
        "trend": trend_data.get("trend", "no_data"),
        "hitl_triggered": trend_data.get("hitl_triggered", False),
        "hitl_response_count": trend_data.get("hitl_response_count", 0),
    }
