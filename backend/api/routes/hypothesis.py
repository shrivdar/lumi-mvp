"""Hypothesis tree endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_knowledge_graphs, get_orchestrators

router = APIRouter(prefix="/research/{research_id}/hypotheses", tags=["hypotheses"])


@router.get("")
async def get_hypothesis_tree(
    research_id: str,
    orchestrators: dict = Depends(get_orchestrators),
) -> dict:
    """Full hypothesis tree state."""
    orch = orchestrators.get(research_id)
    if not orch:
        raise HTTPException(status_code=404, detail="Research session not found")

    tree = orch.tree
    if not tree:
        return {"nodes": [], "root_id": None}

    nodes = [n.model_dump(mode="json") for n in tree._nodes.values()]
    return {
        "root_id": tree._root_id,
        "nodes": nodes,
        "total_visits": tree.total_visits,
        "node_count": tree.node_count,
    }


@router.get("/best")
async def get_best_hypothesis(
    research_id: str,
    orchestrators: dict = Depends(get_orchestrators),
) -> dict:
    """Best hypothesis path."""
    orch = orchestrators.get(research_id)
    if not orch:
        raise HTTPException(status_code=404, detail="Research session not found")

    tree = orch.tree
    if not tree:
        raise HTTPException(status_code=404, detail="Hypothesis tree not available")

    best = tree.get_best_hypothesis()
    if not best:
        return {"best": None, "ranking": []}

    ranking = tree.get_ranking()
    return {
        "best": best.model_dump(mode="json"),
        "ranking": [h.model_dump(mode="json") for h in ranking],
    }


@router.get("/{node_id}")
async def get_hypothesis_node(
    research_id: str,
    node_id: str,
    orchestrators: dict = Depends(get_orchestrators),
    kgs: dict = Depends(get_knowledge_graphs),
) -> dict:
    """Single hypothesis node with supporting/contradicting edges."""
    orch = orchestrators.get(research_id)
    if not orch:
        raise HTTPException(status_code=404, detail="Research session not found")

    tree = orch.tree
    if not tree:
        raise HTTPException(status_code=404, detail="Hypothesis tree not available")

    node = tree._nodes.get(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Hypothesis node not found")

    # Resolve edge references from KG
    kg = kgs.get(research_id)
    supporting = []
    contradicting = []

    if kg:
        for eid in node.supporting_edges:
            edge = kg.get_edge(eid)
            if edge:
                supporting.append(edge.model_dump(mode="json"))
        for eid in node.contradicting_edges:
            edge = kg.get_edge(eid)
            if edge:
                contradicting.append(edge.model_dump(mode="json"))

    result = node.model_dump(mode="json")
    result["supporting_edge_details"] = supporting
    result["contradicting_edge_details"] = contradicting
    return result
