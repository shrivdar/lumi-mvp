"""Knowledge graph query endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_knowledge_graphs
from world_model.knowledge_graph import InMemoryKnowledgeGraph

router = APIRouter(prefix="/research/{research_id}/graph", tags=["graph"])


def _get_kg(research_id: str, kgs: dict = Depends(get_knowledge_graphs)) -> InMemoryKnowledgeGraph:
    kg = kgs.get(research_id)
    if not kg:
        raise HTTPException(status_code=404, detail="Knowledge graph not found")
    return kg


@router.get("")
async def get_full_graph(
    research_id: str,
    format: str = Query("cytoscape", pattern="^(cytoscape|json|summary)$"),
    kgs: dict = Depends(get_knowledge_graphs),
) -> dict:
    """Full KG in cytoscape, json, or summary format."""
    kg = _get_kg(research_id, kgs)
    if format == "cytoscape":
        return {"format": "cytoscape", "data": kg.to_cytoscape()}
    elif format == "json":
        return {"format": "json", "data": kg.to_json()}
    else:
        return {"format": "summary", "data": kg.to_markdown_summary()}


@router.get("/subgraph")
async def get_subgraph(
    research_id: str,
    center: str = Query(..., description="Center node ID"),
    hops: int = Query(2, ge=1, le=5),
    kgs: dict = Depends(get_knowledge_graphs),
) -> dict:
    """Subgraph around a center node."""
    kg = _get_kg(research_id, kgs)
    node = kg.get_node(center)
    if not node:
        raise HTTPException(status_code=404, detail="Center node not found")
    subgraph = kg.get_subgraph(center, hops=hops)
    return {"center": center, "hops": hops, "subgraph": subgraph}


@router.get("/nodes")
async def get_nodes(
    research_id: str,
    type: str | None = None,
    offset: int = 0,
    limit: int = 50,
    kgs: dict = Depends(get_knowledge_graphs),
) -> dict:
    """List nodes, optionally filtered by type."""
    kg = _get_kg(research_id, kgs)
    graph_data = kg.to_json()
    nodes = graph_data.get("nodes", [])

    if type:
        nodes = [n for n in nodes if n.get("type") == type]

    total = len(nodes)
    nodes = nodes[offset : offset + limit]

    return {"nodes": nodes, "total": total, "offset": offset, "limit": limit}


@router.get("/edges")
async def get_edges(
    research_id: str,
    source: str | None = None,
    target: str | None = None,
    relation: str | None = None,
    offset: int = 0,
    limit: int = 50,
    kgs: dict = Depends(get_knowledge_graphs),
) -> dict:
    """List edges, optionally filtered by source/target/relation."""
    kg = _get_kg(research_id, kgs)
    graph_data = kg.to_json()
    edges = graph_data.get("edges", [])

    if source:
        edges = [e for e in edges if e.get("source_id") == source]
    if target:
        edges = [e for e in edges if e.get("target_id") == target]
    if relation:
        edges = [e for e in edges if e.get("relation") == relation]

    total = len(edges)
    edges = edges[offset : offset + limit]

    return {"edges": edges, "total": total, "offset": offset, "limit": limit}


@router.get("/contradictions")
async def get_contradictions(
    research_id: str,
    kgs: dict = Depends(get_knowledge_graphs),
) -> dict:
    """List contradicting edge pairs."""
    kg = _get_kg(research_id, kgs)
    graph_data = kg.to_json()
    edges = graph_data.get("edges", [])

    contradiction_pairs = []
    seen: set[str] = set()

    for edge in edges:
        if edge.get("is_contradiction") and edge.get("contradicted_by"):
            for contra_id in edge["contradicted_by"]:
                pair_key = tuple(sorted([edge["id"], contra_id]))
                if pair_key not in seen:
                    seen.add(pair_key)
                    contra_edge = next((e for e in edges if e.get("id") == contra_id), None)
                    if contra_edge:
                        contradiction_pairs.append({"edge_a": edge, "edge_b": contra_edge})

    return {"contradictions": contradiction_pairs, "count": len(contradiction_pairs)}


@router.get("/stats")
async def get_graph_stats(
    research_id: str,
    kgs: dict = Depends(get_knowledge_graphs),
) -> dict:
    """Graph statistics."""
    kg = _get_kg(research_id, kgs)
    graph_data = kg.to_json()
    nodes = graph_data.get("nodes", [])

    type_distribution: dict[str, int] = {}
    for n in nodes:
        t = n.get("type", "unknown")
        type_distribution[t] = type_distribution.get(t, 0) + 1

    return {
        "node_count": kg.node_count(),
        "edge_count": kg.edge_count(),
        "avg_confidence": kg.avg_confidence(),
        "type_distribution": type_distribution,
    }
