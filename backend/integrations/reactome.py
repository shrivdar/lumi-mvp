"""Reactome API client — pathway search, detail, and analysis."""

from __future__ import annotations

from typing import Any

from core.constants import CACHE_TTL_REACTOME, RATE_LIMIT_REACTOME
from integrations.base_tool import BaseTool

REACTOME_BASE = "https://reactome.org"
CONTENT_BASE = f"{REACTOME_BASE}/ContentService"


class ReactomeTool(BaseTool):
    tool_id = "reactome"
    name = "reactome_search"
    description = "Search Reactome for biological pathways, reactions, and molecular events with species-specific data."
    category = "pathway"
    rate_limit = RATE_LIMIT_REACTOME
    cache_ttl = CACHE_TTL_REACTOME

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "search")
        if action == "search":
            return await self._search(query=kwargs["query"], species=kwargs.get("species", "Homo sapiens"))
        elif action == "pathway":
            return await self._get_pathway(pathway_id=kwargs["pathway_id"])
        elif action == "participants":
            return await self._get_participants(pathway_id=kwargs["pathway_id"])
        elif action == "analyze":
            return await self._analyze_genes(genes=kwargs["genes"])
        raise ValueError(f"Unknown Reactome action: {action}")

    async def _search(self, query: str, species: str = "Homo sapiens") -> dict[str, Any]:
        resp = await self._http.get(
            f"{CONTENT_BASE}/search/query",
            params={"query": query, "species": species, "types": "Pathway", "cluster": "true"},
        )
        resp.raise_for_status()
        data = resp.json()
        entries = data.get("results", [])
        pathways: list[dict[str, Any]] = []
        for group in entries:
            for entry in group.get("entries", []):
                pathways.append({
                    "stable_id": entry.get("stId", ""),
                    "name": entry.get("name", ""),
                    "species": entry.get("species", []),
                    "summary": entry.get("summation", ""),
                    "type": entry.get("exactType", ""),
                })
        return {"pathways": pathways, "count": len(pathways), "query": query}

    async def _get_pathway(self, pathway_id: str) -> dict[str, Any]:
        resp = await self._http.get(f"{CONTENT_BASE}/data/query/{pathway_id}")
        resp.raise_for_status()
        data = resp.json()
        return {
            "pathway_id": pathway_id,
            "name": data.get("displayName", ""),
            "stable_id": data.get("stId", ""),
            "species": data.get("speciesName", ""),
            "summary": (data.get("summation", [{}])[0].get("text", "") if data.get("summation") else ""),
            "has_diagram": data.get("hasDiagram", False),
            "is_in_disease": data.get("isInDisease", False),
            "sub_events": [
                {"stable_id": e.get("stId", ""), "name": e.get("displayName", "")}
                for e in data.get("hasEvent", [])
            ],
        }

    async def _get_participants(self, pathway_id: str) -> dict[str, Any]:
        resp = await self._http.get(f"{CONTENT_BASE}/data/participants/{pathway_id}")
        resp.raise_for_status()
        data = resp.json()
        participants: list[dict[str, str]] = []
        for entity in data:
            participants.append({
                "stable_id": entity.get("stId", ""),
                "name": entity.get("displayName", ""),
                "schema_class": entity.get("schemaClass", ""),
            })
        return {"pathway_id": pathway_id, "participants": participants, "count": len(participants)}

    async def _analyze_genes(self, genes: list[str]) -> dict[str, Any]:
        payload = "\n".join(genes)
        resp = await self._http.post(
            f"{CONTENT_BASE}/identifiers/projection",
            content=payload,
            headers={"Content-Type": "text/plain"},
        )
        resp.raise_for_status()
        data = resp.json()
        enriched = []
        for pathway in data.get("pathways", []):
            enriched.append({
                "stable_id": pathway.get("stId", ""),
                "name": pathway.get("name", ""),
                "p_value": pathway.get("entities", {}).get("pValue"),
                "fdr": pathway.get("entities", {}).get("fdr"),
                "found": pathway.get("entities", {}).get("found", 0),
                "total": pathway.get("entities", {}).get("total", 0),
            })
        return {"enriched_pathways": enriched, "input_genes": genes, "count": len(enriched)}
