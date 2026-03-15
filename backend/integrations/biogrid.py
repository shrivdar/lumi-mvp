"""BioGRID API client — protein-protein interactions."""

from __future__ import annotations

import os
from typing import Any

from core.constants import CACHE_TTL_BIOGRID, RATE_LIMIT_BIOGRID
from integrations.base_tool import BaseTool

BIOGRID_BASE = "https://webservice.thebiogrid.org/interactions"


class BioGRIDTool(BaseTool):
    tool_id = "biogrid"
    name = "biogrid_search"
    description = "Search BioGRID for curated protein-protein, genetic, and chemical interactions."
    category = "network"
    rate_limit = RATE_LIMIT_BIOGRID
    cache_ttl = CACHE_TTL_BIOGRID

    def _get_access_key(self) -> str:
        return os.environ.get("BIOGRID_ACCESS_KEY", "")

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "search")
        if action == "search":
            return await self._search(
                query=kwargs["query"],
                organism=kwargs.get("organism", 9606),
                max_results=kwargs.get("max_results", 100),
            )
        elif action == "interactions":
            return await self._get_interactions(
                gene_list=kwargs["gene_list"],
                organism=kwargs.get("organism", 9606),
                evidence=kwargs.get("evidence", ""),
                max_results=kwargs.get("max_results", 100),
            )
        raise ValueError(f"Unknown BioGRID action: {action}")

    async def _search(self, query: str, organism: int = 9606, max_results: int = 100) -> dict[str, Any]:
        params: dict[str, Any] = {
            "searchNames": "true",
            "geneList": query,
            "organism": organism,
            "format": "json",
            "start": 0,
            "max": min(max_results, 500),
            "includeInteractors": "true",
        }
        access_key = self._get_access_key()
        if access_key:
            params["accessKey"] = access_key

        resp = await self._http.get(BIOGRID_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()

        interactions = [self._normalize_interaction(v) for v in data.values()] if isinstance(data, dict) else []
        return {"interactions": interactions, "count": len(interactions), "query": query}

    async def _get_interactions(
        self, gene_list: list[str], organism: int = 9606,
        evidence: str = "", max_results: int = 100,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "geneList": "|".join(gene_list),
            "organism": organism,
            "format": "json",
            "start": 0,
            "max": min(max_results, 500),
            "includeInteractors": "true",
        }
        if evidence:
            params["evidenceList"] = evidence
        access_key = self._get_access_key()
        if access_key:
            params["accessKey"] = access_key

        resp = await self._http.get(BIOGRID_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()

        interactions = [self._normalize_interaction(v) for v in data.values()] if isinstance(data, dict) else []
        return {"interactions": interactions, "count": len(interactions)}

    @staticmethod
    def _normalize_interaction(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "interaction_id": str(item.get("BIOGRID_INTERACTION_ID", "")),
            "gene_a": item.get("OFFICIAL_SYMBOL_A", ""),
            "gene_b": item.get("OFFICIAL_SYMBOL_B", ""),
            "entrez_a": str(item.get("ENTREZ_GENE_A", "")),
            "entrez_b": str(item.get("ENTREZ_GENE_B", "")),
            "experimental_system": item.get("EXPERIMENTAL_SYSTEM", ""),
            "system_type": item.get("EXPERIMENTAL_SYSTEM_TYPE", ""),
            "organism_a": item.get("ORGANISM_A_ID"),
            "organism_b": item.get("ORGANISM_B_ID"),
            "throughput": item.get("THROUGHPUT", ""),
            "pubmed_id": str(item.get("PUBMED_ID", "")),
        }
