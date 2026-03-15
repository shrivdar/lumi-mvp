"""STRING database API client — protein-protein interactions with confidence scores.

Replaces the auto-generated dynamic/string_db.py with a proper BaseTool implementation.
"""

from __future__ import annotations

from typing import Any

from core.constants import CACHE_TTL_STRING_DB, RATE_LIMIT_STRING_DB
from integrations.base_tool import BaseTool

STRING_BASE = "https://string-db.org/api/json"


class StringDBTool(BaseTool):
    tool_id = "string_db"
    name = "string_db_search"
    description = "Query STRING for known and predicted protein-protein interactions with confidence scores."
    category = "network"
    rate_limit = RATE_LIMIT_STRING_DB
    cache_ttl = CACHE_TTL_STRING_DB

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "network")
        if action == "network":
            return await self._get_network(
                query=kwargs["query"],
                species=kwargs.get("species", 9606),
                limit=kwargs.get("limit", 10),
                required_score=kwargs.get("required_score", 400),
            )
        elif action == "interaction_partners":
            return await self._get_interaction_partners(
                query=kwargs["query"],
                species=kwargs.get("species", 9606),
                limit=kwargs.get("limit", 25),
                required_score=kwargs.get("required_score", 400),
            )
        elif action == "enrichment":
            return await self._get_enrichment(
                identifiers=kwargs["identifiers"],
                species=kwargs.get("species", 9606),
            )
        raise ValueError(f"Unknown STRING action: {action}")

    async def _get_network(
        self, query: str, species: int = 9606, limit: int = 10, required_score: int = 400,
    ) -> dict[str, Any]:
        resp = await self._http.get(
            f"{STRING_BASE}/network",
            params={
                "identifiers": query,
                "species": species,
                "limit": min(limit, 100),
                "required_score": required_score,
                "caller_identity": "yohas",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        interactions = [self._normalize_interaction(item) for item in data]
        return {"query": query, "interactions": interactions, "count": len(interactions)}

    async def _get_interaction_partners(
        self, query: str, species: int = 9606, limit: int = 25, required_score: int = 400,
    ) -> dict[str, Any]:
        resp = await self._http.get(
            f"{STRING_BASE}/interaction_partners",
            params={
                "identifiers": query,
                "species": species,
                "limit": min(limit, 100),
                "required_score": required_score,
                "caller_identity": "yohas",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        interactions = [self._normalize_interaction(item) for item in data]
        return {"query": query, "interactions": interactions, "count": len(interactions)}

    async def _get_enrichment(
        self, identifiers: list[str], species: int = 9606,
    ) -> dict[str, Any]:
        resp = await self._http.get(
            f"{STRING_BASE}/enrichment",
            params={
                "identifiers": "\r".join(identifiers),
                "species": species,
                "caller_identity": "yohas",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        enrichments = []
        for item in data:
            enrichments.append({
                "category": item.get("category", ""),
                "term": item.get("term", ""),
                "description": item.get("description", ""),
                "p_value": item.get("p_value"),
                "fdr": item.get("fdr"),
                "gene_count": item.get("number_of_genes"),
                "genes": item.get("inputGenes", "").split(",") if item.get("inputGenes") else [],
            })
        return {"enrichments": enrichments, "count": len(enrichments)}

    @staticmethod
    def _normalize_interaction(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "protein_a": item.get("preferredName_A", ""),
            "protein_b": item.get("preferredName_B", ""),
            "string_id_a": item.get("stringId_A", ""),
            "string_id_b": item.get("stringId_B", ""),
            "score": item.get("score", 0),
            "nscore": item.get("nscore", 0),
            "fscore": item.get("fscore", 0),
            "pscore": item.get("pscore", 0),
            "ascore": item.get("ascore", 0),
            "escore": item.get("escore", 0),
            "dscore": item.get("dscore", 0),
            "tscore": item.get("tscore", 0),
        }
