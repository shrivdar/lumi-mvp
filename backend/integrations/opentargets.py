"""Open Targets Platform API client — target-disease associations and evidence."""

from __future__ import annotations

from typing import Any

from core.constants import CACHE_TTL_OPENTARGETS, RATE_LIMIT_OPENTARGETS
from integrations.base_tool import BaseTool

OT_BASE = "https://api.platform.opentargets.org/api/v4"


class OpenTargetsTool(BaseTool):
    tool_id = "opentargets"
    name = "opentargets_search"
    description = "Query Open Targets for target-disease associations, evidence, and tractability."
    category = "drug"
    rate_limit = RATE_LIMIT_OPENTARGETS
    cache_ttl = CACHE_TTL_OPENTARGETS

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "associations")
        if action == "associations":
            return await self._get_associations(
                target_id=kwargs.get("target_id", ""),
                disease_id=kwargs.get("disease_id", ""),
                size=kwargs.get("size", 25),
            )
        elif action == "target":
            return await self._get_target(target_id=kwargs["target_id"])
        elif action == "disease":
            return await self._get_disease(disease_id=kwargs["disease_id"])
        elif action == "search":
            return await self._search(query=kwargs["query"], size=kwargs.get("size", 25))
        raise ValueError(f"Unknown Open Targets action: {action}")

    async def _get_associations(
        self, target_id: str = "", disease_id: str = "", size: int = 25,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"size": min(size, 100)}
        if target_id and disease_id:
            resp = await self._http.get(
                f"{OT_BASE}/association/filter",
                params={"target": target_id, "disease": disease_id, **params},
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "associations": [self._normalize_assoc(a) for a in data.get("data", [])],
                "count": data.get("total", 0),
                "target_id": target_id,
                "disease_id": disease_id,
            }
        elif disease_id:
            resp = await self._http.get(
                f"{OT_BASE}/association/filter",
                params={"disease": disease_id, **params},
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "associations": [self._normalize_assoc(a) for a in data.get("data", [])],
                "count": data.get("total", 0),
                "disease_id": disease_id,
            }
        elif target_id:
            resp = await self._http.get(
                f"{OT_BASE}/association/filter",
                params={"target": target_id, **params},
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "associations": [self._normalize_assoc(a) for a in data.get("data", [])],
                "count": data.get("total", 0),
                "target_id": target_id,
            }
        raise ValueError("Must provide target_id and/or disease_id for associations")

    async def _get_target(self, target_id: str) -> dict[str, Any]:
        resp = await self._http.get(f"{OT_BASE}/target/{target_id}")
        resp.raise_for_status()
        data = resp.json()
        return {
            "target": {
                "id": data.get("id", ""),
                "approved_symbol": data.get("approvedSymbol", ""),
                "approved_name": data.get("approvedName", ""),
                "biotype": data.get("biotype", ""),
                "tractability": data.get("tractability", {}),
            },
        }

    async def _get_disease(self, disease_id: str) -> dict[str, Any]:
        resp = await self._http.get(f"{OT_BASE}/disease/{disease_id}")
        resp.raise_for_status()
        data = resp.json()
        return {
            "disease": {
                "id": data.get("id", ""),
                "name": data.get("name", ""),
                "description": data.get("description", ""),
                "therapeutic_areas": [ta.get("name", "") for ta in data.get("therapeuticAreas", [])],
            },
        }

    async def _search(self, query: str, size: int = 25) -> dict[str, Any]:
        resp = await self._http.get(
            f"{OT_BASE}/search", params={"q": query, "size": min(size, 100)},
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for hit in data.get("data", []):
            results.append({
                "id": hit.get("id", ""),
                "name": hit.get("name", ""),
                "entity": hit.get("entity", ""),
                "description": hit.get("description", ""),
                "score": hit.get("score"),
            })
        return {"results": results, "count": data.get("total", 0), "query": query}

    @staticmethod
    def _normalize_assoc(assoc: dict[str, Any]) -> dict[str, Any]:
        target = assoc.get("target", {})
        disease = assoc.get("disease", {})
        return {
            "target_id": target.get("id", assoc.get("targetId", "")),
            "target_symbol": target.get("approvedSymbol", ""),
            "disease_id": disease.get("id", assoc.get("diseaseId", "")),
            "disease_name": disease.get("name", ""),
            "score": assoc.get("score", assoc.get("associationScore", {}).get("overall", 0)),
            "datatype_scores": assoc.get("datatypeScores", {}),
        }
