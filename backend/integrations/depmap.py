"""DepMap Portal API client — CRISPR dependency scores across cancer cell lines."""

from __future__ import annotations

from typing import Any

from core.constants import CACHE_TTL_DEPMAP, RATE_LIMIT_DEPMAP
from integrations.base_tool import BaseTool

DEPMAP_BASE = "https://api.depmap.org/api/v1"


class DepMapTool(BaseTool):
    tool_id = "depmap"
    name = "depmap_search"
    description = "Query DepMap for CRISPR gene dependency scores across cancer cell lines."
    category = "expression"
    rate_limit = RATE_LIMIT_DEPMAP
    cache_ttl = CACHE_TTL_DEPMAP

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "gene_dependency")
        if action == "gene_dependency":
            return await self._gene_dependency(
                gene=kwargs["gene"],
            )
        elif action == "cell_line":
            return await self._cell_line_info(
                cell_line=kwargs["cell_line"],
            )
        elif action == "search":
            return await self._search(query=kwargs["query"])
        raise ValueError(f"Unknown DepMap action: {action}")

    async def _gene_dependency(self, gene: str) -> dict[str, Any]:
        resp = await self._http.get(
            f"{DEPMAP_BASE}/gene_dependency",
            params={"gene": gene},
        )
        resp.raise_for_status()
        data = resp.json()
        dependencies = []
        for item in data.get("data", data) if isinstance(data, dict) else data:
            if isinstance(item, dict):
                dependencies.append({
                    "cell_line": item.get("cell_line_name", item.get("depmap_id", "")),
                    "depmap_id": item.get("depmap_id", ""),
                    "lineage": item.get("lineage", ""),
                    "dependency_score": item.get("dependency", item.get("gene_effect")),
                })
        return {"gene": gene, "dependencies": dependencies, "count": len(dependencies)}

    async def _cell_line_info(self, cell_line: str) -> dict[str, Any]:
        resp = await self._http.get(
            f"{DEPMAP_BASE}/cell_line",
            params={"cell_line": cell_line},
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "cell_line": {
                "depmap_id": data.get("depmap_id", ""),
                "cell_line_name": data.get("cell_line_name", ""),
                "lineage": data.get("lineage", ""),
                "lineage_subtype": data.get("lineage_subtype", ""),
                "primary_disease": data.get("primary_disease", ""),
                "subtype_disease": data.get("subtype_disease", ""),
                "source": data.get("source", ""),
            },
        }

    async def _search(self, query: str) -> dict[str, Any]:
        resp = await self._http.get(
            f"{DEPMAP_BASE}/search",
            params={"q": query},
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("data", data) if isinstance(data, dict) else data:
            if isinstance(item, dict):
                results.append({
                    "id": item.get("id", ""),
                    "name": item.get("name", item.get("label", "")),
                    "type": item.get("type", ""),
                })
        return {"results": results, "count": len(results), "query": query}
