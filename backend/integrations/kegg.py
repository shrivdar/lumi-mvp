"""KEGG API client — pathway and compound search, pathway detail retrieval."""

from __future__ import annotations

from typing import Any

from core.constants import CACHE_TTL_KEGG, RATE_LIMIT_KEGG
from integrations.base_tool import BaseTool

KEGG_BASE = "https://rest.kegg.jp"


class KEGGTool(BaseTool):
    tool_id = "kegg"
    name = "kegg_search"
    description = "Search KEGG for metabolic/signaling pathways, compounds, and enzymes with cross-references."
    category = "pathway"
    rate_limit = RATE_LIMIT_KEGG
    cache_ttl = CACHE_TTL_KEGG

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "search")
        if action == "search":
            return await self._search(
                database=kwargs.get("database", "pathway"),
                query=kwargs["query"],
            )
        elif action == "get":
            return await self._get_entry(entry_id=kwargs["entry_id"])
        elif action == "pathway_genes":
            return await self._pathway_genes(pathway_id=kwargs["pathway_id"])
        elif action == "find_compound":
            return await self._find_compound(query=kwargs["query"])
        raise ValueError(f"Unknown KEGG action: {action}")

    async def _search(self, database: str, query: str) -> dict[str, Any]:
        resp = await self._http.get(f"{KEGG_BASE}/find/{database}/{query}")
        resp.raise_for_status()
        entries = self._parse_tab(resp.text)
        return {"results": entries, "database": database, "query": query, "count": len(entries)}

    async def _get_entry(self, entry_id: str) -> dict[str, Any]:
        resp = await self._http.get(f"{KEGG_BASE}/get/{entry_id}")
        resp.raise_for_status()
        return {"entry_id": entry_id, "data": self._parse_flat(resp.text)}

    async def _pathway_genes(self, pathway_id: str) -> dict[str, Any]:
        resp = await self._http.get(f"{KEGG_BASE}/link/genes/{pathway_id}")
        resp.raise_for_status()
        gene_links = self._parse_tab(resp.text)
        return {"pathway_id": pathway_id, "genes": gene_links, "count": len(gene_links)}

    async def _find_compound(self, query: str) -> dict[str, Any]:
        resp = await self._http.get(f"{KEGG_BASE}/find/compound/{query}")
        resp.raise_for_status()
        entries = self._parse_tab(resp.text)
        return {"results": entries, "query": query, "count": len(entries)}

    @staticmethod
    def _parse_tab(text: str) -> list[dict[str, str]]:
        """Parse KEGG tab-delimited output into list of dicts."""
        results: list[dict[str, str]] = []
        for line in text.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            entry: dict[str, str] = {"id": parts[0].strip()}
            if len(parts) > 1:
                entry["description"] = parts[1].strip()
            results.append(entry)
        return results

    @staticmethod
    def _parse_flat(text: str) -> dict[str, Any]:
        """Parse KEGG flat-file format into structured dict."""
        data: dict[str, Any] = {}
        current_key = ""
        for line in text.split("\n"):
            if line.startswith("///"):
                break
            if line and not line[0].isspace():
                parts = line.split(None, 1)
                current_key = parts[0] if parts else ""
                value = parts[1].strip() if len(parts) > 1 else ""
                if current_key in data:
                    if isinstance(data[current_key], list):
                        data[current_key].append(value)
                    else:
                        data[current_key] = [data[current_key], value]
                else:
                    data[current_key] = value
            elif line.strip() and current_key:
                val = line.strip()
                if isinstance(data.get(current_key), list):
                    data[current_key].append(val)
                elif current_key in data:
                    data[current_key] = [data[current_key], val]
        return data
