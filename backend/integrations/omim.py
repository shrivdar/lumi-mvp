"""OMIM API client — gene-disease relationships (API key required)."""

from __future__ import annotations

import os
from typing import Any

from core.constants import CACHE_TTL_OMIM, RATE_LIMIT_OMIM
from core.exceptions import ToolError
from integrations.base_tool import BaseTool

OMIM_BASE = "https://api.omim.org/api"


class OMIMTool(BaseTool):
    tool_id = "omim"
    name = "omim_search"
    description = "Search OMIM for gene-disease relationships and phenotype descriptions."
    category = "genomics"
    rate_limit = RATE_LIMIT_OMIM
    cache_ttl = CACHE_TTL_OMIM

    def _get_api_key(self) -> str:
        key = os.environ.get("OMIM_API_KEY", "")
        if not key:
            raise ToolError(
                "OMIM API key not configured — set OMIM_API_KEY env var",
                error_code="MISSING_API_KEY",
            )
        return key

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "search")
        if action == "search":
            return await self._search(query=kwargs["query"], max_results=kwargs.get("max_results", 20))
        elif action == "entry":
            return await self._get_entry(mim_number=kwargs["mim_number"])
        elif action == "gene_map":
            return await self._gene_map(query=kwargs["query"])
        raise ValueError(f"Unknown OMIM action: {action}")

    async def _search(self, query: str, max_results: int = 20) -> dict[str, Any]:
        resp = await self._http.get(
            f"{OMIM_BASE}/entry/search",
            params={
                "search": query,
                "limit": min(max_results, 50),
                "format": "json",
                "apiKey": self._get_api_key(),
            },
        )
        resp.raise_for_status()
        data = resp.json()
        search_response = data.get("omim", {}).get("searchResponse", {})
        entries = []
        for item in search_response.get("entryList", []):
            entry = item.get("entry", {})
            entries.append(self._normalize_entry(entry))
        return {
            "entries": entries,
            "count": search_response.get("totalResults", 0),
            "query": query,
        }

    async def _get_entry(self, mim_number: str) -> dict[str, Any]:
        resp = await self._http.get(
            f"{OMIM_BASE}/entry",
            params={
                "mimNumber": mim_number,
                "include": "text,geneMap,clinicalSynopsis",
                "format": "json",
                "apiKey": self._get_api_key(),
            },
        )
        resp.raise_for_status()
        data = resp.json()
        entry_list = data.get("omim", {}).get("entryList", [])
        if not entry_list:
            return {"entry": None, "mim_number": mim_number}
        entry = entry_list[0].get("entry", {})
        return {"entry": self._normalize_entry(entry)}

    async def _gene_map(self, query: str) -> dict[str, Any]:
        resp = await self._http.get(
            f"{OMIM_BASE}/geneMap/search",
            params={
                "search": query,
                "limit": 25,
                "format": "json",
                "apiKey": self._get_api_key(),
            },
        )
        resp.raise_for_status()
        data = resp.json()
        search_response = data.get("omim", {}).get("searchResponse", {})
        gene_maps = []
        for item in search_response.get("geneMapList", []):
            gm = item.get("geneMap", {})
            phenotypes = []
            for pheno in gm.get("phenotypeMapList", []):
                pm = pheno.get("phenotypeMap", {})
                phenotypes.append({
                    "phenotype": pm.get("phenotype", ""),
                    "mim_number": str(pm.get("phenotypeMimNumber", "")),
                    "inheritance": pm.get("phenotypeInheritance", ""),
                    "mapping_key": pm.get("phenotypeMappingKey"),
                })
            gene_maps.append({
                "mim_number": str(gm.get("mimNumber", "")),
                "gene_symbols": gm.get("geneSymbols", ""),
                "gene_name": gm.get("geneName", ""),
                "cyto_location": gm.get("computedCytoLocation", ""),
                "phenotypes": phenotypes,
            })
        return {
            "gene_maps": gene_maps,
            "count": search_response.get("totalResults", 0),
            "query": query,
        }

    @staticmethod
    def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
        titles = entry.get("titles", {})
        gene_map = entry.get("geneMap", {})
        return {
            "mim_number": str(entry.get("mimNumber", "")),
            "title": titles.get("preferredTitle", ""),
            "alternative_titles": titles.get("alternativeTitles", ""),
            "status": entry.get("status", ""),
            "gene_symbols": gene_map.get("geneSymbols", ""),
            "gene_name": gene_map.get("geneName", ""),
            "cyto_location": gene_map.get("computedCytoLocation", ""),
        }
