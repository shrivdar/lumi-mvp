"""Human Phenotype Ontology (HPO) API client — phenotype terms and gene-phenotype links."""

from __future__ import annotations

from typing import Any

from core.constants import CACHE_TTL_HPO, RATE_LIMIT_HPO
from integrations.base_tool import BaseTool

HPO_BASE = "https://ontology.jax.org/api/hp"


class HPOTool(BaseTool):
    tool_id = "hpo"
    name = "hpo_search"
    description = "Search Human Phenotype Ontology for phenotype terms, gene-phenotype links, and disease annotations."
    category = "ontology"
    rate_limit = RATE_LIMIT_HPO
    cache_ttl = CACHE_TTL_HPO

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "search")
        if action == "search":
            return await self._search(query=kwargs["query"], max_results=kwargs.get("max_results", 25))
        elif action == "term":
            return await self._get_term(term_id=kwargs["term_id"])
        elif action == "genes":
            return await self._get_genes(term_id=kwargs["term_id"])
        elif action == "diseases":
            return await self._get_diseases(term_id=kwargs["term_id"])
        raise ValueError(f"Unknown HPO action: {action}")

    async def _search(self, query: str, max_results: int = 25) -> dict[str, Any]:
        resp = await self._http.get(
            f"{HPO_BASE}/search",
            params={"q": query, "max": min(max_results, 100)},
        )
        resp.raise_for_status()
        data = resp.json()
        terms = []
        for item in data.get("terms", []):
            terms.append({
                "id": item.get("id", ""),
                "name": item.get("name", ""),
                "definition": item.get("definition", ""),
                "synonyms": item.get("synonyms", []),
            })
        return {"terms": terms, "count": len(terms), "query": query}

    async def _get_term(self, term_id: str) -> dict[str, Any]:
        resp = await self._http.get(f"{HPO_BASE}/terms/{term_id}")
        resp.raise_for_status()
        data = resp.json()
        return {
            "term": {
                "id": data.get("id", ""),
                "name": data.get("name", ""),
                "definition": data.get("definition", ""),
                "synonyms": data.get("synonyms", []),
                "parents": [
                    {"id": p.get("id", ""), "name": p.get("name", "")}
                    for p in data.get("parents", [])
                ],
                "children": [
                    {"id": c.get("id", ""), "name": c.get("name", "")}
                    for c in data.get("children", [])
                ],
            },
        }

    async def _get_genes(self, term_id: str) -> dict[str, Any]:
        resp = await self._http.get(f"{HPO_BASE}/terms/{term_id}/genes")
        resp.raise_for_status()
        data = resp.json()
        genes = []
        for item in data.get("genes", []):
            genes.append({
                "gene_id": str(item.get("entrezGeneId", "")),
                "gene_symbol": item.get("entrezGeneSymbol", ""),
            })
        return {"genes": genes, "count": len(genes), "term_id": term_id}

    async def _get_diseases(self, term_id: str) -> dict[str, Any]:
        resp = await self._http.get(f"{HPO_BASE}/terms/{term_id}/diseases")
        resp.raise_for_status()
        data = resp.json()
        diseases = []
        for item in data.get("diseases", []):
            diseases.append({
                "disease_id": item.get("diseaseId", ""),
                "disease_name": item.get("diseaseName", ""),
                "db_name": item.get("dbName", ""),
            })
        return {"diseases": diseases, "count": len(diseases), "term_id": term_id}
