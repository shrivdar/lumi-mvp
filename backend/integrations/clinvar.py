"""ClinVar API client — variant clinical significance via NCBI E-utils + VCV API."""

from __future__ import annotations

from typing import Any

from core.constants import CACHE_TTL_CLINVAR, RATE_LIMIT_CLINVAR
from integrations.base_tool import BaseTool

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CLINVAR_VCV_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class ClinVarTool(BaseTool):
    tool_id = "clinvar"
    name = "clinvar_search"
    description = "Search ClinVar for clinical significance of human genomic variants."
    category = "variant"
    rate_limit = RATE_LIMIT_CLINVAR
    cache_ttl = CACHE_TTL_CLINVAR

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "search")
        if action == "search":
            return await self._search(query=kwargs["query"], max_results=kwargs.get("max_results", 20))
        elif action == "fetch":
            return await self._fetch(variant_id=kwargs["variant_id"])
        raise ValueError(f"Unknown ClinVar action: {action}")

    async def _search(self, query: str, max_results: int = 20) -> dict[str, Any]:
        # Step 1: esearch to get IDs
        resp = await self._http.get(
            f"{EUTILS_BASE}/esearch.fcgi",
            params={
                "db": "clinvar",
                "term": query,
                "retmax": min(max_results, 100),
                "retmode": "json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return {"variants": [], "count": 0, "query": query}

        # Step 2: esummary to get variant details
        resp = await self._http.get(
            f"{EUTILS_BASE}/esummary.fcgi",
            params={
                "db": "clinvar",
                "id": ",".join(id_list),
                "retmode": "json",
            },
        )
        resp.raise_for_status()
        summary = resp.json()
        result_data = summary.get("result", {})
        uid_list = result_data.get("uids", [])

        variants = []
        for uid in uid_list:
            entry = result_data.get(uid, {})
            variants.append(self._normalize_variant(uid, entry))

        return {"variants": variants, "count": len(variants), "query": query}

    async def _fetch(self, variant_id: str) -> dict[str, Any]:
        resp = await self._http.get(
            f"{EUTILS_BASE}/esummary.fcgi",
            params={
                "db": "clinvar",
                "id": variant_id,
                "retmode": "json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        result_data = data.get("result", {})
        entry = result_data.get(variant_id, {})
        return {"variant": self._normalize_variant(variant_id, entry)}

    @staticmethod
    def _normalize_variant(uid: str, entry: dict[str, Any]) -> dict[str, Any]:
        # Extract clinical significance from the nested structure
        clinical_sig = entry.get("clinical_significance", {})
        if isinstance(clinical_sig, dict):
            significance = clinical_sig.get("description", "")
            review_status = clinical_sig.get("review_status", "")
            last_evaluated = clinical_sig.get("last_evaluated", "")
        else:
            significance = str(clinical_sig) if clinical_sig else ""
            review_status = ""
            last_evaluated = ""

        genes = []
        for gene in entry.get("genes", []):
            genes.append({
                "symbol": gene.get("symbol", ""),
                "gene_id": str(gene.get("geneid", "")),
            })

        return {
            "uid": uid,
            "title": entry.get("title", ""),
            "variant_type": entry.get("obj_type", ""),
            "clinical_significance": significance,
            "review_status": review_status,
            "last_evaluated": last_evaluated,
            "genes": genes,
            "accession": entry.get("accession", ""),
            "conditions": [
                t.get("trait_name", "") for t in entry.get("trait_set", [])
            ] if isinstance(entry.get("trait_set"), list) else [],
        }
