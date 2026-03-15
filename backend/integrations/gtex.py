"""GTEx Portal API client — tissue-specific gene expression and eQTL data."""

from __future__ import annotations

from typing import Any

from core.constants import CACHE_TTL_GTEX, RATE_LIMIT_GTEX
from integrations.base_tool import BaseTool

GTEX_BASE = "https://gtexportal.org/api/v2"


class GTExTool(BaseTool):
    tool_id = "gtex"
    name = "gtex_search"
    description = "Query GTEx for tissue-specific gene expression and eQTL data."
    category = "expression"
    rate_limit = RATE_LIMIT_GTEX
    cache_ttl = CACHE_TTL_GTEX

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "expression")
        if action == "expression":
            return await self._get_expression(
                gene_id=kwargs["gene_id"],
            )
        elif action == "eqtl":
            return await self._get_eqtl(
                gene_id=kwargs["gene_id"],
                tissue=kwargs.get("tissue", ""),
            )
        elif action == "gene_search":
            return await self._search_gene(query=kwargs["query"])
        raise ValueError(f"Unknown GTEx action: {action}")

    async def _get_expression(self, gene_id: str) -> dict[str, Any]:
        resp = await self._http.get(
            f"{GTEX_BASE}/expression/medianGeneExpression",
            params={"gencodeId": gene_id, "datasetId": "gtex_v8"},
        )
        resp.raise_for_status()
        data = resp.json()
        tissues = []
        for item in data.get("data", []):
            tissues.append({
                "tissue_id": item.get("tissueSiteDetailId", ""),
                "tissue_name": item.get("tissueSiteDetail", ""),
                "median_tpm": item.get("median", 0),
                "unit": "TPM",
            })
        # Sort by expression level descending
        tissues.sort(key=lambda t: t["median_tpm"], reverse=True)
        return {"gene_id": gene_id, "tissues": tissues, "count": len(tissues)}

    async def _get_eqtl(self, gene_id: str, tissue: str = "") -> dict[str, Any]:
        params: dict[str, Any] = {"gencodeId": gene_id, "datasetId": "gtex_v8"}
        if tissue:
            params["tissueSiteDetailId"] = tissue
        resp = await self._http.get(
            f"{GTEX_BASE}/association/singleTissueEqtl",
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()
        eqtls = []
        for item in data.get("data", []):
            eqtls.append({
                "variant_id": item.get("variantId", ""),
                "tissue_id": item.get("tissueSiteDetailId", ""),
                "p_value": item.get("pValue"),
                "nes": item.get("nes"),  # normalized effect size
                "maf": item.get("maf"),
            })
        return {"gene_id": gene_id, "eqtls": eqtls, "count": len(eqtls)}

    async def _search_gene(self, query: str) -> dict[str, Any]:
        resp = await self._http.get(
            f"{GTEX_BASE}/reference/gene",
            params={"geneId": query, "datasetId": "gtex_v8"},
        )
        resp.raise_for_status()
        data = resp.json()
        genes = []
        for item in data.get("data", []):
            genes.append({
                "gencode_id": item.get("gencodeId", ""),
                "gene_symbol": item.get("geneSymbol", ""),
                "description": item.get("description", ""),
                "gene_type": item.get("geneType", ""),
                "chromosome": item.get("chromosome", ""),
                "start": item.get("start"),
                "end": item.get("end"),
            })
        return {"genes": genes, "count": len(genes), "query": query}
