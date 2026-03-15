"""CZ CELLxGENE API client — single-cell reference data via Discover API."""

from __future__ import annotations

from typing import Any

from core.constants import CACHE_TTL_CELLXGENE, RATE_LIMIT_CELLXGENE
from integrations.base_tool import BaseTool

CELLXGENE_BASE = "https://api.cellxgene.cziscience.com"


class CellxGeneTool(BaseTool):
    tool_id = "cellxgene"
    name = "cellxgene_search"
    description = "Search CZ CELLxGENE for single-cell RNA-seq datasets and cell type annotations."
    category = "expression"
    rate_limit = RATE_LIMIT_CELLXGENE
    cache_ttl = CACHE_TTL_CELLXGENE

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "collections")
        if action == "collections":
            return await self._list_collections()
        elif action == "collection":
            return await self._get_collection(collection_id=kwargs["collection_id"])
        elif action == "datasets":
            return await self._list_datasets(
                organism=kwargs.get("organism", ""),
                tissue=kwargs.get("tissue", ""),
                disease=kwargs.get("disease", ""),
            )
        elif action == "gene_expression":
            return await self._get_gene_expression(
                gene=kwargs["gene"],
                organism=kwargs.get("organism", "Homo sapiens"),
            )
        raise ValueError(f"Unknown CELLxGENE action: {action}")

    async def _list_collections(self) -> dict[str, Any]:
        resp = await self._http.get(f"{CELLXGENE_BASE}/curation/v1/collections")
        resp.raise_for_status()
        data = resp.json()
        collections = []
        for item in data if isinstance(data, list) else data.get("collections", []):
            collections.append({
                "collection_id": item.get("collection_id", ""),
                "name": item.get("name", ""),
                "description": item.get("description", "")[:200],
                "doi": item.get("doi", ""),
                "publisher_metadata": item.get("publisher_metadata", {}),
            })
        return {"collections": collections, "count": len(collections)}

    async def _get_collection(self, collection_id: str) -> dict[str, Any]:
        resp = await self._http.get(f"{CELLXGENE_BASE}/curation/v1/collections/{collection_id}")
        resp.raise_for_status()
        data = resp.json()
        datasets = []
        for ds in data.get("datasets", []):
            datasets.append({
                "dataset_id": ds.get("dataset_id", ""),
                "name": ds.get("name", ""),
                "organism": [o.get("label", "") for o in ds.get("organism", [])],
                "tissue": [t.get("label", "") for t in ds.get("tissue", [])],
                "disease": [d.get("label", "") for d in ds.get("disease", [])],
                "cell_count": ds.get("cell_count"),
                "assay": [a.get("label", "") for a in ds.get("assay", [])],
            })
        return {
            "collection_id": collection_id,
            "name": data.get("name", ""),
            "description": data.get("description", ""),
            "datasets": datasets,
            "count": len(datasets),
        }

    async def _list_datasets(
        self, organism: str = "", tissue: str = "", disease: str = "",
    ) -> dict[str, Any]:
        resp = await self._http.get(f"{CELLXGENE_BASE}/curation/v1/datasets")
        resp.raise_for_status()
        data = resp.json()
        datasets_list = data if isinstance(data, list) else data.get("datasets", [])

        # Client-side filtering
        filtered = []
        for ds in datasets_list:
            if organism:
                org_labels = [o.get("label", "").lower() for o in ds.get("organism", [])]
                if not any(organism.lower() in lab for lab in org_labels):
                    continue
            if tissue:
                tissue_labels = [t.get("label", "").lower() for t in ds.get("tissue", [])]
                if not any(tissue.lower() in lab for lab in tissue_labels):
                    continue
            if disease:
                disease_labels = [d.get("label", "").lower() for d in ds.get("disease", [])]
                if not any(disease.lower() in lab for lab in disease_labels):
                    continue
            filtered.append({
                "dataset_id": ds.get("dataset_id", ""),
                "name": ds.get("name", ""),
                "organism": [o.get("label", "") for o in ds.get("organism", [])],
                "tissue": [t.get("label", "") for t in ds.get("tissue", [])],
                "disease": [d.get("label", "") for d in ds.get("disease", [])],
                "cell_count": ds.get("cell_count"),
                "assay": [a.get("label", "") for a in ds.get("assay", [])],
            })
            if len(filtered) >= 50:
                break

        return {"datasets": filtered, "count": len(filtered)}

    async def _get_gene_expression(self, gene: str, organism: str = "Homo sapiens") -> dict[str, Any]:
        resp = await self._http.post(
            f"{CELLXGENE_BASE}/wmg/v2/query",
            json={
                "filter": {"gene_ontology_term_ids": [], "organism_ontology_term_id": ""},
                "gene_symbols": [gene],
                "compare": "tissue",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        expression = []
        for item in data.get("expression_summary", data.get("data", [])):
            if isinstance(item, dict):
                expression.append({
                    "tissue": item.get("tissue", item.get("label", "")),
                    "mean_expression": item.get("mean", item.get("me")),
                    "cell_count": item.get("n"),
                })
        return {"gene": gene, "expression": expression, "count": len(expression)}
