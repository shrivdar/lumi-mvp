"""MyGene.info API client — gene annotation and query service."""

from __future__ import annotations

from typing import Any

from core.constants import CACHE_TTL_MYGENE, RATE_LIMIT_MYGENE
from integrations.base_tool import BaseTool

MYGENE_BASE = "https://mygene.info/v3"


class MyGeneTool(BaseTool):
    tool_id = "mygene"
    name = "mygene_search"
    description = (
        "Search MyGene.info for gene annotations — symbols, aliases, coordinates, GO terms, pathways."
    )
    category = "genomics"
    rate_limit = RATE_LIMIT_MYGENE
    cache_ttl = CACHE_TTL_MYGENE

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "search")
        if action == "search":
            return await self._search(query=kwargs["query"], species=kwargs.get("species", "human"))
        elif action == "gene":
            return await self._get_gene(gene_id=kwargs["gene_id"])
        elif action == "batch":
            return await self._batch_query(gene_ids=kwargs["gene_ids"])
        raise ValueError(f"Unknown MyGene action: {action}")

    async def _search(self, query: str, species: str = "human") -> dict[str, Any]:
        resp = await self._http.get(
            f"{MYGENE_BASE}/query",
            params={
                "q": query,
                "species": species,
                "fields": (
                    "symbol,name,alias,entrezgene,ensembl.gene,uniprot.Swiss-Prot,"
                    "go,pathway.kegg,genomic_pos,type_of_gene,summary"
                ),
                "size": 25,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        hits = [self._normalize(h) for h in data.get("hits", [])]
        return {"genes": hits, "total": data.get("total", len(hits)), "query": query}

    async def _get_gene(self, gene_id: str) -> dict[str, Any]:
        resp = await self._http.get(
            f"{MYGENE_BASE}/gene/{gene_id}",
            params={
                "fields": (
                    "symbol,name,alias,entrezgene,ensembl.gene,uniprot.Swiss-Prot,"
                    "go,pathway.kegg,genomic_pos,type_of_gene,summary,interpro,homologene"
                ),
            },
        )
        resp.raise_for_status()
        return {"gene": self._normalize(resp.json())}

    async def _batch_query(self, gene_ids: list[str]) -> dict[str, Any]:
        resp = await self._http.post(
            f"{MYGENE_BASE}/gene",
            json={
                "ids": gene_ids,
                "fields": (
                    "symbol,name,alias,entrezgene,ensembl.gene,uniprot.Swiss-Prot,"
                    "go,pathway.kegg,type_of_gene,summary"
                ),
            },
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            genes = [self._normalize(g) for g in data]
        else:
            genes = [self._normalize(data)]
        return {"genes": genes, "count": len(genes)}

    @staticmethod
    def _normalize(hit: dict[str, Any]) -> dict[str, Any]:
        ensembl = hit.get("ensembl", {})
        if isinstance(ensembl, list):
            ensembl = ensembl[0] if ensembl else {}
        uniprot = hit.get("uniprot", {})
        swiss_prot = uniprot.get("Swiss-Prot", "")
        if isinstance(swiss_prot, list):
            swiss_prot = swiss_prot[0] if swiss_prot else ""

        genomic_pos = hit.get("genomic_pos", {})
        if isinstance(genomic_pos, list):
            genomic_pos = genomic_pos[0] if genomic_pos else {}

        # GO terms
        go_data = hit.get("go", {})
        go_terms: list[dict[str, str]] = []
        for category in ("BP", "MF", "CC"):
            entries = go_data.get(category, [])
            if isinstance(entries, dict):
                entries = [entries]
            for e in entries:
                if isinstance(e, dict):
                    go_terms.append({
                        "id": e.get("id", ""),
                        "term": e.get("term", ""),
                        "category": category,
                    })

        # KEGG pathways
        kegg = hit.get("pathway", {}).get("kegg", [])
        if isinstance(kegg, dict):
            kegg = [kegg]
        pathways = [{"id": p.get("id", ""), "name": p.get("name", "")} for p in kegg if isinstance(p, dict)]

        return {
            "gene_id": str(hit.get("_id", hit.get("entrezgene", ""))),
            "symbol": hit.get("symbol", ""),
            "name": hit.get("name", ""),
            "aliases": hit.get("alias", []) if isinstance(hit.get("alias"), list) else [hit.get("alias", "")],
            "entrez_id": str(hit.get("entrezgene", "")),
            "ensembl_id": ensembl.get("gene", ""),
            "uniprot_id": swiss_prot,
            "type_of_gene": hit.get("type_of_gene", ""),
            "summary": hit.get("summary", ""),
            "genomic_position": {
                "chr": str(genomic_pos.get("chr", "")),
                "start": genomic_pos.get("start"),
                "end": genomic_pos.get("end"),
                "strand": genomic_pos.get("strand"),
            },
            "go_terms": go_terms,
            "kegg_pathways": pathways,
        }
