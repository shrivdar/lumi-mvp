"""UniProt REST API client — protein search, entry retrieval, sequence/function data."""

from __future__ import annotations

from typing import Any

from core.constants import CACHE_TTL_UNIPROT, RATE_LIMIT_UNIPROT
from integrations.base_tool import BaseTool

UNIPROT_BASE = "https://rest.uniprot.org"


class UniProtTool(BaseTool):
    tool_id = "uniprot"
    name = "uniprot_search"
    description = "Search UniProt for protein entries with sequences, functions, structures, and cross-references."
    category = "protein"
    rate_limit = RATE_LIMIT_UNIPROT
    cache_ttl = CACHE_TTL_UNIPROT

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "search")
        if action == "search":
            return await self._search(query=kwargs["query"], max_results=kwargs.get("max_results", 25))
        elif action == "entry":
            return await self._get_entry(accession=kwargs["accession"])
        elif action == "features":
            return await self._get_features(accession=kwargs["accession"])
        raise ValueError(f"Unknown UniProt action: {action}")

    async def _search(self, query: str, max_results: int = 25) -> dict[str, Any]:
        resp = await self._http.get(
            f"{UNIPROT_BASE}/uniprotkb/search",
            params={
                "query": query,
                "format": "json",
                "size": min(max_results, 100),
                "fields": (
                    "accession,id,protein_name,gene_names,organism_name,length,"
                    "sequence,ft_domain,xref_pdb,go_p,go_f,go_c,cc_function"
                ),
            },
        )
        resp.raise_for_status()
        data = resp.json()
        entries = [self._normalize(r) for r in data.get("results", [])]
        return {"entries": entries, "total_count": len(entries), "query": query}

    async def _get_entry(self, accession: str) -> dict[str, Any]:
        resp = await self._http.get(f"{UNIPROT_BASE}/uniprotkb/{accession}.json")
        resp.raise_for_status()
        return {"entry": self._normalize(resp.json())}

    async def _get_features(self, accession: str) -> dict[str, Any]:
        resp = await self._http.get(f"{UNIPROT_BASE}/uniprotkb/{accession}.json")
        resp.raise_for_status()
        data = resp.json()
        features = data.get("features", [])
        return {
            "accession": accession,
            "features": [
                {
                    "type": f.get("type", ""),
                    "description": f.get("description", ""),
                    "location": {
                        "start": f.get("location", {}).get("start", {}).get("value"),
                        "end": f.get("location", {}).get("end", {}).get("value"),
                    },
                }
                for f in features
            ],
        }

    @staticmethod
    def _normalize(entry: dict[str, Any]) -> dict[str, Any]:
        protein_desc = entry.get("proteinDescription", {})
        rec_name = protein_desc.get("recommendedName", {})
        full_name = rec_name.get("fullName", {}).get("value", "")

        genes = entry.get("genes", [])
        gene_names = [g.get("geneName", {}).get("value", "") for g in genes]

        organism = entry.get("organism", {})
        org_name = organism.get("scientificName", "")

        sequence_data = entry.get("sequence", {})

        # PDB cross-refs
        pdb_ids = []
        for xref in entry.get("uniProtKBCrossReferences", []):
            if xref.get("database") == "PDB":
                pdb_ids.append(xref.get("id", ""))

        # GO annotations
        go_terms: list[dict[str, str]] = []
        for xref in entry.get("uniProtKBCrossReferences", []):
            if xref.get("database") == "GO":
                props = {p.get("key", ""): p.get("value", "") for p in xref.get("properties", [])}
                go_terms.append({
                    "id": xref.get("id", ""),
                    "term": props.get("GoTerm", ""),
                    "source": props.get("GoEvidenceType", ""),
                })

        # Function comments
        function_text = ""
        for comment in entry.get("comments", []):
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    function_text = texts[0].get("value", "")
                    break

        return {
            "accession": entry.get("primaryAccession", ""),
            "entry_name": entry.get("uniProtkbId", ""),
            "protein_name": full_name,
            "gene_names": gene_names,
            "organism": org_name,
            "length": sequence_data.get("length"),
            "sequence": sequence_data.get("value", ""),
            "molecular_weight": sequence_data.get("molWeight"),
            "pdb_ids": pdb_ids,
            "go_terms": go_terms,
            "function": function_text,
        }
