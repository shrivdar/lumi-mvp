"""Genomics MCP Server — NCBI Gene, Ensembl, UCSC, gnomAD, ClinVar, dbSNP, SRA, Taxonomy.

Covers ~30 Biomni-equivalent tools for genomic data access.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

from base_server import APIClient, Server, error_result, make_tool, start, text_result

server = Server("genomics")

# ---------------------------------------------------------------------------
# API Clients
# ---------------------------------------------------------------------------

NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

ncbi = APIClient(base_url=NCBI_BASE, rate_limit=3 if not NCBI_API_KEY else 10)
ensembl = APIClient(base_url="https://rest.ensembl.org", headers={"Content-Type": "application/json"}, rate_limit=15)
ucsc = APIClient(base_url="https://api.genome.ucsc.edu", rate_limit=5)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list:
    return [
        make_tool("ncbi_gene_search", "Search NCBI Gene database by keyword, symbol, or organism.",
                   {"query": {"type": "string", "description": "Search query (gene name, symbol, or keyword)"},
                    "organism": {"type": "string", "description": "Organism filter (e.g. 'Homo sapiens')", "default": "Homo sapiens"},
                    "max_results": {"type": "integer", "description": "Max results to return", "default": 10}},
                   required=["query"]),

        make_tool("ncbi_gene_info", "Get detailed gene information from NCBI Gene by gene ID.",
                   {"gene_id": {"type": "string", "description": "NCBI Gene ID (e.g. '7157' for TP53)"}},
                   required=["gene_id"]),

        make_tool("ncbi_snp_lookup", "Look up SNP details from dbSNP by rsID.",
                   {"rsid": {"type": "string", "description": "dbSNP rsID (e.g. 'rs334')"}},
                   required=["rsid"]),

        make_tool("ncbi_clinvar_search", "Search ClinVar for variant clinical significance.",
                   {"query": {"type": "string", "description": "Gene name, variant, or condition"},
                    "max_results": {"type": "integer", "description": "Max results", "default": 10}},
                   required=["query"]),

        make_tool("ncbi_sra_search", "Search NCBI SRA for sequencing datasets.",
                   {"query": {"type": "string", "description": "Search query for SRA"},
                    "max_results": {"type": "integer", "description": "Max results", "default": 10}},
                   required=["query"]),

        make_tool("ncbi_taxonomy_lookup", "Look up taxonomic information for an organism.",
                   {"query": {"type": "string", "description": "Organism name or taxonomy ID"}},
                   required=["query"]),

        make_tool("ensembl_gene_lookup", "Look up gene by symbol or stable ID via Ensembl REST.",
                   {"identifier": {"type": "string", "description": "Gene symbol or Ensembl ID (e.g. 'BRCA2' or 'ENSG00000139618')"},
                    "species": {"type": "string", "description": "Species", "default": "homo_sapiens"}},
                   required=["identifier"]),

        make_tool("ensembl_sequence", "Retrieve genomic/transcript/protein sequence from Ensembl.",
                   {"identifier": {"type": "string", "description": "Ensembl stable ID"},
                    "seq_type": {"type": "string", "description": "Sequence type: genomic, cdna, cds, protein", "default": "protein"}},
                   required=["identifier"]),

        make_tool("ensembl_variant_effect", "Predict variant consequences using Ensembl VEP.",
                   {"variant": {"type": "string", "description": "HGVS notation or 'chr:pos:ref:alt' format"},
                    "species": {"type": "string", "description": "Species", "default": "homo_sapiens"}},
                   required=["variant"]),

        make_tool("ensembl_regulatory", "Fetch regulatory features for a genomic region.",
                   {"region": {"type": "string", "description": "Region in 'chr:start-end' format (e.g. '17:7668402-7687550')"},
                    "species": {"type": "string", "description": "Species", "default": "homo_sapiens"}},
                   required=["region"]),

        make_tool("ucsc_track_data", "Get track data from UCSC Genome Browser for a genomic region.",
                   {"genome": {"type": "string", "description": "Genome assembly (e.g. 'hg38')", "default": "hg38"},
                    "track": {"type": "string", "description": "Track name (e.g. 'knownGene', 'snp151')"},
                    "chrom": {"type": "string", "description": "Chromosome (e.g. 'chr17')"},
                    "start": {"type": "integer", "description": "Start position"},
                    "end": {"type": "integer", "description": "End position"}},
                   required=["track", "chrom", "start", "end"]),

        make_tool("ucsc_blat", "Run BLAT sequence search against UCSC genomes.",
                   {"sequence": {"type": "string", "description": "DNA or protein sequence"},
                    "genome": {"type": "string", "description": "Genome assembly", "default": "hg38"},
                    "query_type": {"type": "string", "description": "Query type: DNA, protein, translated RNA, translated DNA", "default": "DNA"}},
                   required=["sequence"]),

        make_tool("gnomad_variant", "Query gnomAD for population allele frequencies and constraint metrics.",
                   {"variant": {"type": "string", "description": "Variant in 'chr-pos-ref-alt' format or rsID"},
                    "dataset": {"type": "string", "description": "Dataset version", "default": "gnomad_r4"}},
                   required=["variant"]),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    try:
        params = {"api_key": NCBI_API_KEY} if NCBI_API_KEY else {}

        if name == "ncbi_gene_search":
            query = arguments["query"]
            organism = arguments.get("organism", "Homo sapiens")
            max_results = arguments.get("max_results", 10)
            resp = await ncbi.get("/esearch.fcgi", params={
                **params, "db": "gene", "term": f"{query}[Gene Name] AND {organism}[Organism]",
                "retmax": max_results, "retmode": "json",
            })
            data = resp.json()
            ids = data.get("esearchresult", {}).get("idlist", [])
            if not ids:
                return text_result({"genes": [], "count": 0})
            # Fetch summaries
            summary_resp = await ncbi.get("/esummary.fcgi", params={
                **params, "db": "gene", "id": ",".join(ids), "retmode": "json",
            })
            summary = summary_resp.json().get("result", {})
            genes = []
            for gid in ids:
                info = summary.get(gid, {})
                genes.append({
                    "gene_id": gid,
                    "symbol": info.get("name", ""),
                    "description": info.get("description", ""),
                    "organism": info.get("organism", {}).get("scientificname", ""),
                    "chromosome": info.get("chromosome", ""),
                    "map_location": info.get("maplocation", ""),
                })
            return text_result({"genes": genes, "count": len(genes)})

        elif name == "ncbi_gene_info":
            gene_id = arguments["gene_id"]
            resp = await ncbi.get("/esummary.fcgi", params={
                **params, "db": "gene", "id": gene_id, "retmode": "json",
            })
            result = resp.json().get("result", {}).get(gene_id, {})
            return text_result({"gene": {
                "gene_id": gene_id,
                "symbol": result.get("name", ""),
                "description": result.get("description", ""),
                "organism": result.get("organism", {}).get("scientificname", ""),
                "chromosome": result.get("chromosome", ""),
                "map_location": result.get("maplocation", ""),
                "gene_type": result.get("geneticresource", ""),
                "summary": result.get("summary", ""),
                "aliases": result.get("otheraliases", "").split(", ") if result.get("otheraliases") else [],
            }})

        elif name == "ncbi_snp_lookup":
            rsid = arguments["rsid"].lstrip("rs")
            resp = await ncbi.get("/esummary.fcgi", params={
                **params, "db": "snp", "id": rsid, "retmode": "json",
            })
            result = resp.json().get("result", {}).get(rsid, {})
            return text_result({"snp": {
                "rsid": f"rs{rsid}",
                "chromosome": result.get("chr", ""),
                "position": result.get("chrpos", ""),
                "clinical_significance": result.get("clinical_significance", ""),
                "functional_class": result.get("fxn_class", ""),
                "global_maf": result.get("global_maf", ""),
                "genes": result.get("genes", []),
            }})

        elif name == "ncbi_clinvar_search":
            query = arguments["query"]
            max_results = arguments.get("max_results", 10)
            resp = await ncbi.get("/esearch.fcgi", params={
                **params, "db": "clinvar", "term": query, "retmax": max_results, "retmode": "json",
            })
            ids = resp.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return text_result({"variants": [], "count": 0})
            summary_resp = await ncbi.get("/esummary.fcgi", params={
                **params, "db": "clinvar", "id": ",".join(ids), "retmode": "json",
            })
            result = summary_resp.json().get("result", {})
            variants = []
            for vid in ids:
                info = result.get(vid, {})
                variants.append({
                    "uid": vid,
                    "title": info.get("title", ""),
                    "clinical_significance": info.get("clinical_significance", {}).get("description", ""),
                    "gene": info.get("genes", [{}])[0].get("symbol", "") if info.get("genes") else "",
                    "condition": info.get("trait_set", [{}])[0].get("trait_name", "") if info.get("trait_set") else "",
                })
            return text_result({"variants": variants, "count": len(variants)})

        elif name == "ncbi_sra_search":
            query = arguments["query"]
            max_results = arguments.get("max_results", 10)
            resp = await ncbi.get("/esearch.fcgi", params={
                **params, "db": "sra", "term": query, "retmax": max_results, "retmode": "json",
            })
            ids = resp.json().get("esearchresult", {}).get("idlist", [])
            return text_result({"sra_ids": ids, "count": len(ids)})

        elif name == "ncbi_taxonomy_lookup":
            query = arguments["query"]
            resp = await ncbi.get("/esearch.fcgi", params={
                **params, "db": "taxonomy", "term": query, "retmax": 5, "retmode": "json",
            })
            ids = resp.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return text_result({"taxa": [], "count": 0})
            summary_resp = await ncbi.get("/esummary.fcgi", params={
                **params, "db": "taxonomy", "id": ",".join(ids), "retmode": "json",
            })
            result = summary_resp.json().get("result", {})
            taxa = []
            for tid in ids:
                info = result.get(tid, {})
                taxa.append({
                    "tax_id": tid,
                    "scientific_name": info.get("scientificname", ""),
                    "common_name": info.get("commonname", ""),
                    "rank": info.get("rank", ""),
                    "division": info.get("division", ""),
                })
            return text_result({"taxa": taxa, "count": len(taxa)})

        elif name == "ensembl_gene_lookup":
            identifier = arguments["identifier"]
            species = arguments.get("species", "homo_sapiens")
            # Try symbol lookup first, fall back to ID lookup
            if identifier.startswith("ENSG"):
                resp = await ensembl.get(f"/lookup/id/{identifier}", params={"expand": 1})
            else:
                resp = await ensembl.get(f"/lookup/symbol/{species}/{identifier}", params={"expand": 1})
            data = resp.json()
            return text_result({"gene": {
                "id": data.get("id", ""),
                "display_name": data.get("display_name", ""),
                "description": data.get("description", ""),
                "biotype": data.get("biotype", ""),
                "species": data.get("species", ""),
                "chromosome": data.get("seq_region_name", ""),
                "start": data.get("start"),
                "end": data.get("end"),
                "strand": data.get("strand"),
                "assembly": data.get("assembly_name", ""),
            }})

        elif name == "ensembl_sequence":
            identifier = arguments["identifier"]
            seq_type = arguments.get("seq_type", "protein")
            resp = await ensembl.get(f"/sequence/id/{identifier}", params={"type": seq_type},
                                     headers={"Content-Type": "application/json"})
            data = resp.json()
            return text_result({"sequence": {
                "id": data.get("id", ""),
                "seq": data.get("seq", ""),
                "molecule": data.get("molecule", ""),
                "description": data.get("desc", ""),
            }})

        elif name == "ensembl_variant_effect":
            variant = arguments["variant"]
            species = arguments.get("species", "homo_sapiens")
            resp = await ensembl.get(f"/vep/{species}/hgvs/{variant}", params={"content-type": "application/json"})
            data = resp.json()
            if isinstance(data, list) and data:
                vep = data[0]
                consequences = []
                for tc in vep.get("transcript_consequences", []):
                    consequences.append({
                        "gene_symbol": tc.get("gene_symbol", ""),
                        "consequence": ", ".join(tc.get("consequence_terms", [])),
                        "impact": tc.get("impact", ""),
                        "biotype": tc.get("biotype", ""),
                        "sift": tc.get("sift_prediction", ""),
                        "polyphen": tc.get("polyphen_prediction", ""),
                    })
                return text_result({"variant": variant, "consequences": consequences})
            return text_result({"variant": variant, "consequences": []})

        elif name == "ensembl_regulatory":
            region = arguments["region"]
            species = arguments.get("species", "homo_sapiens")
            resp = await ensembl.get(f"/overlap/region/{species}/{region}",
                                     params={"feature": "regulatory", "content-type": "application/json"})
            features = resp.json()
            results = [{"id": f.get("id", ""), "type": f.get("feature_type", ""),
                        "start": f.get("start"), "end": f.get("end"),
                        "description": f.get("description", "")} for f in features[:20]]
            return text_result({"features": results, "count": len(features)})

        elif name == "ucsc_track_data":
            genome = arguments.get("genome", "hg38")
            track = arguments["track"]
            chrom = arguments["chrom"]
            start_pos = arguments["start"]
            end_pos = arguments["end"]
            resp = await ucsc.get(f"/getData/track", params={
                "genome": genome, "track": track, "chrom": chrom, "start": start_pos, "end": end_pos,
            })
            return text_result(resp.json())

        elif name == "ucsc_blat":
            sequence = arguments["sequence"]
            genome = arguments.get("genome", "hg38")
            query_type = arguments.get("query_type", "DNA")
            resp = await ucsc.get(f"/search", params={
                "genome": genome, "search": sequence, "type": query_type,
            })
            return text_result(resp.json())

        elif name == "gnomad_variant":
            variant = arguments["variant"]
            dataset = arguments.get("dataset", "gnomad_r4")
            gnomad_client = APIClient(base_url="https://gnomad.broadinstitute.org/api", rate_limit=2)
            # Use GraphQL API
            query = """query($variantId: String!, $dataset: DatasetId!) {
                variant(variantId: $variantId, dataset: $dataset) {
                    variant_id
                    chrom
                    pos
                    ref
                    alt
                    exome { ac an af }
                    genome { ac an af }
                }
            }"""
            resp = await gnomad_client.post("", json={"query": query, "variables": {"variantId": variant, "dataset": dataset}})
            await gnomad_client.close()
            return text_result(resp.json())

        return error_result(f"Unknown tool: {name}")
    except Exception as exc:
        return error_result(f"{name} failed: {exc}")


if __name__ == "__main__":
    start(server)
