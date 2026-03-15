"""Gene Expression MCP Server — GEO, GTEx, Human Protein Atlas, GeneBass, RummaGEO.

Covers expression profiling datasets, tissue-level expression, eQTLs, subcellular
localization, pathology expression, phenotype associations, and gene set enrichment.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

from base_server import APIClient, Server, error_result, make_tool, start, text_result

server = Server("gene-expression")

# ---------------------------------------------------------------------------
# API Clients
# ---------------------------------------------------------------------------

NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

ncbi = APIClient(
    base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
    rate_limit=3 if not NCBI_API_KEY else 10,
)
gtex = APIClient(
    base_url="https://gtexportal.org/api/v2",
    headers={"Accept": "application/json"},
    rate_limit=5,
)
hpa = APIClient(
    base_url="https://www.proteinatlas.org/api",
    headers={"Accept": "application/json"},
    rate_limit=5,
)
genebass = APIClient(
    base_url="https://app.genebass.org/api",
    headers={"Accept": "application/json"},
    rate_limit=3,
)
rummageo = APIClient(
    base_url="https://rummageo.com/api",
    headers={"Accept": "application/json"},
    rate_limit=3,
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@server.list_tools()
async def list_tools() -> list:
    return [
        make_tool(
            "geo_search",
            "Search NCBI GEO for gene expression datasets by keyword, organism, or platform.",
            {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g. 'BRCA1 breast cancer', 'RNA-seq glioblastoma')",
                },
                "organism": {
                    "type": "string",
                    "description": "Organism filter (e.g. 'Homo sapiens')",
                    "default": "Homo sapiens",
                },
                "dataset_type": {
                    "type": "string",
                    "description": "GEO type filter: gse (series), gds (dataset), gpl (platform), gsm (sample)",
                    "default": "gse",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return",
                    "default": 10,
                },
            },
            required=["query"],
        ),
        make_tool(
            "geo_dataset",
            "Get detailed information for a GEO dataset by accession (GSE or GDS ID).",
            {
                "accession": {
                    "type": "string",
                    "description": "GEO accession ID (e.g. 'GSE12345' or 'GDS1234')",
                },
            },
            required=["accession"],
        ),
        make_tool(
            "geo_samples",
            "Get sample information (GSM) for a GEO series, including conditions and metadata.",
            {
                "accession": {
                    "type": "string",
                    "description": "GEO series accession (e.g. 'GSE12345')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max samples to return",
                    "default": 20,
                },
            },
            required=["accession"],
        ),
        make_tool(
            "gtex_gene_expression",
            "Query GTEx for median gene expression (TPM) across human tissues.",
            {
                "gene_id": {
                    "type": "string",
                    "description": "Ensembl gene ID (e.g. 'ENSG00000141510') or gene symbol (e.g. 'TP53')",
                },
                "tissue": {
                    "type": "string",
                    "description": "Optional tissue filter (e.g. 'Brain', 'Liver'). Omit for all tissues.",
                },
            },
            required=["gene_id"],
        ),
        make_tool(
            "gtex_eqtl",
            "Query GTEx for expression quantitative trait loci (eQTLs) associated with a gene.",
            {
                "gene_id": {
                    "type": "string",
                    "description": "Ensembl gene ID (e.g. 'ENSG00000141510')",
                },
                "tissue": {
                    "type": "string",
                    "description": "GTEx tissue ID (e.g. 'Liver', 'Brain_Cortex')",
                },
            },
            required=["gene_id", "tissue"],
        ),
        make_tool(
            "hpa_tissue_expression",
            "Get tissue-level protein and RNA expression data from the Human Protein Atlas.",
            {
                "gene": {
                    "type": "string",
                    "description": "Gene symbol (e.g. 'TP53', 'EGFR')",
                },
            },
            required=["gene"],
        ),
        make_tool(
            "hpa_subcellular",
            "Get subcellular localization data from the Human Protein Atlas.",
            {
                "gene": {
                    "type": "string",
                    "description": "Gene symbol (e.g. 'TP53')",
                },
            },
            required=["gene"],
        ),
        make_tool(
            "hpa_pathology",
            "Get pathology/cancer expression data from the Human Protein Atlas.",
            {
                "gene": {
                    "type": "string",
                    "description": "Gene symbol (e.g. 'TP53')",
                },
            },
            required=["gene"],
        ),
        make_tool(
            "genebass_gene",
            "Query GeneBass (UK Biobank exome) for gene-level phenotype associations.",
            {
                "gene": {
                    "type": "string",
                    "description": "Gene symbol (e.g. 'PCSK9', 'BRCA1')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max phenotype associations to return",
                    "default": 20,
                },
            },
            required=["gene"],
        ),
        make_tool(
            "rummageo_search",
            "Search RummaGEO for gene set enrichment across GEO datasets. Finds datasets where a given gene set is significantly up- or down-regulated.",
            {
                "genes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of gene symbols to search (e.g. ['TP53', 'BRCA1', 'MDM2'])",
                },
                "direction": {
                    "type": "string",
                    "description": "Expression direction: 'up', 'down', or 'both'",
                    "default": "both",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return",
                    "default": 10,
                },
            },
            required=["genes"],
        ),
    ]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    try:
        ncbi_params = {"api_key": NCBI_API_KEY} if NCBI_API_KEY else {}

        # ---- GEO Search ----
        if name == "geo_search":
            query = arguments["query"]
            organism = arguments.get("organism", "Homo sapiens")
            dataset_type = arguments.get("dataset_type", "gse")
            max_results = arguments.get("max_results", 10)

            db = "gds"
            type_filter = {
                "gse": "[Entry Type]",
                "gds": "[Entry Type]",
                "gpl": "[Entry Type]",
                "gsm": "[Entry Type]",
            }
            term_parts = [query]
            if organism:
                term_parts.append(f"{organism}[Organism]")
            if dataset_type in type_filter:
                term_parts.append(f"{dataset_type}{type_filter[dataset_type]}")
            term = " AND ".join(term_parts)

            resp = await ncbi.get(
                "/esearch.fcgi",
                params={
                    **ncbi_params,
                    "db": db,
                    "term": term,
                    "retmax": max_results,
                    "retmode": "json",
                },
            )
            data = resp.json()
            ids = data.get("esearchresult", {}).get("idlist", [])
            if not ids:
                return text_result({"datasets": [], "count": 0, "query": term})

            summary_resp = await ncbi.get(
                "/esummary.fcgi",
                params={
                    **ncbi_params,
                    "db": db,
                    "id": ",".join(ids),
                    "retmode": "json",
                },
            )
            result = summary_resp.json().get("result", {})
            datasets = []
            for uid in ids:
                info = result.get(uid, {})
                datasets.append(
                    {
                        "uid": uid,
                        "accession": info.get("Accession", info.get("accession", "")),
                        "title": info.get("title", ""),
                        "summary": info.get("summary", "")[:500],
                        "organism": info.get("taxon", ""),
                        "type": info.get("entrytype", info.get("gdsType", "")),
                        "platform": info.get("GPL", info.get("gpl", "")),
                        "sample_count": info.get("n_samples", info.get("samplecount", "")),
                        "pub_date": info.get("PDAT", info.get("pdat", "")),
                    }
                )
            return text_result({"datasets": datasets, "count": len(datasets)})

        # ---- GEO Dataset Details ----
        elif name == "geo_dataset":
            accession = arguments["accession"].strip().upper()
            db = "gds"

            # Search for the accession
            resp = await ncbi.get(
                "/esearch.fcgi",
                params={
                    **ncbi_params,
                    "db": db,
                    "term": f"{accession}[Accession]",
                    "retmax": 1,
                    "retmode": "json",
                },
            )
            ids = resp.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return error_result(f"GEO accession not found: {accession}")

            summary_resp = await ncbi.get(
                "/esummary.fcgi",
                params={
                    **ncbi_params,
                    "db": db,
                    "id": ids[0],
                    "retmode": "json",
                },
            )
            info = summary_resp.json().get("result", {}).get(ids[0], {})
            return text_result(
                {
                    "accession": accession,
                    "uid": ids[0],
                    "title": info.get("title", ""),
                    "summary": info.get("summary", ""),
                    "organism": info.get("taxon", ""),
                    "type": info.get("entrytype", info.get("gdsType", "")),
                    "platform": info.get("GPL", info.get("gpl", "")),
                    "sample_count": info.get("n_samples", info.get("samplecount", "")),
                    "pub_date": info.get("PDAT", info.get("pdat", "")),
                    "feature_count": info.get("FTPLink", info.get("ftplink", "")),
                    "supplementary_files": info.get("suppFile", info.get("supplementaryfile", "")),
                    "pubmed_ids": info.get("PubMedIds", info.get("pubmedids", [])),
                    "contact": info.get("Contact", info.get("contact", "")),
                }
            )

        # ---- GEO Samples ----
        elif name == "geo_samples":
            accession = arguments["accession"].strip().upper()
            max_results = arguments.get("max_results", 20)

            # Search for the GSE to get associated samples
            resp = await ncbi.get(
                "/esearch.fcgi",
                params={
                    **ncbi_params,
                    "db": "gds",
                    "term": f"{accession}[Accession]",
                    "retmax": 1,
                    "retmode": "json",
                },
            )
            ids = resp.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return error_result(f"GEO series not found: {accession}")

            # Get the series summary which includes sample info
            summary_resp = await ncbi.get(
                "/esummary.fcgi",
                params={
                    **ncbi_params,
                    "db": "gds",
                    "id": ids[0],
                    "retmode": "json",
                },
            )
            info = summary_resp.json().get("result", {}).get(ids[0], {})
            samples_raw = info.get("Samples", info.get("samples", []))

            samples = []
            for s in samples_raw[:max_results]:
                samples.append(
                    {
                        "accession": s.get("Accession", s.get("accession", "")),
                        "title": s.get("Title", s.get("title", "")),
                    }
                )

            # If no embedded samples, search for GSM entries linked to this series
            if not samples:
                gsm_resp = await ncbi.get(
                    "/esearch.fcgi",
                    params={
                        **ncbi_params,
                        "db": "gds",
                        "term": f"{accession}[Accession] AND gsm[Entry Type]",
                        "retmax": max_results,
                        "retmode": "json",
                    },
                )
                gsm_ids = gsm_resp.json().get("esearchresult", {}).get("idlist", [])
                if gsm_ids:
                    gsm_summary = await ncbi.get(
                        "/esummary.fcgi",
                        params={
                            **ncbi_params,
                            "db": "gds",
                            "id": ",".join(gsm_ids),
                            "retmode": "json",
                        },
                    )
                    gsm_result = gsm_summary.json().get("result", {})
                    for gid in gsm_ids:
                        gs = gsm_result.get(gid, {})
                        samples.append(
                            {
                                "accession": gs.get("Accession", gs.get("accession", "")),
                                "title": gs.get("title", ""),
                                "organism": gs.get("taxon", ""),
                                "platform": gs.get("GPL", gs.get("gpl", "")),
                            }
                        )

            return text_result(
                {
                    "series": accession,
                    "series_title": info.get("title", ""),
                    "total_samples": info.get("n_samples", info.get("samplecount", len(samples))),
                    "samples": samples,
                }
            )

        # ---- GTEx Gene Expression ----
        elif name == "gtex_gene_expression":
            gene_id = arguments["gene_id"]
            tissue = arguments.get("tissue")

            params = {"geneId": gene_id, "datasetId": "gtex_v8"}
            if tissue:
                params["tissueSiteDetailId"] = tissue

            resp = await gtex.get("/expression/medianGeneExpression", params=params)
            data = resp.json()

            items = data.get("data", data) if isinstance(data, dict) else data
            if not isinstance(items, list):
                items = [items] if items else []

            expression = []
            for entry in items:
                expression.append(
                    {
                        "tissue": entry.get("tissueSiteDetailId", entry.get("tissueSiteDetail", "")),
                        "median_tpm": entry.get("median", entry.get("medianTPM", None)),
                        "gene_id": entry.get("gencodeId", entry.get("geneId", gene_id)),
                        "gene_symbol": entry.get("geneSymbol", entry.get("geneSymbolUpper", "")),
                        "sample_count": entry.get("numSamples", None),
                    }
                )

            # Sort by expression level descending
            expression.sort(key=lambda x: x.get("median_tpm") or 0, reverse=True)

            return text_result(
                {
                    "gene": gene_id,
                    "tissues": expression,
                    "count": len(expression),
                }
            )

        # ---- GTEx eQTL ----
        elif name == "gtex_eqtl":
            gene_id = arguments["gene_id"]
            tissue = arguments["tissue"]

            resp = await gtex.get(
                "/association/singleTissueEqtl",
                params={
                    "geneId": gene_id,
                    "tissueSiteDetailId": tissue,
                    "datasetId": "gtex_v8",
                },
            )
            data = resp.json()

            items = data.get("data", data) if isinstance(data, dict) else data
            if not isinstance(items, list):
                items = [items] if items else []

            eqtls = []
            for entry in items:
                eqtls.append(
                    {
                        "variant_id": entry.get("variantId", entry.get("snpId", "")),
                        "gene_id": entry.get("gencodeId", entry.get("geneId", gene_id)),
                        "gene_symbol": entry.get("geneSymbol", entry.get("geneSymbolUpper", "")),
                        "tissue": entry.get("tissueSiteDetailId", tissue),
                        "pvalue": entry.get("pValue", entry.get("pval", None)),
                        "nes": entry.get("nes", entry.get("slope", None)),
                        "effect_size": entry.get("effectSize", None),
                    }
                )

            # Sort by p-value ascending (most significant first)
            eqtls.sort(key=lambda x: x.get("pvalue") or 1.0)

            return text_result(
                {
                    "gene": gene_id,
                    "tissue": tissue,
                    "eqtls": eqtls[:50],
                    "count": len(eqtls),
                }
            )

        # ---- HPA Tissue Expression ----
        elif name == "hpa_tissue_expression":
            gene = arguments["gene"].upper()

            resp = await hpa.get(f"/search_download.php", params={
                "search": gene,
                "format": "json",
                "columns": "g,t,rnats,up",
                "compress": "no",
            })
            data = resp.json()

            if not data or not isinstance(data, list):
                return text_result({"gene": gene, "tissues": [], "count": 0})

            # Find the matching gene entry
            gene_data = None
            for entry in data:
                if entry.get("Gene", "").upper() == gene or entry.get("Gene name", "").upper() == gene:
                    gene_data = entry
                    break
            if not gene_data and data:
                gene_data = data[0]

            tissues = []
            # Parse RNA tissue expression
            rna_tissues = gene_data.get("RNA tissue specific nTPM", gene_data.get("RNA TS TPM", ""))
            if isinstance(rna_tissues, str) and rna_tissues:
                for pair in rna_tissues.split(";"):
                    if ":" in pair:
                        tissue_name, value = pair.rsplit(":", 1)
                        try:
                            tissues.append({"tissue": tissue_name.strip(), "ntpm": float(value.strip())})
                        except ValueError:
                            tissues.append({"tissue": tissue_name.strip(), "ntpm": value.strip()})
            elif isinstance(rna_tissues, dict):
                for tissue_name, value in rna_tissues.items():
                    tissues.append({"tissue": tissue_name, "ntpm": value})

            # Sort by expression level
            tissues.sort(key=lambda x: float(x.get("ntpm", 0)) if isinstance(x.get("ntpm"), (int, float)) else 0, reverse=True)

            # Parse protein expression (UP = tissue expression reliability + level)
            protein_data = gene_data.get("UP", gene_data.get("Uniprot", ""))

            return text_result(
                {
                    "gene": gene,
                    "ensembl_id": gene_data.get("Ensembl", ""),
                    "gene_name": gene_data.get("Gene", gene_data.get("Gene name", "")),
                    "gene_description": gene_data.get("Gene description", ""),
                    "rna_tissues": tissues,
                    "protein_expression": protein_data,
                    "count": len(tissues),
                }
            )

        # ---- HPA Subcellular ----
        elif name == "hpa_subcellular":
            gene = arguments["gene"].upper()

            resp = await hpa.get(f"/search_download.php", params={
                "search": gene,
                "format": "json",
                "columns": "g,scl,scml",
                "compress": "no",
            })
            data = resp.json()

            if not data or not isinstance(data, list):
                return text_result({"gene": gene, "locations": []})

            gene_data = None
            for entry in data:
                if entry.get("Gene", "").upper() == gene or entry.get("Gene name", "").upper() == gene:
                    gene_data = entry
                    break
            if not gene_data and data:
                gene_data = data[0]

            # Parse subcellular localization
            main_location = gene_data.get("Subcellular main location", gene_data.get("Main location", ""))
            additional_location = gene_data.get("Subcellular additional location", gene_data.get("Additional location", ""))

            main_locs = [loc.strip() for loc in main_location.split(";") if loc.strip()] if isinstance(main_location, str) else []
            additional_locs = [loc.strip() for loc in additional_location.split(";") if loc.strip()] if isinstance(additional_location, str) else []

            return text_result(
                {
                    "gene": gene,
                    "ensembl_id": gene_data.get("Ensembl", ""),
                    "main_locations": main_locs,
                    "additional_locations": additional_locs,
                    "reliability": gene_data.get("Reliability (IF)", gene_data.get("Reliability", "")),
                }
            )

        # ---- HPA Pathology ----
        elif name == "hpa_pathology":
            gene = arguments["gene"].upper()

            resp = await hpa.get(f"/search_download.php", params={
                "search": gene,
                "format": "json",
                "columns": "g,patd,prna",
                "compress": "no",
            })
            data = resp.json()

            if not data or not isinstance(data, list):
                return text_result({"gene": gene, "cancers": []})

            gene_data = None
            for entry in data:
                if entry.get("Gene", "").upper() == gene or entry.get("Gene name", "").upper() == gene:
                    gene_data = entry
                    break
            if not gene_data and data:
                gene_data = data[0]

            # Parse pathology data
            cancers = []
            pathology_data = gene_data.get("Pathology diagnostic", gene_data.get("Pathology prognostics", ""))
            if isinstance(pathology_data, str) and pathology_data:
                for entry_str in pathology_data.split(";"):
                    entry_str = entry_str.strip()
                    if entry_str:
                        cancers.append(entry_str)
            elif isinstance(pathology_data, dict):
                for cancer_type, details in pathology_data.items():
                    cancers.append({"cancer": cancer_type, "data": details})

            # Parse RNA cancer expression
            rna_cancer = gene_data.get("RNA cancer specific FPKM", gene_data.get("Pathology RNA", ""))
            rna_cancers = []
            if isinstance(rna_cancer, str) and rna_cancer:
                for pair in rna_cancer.split(";"):
                    if ":" in pair:
                        cancer_name, value = pair.rsplit(":", 1)
                        try:
                            rna_cancers.append({"cancer": cancer_name.strip(), "fpkm": float(value.strip())})
                        except ValueError:
                            rna_cancers.append({"cancer": cancer_name.strip(), "fpkm": value.strip()})
            elif isinstance(rna_cancer, dict):
                for cancer_name, value in rna_cancer.items():
                    rna_cancers.append({"cancer": cancer_name, "fpkm": value})

            rna_cancers.sort(
                key=lambda x: float(x.get("fpkm", 0)) if isinstance(x.get("fpkm"), (int, float)) else 0,
                reverse=True,
            )

            return text_result(
                {
                    "gene": gene,
                    "ensembl_id": gene_data.get("Ensembl", ""),
                    "pathology_diagnostic": cancers,
                    "rna_cancer_expression": rna_cancers,
                }
            )

        # ---- GeneBass Gene ----
        elif name == "genebass_gene":
            gene = arguments["gene"]
            max_results = arguments.get("max_results", 20)

            resp = await genebass.get(
                "/v1/results",
                params={"gene": gene, "limit": max_results},
            )
            data = resp.json()

            items = data.get("results", data.get("data", data))
            if not isinstance(items, list):
                items = [items] if items else []

            associations = []
            for entry in items:
                associations.append(
                    {
                        "gene": entry.get("gene", entry.get("gene_symbol", gene)),
                        "phenotype": entry.get("phenotype", entry.get("description", "")),
                        "phenotype_code": entry.get("phenocode", entry.get("pheno", "")),
                        "pvalue": entry.get("pval", entry.get("Pvalue", None)),
                        "beta": entry.get("BETA", entry.get("beta", None)),
                        "se": entry.get("SE", entry.get("se", None)),
                        "n_cases": entry.get("n_cases", entry.get("N_case", None)),
                        "n_controls": entry.get("n_controls", entry.get("N_ctrl", None)),
                        "annotation": entry.get("annotation", entry.get("consequence", "")),
                    }
                )

            associations.sort(key=lambda x: x.get("pvalue") or 1.0)

            return text_result(
                {
                    "gene": gene,
                    "associations": associations,
                    "count": len(associations),
                }
            )

        # ---- RummaGEO Search ----
        elif name == "rummageo_search":
            genes = arguments["genes"]
            direction = arguments.get("direction", "both")
            max_results = arguments.get("max_results", 10)

            gene_str = "\n".join(genes)

            resp = await rummageo.post(
                "/enrich",
                json={
                    "genes": gene_str,
                    "direction": direction,
                    "limit": max_results,
                },
            )
            data = resp.json()

            items = data.get("results", data.get("data", data))
            if not isinstance(items, list):
                items = [items] if items else []

            results = []
            for entry in items:
                results.append(
                    {
                        "geo_accession": entry.get("gse", entry.get("accession", "")),
                        "title": entry.get("title", ""),
                        "description": entry.get("description", entry.get("summary", ""))[:500],
                        "direction": entry.get("direction", entry.get("regulation", "")),
                        "pvalue": entry.get("pvalue", entry.get("pval", None)),
                        "adj_pvalue": entry.get("adj_pvalue", entry.get("fdr", None)),
                        "overlap": entry.get("overlap", entry.get("n_overlap", None)),
                        "overlap_genes": entry.get("overlap_genes", entry.get("genes", [])),
                        "organism": entry.get("organism", entry.get("species", "")),
                        "platform": entry.get("platform", entry.get("gpl", "")),
                    }
                )

            return text_result(
                {
                    "query_genes": genes,
                    "direction": direction,
                    "results": results,
                    "count": len(results),
                }
            )

        return error_result(f"Unknown tool: {name}")
    except Exception as exc:
        return error_result(f"{name} failed: {exc}")


if __name__ == "__main__":
    start(server)
