"""Regulatory/Epigenomics MCP Server — ENCODE, SCREEN, JASPAR, ReMap, miRTarBase, miRDB.

Covers cis-regulatory elements, transcription factor binding, chromatin accessibility,
miRNA-target interactions, and epigenomic annotation datasets.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

from base_server import APIClient, Server, error_result, make_tool, start, text_result

server = Server("regulatory-epigenomics")

# ---------------------------------------------------------------------------
# API Clients
# ---------------------------------------------------------------------------

encode = APIClient(
    base_url="https://www.encodeproject.org",
    headers={"Accept": "application/json"},
    rate_limit=5,
)
screen = APIClient(
    base_url="https://screen.encodeproject.org/api",
    headers={"Accept": "application/json"},
    rate_limit=5,
)
jaspar = APIClient(
    base_url="https://jaspar.elixir.no/api/v1",
    headers={"Accept": "application/json"},
    rate_limit=5,
)
remap = APIClient(
    base_url="https://remap2022.univ-amu.fr/api",
    headers={"Accept": "application/json"},
    rate_limit=3,
)
mirtarbase = APIClient(
    base_url="https://mirtarbase.cuhk.edu.cn/api",
    headers={"Accept": "application/json"},
    rate_limit=3,
)
mirdb = APIClient(
    base_url="https://mirdb.org/api",
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
            "encode_search",
            "Search the ENCODE portal for experiments, datasets, biosamples, or other objects. "
            "Supports full-text search with optional type and organism filters.",
            {
                "query": {"type": "string", "description": "Search query (e.g. 'CTCF ChIP-seq', 'H3K27ac')"},
                "type": {
                    "type": "string",
                    "description": "ENCODE object type filter (e.g. 'Experiment', 'Dataset', 'Biosample')",
                    "default": "Experiment",
                },
                "organism": {
                    "type": "string",
                    "description": "Organism filter (e.g. 'Homo sapiens')",
                    "default": "Homo sapiens",
                },
                "assay_title": {
                    "type": "string",
                    "description": "Assay type filter (e.g. 'ChIP-seq', 'ATAC-seq', 'Hi-C')",
                },
                "max_results": {"type": "integer", "description": "Max results to return", "default": 10},
            },
            required=["query"],
        ),
        make_tool(
            "encode_experiment",
            "Get detailed information about a specific ENCODE experiment including files, "
            "biosamples, targets, and quality metrics.",
            {
                "accession": {
                    "type": "string",
                    "description": "ENCODE experiment accession (e.g. 'ENCSR000BGZ')",
                },
            },
            required=["accession"],
        ),
        make_tool(
            "screen_ccres",
            "Search ENCODE SCREEN for candidate cis-regulatory elements (cCREs) by genomic "
            "coordinates, gene name, or accession. Returns element classification, "
            "epigenetic signals, and linked genes.",
            {
                "query": {
                    "type": "string",
                    "description": "Gene symbol, SCREEN accession (e.g. 'EH38E1516972'), or coordinates ('chr1:10000-20000')",
                },
                "assembly": {
                    "type": "string",
                    "description": "Genome assembly",
                    "default": "GRCh38",
                },
                "biosample": {
                    "type": "string",
                    "description": "Biosample/cell type filter (e.g. 'K562', 'GM12878')",
                },
                "element_type": {
                    "type": "string",
                    "description": "cCRE classification filter: PLS (promoter-like), pELS (proximal enhancer-like), dELS (distal enhancer-like), CTCF-only, DNase-H3K4me3",
                },
                "max_results": {"type": "integer", "description": "Max results", "default": 10},
            },
            required=["query"],
        ),
        make_tool(
            "jaspar_matrix",
            "Get a transcription factor binding profile (position weight matrix) from JASPAR "
            "by matrix ID or TF name. Returns the frequency matrix and consensus sequence.",
            {
                "query": {
                    "type": "string",
                    "description": "JASPAR matrix ID (e.g. 'MA0139.1') or TF name (e.g. 'CTCF')",
                },
                "collection": {
                    "type": "string",
                    "description": "JASPAR collection: CORE, CNE, PHYLOFACTS, SPLICE, POLII, FAM, UNVALIDATED",
                    "default": "CORE",
                },
                "tax_group": {
                    "type": "string",
                    "description": "Taxonomic group filter (e.g. 'vertebrates', 'plants', 'insects')",
                    "default": "vertebrates",
                },
            },
            required=["query"],
        ),
        make_tool(
            "jaspar_scan",
            "Scan a DNA sequence for transcription factor binding sites using JASPAR profiles. "
            "Returns predicted binding sites with positions and scores.",
            {
                "sequence": {
                    "type": "string",
                    "description": "DNA sequence to scan (ACGT characters)",
                },
                "matrix_id": {
                    "type": "string",
                    "description": "JASPAR matrix ID to scan with (e.g. 'MA0139.1' for CTCF). If omitted, scans with top vertebrate matrices.",
                },
                "threshold": {
                    "type": "number",
                    "description": "Relative score threshold (0-1, higher is more stringent)",
                    "default": 0.8,
                },
            },
            required=["sequence"],
        ),
        make_tool(
            "remap_search",
            "Search ReMap for transcription factor and regulatory region annotations. "
            "Provides ChIP-seq peak data across cell types and conditions.",
            {
                "tf_name": {
                    "type": "string",
                    "description": "Transcription factor name (e.g. 'CTCF', 'TP53')",
                },
                "region": {
                    "type": "string",
                    "description": "Genomic region in 'chr:start-end' format (e.g. 'chr1:10000-20000')",
                },
                "cell_type": {
                    "type": "string",
                    "description": "Cell type or tissue filter (e.g. 'K562', 'HepG2')",
                },
                "species": {
                    "type": "string",
                    "description": "Species",
                    "default": "Homo sapiens",
                },
                "max_results": {"type": "integer", "description": "Max results", "default": 20},
            },
            required=[],
        ),
        make_tool(
            "mirtarbase_search",
            "Search miRTarBase for experimentally validated miRNA-target interactions. "
            "Returns interaction details, validation methods, and supporting references.",
            {
                "mirna": {
                    "type": "string",
                    "description": "miRNA name (e.g. 'hsa-miR-21-5p')",
                },
                "target_gene": {
                    "type": "string",
                    "description": "Target gene symbol (e.g. 'PTEN', 'TP53')",
                },
                "species": {
                    "type": "string",
                    "description": "Species filter (e.g. 'Homo sapiens')",
                    "default": "Homo sapiens",
                },
                "strong_evidence_only": {
                    "type": "boolean",
                    "description": "Only return interactions with strong experimental evidence (reporter assay, western blot, qPCR)",
                    "default": False,
                },
                "max_results": {"type": "integer", "description": "Max results", "default": 20},
            },
            required=[],
        ),
        make_tool(
            "mirdb_predict",
            "Predict miRNA targets using miRDB or look up predicted targets for a specific miRNA. "
            "Returns predicted target genes with scores.",
            {
                "mirna": {
                    "type": "string",
                    "description": "miRNA name (e.g. 'hsa-miR-21-5p')",
                },
                "target_gene": {
                    "type": "string",
                    "description": "Target gene symbol to check prediction score against",
                },
                "species": {
                    "type": "string",
                    "description": "Species: human, mouse, rat, dog, chicken",
                    "default": "human",
                },
                "score_threshold": {
                    "type": "integer",
                    "description": "Minimum prediction score (50-100, higher is more confident)",
                    "default": 80,
                },
                "max_results": {"type": "integer", "description": "Max results", "default": 20},
            },
            required=[],
        ),
        make_tool(
            "encode_biosample",
            "Search ENCODE for biosamples (cell lines, tissues, primary cells) with metadata "
            "including organism, biosample type, and available experiments.",
            {
                "query": {
                    "type": "string",
                    "description": "Biosample search query (e.g. 'K562', 'liver', 'neural crest')",
                },
                "organism": {
                    "type": "string",
                    "description": "Organism filter",
                    "default": "Homo sapiens",
                },
                "biosample_type": {
                    "type": "string",
                    "description": "Type filter: cell_line, tissue, primary_cell, in_vitro_differentiated_cells",
                },
                "max_results": {"type": "integer", "description": "Max results", "default": 10},
            },
            required=["query"],
        ),
        make_tool(
            "encode_annotation",
            "Get annotation data from ENCODE including gene annotations, regulatory elements, "
            "and reference epigenomes for a specific accession or search term.",
            {
                "accession": {
                    "type": "string",
                    "description": "ENCODE annotation accession (e.g. 'ENCSR000BGZ')",
                },
                "query": {
                    "type": "string",
                    "description": "Search term for annotations (e.g. 'H3K27ac human')",
                },
                "annotation_type": {
                    "type": "string",
                    "description": "Annotation type filter (e.g. 'candidate regulatory elements', 'gene expression')",
                },
                "max_results": {"type": "integer", "description": "Max results", "default": 10},
            },
            required=[],
        ),
    ]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    try:
        # ---- ENCODE Search ------------------------------------------------
        if name == "encode_search":
            query = arguments["query"]
            obj_type = arguments.get("type", "Experiment")
            organism = arguments.get("organism", "Homo sapiens")
            assay_title = arguments.get("assay_title")
            max_results = arguments.get("max_results", 10)

            params: dict = {
                "searchTerm": query,
                "type": obj_type,
                "organism.scientific_name": organism,
                "limit": max_results,
                "format": "json",
            }
            if assay_title:
                params["assay_title"] = assay_title

            resp = await encode.get("/search/", params=params)
            data = resp.json()
            graph = data.get("@graph", [])
            results = []
            for item in graph[:max_results]:
                results.append({
                    "accession": item.get("accession", ""),
                    "description": item.get("description", ""),
                    "assay_title": item.get("assay_title", ""),
                    "biosample_summary": item.get("biosample_summary", ""),
                    "target": item.get("target", {}).get("label", "") if isinstance(item.get("target"), dict) else item.get("target", ""),
                    "status": item.get("status", ""),
                    "date_released": item.get("date_released", ""),
                    "lab": item.get("lab", {}).get("title", "") if isinstance(item.get("lab"), dict) else item.get("lab", ""),
                    "award": item.get("award", {}).get("project", "") if isinstance(item.get("award"), dict) else "",
                })
            return text_result({
                "results": results,
                "count": data.get("total", len(results)),
                "query": query,
            })

        # ---- ENCODE Experiment Detail -------------------------------------
        elif name == "encode_experiment":
            accession = arguments["accession"]
            resp = await encode.get(f"/experiments/{accession}/", params={"format": "json"})
            data = resp.json()

            files = []
            for f in data.get("files", [])[:20]:
                file_path = f if isinstance(f, str) else f.get("@id", "")
                files.append(file_path)

            # Fetch summary file info if embedded
            file_details = []
            for f in data.get("files", [])[:20]:
                if isinstance(f, dict):
                    file_details.append({
                        "accession": f.get("accession", ""),
                        "file_type": f.get("file_type", ""),
                        "output_type": f.get("output_type", ""),
                        "assembly": f.get("assembly", ""),
                        "status": f.get("status", ""),
                        "href": f.get("href", ""),
                    })

            replicates = []
            for rep in data.get("replicates", [])[:10]:
                if isinstance(rep, dict):
                    lib = rep.get("library", {})
                    bio = lib.get("biosample", {}) if isinstance(lib, dict) else {}
                    replicates.append({
                        "biological_replicate": rep.get("biological_replicate_number"),
                        "technical_replicate": rep.get("technical_replicate_number"),
                        "biosample": bio.get("summary", "") if isinstance(bio, dict) else "",
                    })

            return text_result({
                "accession": data.get("accession", accession),
                "description": data.get("description", ""),
                "assay_title": data.get("assay_title", ""),
                "assay_term_name": data.get("assay_term_name", ""),
                "biosample_summary": data.get("biosample_summary", ""),
                "target": data.get("target", {}).get("label", "") if isinstance(data.get("target"), dict) else data.get("target", ""),
                "status": data.get("status", ""),
                "date_released": data.get("date_released", ""),
                "lab": data.get("lab", {}).get("title", "") if isinstance(data.get("lab"), dict) else "",
                "files": file_details if file_details else files,
                "replicates": replicates,
                "references": [r.get("PMID", r) if isinstance(r, dict) else r for r in data.get("references", [])],
            })

        # ---- SCREEN cCREs -------------------------------------------------
        elif name == "screen_ccres":
            query = arguments["query"]
            assembly = arguments.get("assembly", "GRCh38")
            biosample = arguments.get("biosample")
            element_type = arguments.get("element_type")
            max_results = arguments.get("max_results", 10)

            params: dict = {"assembly": assembly}

            # Detect query type: coordinates vs accession vs gene name
            if ":" in query and "-" in query:
                # Genomic coordinates: chr1:10000-20000
                parts = query.replace(",", "").split(":")
                chrom = parts[0]
                start_end = parts[1].split("-")
                params["chrom"] = chrom
                params["start"] = int(start_end[0])
                params["end"] = int(start_end[1])
                endpoint = "/search"
            elif query.startswith("EH"):
                # SCREEN accession
                params["accession"] = query
                endpoint = "/search"
            else:
                # Gene name
                params["gene"] = query
                endpoint = "/search"

            if biosample:
                params["cellType"] = biosample
            if element_type:
                params["element_type"] = element_type

            resp = await screen.get(endpoint, params=params)
            data = resp.json()

            ccres = []
            items = data if isinstance(data, list) else data.get("data", data.get("ccres", data.get("results", [])))
            if isinstance(items, list):
                for item in items[:max_results]:
                    ccres.append({
                        "accession": item.get("accession", item.get("rDHS", "")),
                        "chromosome": item.get("chrom", item.get("chromosome", "")),
                        "start": item.get("start", ""),
                        "end": item.get("end", ""),
                        "element_type": item.get("group", item.get("element_type", item.get("ccre_group", ""))),
                        "dnase_zscore": item.get("dnase_zscore", item.get("dnase", "")),
                        "h3k4me3_zscore": item.get("h3k4me3_zscore", item.get("h3k4me3", "")),
                        "h3k27ac_zscore": item.get("h3k27ac_zscore", item.get("h3k27ac", "")),
                        "ctcf_zscore": item.get("ctcf_zscore", item.get("ctcf", "")),
                        "linked_genes": item.get("linked_genes", item.get("genesallpc", {}).get("all", [])) if isinstance(item.get("linked_genes", item.get("genesallpc")), (dict, list)) else [],
                    })

            return text_result({
                "ccres": ccres,
                "count": len(ccres),
                "assembly": assembly,
                "query": query,
            })

        # ---- JASPAR Matrix -------------------------------------------------
        elif name == "jaspar_matrix":
            query = arguments["query"]
            collection = arguments.get("collection", "CORE")
            tax_group = arguments.get("tax_group", "vertebrates")

            # If query looks like a matrix ID (e.g. MA0139.1), fetch directly
            if query.startswith("MA") and "." in query:
                resp = await jaspar.get(f"/matrix/{query}/")
                data = resp.json()
                return text_result({
                    "matrix_id": data.get("matrix_id", ""),
                    "name": data.get("name", ""),
                    "collection": data.get("collection", ""),
                    "tax_group": data.get("tax_group", []),
                    "class": data.get("class", []),
                    "family": data.get("family", []),
                    "species": data.get("species", []),
                    "pfm": data.get("pfm", {}),
                    "sequence_logo": data.get("sequence_logo", ""),
                    "uniprot_ids": data.get("uniprot_ids", []),
                })
            else:
                # Search by TF name
                resp = await jaspar.get("/matrix/", params={
                    "name": query,
                    "collection": collection,
                    "tax_group": tax_group,
                    "page_size": 10,
                })
                data = resp.json()
                results_list = data.get("results", [])
                matrices = []
                for m in results_list:
                    matrices.append({
                        "matrix_id": m.get("matrix_id", ""),
                        "name": m.get("name", ""),
                        "collection": m.get("collection", ""),
                        "tax_group": m.get("tax_group", []),
                        "class": m.get("class", []),
                        "family": m.get("family", []),
                        "species": m.get("species", []),
                        "sequence_logo": m.get("sequence_logo", ""),
                    })
                return text_result({
                    "matrices": matrices,
                    "count": data.get("count", len(matrices)),
                    "query": query,
                })

        # ---- JASPAR Scan ---------------------------------------------------
        elif name == "jaspar_scan":
            sequence = arguments["sequence"].upper().strip()
            matrix_id = arguments.get("matrix_id")
            threshold = arguments.get("threshold", 0.8)

            # Validate sequence
            if not all(c in "ACGTN" for c in sequence):
                return error_result("Invalid DNA sequence: only A, C, G, T, N characters are allowed")

            if len(sequence) < 10:
                return error_result("Sequence too short: minimum 10 nucleotides required for scanning")

            # Convert threshold from relative (0-1) to percentage string for API
            threshold_pct = f"{threshold * 100:.1f}%"

            if matrix_id:
                # Scan with specific matrix
                resp = await jaspar.post(
                    f"/matrix/{matrix_id}/scan",
                    json={
                        "sequence": sequence,
                        "rel_score_threshold": threshold,
                    },
                )
            else:
                # Scan against all CORE vertebrate profiles - use top matrices
                resp = await jaspar.post(
                    "/scan",
                    json={
                        "sequence": sequence,
                        "rel_score_threshold": threshold,
                        "collection": "CORE",
                        "tax_group": "vertebrates",
                    },
                )

            data = resp.json()

            sites = []
            # Handle various response formats
            scan_results = data if isinstance(data, list) else data.get("results", data.get("sites", []))
            if isinstance(scan_results, list):
                for site in scan_results[:50]:
                    sites.append({
                        "matrix_id": site.get("matrix_id", site.get("profile_id", "")),
                        "tf_name": site.get("name", site.get("tf_name", "")),
                        "start": site.get("start", ""),
                        "end": site.get("end", ""),
                        "strand": site.get("strand", ""),
                        "score": site.get("score", site.get("rel_score", "")),
                        "sequence": site.get("sequence", site.get("matched_sequence", "")),
                    })
            elif isinstance(scan_results, dict):
                # Response keyed by matrix ID
                for mid, hits in scan_results.items():
                    if isinstance(hits, list):
                        for site in hits:
                            sites.append({
                                "matrix_id": mid,
                                "tf_name": site.get("name", site.get("tf_name", "")),
                                "start": site.get("start", ""),
                                "end": site.get("end", ""),
                                "strand": site.get("strand", ""),
                                "score": site.get("score", site.get("rel_score", "")),
                                "sequence": site.get("sequence", site.get("matched_sequence", "")),
                            })

            return text_result({
                "sites": sites,
                "count": len(sites),
                "sequence_length": len(sequence),
                "threshold": threshold,
            })

        # ---- ReMap Search --------------------------------------------------
        elif name == "remap_search":
            tf_name = arguments.get("tf_name")
            region = arguments.get("region")
            cell_type = arguments.get("cell_type")
            species = arguments.get("species", "Homo sapiens")
            max_results = arguments.get("max_results", 20)

            if not tf_name and not region:
                return error_result("At least one of 'tf_name' or 'region' is required")

            params: dict = {"species": species}
            if tf_name:
                params["tf"] = tf_name
            if cell_type:
                params["biotype"] = cell_type

            if region:
                # Parse genomic coordinates for region query
                params["region"] = region
                resp = await remap.get("/search/region", params=params)
            else:
                resp = await remap.get("/search/tf", params=params)

            data = resp.json()
            entries = data if isinstance(data, list) else data.get("results", data.get("peaks", data.get("data", [])))

            results = []
            if isinstance(entries, list):
                for entry in entries[:max_results]:
                    results.append({
                        "tf_name": entry.get("tf", entry.get("transcription_factor", tf_name or "")),
                        "cell_type": entry.get("biotype", entry.get("cell_type", "")),
                        "experiment_id": entry.get("experiment_id", entry.get("id", "")),
                        "chromosome": entry.get("chrom", entry.get("chromosome", "")),
                        "start": entry.get("start", ""),
                        "end": entry.get("end", ""),
                        "score": entry.get("score", ""),
                        "source": entry.get("source", entry.get("study", "")),
                    })

            return text_result({
                "results": results,
                "count": len(results),
                "query": {"tf_name": tf_name, "region": region, "cell_type": cell_type},
            })

        # ---- miRTarBase Search ---------------------------------------------
        elif name == "mirtarbase_search":
            mirna = arguments.get("mirna")
            target_gene = arguments.get("target_gene")
            species = arguments.get("species", "Homo sapiens")
            strong_only = arguments.get("strong_evidence_only", False)
            max_results = arguments.get("max_results", 20)

            if not mirna and not target_gene:
                return error_result("At least one of 'mirna' or 'target_gene' is required")

            params: dict = {"species": species}
            if mirna:
                params["miRNA"] = mirna
            if target_gene:
                params["target"] = target_gene
            if strong_only:
                params["strong_evidence"] = "true"

            resp = await mirtarbase.get("/search", params=params)
            data = resp.json()

            interactions_raw = data if isinstance(data, list) else data.get("results", data.get("interactions", data.get("data", [])))
            interactions = []
            if isinstance(interactions_raw, list):
                for item in interactions_raw[:max_results]:
                    interactions.append({
                        "mirtarbase_id": item.get("mirtarbase_id", item.get("id", "")),
                        "mirna": item.get("miRNA", item.get("mirna", "")),
                        "target_gene": item.get("target_gene", item.get("gene_symbol", "")),
                        "target_gene_id": item.get("target_gene_id", item.get("entrez_id", "")),
                        "species": item.get("species", ""),
                        "experiments": item.get("experiments", item.get("evidence", "")),
                        "support_type": item.get("support_type", item.get("validation_type", "")),
                        "references": item.get("references", item.get("pmid", [])),
                    })

            return text_result({
                "interactions": interactions,
                "count": len(interactions),
                "query": {"mirna": mirna, "target_gene": target_gene},
            })

        # ---- miRDB Predict -------------------------------------------------
        elif name == "mirdb_predict":
            mirna = arguments.get("mirna")
            target_gene = arguments.get("target_gene")
            species = arguments.get("species", "human")
            score_threshold = arguments.get("score_threshold", 80)
            max_results = arguments.get("max_results", 20)

            if not mirna and not target_gene:
                return error_result("At least one of 'mirna' or 'target_gene' is required")

            params: dict = {"species": species}
            if mirna:
                params["mirna"] = mirna
            if target_gene:
                params["target"] = target_gene
            if score_threshold:
                params["score_cutoff"] = score_threshold

            resp = await mirdb.get("/predict", params=params)
            data = resp.json()

            predictions_raw = data if isinstance(data, list) else data.get("results", data.get("predictions", data.get("data", [])))
            predictions = []
            if isinstance(predictions_raw, list):
                for item in predictions_raw[:max_results]:
                    score = item.get("score", item.get("target_score", 0))
                    if isinstance(score, (int, float)) and score < score_threshold:
                        continue
                    predictions.append({
                        "mirna": item.get("miRNA", item.get("mirna", mirna or "")),
                        "target_gene": item.get("target_gene", item.get("gene_symbol", "")),
                        "target_gene_id": item.get("gene_id", item.get("refseq_id", "")),
                        "score": score,
                        "seed_location": item.get("seed_location", ""),
                    })

            return text_result({
                "predictions": predictions,
                "count": len(predictions),
                "species": species,
                "score_threshold": score_threshold,
                "query": {"mirna": mirna, "target_gene": target_gene},
            })

        # ---- ENCODE Biosample ----------------------------------------------
        elif name == "encode_biosample":
            query = arguments["query"]
            organism = arguments.get("organism", "Homo sapiens")
            biosample_type = arguments.get("biosample_type")
            max_results = arguments.get("max_results", 10)

            params: dict = {
                "searchTerm": query,
                "type": "Biosample",
                "organism.scientific_name": organism,
                "limit": max_results,
                "format": "json",
            }
            if biosample_type:
                params["biosample_ontology.classification"] = biosample_type

            resp = await encode.get("/search/", params=params)
            data = resp.json()
            graph = data.get("@graph", [])

            biosamples = []
            for item in graph[:max_results]:
                biosamples.append({
                    "accession": item.get("accession", ""),
                    "summary": item.get("summary", ""),
                    "biosample_ontology": item.get("biosample_ontology", {}).get("term_name", "") if isinstance(item.get("biosample_ontology"), dict) else "",
                    "classification": item.get("biosample_ontology", {}).get("classification", "") if isinstance(item.get("biosample_ontology"), dict) else "",
                    "organism": item.get("organism", {}).get("scientific_name", "") if isinstance(item.get("organism"), dict) else "",
                    "source": item.get("source", {}).get("title", "") if isinstance(item.get("source"), dict) else "",
                    "life_stage": item.get("life_stage", ""),
                    "age": item.get("age", ""),
                    "sex": item.get("sex", ""),
                    "status": item.get("status", ""),
                })

            return text_result({
                "biosamples": biosamples,
                "count": data.get("total", len(biosamples)),
                "query": query,
            })

        # ---- ENCODE Annotation ---------------------------------------------
        elif name == "encode_annotation":
            accession = arguments.get("accession")
            query = arguments.get("query")
            annotation_type = arguments.get("annotation_type")
            max_results = arguments.get("max_results", 10)

            if accession:
                # Direct accession lookup
                resp = await encode.get(f"/annotations/{accession}/", params={"format": "json"})
                data = resp.json()
                return text_result({
                    "accession": data.get("accession", accession),
                    "description": data.get("description", ""),
                    "annotation_type": data.get("annotation_type", ""),
                    "biosample_ontology": data.get("biosample_ontology", {}).get("term_name", "") if isinstance(data.get("biosample_ontology"), dict) else "",
                    "organism": data.get("organism", {}).get("scientific_name", "") if isinstance(data.get("organism"), dict) else "",
                    "software_used": [s.get("title", s) if isinstance(s, dict) else s for s in data.get("software_used", [])],
                    "files": [f.get("@id", f) if isinstance(f, dict) else f for f in data.get("files", [])[:15]],
                    "status": data.get("status", ""),
                    "date_released": data.get("date_released", ""),
                    "references": [r.get("PMID", r) if isinstance(r, dict) else r for r in data.get("references", [])],
                })
            elif query:
                # Search annotations
                params_ann: dict = {
                    "searchTerm": query,
                    "type": "Annotation",
                    "limit": max_results,
                    "format": "json",
                }
                if annotation_type:
                    params_ann["annotation_type"] = annotation_type

                resp = await encode.get("/search/", params=params_ann)
                data = resp.json()
                graph = data.get("@graph", [])

                annotations = []
                for item in graph[:max_results]:
                    annotations.append({
                        "accession": item.get("accession", ""),
                        "description": item.get("description", ""),
                        "annotation_type": item.get("annotation_type", ""),
                        "biosample_summary": item.get("biosample_summary", ""),
                        "organism": item.get("organism", {}).get("scientific_name", "") if isinstance(item.get("organism"), dict) else "",
                        "status": item.get("status", ""),
                        "date_released": item.get("date_released", ""),
                    })

                return text_result({
                    "annotations": annotations,
                    "count": data.get("total", len(annotations)),
                    "query": query,
                })
            else:
                return error_result("At least one of 'accession' or 'query' is required")

        return error_result(f"Unknown tool: {name}")
    except Exception as exc:
        return error_result(f"{name} failed: {exc}")


if __name__ == "__main__":
    start(server)
