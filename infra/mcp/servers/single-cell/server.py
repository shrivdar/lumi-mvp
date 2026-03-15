"""Single Cell MCP Server — CZ CELLxGENE, Human Cell Atlas, CellMarker 2.0, PanglaoDB.

Covers 10 tools for single-cell transcriptomics data access: dataset discovery,
gene expression queries, cell type annotations, and marker gene lookups.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

from base_server import APIClient, Server, error_result, make_tool, start, text_result

server = Server("single-cell")

# ---------------------------------------------------------------------------
# API Clients
# ---------------------------------------------------------------------------

cellxgene = APIClient(
    base_url="https://api.cellxgene.cziscience.com",
    headers={"Content-Type": "application/json"},
    rate_limit=5,
)
hca = APIClient(
    base_url="https://service.azul.data.humancellatlas.org",
    headers={"Content-Type": "application/json"},
    rate_limit=5,
)
cellmarker = APIClient(
    base_url="http://bio-bigdata.hrbmu.edu.cn/CellMarker/api",
    rate_limit=2,
)
panglao = APIClient(
    base_url="https://panglaodb.se/api",
    rate_limit=2,
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@server.list_tools()
async def list_tools() -> list:
    return [
        make_tool(
            "cellxgene_datasets",
            "Search CZ CELLxGENE Discover for single-cell datasets by disease, tissue, organism, or free text.",
            {
                "disease": {
                    "type": "string",
                    "description": "Disease ontology term or label (e.g. 'lung adenocarcinoma', 'COVID-19')",
                },
                "tissue": {
                    "type": "string",
                    "description": "Tissue ontology term or label (e.g. 'lung', 'brain')",
                },
                "organism": {
                    "type": "string",
                    "description": "Organism (e.g. 'Homo sapiens', 'Mus musculus')",
                    "default": "Homo sapiens",
                },
                "gene": {
                    "type": "string",
                    "description": "Gene symbol to filter datasets that assayed this gene",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max datasets to return",
                    "default": 10,
                },
            },
            required=[],
        ),
        make_tool(
            "cellxgene_gene_expression",
            "Query gene expression across cell types in CZ CELLxGENE Census. Returns mean expression per cell type.",
            {
                "gene": {
                    "type": "string",
                    "description": "Gene symbol (e.g. 'CD8A', 'FOXP3', 'EPCAM')",
                },
                "organism": {
                    "type": "string",
                    "description": "Organism name",
                    "default": "Homo sapiens",
                },
                "tissue": {
                    "type": "string",
                    "description": "Tissue to filter by (optional)",
                },
                "dataset_id": {
                    "type": "string",
                    "description": "Specific dataset ID to query within",
                },
            },
            required=["gene"],
        ),
        make_tool(
            "cellxgene_cell_types",
            "Get cell type annotations and counts from CZ CELLxGENE for a dataset or tissue.",
            {
                "dataset_id": {
                    "type": "string",
                    "description": "CELLxGENE dataset ID",
                },
                "tissue": {
                    "type": "string",
                    "description": "Tissue to get cell types for",
                },
                "organism": {
                    "type": "string",
                    "description": "Organism",
                    "default": "Homo sapiens",
                },
            },
            required=[],
        ),
        make_tool(
            "hca_projects",
            "Search Human Cell Atlas Data Coordination Platform for projects by keyword, organ, or disease.",
            {
                "query": {
                    "type": "string",
                    "description": "Free-text search query",
                },
                "organ": {
                    "type": "string",
                    "description": "Organ filter (e.g. 'brain', 'heart', 'lung')",
                },
                "disease": {
                    "type": "string",
                    "description": "Disease filter",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max projects to return",
                    "default": 10,
                },
            },
            required=[],
        ),
        make_tool(
            "hca_samples",
            "Get sample and specimen information from an HCA project.",
            {
                "project_id": {
                    "type": "string",
                    "description": "HCA project UUID",
                },
                "organ": {
                    "type": "string",
                    "description": "Filter samples by organ",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max samples to return",
                    "default": 25,
                },
            },
            required=["project_id"],
        ),
        make_tool(
            "cellmarker_search",
            "Search CellMarker 2.0 database for cell type markers by gene, cell type, tissue, or cancer type.",
            {
                "query": {
                    "type": "string",
                    "description": "Search term — gene symbol, cell type name, tissue, or cancer type",
                },
                "species": {
                    "type": "string",
                    "description": "Species filter",
                    "default": "Human",
                },
                "query_type": {
                    "type": "string",
                    "description": "Type of query: 'gene', 'cell_type', 'tissue', 'cancer'",
                    "default": "gene",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return",
                    "default": 20,
                },
            },
            required=["query"],
        ),
        make_tool(
            "cellmarker_cell_type",
            "Get all known markers for a specific cell type from CellMarker 2.0.",
            {
                "cell_type": {
                    "type": "string",
                    "description": "Cell type name (e.g. 'T cell', 'Macrophage', 'Fibroblast')",
                },
                "species": {
                    "type": "string",
                    "description": "Species filter",
                    "default": "Human",
                },
                "tissue": {
                    "type": "string",
                    "description": "Tissue context for markers (optional)",
                },
            },
            required=["cell_type"],
        ),
        make_tool(
            "panglao_markers",
            "Get cell type marker genes from PanglaoDB for a given cell type.",
            {
                "cell_type": {
                    "type": "string",
                    "description": "Cell type name (e.g. 'T cells', 'B cells', 'Neurons')",
                },
                "species": {
                    "type": "string",
                    "description": "Species: 'Hs' (human), 'Mm' (mouse), or 'Hs Mm' (both)",
                    "default": "Hs",
                },
            },
            required=["cell_type"],
        ),
        make_tool(
            "panglao_cell_types",
            "Search PanglaoDB for cell types associated with a given tissue or organ.",
            {
                "tissue": {
                    "type": "string",
                    "description": "Tissue or organ name (e.g. 'Brain', 'Lung', 'Liver')",
                },
                "species": {
                    "type": "string",
                    "description": "Species: 'Hs' (human), 'Mm' (mouse), or 'Hs Mm' (both)",
                    "default": "Hs",
                },
            },
            required=["tissue"],
        ),
        make_tool(
            "tabula_sapiens",
            "Query Tabula Sapiens reference atlas for cell type information by organ or cell type.",
            {
                "organ": {
                    "type": "string",
                    "description": "Organ to query (e.g. 'Lung', 'Heart', 'Liver', 'Blood')",
                },
                "cell_type": {
                    "type": "string",
                    "description": "Cell type to look up across organs",
                },
                "gene": {
                    "type": "string",
                    "description": "Gene to query expression for across Tabula Sapiens cell types",
                },
            },
            required=[],
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    try:
        # ---------------------------------------------------------------
        # CZ CELLxGENE Discover
        # ---------------------------------------------------------------
        if name == "cellxgene_datasets":
            organism = arguments.get("organism", "Homo sapiens")
            max_results = arguments.get("max_results", 10)

            # Build filter for the /curation/v1/collections endpoint
            params: dict = {}
            resp = await cellxgene.get("/curation/v1/collections", params=params)
            collections = resp.json()

            # Client-side filtering since the collections API returns all
            results = []
            disease_filter = arguments.get("disease", "").lower()
            tissue_filter = arguments.get("tissue", "").lower()
            gene_filter = arguments.get("gene", "").lower()

            for collection in collections:
                datasets = collection.get("datasets", [])
                for ds in datasets:
                    # Filter by organism
                    ds_organisms = [
                        o.get("label", "").lower()
                        for o in ds.get("organism", [])
                    ]
                    if organism.lower() not in ds_organisms:
                        continue
                    # Filter by disease
                    if disease_filter:
                        ds_diseases = [
                            d.get("label", "").lower()
                            for d in ds.get("disease", [])
                        ]
                        if not any(disease_filter in d for d in ds_diseases):
                            continue
                    # Filter by tissue
                    if tissue_filter:
                        ds_tissues = [
                            t.get("label", "").lower()
                            for t in ds.get("tissue", [])
                        ]
                        if not any(tissue_filter in t for t in ds_tissues):
                            continue

                    results.append({
                        "dataset_id": ds.get("dataset_id", ""),
                        "collection_id": collection.get("collection_id", ""),
                        "title": collection.get("name", ""),
                        "description": collection.get("description", "")[:300],
                        "disease": [d.get("label", "") for d in ds.get("disease", [])],
                        "tissue": [t.get("label", "") for t in ds.get("tissue", [])],
                        "organism": [o.get("label", "") for o in ds.get("organism", [])],
                        "assay": [a.get("label", "") for a in ds.get("assay", [])],
                        "cell_count": ds.get("cell_count", 0),
                        "donor_id": ds.get("donor_id", ""),
                    })

                    if len(results) >= max_results:
                        break
                if len(results) >= max_results:
                    break

            return text_result({"datasets": results, "count": len(results)})

        elif name == "cellxgene_gene_expression":
            gene = arguments["gene"]
            organism = arguments.get("organism", "Homo sapiens")
            tissue = arguments.get("tissue")
            dataset_id = arguments.get("dataset_id")

            # Use the CELLxGENE Census API / wmg (where's my gene) endpoint
            payload: dict = {
                "filter": {
                    "gene_ontology_term_ids": [],
                    "organism_ontology_term_id": (
                        "NCBITaxon:9606" if "sapiens" in organism.lower() else "NCBITaxon:10090"
                    ),
                },
                "is_rollup": True,
                "compare": "cell_type",
            }

            # Search for the gene via the gene info endpoint first
            gene_resp = await cellxgene.get(
                "/wmg/v2/gene_info",
                params={"gene": gene},
            )
            gene_data = gene_resp.json()
            gene_terms = gene_data if isinstance(gene_data, list) else gene_data.get("genes", [])

            if not gene_terms:
                return text_result({
                    "gene": gene,
                    "message": f"Gene '{gene}' not found in CELLxGENE. Check the symbol.",
                    "expression": [],
                })

            # Use the first matched gene term
            gene_term_id = (
                gene_terms[0].get("gene_ontology_term_id", "")
                if isinstance(gene_terms[0], dict)
                else gene_terms[0]
            )

            # Query expression via the marker genes or wmg primary filter endpoint
            filter_params: dict = {
                "gene_ontology_term_ids": gene_term_id,
                "organism_ontology_term_id": (
                    "NCBITaxon:9606" if "sapiens" in organism.lower() else "NCBITaxon:10090"
                ),
            }
            if tissue:
                filter_params["tissue_ontology_term_ids"] = tissue
            if dataset_id:
                filter_params["dataset_ids"] = dataset_id

            expr_resp = await cellxgene.get("/wmg/v2/query", params=filter_params)
            expr_data = expr_resp.json()

            # Parse expression results
            expression_results = []
            snapshot_data = expr_data.get("snapshot_id") or expr_data
            term_id_labels = expr_data.get("term_id_labels", {})

            if isinstance(expr_data, dict) and "expression_summary" in expr_data:
                for entry in expr_data["expression_summary"][:50]:
                    expression_results.append({
                        "cell_type": entry.get("cell_type_ontology_term_id", ""),
                        "cell_type_label": entry.get("label", entry.get("cell_type", "")),
                        "mean_expression": entry.get("me", entry.get("mean", 0)),
                        "fraction_expressing": entry.get("pc", entry.get("pct", 0)),
                        "n_cells": entry.get("n", 0),
                    })
            else:
                # Return the raw structure if format differs
                expression_results = expr_data

            return text_result({
                "gene": gene,
                "gene_ontology_term_id": gene_term_id,
                "organism": organism,
                "expression": expression_results,
            })

        elif name == "cellxgene_cell_types":
            dataset_id = arguments.get("dataset_id")
            tissue = arguments.get("tissue")
            organism = arguments.get("organism", "Homo sapiens")

            organism_id = (
                "NCBITaxon:9606" if "sapiens" in organism.lower() else "NCBITaxon:10090"
            )

            if dataset_id:
                # Fetch dataset-specific metadata
                resp = await cellxgene.get(f"/curation/v1/datasets/{dataset_id}")
                ds = resp.json()
                cell_types = []
                for ct in ds.get("cell_type", []):
                    cell_types.append({
                        "cell_type_ontology_term_id": ct.get("ontology_term_id", ""),
                        "label": ct.get("label", ""),
                    })
                return text_result({
                    "dataset_id": dataset_id,
                    "cell_types": cell_types,
                    "cell_count": ds.get("cell_count", 0),
                    "tissue": [t.get("label", "") for t in ds.get("tissue", [])],
                })

            # Query cell types across the corpus filtered by tissue
            params = {"organism_ontology_term_id": organism_id}
            if tissue:
                params["tissue"] = tissue

            resp = await cellxgene.get("/wmg/v2/cell_types", params=params)
            data = resp.json()

            cell_types = []
            items = data if isinstance(data, list) else data.get("cell_types", [])
            for ct in items[:100]:
                if isinstance(ct, dict):
                    cell_types.append({
                        "cell_type_ontology_term_id": ct.get("cell_type_ontology_term_id", ct.get("id", "")),
                        "label": ct.get("label", ct.get("name", "")),
                        "count": ct.get("total_count", ct.get("count", 0)),
                    })
                elif isinstance(ct, str):
                    cell_types.append({"label": ct})

            return text_result({
                "tissue": tissue or "all",
                "organism": organism,
                "cell_types": cell_types,
                "count": len(cell_types),
            })

        # ---------------------------------------------------------------
        # Human Cell Atlas
        # ---------------------------------------------------------------
        elif name == "hca_projects":
            query = arguments.get("query", "")
            organ = arguments.get("organ", "")
            disease = arguments.get("disease", "")
            max_results = arguments.get("max_results", 10)

            # Build Azul catalog search
            params: dict = {"catalog": "dcp2", "size": max_results}
            filters: dict = {}
            if organ:
                filters["organ"] = {"is": [organ]}
            if disease:
                filters["disease"] = {"is": [disease]}
            if filters:
                import json as _json
                params["filters"] = _json.dumps(filters)

            search_url = "/index/projects"
            if query:
                params["filters"] = params.get("filters", "{}").rstrip("}")
                if params["filters"] != "{":
                    params["filters"] += ","
                else:
                    params["filters"] = "{"
                params["filters"] += f'"projectTitle":{{"is":["{query}"]}}}}'

            resp = await hca.get(search_url, params=params)
            data = resp.json()

            projects = []
            hits = data.get("hits", [])
            for hit in hits[:max_results]:
                proj_list = hit.get("projects", [{}])
                proj = proj_list[0] if proj_list else {}
                samples = hit.get("samples", [{}])
                sample = samples[0] if samples else {}
                protocols = hit.get("protocols", [{}])
                protocol = protocols[0] if protocols else {}

                projects.append({
                    "project_id": proj.get("projectId", ""),
                    "title": proj.get("projectTitle", ""),
                    "description": proj.get("projectDescription", "")[:300],
                    "laboratory": proj.get("laboratory", []),
                    "organs": sample.get("organ", []),
                    "organ_parts": sample.get("organPart", []),
                    "diseases": sample.get("disease", []),
                    "species": sample.get("species", []),
                    "library_construction": protocol.get("libraryConstructionApproach", []),
                    "paired_end": protocol.get("pairedEnd", []),
                    "cell_count": hit.get("cellSuspensions", [{}])[0].get("totalCells", 0)
                    if hit.get("cellSuspensions")
                    else 0,
                })

            return text_result({"projects": projects, "count": len(projects)})

        elif name == "hca_samples":
            project_id = arguments["project_id"]
            organ = arguments.get("organ", "")
            max_results = arguments.get("max_results", 25)

            import json as _json

            filters: dict = {"projectId": {"is": [project_id]}}
            if organ:
                filters["organ"] = {"is": [organ]}

            params = {
                "catalog": "dcp2",
                "size": max_results,
                "filters": _json.dumps(filters),
            }

            resp = await hca.get("/index/samples", params=params)
            data = resp.json()

            samples = []
            for hit in data.get("hits", [])[:max_results]:
                sample_list = hit.get("samples", [{}])
                sample = sample_list[0] if sample_list else {}
                donor_list = hit.get("donorOrganisms", [{}])
                donor = donor_list[0] if donor_list else {}

                samples.append({
                    "sample_id": sample.get("id", [""])[0] if isinstance(sample.get("id"), list) else sample.get("id", ""),
                    "organ": sample.get("organ", []),
                    "organ_part": sample.get("organPart", []),
                    "disease": sample.get("disease", []),
                    "preservation_method": sample.get("preservationMethod", []),
                    "source": sample.get("source", []),
                    "donor_species": donor.get("species", []),
                    "donor_sex": donor.get("biologicalSex", []),
                    "donor_age": donor.get("donorAge", []),
                    "donor_disease": donor.get("disease", []),
                })

            return text_result({
                "project_id": project_id,
                "samples": samples,
                "count": len(samples),
            })

        # ---------------------------------------------------------------
        # CellMarker 2.0
        # ---------------------------------------------------------------
        elif name == "cellmarker_search":
            query = arguments["query"]
            species = arguments.get("species", "Human")
            query_type = arguments.get("query_type", "gene")
            max_results = arguments.get("max_results", 20)

            # CellMarker 2.0 API search
            params = {
                "species": species,
                "keyword": query,
                "type": query_type,
                "page": 1,
                "limit": max_results,
            }
            resp = await cellmarker.get("/search", params=params)
            data = resp.json()

            markers = []
            items = data if isinstance(data, list) else data.get("data", data.get("results", []))
            for item in items[:max_results]:
                markers.append({
                    "cell_name": item.get("cell_name", item.get("cellName", "")),
                    "cell_type": item.get("cell_type", item.get("cellType", "")),
                    "tissue_type": item.get("tissue_type", item.get("tissueType", "")),
                    "cancer_type": item.get("cancer_type", item.get("cancerType", "")),
                    "marker": item.get("marker", item.get("Symbol", item.get("gene_symbol", ""))),
                    "species": item.get("species", species),
                    "source": item.get("source", item.get("pmid", "")),
                    "evidence": item.get("evidence", item.get("technology", "")),
                })

            return text_result({"query": query, "markers": markers, "count": len(markers)})

        elif name == "cellmarker_cell_type":
            cell_type = arguments["cell_type"]
            species = arguments.get("species", "Human")
            tissue = arguments.get("tissue", "")

            params: dict = {
                "species": species,
                "keyword": cell_type,
                "type": "cell_type",
                "page": 1,
                "limit": 50,
            }
            if tissue:
                params["tissue"] = tissue

            resp = await cellmarker.get("/search", params=params)
            data = resp.json()

            items = data if isinstance(data, list) else data.get("data", data.get("results", []))

            # Aggregate unique markers
            marker_genes: dict[str, dict] = {}
            for item in items:
                gene = item.get("marker", item.get("Symbol", item.get("gene_symbol", "")))
                if not gene:
                    continue
                if gene not in marker_genes:
                    marker_genes[gene] = {
                        "gene": gene,
                        "tissues": [],
                        "cancer_types": [],
                        "sources": [],
                        "count": 0,
                    }
                tissue_val = item.get("tissue_type", item.get("tissueType", ""))
                cancer_val = item.get("cancer_type", item.get("cancerType", ""))
                source_val = item.get("source", item.get("pmid", ""))
                if tissue_val and tissue_val not in marker_genes[gene]["tissues"]:
                    marker_genes[gene]["tissues"].append(tissue_val)
                if cancer_val and cancer_val not in marker_genes[gene]["cancer_types"]:
                    marker_genes[gene]["cancer_types"].append(cancer_val)
                if source_val and source_val not in marker_genes[gene]["sources"]:
                    marker_genes[gene]["sources"].append(str(source_val))
                marker_genes[gene]["count"] += 1

            # Sort by evidence count
            sorted_markers = sorted(marker_genes.values(), key=lambda x: x["count"], reverse=True)

            return text_result({
                "cell_type": cell_type,
                "species": species,
                "tissue": tissue or "all",
                "markers": sorted_markers,
                "count": len(sorted_markers),
            })

        # ---------------------------------------------------------------
        # PanglaoDB
        # ---------------------------------------------------------------
        elif name == "panglao_markers":
            cell_type = arguments["cell_type"]
            species = arguments.get("species", "Hs")

            params = {
                "cell_type": cell_type,
                "species": species,
            }
            resp = await panglao.get("/markers", params=params)
            data = resp.json()

            markers = []
            items = data if isinstance(data, list) else data.get("markers", data.get("data", []))
            for item in items:
                markers.append({
                    "gene_symbol": item.get("official gene symbol", item.get("gene_symbol", item.get("symbol", ""))),
                    "cell_type": item.get("cell type", item.get("cell_type", cell_type)),
                    "organ": item.get("organ", item.get("tissue", "")),
                    "species": item.get("species", species),
                    "ubiquitousness": item.get("ubiquitousness index", item.get("ubiquitousness", "")),
                    "sensitivity": item.get("sensitivity_human", item.get("sensitivity", "")),
                    "specificity": item.get("specificity_human", item.get("specificity", "")),
                    "canonical_marker": item.get("canonical marker", item.get("canonical", "")),
                })

            return text_result({
                "cell_type": cell_type,
                "species": species,
                "markers": markers,
                "count": len(markers),
            })

        elif name == "panglao_cell_types":
            tissue = arguments["tissue"]
            species = arguments.get("species", "Hs")

            params = {
                "tissue": tissue,
                "species": species,
            }
            resp = await panglao.get("/cell_types", params=params)
            data = resp.json()

            cell_types = []
            items = data if isinstance(data, list) else data.get("cell_types", data.get("data", []))
            for item in items:
                if isinstance(item, dict):
                    cell_types.append({
                        "cell_type": item.get("cell type", item.get("cell_type", item.get("name", ""))),
                        "organ": item.get("organ", tissue),
                        "species": item.get("species", species),
                        "marker_count": item.get("marker_count", item.get("n_markers", "")),
                        "germ_layer": item.get("germ layer", item.get("germ_layer", "")),
                    })
                elif isinstance(item, str):
                    cell_types.append({"cell_type": item, "organ": tissue})

            return text_result({
                "tissue": tissue,
                "species": species,
                "cell_types": cell_types,
                "count": len(cell_types),
            })

        # ---------------------------------------------------------------
        # Tabula Sapiens
        # ---------------------------------------------------------------
        elif name == "tabula_sapiens":
            organ = arguments.get("organ", "")
            cell_type = arguments.get("cell_type", "")
            gene = arguments.get("gene", "")

            if not organ and not cell_type and not gene:
                return error_result(
                    "Provide at least one of: organ, cell_type, or gene."
                )

            # Tabula Sapiens data is accessible via CELLxGENE — query the
            # known Tabula Sapiens collection.
            TS_COLLECTION_ID = "e5f58829-1a66-40b5-a624-9046778e74f5"

            resp = await cellxgene.get(f"/curation/v1/collections/{TS_COLLECTION_ID}")
            collection = resp.json()

            datasets = collection.get("datasets", [])
            results = []

            for ds in datasets:
                ds_organs = [t.get("label", "").lower() for t in ds.get("tissue", [])]
                ds_cell_types = [ct.get("label", "").lower() for ct in ds.get("cell_type", [])]

                # Apply organ filter
                if organ and not any(organ.lower() in o for o in ds_organs):
                    continue

                # Collect cell type info
                cell_type_list = [
                    {
                        "ontology_term_id": ct.get("ontology_term_id", ""),
                        "label": ct.get("label", ""),
                    }
                    for ct in ds.get("cell_type", [])
                ]

                # Apply cell type filter
                if cell_type:
                    cell_type_list = [
                        ct
                        for ct in cell_type_list
                        if cell_type.lower() in ct.get("label", "").lower()
                    ]
                    if not cell_type_list:
                        continue

                results.append({
                    "dataset_id": ds.get("dataset_id", ""),
                    "tissue": [t.get("label", "") for t in ds.get("tissue", [])],
                    "assay": [a.get("label", "") for a in ds.get("assay", [])],
                    "cell_count": ds.get("cell_count", 0),
                    "cell_types": cell_type_list[:50],
                    "cell_type_count": len(ds.get("cell_type", [])),
                })

            # If a gene was requested, also query expression
            gene_expression = None
            if gene and results:
                first_ds = results[0].get("dataset_id", "")
                try:
                    expr_resp = await cellxgene.get(
                        "/wmg/v2/gene_info",
                        params={"gene": gene},
                    )
                    gene_expression = {
                        "gene": gene,
                        "info": expr_resp.json(),
                        "note": "Use cellxgene_gene_expression tool with a dataset_id for detailed per-cell-type expression.",
                    }
                except Exception:
                    gene_expression = {
                        "gene": gene,
                        "note": "Gene info lookup failed. Use cellxgene_gene_expression for detailed queries.",
                    }

            return text_result({
                "collection": "Tabula Sapiens",
                "collection_id": TS_COLLECTION_ID,
                "datasets": results,
                "count": len(results),
                "gene_expression": gene_expression,
            })

        return error_result(f"Unknown tool: {name}")
    except Exception as exc:
        return error_result(f"{name} failed: {exc}")


if __name__ == "__main__":
    start(server)
