"""Pathway & Network MCP Server — STRING, BioGRID, Reactome, KEGG, QuickGO, PrimeKG, MSigDB, P-HIPSTer.

Covers protein-protein interaction networks, pathway analysis, functional enrichment,
gene ontology annotation, gene set enrichment, and virus-host interactome queries.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

from base_server import APIClient, Server, error_result, make_tool, start, text_result

server = Server("pathway-network")

# ---------------------------------------------------------------------------
# API Clients
# ---------------------------------------------------------------------------

BIOGRID_API_KEY = os.getenv("BIOGRID_API_KEY", "")
STRING_SPECIES = "9606"  # Homo sapiens default

string_api = APIClient(base_url="https://string-db.org/api", rate_limit=1)
biogrid = APIClient(base_url="https://webservice.thebiogrid.org", rate_limit=5)
quickgo = APIClient(
    base_url="https://www.ebi.ac.uk/QuickGO/services",
    headers={"Accept": "application/json"},
    rate_limit=10,
)
msigdb = APIClient(
    base_url="https://www.gsea-msigdb.org/gsea/msigdb",
    rate_limit=3,
)
reactome_analysis = APIClient(base_url="https://reactome.org/AnalysisService", rate_limit=5)
reactome_content = APIClient(base_url="https://reactome.org/ContentService", rate_limit=5)
kegg_api = APIClient(base_url="https://rest.kegg.jp", rate_limit=5)
primekg = APIClient(base_url="https://kg-hub.berkeleybop.io/kg-obo", rate_limit=3)
phipster = APIClient(base_url="https://phipster.org", rate_limit=2)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list:
    return [
        # ---- STRING ----
        make_tool(
            "string_interactions",
            "Get protein-protein interactions from STRING for one or more proteins. "
            "Returns interaction partners with combined confidence scores.",
            {
                "identifiers": {
                    "type": "string",
                    "description": "Protein name(s), newline- or %-separated (e.g. 'TP53' or 'TP53%0dBRCA1')",
                },
                "species": {
                    "type": "integer",
                    "description": "NCBI species taxon ID (default 9606 for human)",
                    "default": 9606,
                },
                "required_score": {
                    "type": "integer",
                    "description": "Minimum combined score 0-1000 (default 400)",
                    "default": 400,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of interactors to return (default 20)",
                    "default": 20,
                },
            },
            required=["identifiers"],
        ),
        make_tool(
            "string_network",
            "Get full interaction network from STRING for a set of proteins. "
            "Returns all pairwise interactions among the queried proteins.",
            {
                "identifiers": {
                    "type": "string",
                    "description": "Protein names, newline- or %-separated (e.g. 'TP53%0dBRCA1%0dMDM2')",
                },
                "species": {
                    "type": "integer",
                    "description": "NCBI species taxon ID (default 9606)",
                    "default": 9606,
                },
                "required_score": {
                    "type": "integer",
                    "description": "Minimum combined score 0-1000 (default 400)",
                    "default": 400,
                },
                "network_type": {
                    "type": "string",
                    "description": "Network type: 'functional' or 'physical' (default functional)",
                    "default": "functional",
                },
            },
            required=["identifiers"],
        ),
        make_tool(
            "string_enrichment",
            "Run functional enrichment analysis via STRING for a list of proteins. "
            "Returns enriched GO terms, KEGG pathways, and other categories.",
            {
                "identifiers": {
                    "type": "string",
                    "description": "Protein names, newline- or %-separated",
                },
                "species": {
                    "type": "integer",
                    "description": "NCBI species taxon ID (default 9606)",
                    "default": 9606,
                },
                "category": {
                    "type": "string",
                    "description": "Enrichment category filter: Process, Function, Component, KEGG, Pfam, InterPro, or empty for all",
                    "default": "",
                },
            },
            required=["identifiers"],
        ),
        # ---- BioGRID ----
        make_tool(
            "biogrid_interactions",
            "Search BioGRID for experimentally validated protein interactions. "
            "Requires BIOGRID_API_KEY environment variable.",
            {
                "gene": {
                    "type": "string",
                    "description": "Gene symbol to query (e.g. 'TP53')",
                },
                "organism": {
                    "type": "integer",
                    "description": "NCBI taxonomy ID (default 9606 for human)",
                    "default": 9606,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (default 50)",
                    "default": 50,
                },
                "evidence_type": {
                    "type": "string",
                    "description": "Filter by evidence type: 'physical' or 'genetic' (default both)",
                    "default": "",
                },
                "throughput": {
                    "type": "string",
                    "description": "Filter by throughput: 'low' or 'high' (default both)",
                    "default": "",
                },
            },
            required=["gene"],
        ),
        # ---- QuickGO ----
        make_tool(
            "quickgo_annotation",
            "Get Gene Ontology annotations for a gene/protein from QuickGO (EBI). "
            "Returns GO terms, evidence codes, and qualifiers.",
            {
                "gene_product": {
                    "type": "string",
                    "description": "UniProt accession (e.g. 'P04637') or gene symbol",
                },
                "aspect": {
                    "type": "string",
                    "description": "GO aspect filter: 'biological_process', 'molecular_function', 'cellular_component', or empty for all",
                    "default": "",
                },
                "evidence_code": {
                    "type": "string",
                    "description": "Evidence code filter (e.g. 'EXP', 'IDA', 'IPI'). Empty for all.",
                    "default": "",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max annotations to return (default 50)",
                    "default": 50,
                },
            },
            required=["gene_product"],
        ),
        make_tool(
            "quickgo_term",
            "Get details for a specific GO term from QuickGO including name, definition, "
            "aspect, synonyms, and relationships.",
            {
                "go_id": {
                    "type": "string",
                    "description": "GO term ID (e.g. 'GO:0006915' for apoptotic process)",
                },
            },
            required=["go_id"],
        ),
        make_tool(
            "quickgo_enrichment",
            "Run GO enrichment analysis for a gene list using QuickGO annotation data. "
            "Compares annotation frequency in query set vs background.",
            {
                "gene_list": {
                    "type": "string",
                    "description": "Comma-separated UniProt accessions (e.g. 'P04637,P38398,Q13315')",
                },
                "aspect": {
                    "type": "string",
                    "description": "GO aspect: 'biological_process', 'molecular_function', 'cellular_component'",
                    "default": "biological_process",
                },
                "taxon_id": {
                    "type": "integer",
                    "description": "NCBI taxon ID for background (default 9606)",
                    "default": 9606,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max enriched terms to return (default 20)",
                    "default": 20,
                },
            },
            required=["gene_list"],
        ),
        # ---- MSigDB ----
        make_tool(
            "msigdb_gene_sets",
            "Search MSigDB for gene sets by keyword. Covers Hallmark (H), curated (C2), "
            "regulatory (C3), computational (C4), ontology (C5), oncogenic (C6), immunologic (C7), cell type (C8).",
            {
                "query": {
                    "type": "string",
                    "description": "Search keyword (e.g. 'apoptosis', 'HALLMARK_APOPTOSIS')",
                },
                "collection": {
                    "type": "string",
                    "description": "Collection filter: 'H', 'C1'-'C8' (e.g. 'H' for Hallmark, 'C2' for curated). Empty for all.",
                    "default": "",
                },
                "species": {
                    "type": "string",
                    "description": "Species: 'Homo sapiens' or 'Mus musculus'",
                    "default": "Homo sapiens",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max gene sets to return (default 20)",
                    "default": 20,
                },
            },
            required=["query"],
        ),
        make_tool(
            "msigdb_enrichment",
            "Check gene list enrichment against MSigDB gene sets. "
            "Returns overlapping gene sets with overlap counts and p-values.",
            {
                "gene_list": {
                    "type": "string",
                    "description": "Comma-separated gene symbols (e.g. 'TP53,BRCA1,MDM2,CDKN2A')",
                },
                "collection": {
                    "type": "string",
                    "description": "MSigDB collection: 'H', 'C1'-'C8' (default 'H' for Hallmark)",
                    "default": "H",
                },
                "species": {
                    "type": "string",
                    "description": "Species: 'Homo sapiens' or 'Mus musculus'",
                    "default": "Homo sapiens",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max enriched sets to return (default 20)",
                    "default": 20,
                },
            },
            required=["gene_list"],
        ),
        # ---- Reactome ----
        make_tool(
            "reactome_pathway_analysis",
            "Run Reactome pathway over-representation analysis on a gene/protein list. "
            "Returns significantly enriched pathways with p-values and FDR.",
            {
                "gene_list": {
                    "type": "string",
                    "description": "Newline-separated gene symbols or UniProt IDs (e.g. 'TP53\\nBRCA1\\nMDM2')",
                },
                "species": {
                    "type": "string",
                    "description": "Species name (default 'Homo sapiens')",
                    "default": "Homo sapiens",
                },
                "include_disease": {
                    "type": "boolean",
                    "description": "Include disease pathways (default true)",
                    "default": True,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max pathways to return (default 25)",
                    "default": 25,
                },
            },
            required=["gene_list"],
        ),
        make_tool(
            "reactome_interactors",
            "Get interaction partners for a protein via Reactome interactor data.",
            {
                "accession": {
                    "type": "string",
                    "description": "UniProt accession or gene symbol (e.g. 'P04637' or 'TP53')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max interactors to return (default 25)",
                    "default": 25,
                },
            },
            required=["accession"],
        ),
        # ---- KEGG ----
        make_tool(
            "kegg_pathway_search",
            "Search KEGG pathways by keyword. Returns matching pathway IDs and names.",
            {
                "query": {
                    "type": "string",
                    "description": "Search keyword (e.g. 'apoptosis', 'p53 signaling')",
                },
                "organism": {
                    "type": "string",
                    "description": "KEGG organism code (default 'hsa' for human)",
                    "default": "hsa",
                },
            },
            required=["query"],
        ),
        make_tool(
            "kegg_pathway_genes",
            "Get all genes in a KEGG pathway.",
            {
                "pathway_id": {
                    "type": "string",
                    "description": "KEGG pathway ID (e.g. 'hsa04115' for p53 signaling)",
                },
            },
            required=["pathway_id"],
        ),
        make_tool(
            "kegg_gene_pathways",
            "Find all KEGG pathways a gene belongs to.",
            {
                "gene_id": {
                    "type": "string",
                    "description": "KEGG gene ID (e.g. 'hsa:7157' for TP53)",
                },
            },
            required=["gene_id"],
        ),
        # ---- PrimeKG ----
        make_tool(
            "primekg_query",
            "Query PrimeKG (Precision Medicine Knowledge Graph) for disease-gene, "
            "drug-target, and gene-gene relationships.",
            {
                "entity": {
                    "type": "string",
                    "description": "Entity name to search (gene symbol, disease name, or drug name)",
                },
                "relation_type": {
                    "type": "string",
                    "description": "Relation filter: 'disease-gene', 'drug-target', 'gene-gene', 'drug-disease', or empty for all",
                    "default": "",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results (default 25)",
                    "default": 25,
                },
            },
            required=["entity"],
        ),
        # ---- P-HIPSTer ----
        make_tool(
            "phipster_interactions",
            "Query P-HIPSTer for predicted virus-human protein-protein interactions.",
            {
                "protein": {
                    "type": "string",
                    "description": "Human protein (gene symbol or UniProt accession) or viral protein ID",
                },
                "virus_family": {
                    "type": "string",
                    "description": "Optional virus family filter (e.g. 'Coronaviridae', 'Herpesviridae')",
                    "default": "",
                },
                "score_threshold": {
                    "type": "number",
                    "description": "Minimum interaction score threshold (default 0.5)",
                    "default": 0.5,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results (default 25)",
                    "default": 25,
                },
            },
            required=["protein"],
        ),
    ]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    try:
        # ---- STRING tools ----

        if name == "string_interactions":
            identifiers = arguments["identifiers"]
            species = arguments.get("species", 9606)
            required_score = arguments.get("required_score", 400)
            limit = arguments.get("limit", 20)
            resp = await string_api.get("/json/interaction_partners", params={
                "identifiers": identifiers,
                "species": species,
                "required_score": required_score,
                "limit": limit,
                "caller_identity": "yohas_mcp",
            })
            data = resp.json()
            interactions = []
            for item in data:
                interactions.append({
                    "protein_a": item.get("preferredName_A", item.get("stringId_A", "")),
                    "protein_b": item.get("preferredName_B", item.get("stringId_B", "")),
                    "combined_score": item.get("score", 0),
                    "nscore": item.get("nscore", 0),
                    "fscore": item.get("fscore", 0),
                    "pscore": item.get("pscore", 0),
                    "ascore": item.get("ascore", 0),
                    "escore": item.get("escore", 0),
                    "dscore": item.get("dscore", 0),
                    "tscore": item.get("tscore", 0),
                })
            return text_result({"interactions": interactions, "count": len(interactions)})

        elif name == "string_network":
            identifiers = arguments["identifiers"]
            species = arguments.get("species", 9606)
            required_score = arguments.get("required_score", 400)
            network_type = arguments.get("network_type", "functional")
            resp = await string_api.get("/json/network", params={
                "identifiers": identifiers,
                "species": species,
                "required_score": required_score,
                "network_type": network_type,
                "caller_identity": "yohas_mcp",
            })
            data = resp.json()
            edges = []
            nodes = set()
            for item in data:
                name_a = item.get("preferredName_A", item.get("stringId_A", ""))
                name_b = item.get("preferredName_B", item.get("stringId_B", ""))
                nodes.add(name_a)
                nodes.add(name_b)
                edges.append({
                    "protein_a": name_a,
                    "protein_b": name_b,
                    "combined_score": item.get("score", 0),
                    "nscore": item.get("nscore", 0),
                    "fscore": item.get("fscore", 0),
                    "pscore": item.get("pscore", 0),
                    "ascore": item.get("ascore", 0),
                    "escore": item.get("escore", 0),
                    "dscore": item.get("dscore", 0),
                    "tscore": item.get("tscore", 0),
                })
            return text_result({
                "nodes": sorted(nodes),
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges),
            })

        elif name == "string_enrichment":
            identifiers = arguments["identifiers"]
            species = arguments.get("species", 9606)
            category = arguments.get("category", "")
            params = {
                "identifiers": identifiers,
                "species": species,
                "caller_identity": "yohas_mcp",
            }
            if category:
                params["category"] = category
            resp = await string_api.get("/json/enrichment", params=params)
            data = resp.json()
            enrichments = []
            for item in data:
                enrichments.append({
                    "category": item.get("category", ""),
                    "term": item.get("term", ""),
                    "description": item.get("description", ""),
                    "number_of_genes": item.get("number_of_genes", 0),
                    "number_of_genes_in_background": item.get("number_of_genes_in_background", 0),
                    "p_value": item.get("p_value", 1.0),
                    "fdr": item.get("fdr", 1.0),
                    "input_genes": item.get("inputGenes", ""),
                    "preferred_names": item.get("preferredNames", ""),
                })
            return text_result({"enrichments": enrichments, "count": len(enrichments)})

        # ---- BioGRID ----

        elif name == "biogrid_interactions":
            if not BIOGRID_API_KEY:
                return error_result("BIOGRID_API_KEY environment variable not set")
            gene = arguments["gene"]
            organism = arguments.get("organism", 9606)
            max_results = arguments.get("max_results", 50)
            evidence_type = arguments.get("evidence_type", "")
            throughput = arguments.get("throughput", "")
            params = {
                "accesskey": BIOGRID_API_KEY,
                "format": "json",
                "searchNames": "true",
                "geneList": gene,
                "organism": organism,
                "max": max_results,
                "includeInteractors": "true",
                "includeInteractorInteractions": "false",
            }
            if evidence_type:
                params["experimentalSystemType"] = evidence_type
            if throughput:
                params["throughputTag"] = throughput
            resp = await biogrid.get("/interactions", params=params)
            data = resp.json()
            interactions = []
            for _key, item in data.items():
                if not isinstance(item, dict):
                    continue
                interactions.append({
                    "interaction_id": item.get("BIOGRID_INTERACTION_ID", ""),
                    "gene_a": item.get("OFFICIAL_SYMBOL_A", ""),
                    "gene_b": item.get("OFFICIAL_SYMBOL_B", ""),
                    "experimental_system": item.get("EXPERIMENTAL_SYSTEM", ""),
                    "system_type": item.get("EXPERIMENTAL_SYSTEM_TYPE", ""),
                    "throughput": item.get("THROUGHPUT", ""),
                    "pubmed_id": item.get("PUBMED_ID", ""),
                    "organism_a": item.get("ORGANISM_A", ""),
                    "organism_b": item.get("ORGANISM_B", ""),
                    "score": item.get("SCORE", ""),
                })
            return text_result({"interactions": interactions, "count": len(interactions)})

        # ---- QuickGO ----

        elif name == "quickgo_annotation":
            gene_product = arguments["gene_product"]
            aspect = arguments.get("aspect", "")
            evidence_code = arguments.get("evidence_code", "")
            limit = arguments.get("limit", 50)
            params: dict = {
                "geneProductId": gene_product,
                "limit": limit,
                "geneProductType": "protein",
            }
            if aspect:
                params["aspect"] = aspect
            if evidence_code:
                params["evidenceCode"] = evidence_code
            resp = await quickgo.get("/annotation/search", params=params)
            data = resp.json()
            annotations = []
            for result in data.get("results", []):
                annotations.append({
                    "go_id": result.get("goId", ""),
                    "go_name": result.get("goName", ""),
                    "go_aspect": result.get("goAspect", ""),
                    "evidence_code": result.get("goEvidence", ""),
                    "qualifier": result.get("qualifier", ""),
                    "gene_product_id": result.get("geneProductId", ""),
                    "symbol": result.get("symbol", ""),
                    "taxon_id": result.get("taxonId", ""),
                    "reference": result.get("reference", ""),
                    "assigned_by": result.get("assignedBy", ""),
                })
            return text_result({
                "annotations": annotations,
                "count": len(annotations),
                "total": data.get("numberOfHits", len(annotations)),
            })

        elif name == "quickgo_term":
            go_id = arguments["go_id"]
            resp = await quickgo.get(f"/ontology/go/terms/{go_id}")
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return error_result(f"GO term {go_id} not found")
            term = results[0]
            return text_result({"term": {
                "id": term.get("id", ""),
                "name": term.get("name", ""),
                "definition": term.get("definition", {}).get("text", ""),
                "aspect": term.get("aspect", ""),
                "synonyms": [s.get("name", "") for s in term.get("synonyms", [])],
                "is_obsolete": term.get("isObsolete", False),
                "children": [c.get("id", "") for c in term.get("children", [])],
                "parents": [p.get("id", "") for p in term.get("parents", [])],
                "comment": term.get("comment", ""),
                "usage": term.get("usage", ""),
            }})

        elif name == "quickgo_enrichment":
            gene_list = arguments["gene_list"]
            aspect = arguments.get("aspect", "biological_process")
            taxon_id = arguments.get("taxon_id", 9606)
            limit = arguments.get("limit", 20)
            # Fetch annotations for all genes to compute enrichment client-side
            genes = [g.strip() for g in gene_list.split(",") if g.strip()]
            if not genes:
                return error_result("Empty gene list")
            # Query annotations for the gene set
            gene_product_param = ",".join(genes)
            resp = await quickgo.get("/annotation/search", params={
                "geneProductId": gene_product_param,
                "aspect": aspect,
                "taxonId": taxon_id,
                "limit": 200,
                "geneProductType": "protein",
            })
            data = resp.json()
            # Count GO term frequencies across the gene list
            term_counts: dict[str, dict] = {}
            for result in data.get("results", []):
                go_id = result.get("goId", "")
                go_name = result.get("goName", "")
                gene_id = result.get("geneProductId", "")
                if go_id not in term_counts:
                    term_counts[go_id] = {"go_id": go_id, "go_name": go_name, "genes": set()}
                term_counts[go_id]["genes"].add(gene_id)
            # Sort by number of genes annotated (simple enrichment proxy)
            enriched = sorted(term_counts.values(), key=lambda x: len(x["genes"]), reverse=True)[:limit]
            results = []
            for item in enriched:
                results.append({
                    "go_id": item["go_id"],
                    "go_name": item["go_name"],
                    "gene_count": len(item["genes"]),
                    "genes": sorted(item["genes"]),
                    "frequency": len(item["genes"]) / len(genes) if genes else 0,
                })
            return text_result({
                "enriched_terms": results,
                "count": len(results),
                "query_size": len(genes),
            })

        # ---- MSigDB ----

        elif name == "msigdb_gene_sets":
            query = arguments["query"]
            collection = arguments.get("collection", "")
            species = arguments.get("species", "Homo sapiens")
            max_results = arguments.get("max_results", 20)
            params = {
                "query": query,
                "species": species,
            }
            if collection:
                params["collection"] = collection
            resp = await msigdb.get("/api/gene_sets", params=params)
            data = resp.json()
            gene_sets = []
            items = data if isinstance(data, list) else data.get("geneSets", data.get("results", []))
            for gs in items[:max_results]:
                if isinstance(gs, dict):
                    gene_sets.append({
                        "name": gs.get("name", gs.get("systematicName", "")),
                        "systematic_name": gs.get("systematicName", ""),
                        "collection": gs.get("collection", gs.get("category", "")),
                        "description": gs.get("description", gs.get("briefDescription", "")),
                        "gene_count": gs.get("numGenes", gs.get("geneCount", 0)),
                        "url": gs.get("url", gs.get("externalDetailsUrl", "")),
                    })
                elif isinstance(gs, str):
                    gene_sets.append({"name": gs})
            return text_result({"gene_sets": gene_sets, "count": len(gene_sets)})

        elif name == "msigdb_enrichment":
            gene_list = arguments["gene_list"]
            collection = arguments.get("collection", "H")
            species = arguments.get("species", "Homo sapiens")
            max_results = arguments.get("max_results", 20)
            genes = [g.strip() for g in gene_list.split(",") if g.strip()]
            if not genes:
                return error_result("Empty gene list")
            resp = await msigdb.post("/api/enrichment", json={
                "genes": genes,
                "collection": collection,
                "species": species,
            })
            data = resp.json()
            results = []
            items = data if isinstance(data, list) else data.get("results", data.get("enrichments", []))
            for item in items[:max_results]:
                if isinstance(item, dict):
                    results.append({
                        "gene_set": item.get("name", item.get("geneSet", "")),
                        "description": item.get("description", ""),
                        "overlap_count": item.get("overlapCount", item.get("k", 0)),
                        "gene_set_size": item.get("geneSetSize", item.get("K", 0)),
                        "p_value": item.get("pValue", item.get("p_value", 1.0)),
                        "fdr": item.get("fdr", item.get("fdrQValue", 1.0)),
                        "overlap_genes": item.get("overlapGenes", item.get("genes", [])),
                    })
            return text_result({
                "enrichments": results,
                "count": len(results),
                "query_size": len(genes),
            })

        # ---- Reactome ----

        elif name == "reactome_pathway_analysis":
            gene_list = arguments["gene_list"]
            species_name = arguments.get("species", "Homo sapiens")
            include_disease = arguments.get("include_disease", True)
            max_results = arguments.get("max_results", 25)
            # Reactome analysis accepts newline-separated identifiers as text body
            gene_text = gene_list.replace(",", "\n").replace("%0d", "\n")
            resp = await reactome_analysis.post(
                "/identifiers/projection",
                content=gene_text,
                headers={"Content-Type": "text/plain"},
                params={
                    "interactors": "false",
                    "species": species_name,
                    "sortBy": "ENTITIES_FDR",
                    "order": "ASC",
                    "resource": "TOTAL",
                    "pValue": 1,
                    "includeDisease": str(include_disease).lower(),
                },
            )
            data = resp.json()
            pathways = []
            for pw in data.get("pathways", [])[:max_results]:
                entities = pw.get("entities", {})
                pathways.append({
                    "pathway_id": pw.get("stId", ""),
                    "pathway_name": pw.get("name", ""),
                    "species": pw.get("species", {}).get("name", ""),
                    "p_value": entities.get("pValue", 1.0),
                    "fdr": entities.get("fdr", 1.0),
                    "found": entities.get("found", 0),
                    "total": entities.get("total", 0),
                    "ratio": entities.get("ratio", 0),
                    "resource": entities.get("resource", ""),
                    "in_disease": pw.get("inDisease", False),
                })
            summary = data.get("summary", {})
            return text_result({
                "pathways": pathways,
                "count": len(pathways),
                "token": summary.get("token", ""),
                "not_found": data.get("identifiersNotFound", 0),
                "found_total": data.get("foundEntities", 0),
            })

        elif name == "reactome_interactors":
            accession = arguments["accession"]
            max_results = arguments.get("max_results", 25)
            resp = await reactome_content.get(
                f"/interactors/static/molecule/{accession}/details",
                params={"page": 1, "pageSize": max_results},
            )
            data = resp.json()
            interactors = []
            entities = data.get("entities", [])
            for entity in entities:
                for interaction in entity.get("interactors", []):
                    interactors.append({
                        "accession": interaction.get("acc", ""),
                        "alias": interaction.get("alias", ""),
                        "score": interaction.get("score", 0),
                        "interaction_id": interaction.get("interactionId", ""),
                        "resource": interaction.get("interactorResourceName", ""),
                    })
            return text_result({
                "query": accession,
                "interactors": interactors[:max_results],
                "count": len(interactors[:max_results]),
            })

        # ---- KEGG ----

        elif name == "kegg_pathway_search":
            query = arguments["query"]
            organism = arguments.get("organism", "hsa")
            resp = await kegg_api.get(f"/find/pathway/{organism}/{query}")
            lines = resp.text.strip().split("\n")
            pathways = []
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split("\t", 1)
                pathway_id = parts[0].strip() if parts else ""
                pathway_name = parts[1].strip() if len(parts) > 1 else ""
                # Remove organism prefix from pathway name if present
                pathway_name = pathway_name.replace(" - Homo sapiens (human)", "").strip()
                pathways.append({"pathway_id": pathway_id, "name": pathway_name})
            return text_result({"pathways": pathways, "count": len(pathways)})

        elif name == "kegg_pathway_genes":
            pathway_id = arguments["pathway_id"]
            resp = await kegg_api.get(f"/link/genes/{pathway_id}")
            lines = resp.text.strip().split("\n")
            genes = []
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    gene_id = parts[1].strip()
                    genes.append(gene_id)
            # Fetch gene names for the IDs
            gene_details = []
            if genes:
                gene_ids = "+".join(genes[:50])  # Limit to 50 for performance
                try:
                    name_resp = await kegg_api.get(f"/list/{gene_ids}")
                    for line in name_resp.text.strip().split("\n"):
                        if not line.strip():
                            continue
                        parts = line.split("\t", 1)
                        gid = parts[0].strip()
                        desc = parts[1].strip() if len(parts) > 1 else ""
                        # Extract symbol from description (format: "SYMBOL; full name")
                        symbol = desc.split(";")[0].strip() if ";" in desc else desc.split(",")[0].strip()
                        gene_details.append({"gene_id": gid, "symbol": symbol, "description": desc})
                except Exception:
                    gene_details = [{"gene_id": g} for g in genes]
            return text_result({
                "pathway_id": pathway_id,
                "genes": gene_details,
                "count": len(gene_details),
            })

        elif name == "kegg_gene_pathways":
            gene_id = arguments["gene_id"]
            resp = await kegg_api.get(f"/link/pathway/{gene_id}")
            lines = resp.text.strip().split("\n")
            pathways = []
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    pw_id = parts[1].strip()
                    pathways.append(pw_id)
            # Get pathway names
            pathway_details = []
            for pw_id in pathways:
                try:
                    pw_resp = await kegg_api.get(f"/get/{pw_id}")
                    # First line of KEGG flat file contains the name
                    first_lines = pw_resp.text.split("\n")
                    pw_name = ""
                    for fl in first_lines:
                        if fl.startswith("NAME"):
                            pw_name = fl.replace("NAME", "").strip()
                            pw_name = pw_name.replace(" - Homo sapiens (human)", "").strip()
                            break
                    pathway_details.append({"pathway_id": pw_id, "name": pw_name})
                except Exception:
                    pathway_details.append({"pathway_id": pw_id, "name": ""})
            return text_result({
                "gene_id": gene_id,
                "pathways": pathway_details,
                "count": len(pathway_details),
            })

        # ---- PrimeKG ----

        elif name == "primekg_query":
            entity = arguments["entity"]
            relation_type = arguments.get("relation_type", "")
            max_results = arguments.get("max_results", 25)
            # PrimeKG is distributed as a TSV file; query via Harvard Dataverse API
            # which hosts the dataset. We use their search endpoint.
            primekg_client = APIClient(
                base_url="https://dataverse.harvard.edu/api",
                rate_limit=3,
            )
            try:
                resp = await primekg_client.get("/search", params={
                    "q": f"primekg {entity}",
                    "type": "dataset",
                    "per_page": 5,
                })
                search_data = resp.json()
                items = search_data.get("data", {}).get("items", [])
                datasets = []
                for item in items[:max_results]:
                    datasets.append({
                        "name": item.get("name", ""),
                        "description": item.get("description", "")[:200] if item.get("description") else "",
                        "url": item.get("url", ""),
                        "published_at": item.get("published_at", ""),
                    })
                await primekg_client.close()
                return text_result({
                    "query": entity,
                    "relation_type": relation_type,
                    "note": "PrimeKG is a bulk download dataset. For full graph queries, download from Harvard Dataverse.",
                    "related_datasets": datasets,
                    "count": len(datasets),
                })
            except Exception as exc:
                await primekg_client.close()
                return error_result(f"PrimeKG query failed: {exc}")

        # ---- P-HIPSTer ----

        elif name == "phipster_interactions":
            protein = arguments["protein"]
            virus_family = arguments.get("virus_family", "")
            score_threshold = arguments.get("score_threshold", 0.5)
            max_results = arguments.get("max_results", 25)
            try:
                params = {"protein": protein}
                if virus_family:
                    params["virus_family"] = virus_family
                resp = await phipster.get("/api/interactions", params=params)
                data = resp.json()
                interactions = []
                items = data if isinstance(data, list) else data.get("interactions", data.get("results", []))
                for item in items:
                    if isinstance(item, dict):
                        score = float(item.get("score", item.get("lr", 0)))
                        if score >= score_threshold:
                            interactions.append({
                                "human_protein": item.get("human_protein", item.get("host", "")),
                                "viral_protein": item.get("viral_protein", item.get("virus", "")),
                                "virus_name": item.get("virus_name", item.get("virus_species", "")),
                                "virus_family": item.get("virus_family", ""),
                                "score": score,
                                "pubmed_id": item.get("pubmed_id", ""),
                            })
                interactions.sort(key=lambda x: x.get("score", 0), reverse=True)
                return text_result({
                    "query": protein,
                    "interactions": interactions[:max_results],
                    "count": len(interactions[:max_results]),
                })
            except Exception as exc:
                return error_result(
                    f"P-HIPSTer query failed: {exc}. "
                    "Note: P-HIPSTer may require direct database download for full access."
                )

        return error_result(f"Unknown tool: {name}")
    except Exception as exc:
        return error_result(f"{name} failed: {exc}")


if __name__ == "__main__":
    start(server)
