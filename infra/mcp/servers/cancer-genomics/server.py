"""Cancer Genomics MCP Server — cBioPortal, COSMIC, DepMap, LINCS L1000, TxGNN.

Covers cancer-specific genomic data: somatic mutations (TCGA/MSK-IMPACT),
copy number alterations, gene dependencies, perturbation signatures, and
cancer gene census queries.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

from base_server import APIClient, Server, error_result, make_tool, start, text_result

server = Server("cancer-genomics")

# ---------------------------------------------------------------------------
# API Clients
# ---------------------------------------------------------------------------

COSMIC_API_KEY = os.getenv("COSMIC_API_KEY", "")

cbioportal = APIClient(
    base_url="https://www.cbioportal.org/api",
    headers={"Accept": "application/json"},
    rate_limit=5,
)
cosmic = APIClient(
    base_url="https://cancer.sanger.ac.uk/cosmic/api",
    headers={
        "Accept": "application/json",
        **({"Authorization": f"Bearer {COSMIC_API_KEY}"} if COSMIC_API_KEY else {}),
    },
    rate_limit=3,
)
depmap = APIClient(
    base_url="https://depmap.org/portal/api",
    headers={"Accept": "application/json"},
    rate_limit=3,
)
lincs = APIClient(
    base_url="https://maayanlab.cloud/sigcom-lincs/api",
    headers={"Accept": "application/json"},
    rate_limit=5,
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@server.list_tools()
async def list_tools() -> list:
    return [
        # -- cBioPortal -------------------------------------------------------
        make_tool(
            "cbioportal_studies",
            "List cancer genomics studies from cBioPortal (TCGA, MSK-IMPACT, etc.). "
            "Optionally filter by keyword.",
            {
                "keyword": {
                    "type": "string",
                    "description": "Optional keyword to filter studies (e.g. 'lung', 'TCGA', 'MSK')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max studies to return",
                    "default": 20,
                },
            },
            required=[],
        ),
        make_tool(
            "cbioportal_mutations",
            "Get somatic mutations for a gene in a specific cBioPortal study. "
            "Returns mutation type, protein change, and variant allele frequency.",
            {
                "study_id": {
                    "type": "string",
                    "description": "cBioPortal study ID (e.g. 'brca_tcga_pan_can_atlas_2018')",
                },
                "gene": {
                    "type": "string",
                    "description": "Hugo gene symbol (e.g. 'TP53', 'KRAS', 'BRAF')",
                },
                "molecular_profile_id": {
                    "type": "string",
                    "description": "Molecular profile ID. If omitted, defaults to '<study_id>_mutations'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max mutations to return",
                    "default": 50,
                },
            },
            required=["study_id", "gene"],
        ),
        make_tool(
            "cbioportal_cna",
            "Get copy number alterations for a gene in a cBioPortal study. "
            "Returns discrete CNA values (-2=homodel, -1=hetloss, 0=diploid, 1=gain, 2=amp).",
            {
                "study_id": {
                    "type": "string",
                    "description": "cBioPortal study ID",
                },
                "gene": {
                    "type": "string",
                    "description": "Hugo gene symbol (e.g. 'ERBB2', 'MYC')",
                },
                "molecular_profile_id": {
                    "type": "string",
                    "description": "Molecular profile ID. If omitted, defaults to '<study_id>_gistic'",
                },
            },
            required=["study_id", "gene"],
        ),
        make_tool(
            "cbioportal_clinical",
            "Get clinical data (overall survival, stage, subtype, etc.) for patients in a study.",
            {
                "study_id": {
                    "type": "string",
                    "description": "cBioPortal study ID",
                },
                "clinical_attribute": {
                    "type": "string",
                    "description": "Specific clinical attribute ID to filter (e.g. 'OS_STATUS', 'CANCER_TYPE_DETAILED'). "
                    "If omitted, returns all available clinical attributes for the study.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max patient records to return",
                    "default": 50,
                },
            },
            required=["study_id"],
        ),
        # -- COSMIC -----------------------------------------------------------
        make_tool(
            "cosmic_search",
            "Search the COSMIC database for somatic mutations associated with a gene or variant.",
            {
                "query": {
                    "type": "string",
                    "description": "Gene symbol, mutation ID, or search term (e.g. 'BRAF', 'COSV57937647')",
                },
                "search_type": {
                    "type": "string",
                    "description": "Type of search: 'gene', 'mutation', or 'tumour_site'",
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
            "cosmic_gene_census",
            "Query the COSMIC Cancer Gene Census for known cancer driver genes. "
            "Returns tier, hallmarks, tumour types, and mutation types.",
            {
                "gene": {
                    "type": "string",
                    "description": "Hugo gene symbol (e.g. 'TP53', 'EGFR'). "
                    "If omitted, returns an overview of cancer gene census entries.",
                },
                "tier": {
                    "type": "integer",
                    "description": "Filter by tier (1 or 2). Tier 1 = strong evidence, Tier 2 = emerging.",
                },
            },
            required=[],
        ),
        # -- DepMap ------------------------------------------------------------
        make_tool(
            "depmap_gene_dependency",
            "Query DepMap (Broad) for gene dependency scores (CERES/Chronos). "
            "Lower scores indicate the gene is more essential for cell survival.",
            {
                "gene": {
                    "type": "string",
                    "description": "Hugo gene symbol (e.g. 'KRAS', 'MYC')",
                },
                "cell_line": {
                    "type": "string",
                    "description": "Optional DepMap cell line ID or name to filter (e.g. 'ACH-000001')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max cell lines to return dependency scores for",
                    "default": 20,
                },
            },
            required=["gene"],
        ),
        make_tool(
            "depmap_cell_lines",
            "Search DepMap cell lines by name, lineage, or disease.",
            {
                "query": {
                    "type": "string",
                    "description": "Search term (cell line name, cancer type, or tissue)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results",
                    "default": 20,
                },
            },
            required=["query"],
        ),
        # -- LINCS L1000 ------------------------------------------------------
        make_tool(
            "lincs_l1000_signatures",
            "Query LINCS L1000 for gene expression perturbation signatures. "
            "Find how a drug, gene knockdown, or over-expression changes gene expression.",
            {
                "query": {
                    "type": "string",
                    "description": "Search query — gene symbol, perturbation name, or compound",
                },
                "pert_type": {
                    "type": "string",
                    "description": "Perturbation type: 'trt_cp' (compound), 'trt_sh' (shRNA), "
                    "'trt_oe' (overexpression), 'trt_xpr' (CRISPR)",
                },
                "cell_line": {
                    "type": "string",
                    "description": "Filter by cell line (e.g. 'A549', 'MCF7')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max signatures to return",
                    "default": 20,
                },
            },
            required=["query"],
        ),
        make_tool(
            "lincs_l1000_compounds",
            "Search LINCS L1000 for compounds/drugs and their perturbation metadata.",
            {
                "query": {
                    "type": "string",
                    "description": "Compound name or partial match (e.g. 'vorinostat', 'imatinib')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results",
                    "default": 20,
                },
            },
            required=["query"],
        ),
    ]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    try:
        # -- cBioPortal -------------------------------------------------------
        if name == "cbioportal_studies":
            keyword = arguments.get("keyword", "")
            max_results = arguments.get("max_results", 20)
            resp = await cbioportal.get("/studies", params={"projection": "SUMMARY"})
            studies = resp.json()
            if keyword:
                kw = keyword.lower()
                studies = [
                    s for s in studies
                    if kw in s.get("name", "").lower()
                    or kw in s.get("description", "").lower()
                    or kw in s.get("cancerTypeId", "").lower()
                    or kw in s.get("studyId", "").lower()
                ]
            results = []
            for s in studies[:max_results]:
                results.append({
                    "study_id": s.get("studyId", ""),
                    "name": s.get("name", ""),
                    "description": s.get("description", ""),
                    "cancer_type": s.get("cancerTypeId", ""),
                    "sample_count": s.get("allSampleCount", 0),
                    "citation": s.get("citation", ""),
                    "pmid": s.get("pmid", ""),
                })
            return text_result({"studies": results, "count": len(results), "total_available": len(studies)})

        elif name == "cbioportal_mutations":
            study_id = arguments["study_id"]
            gene = arguments["gene"]
            profile_id = arguments.get("molecular_profile_id", f"{study_id}_mutations")
            max_results = arguments.get("max_results", 50)

            # First resolve the Entrez Gene ID for the Hugo symbol
            gene_resp = await cbioportal.get(f"/genes/{gene}")
            gene_data = gene_resp.json()
            entrez_id = gene_data.get("entrezGeneId")

            # Fetch mutations for this gene in the molecular profile
            resp = await cbioportal.get(
                f"/molecular-profiles/{profile_id}/mutations",
                params={
                    "entrezGeneId": entrez_id,
                    "projection": "DETAILED",
                    "pageSize": max_results,
                    "pageNumber": 0,
                },
            )
            mutations_raw = resp.json()
            mutations = []
            for m in mutations_raw[:max_results]:
                mutations.append({
                    "sample_id": m.get("sampleId", ""),
                    "patient_id": m.get("patientId", ""),
                    "mutation_type": m.get("mutationType", ""),
                    "protein_change": m.get("proteinChange", ""),
                    "mutation_status": m.get("mutationStatus", ""),
                    "variant_type": m.get("variantType", ""),
                    "chromosome": m.get("chr", ""),
                    "start_position": m.get("startPosition"),
                    "end_position": m.get("endPosition"),
                    "ref_allele": m.get("referenceAllele", ""),
                    "var_allele": m.get("variantAllele", ""),
                    "vaf": m.get("tumorAltCount", 0) / max(m.get("tumorRefCount", 0) + m.get("tumorAltCount", 1), 1),
                    "ncbi_build": m.get("ncbiBuild", ""),
                    "keyword": m.get("keyword", ""),
                })
            return text_result({
                "gene": gene,
                "study_id": study_id,
                "molecular_profile_id": profile_id,
                "mutations": mutations,
                "count": len(mutations),
            })

        elif name == "cbioportal_cna":
            study_id = arguments["study_id"]
            gene = arguments["gene"]
            profile_id = arguments.get("molecular_profile_id", f"{study_id}_gistic")

            gene_resp = await cbioportal.get(f"/genes/{gene}")
            gene_data = gene_resp.json()
            entrez_id = gene_data.get("entrezGeneId")

            resp = await cbioportal.get(
                f"/molecular-profiles/{profile_id}/discrete-copy-number",
                params={
                    "entrezGeneId": entrez_id,
                    "projection": "DETAILED",
                    "pageSize": 100,
                    "pageNumber": 0,
                },
            )
            cna_raw = resp.json()
            cna_labels = {-2: "homozygous_deletion", -1: "heterozygous_loss", 0: "diploid", 1: "gain", 2: "amplification"}
            alterations = []
            for c in cna_raw:
                alt_val = c.get("alteration", 0)
                alterations.append({
                    "sample_id": c.get("sampleId", ""),
                    "patient_id": c.get("patientId", ""),
                    "alteration": alt_val,
                    "alteration_label": cna_labels.get(alt_val, "unknown"),
                })
            # Summarize
            summary = {}
            for a in alterations:
                label = a["alteration_label"]
                summary[label] = summary.get(label, 0) + 1
            return text_result({
                "gene": gene,
                "study_id": study_id,
                "molecular_profile_id": profile_id,
                "alterations": alterations,
                "summary": summary,
                "count": len(alterations),
            })

        elif name == "cbioportal_clinical":
            study_id = arguments["study_id"]
            clinical_attr = arguments.get("clinical_attribute")
            max_results = arguments.get("max_results", 50)

            if not clinical_attr:
                # Return available clinical attributes for the study
                resp = await cbioportal.get(f"/studies/{study_id}/clinical-attributes")
                attrs = resp.json()
                results = []
                for a in attrs:
                    results.append({
                        "attribute_id": a.get("clinicalAttributeId", ""),
                        "display_name": a.get("displayName", ""),
                        "description": a.get("description", ""),
                        "datatype": a.get("datatype", ""),
                        "patient_attribute": a.get("patientAttribute", False),
                    })
                return text_result({"study_id": study_id, "clinical_attributes": results, "count": len(results)})

            # Fetch specific clinical data
            resp = await cbioportal.get(
                f"/studies/{study_id}/clinical-data",
                params={
                    "clinicalDataType": "PATIENT",
                    "attributeId": clinical_attr,
                    "projection": "SUMMARY",
                    "pageSize": max_results,
                    "pageNumber": 0,
                },
            )
            data = resp.json()
            records = []
            for d in data[:max_results]:
                records.append({
                    "patient_id": d.get("patientId", ""),
                    "attribute_id": d.get("clinicalAttributeId", ""),
                    "value": d.get("value", ""),
                })
            return text_result({
                "study_id": study_id,
                "clinical_attribute": clinical_attr,
                "records": records,
                "count": len(records),
            })

        # -- COSMIC -----------------------------------------------------------
        elif name == "cosmic_search":
            query = arguments["query"]
            search_type = arguments.get("search_type", "gene")
            max_results = arguments.get("max_results", 20)

            if search_type == "gene":
                resp = await cosmic.get(f"/v1/genes/{query}/mutations", params={"page": 1, "per_page": max_results})
                data = resp.json()
                mutations = []
                items = data if isinstance(data, list) else data.get("data", data.get("mutations", []))
                for m in items[:max_results]:
                    mutations.append({
                        "mutation_id": m.get("id", m.get("mutation_id", "")),
                        "genomic_mutation": m.get("genomic_mutation_id", m.get("genomic_mutation", "")),
                        "cds_mutation": m.get("mutation_cds", m.get("cds_mutation", "")),
                        "aa_mutation": m.get("mutation_aa", m.get("aa_mutation", "")),
                        "primary_site": m.get("primary_site", ""),
                        "primary_histology": m.get("primary_histology", ""),
                        "fathmm_prediction": m.get("fathmm_prediction", ""),
                        "fathmm_score": m.get("fathmm_score", ""),
                        "sample_count": m.get("count", m.get("sample_count", "")),
                    })
                return text_result({"gene": query, "mutations": mutations, "count": len(mutations)})

            elif search_type == "mutation":
                resp = await cosmic.get(f"/v1/mutations/{query}")
                data = resp.json()
                return text_result({"mutation": data})

            elif search_type == "tumour_site":
                resp = await cosmic.get("/v1/tumour-sites", params={"q": query, "per_page": max_results})
                data = resp.json()
                sites = data if isinstance(data, list) else data.get("data", [])
                return text_result({"tumour_sites": sites[:max_results], "count": len(sites)})

            return error_result(f"Unknown search_type: {search_type}. Use 'gene', 'mutation', or 'tumour_site'.")

        elif name == "cosmic_gene_census":
            gene = arguments.get("gene")
            tier = arguments.get("tier")

            if gene:
                resp = await cosmic.get(f"/v1/cancer-gene-census/{gene}")
                data = resp.json()
                entry = data if isinstance(data, dict) else (data[0] if isinstance(data, list) and data else {})
                return text_result({"gene": gene, "census_entry": {
                    "gene_symbol": entry.get("gene_symbol", entry.get("Gene Symbol", gene)),
                    "tier": entry.get("tier", entry.get("Tier", "")),
                    "hallmark": entry.get("hallmark", entry.get("Hallmark", "")),
                    "role_in_cancer": entry.get("role_in_cancer", entry.get("Role in Cancer", "")),
                    "mutation_types": entry.get("mutation_types", entry.get("Mutation Types", "")),
                    "tumour_types_somatic": entry.get("tumour_types_somatic", entry.get("Tumour Types(Somatic)", "")),
                    "tumour_types_germline": entry.get("tumour_types_germline", entry.get("Tumour Types(Germline)", "")),
                    "translocation_partner": entry.get("translocation_partner", entry.get("Translocation Partner", "")),
                    "synonyms": entry.get("synonyms", entry.get("Synonyms", "")),
                }})

            # List census entries, optionally filtered by tier
            params: dict = {"per_page": 50, "page": 1}
            if tier:
                params["tier"] = tier
            resp = await cosmic.get("/v1/cancer-gene-census", params=params)
            data = resp.json()
            entries = data if isinstance(data, list) else data.get("data", [])
            results = []
            for e in entries:
                results.append({
                    "gene_symbol": e.get("gene_symbol", e.get("Gene Symbol", "")),
                    "tier": e.get("tier", e.get("Tier", "")),
                    "role_in_cancer": e.get("role_in_cancer", e.get("Role in Cancer", "")),
                })
            return text_result({"census_genes": results, "count": len(results)})

        # -- DepMap ------------------------------------------------------------
        elif name == "depmap_gene_dependency":
            gene = arguments["gene"]
            cell_line = arguments.get("cell_line")
            max_results = arguments.get("max_results", 20)

            params: dict = {"gene": gene, "limit": max_results}
            if cell_line:
                params["cell_line"] = cell_line
            resp = await depmap.get("/v1/gene-dependency", params=params)
            data = resp.json()
            items = data if isinstance(data, list) else data.get("data", data.get("dependencies", []))
            results = []
            for d in items[:max_results]:
                results.append({
                    "cell_line_id": d.get("depmap_id", d.get("cell_line_id", "")),
                    "cell_line_name": d.get("cell_line_name", d.get("cell_line_display_name", "")),
                    "lineage": d.get("lineage", d.get("primary_disease", "")),
                    "dependency_score": d.get("dependency", d.get("gene_effect", d.get("dependency_score", ""))),
                    "dataset": d.get("dataset", ""),
                })
            return text_result({
                "gene": gene,
                "dependencies": results,
                "count": len(results),
                "note": "Negative scores indicate gene essentiality; scores < -0.5 suggest strong dependency.",
            })

        elif name == "depmap_cell_lines":
            query = arguments["query"]
            max_results = arguments.get("max_results", 20)

            resp = await depmap.get("/v1/cell-lines", params={"q": query, "limit": max_results})
            data = resp.json()
            items = data if isinstance(data, list) else data.get("data", data.get("cell_lines", []))
            results = []
            for cl in items[:max_results]:
                results.append({
                    "depmap_id": cl.get("depmap_id", cl.get("DepMap_ID", "")),
                    "cell_line_name": cl.get("cell_line_name", cl.get("stripped_cell_line_name", "")),
                    "lineage": cl.get("lineage", cl.get("primary_disease", "")),
                    "lineage_subtype": cl.get("lineage_subtype", cl.get("disease_subtype", "")),
                    "primary_disease": cl.get("primary_disease", ""),
                    "source": cl.get("source", ""),
                    "sex": cl.get("sex", ""),
                    "culture_type": cl.get("culture_type", cl.get("culture_medium", "")),
                })
            return text_result({"cell_lines": results, "count": len(results)})

        # -- LINCS L1000 ------------------------------------------------------
        elif name == "lincs_l1000_signatures":
            query = arguments["query"]
            pert_type = arguments.get("pert_type")
            cell_line = arguments.get("cell_line")
            max_results = arguments.get("max_results", 20)

            search_params: dict = {"search": query, "limit": max_results}
            if pert_type:
                search_params["filter"] = f'{{"where":{{"pert_type":"{pert_type}"}}}}'
            resp = await lincs.get("/v1/signatures", params=search_params)
            data = resp.json()
            items = data if isinstance(data, list) else data.get("data", data.get("signatures", []))
            signatures = []
            for sig in items[:max_results]:
                sig_cell = sig.get("cell_id", sig.get("cell_line", ""))
                if cell_line and cell_line.lower() not in str(sig_cell).lower():
                    continue
                signatures.append({
                    "signature_id": sig.get("sig_id", sig.get("id", "")),
                    "perturbation": sig.get("pert_name", sig.get("pert_iname", "")),
                    "pert_type": sig.get("pert_type", ""),
                    "cell_line": sig_cell,
                    "dose": sig.get("pert_dose", sig.get("dose", "")),
                    "time": sig.get("pert_time", sig.get("time", "")),
                    "pert_id": sig.get("pert_id", ""),
                    "is_touchstone": sig.get("is_touchstone", ""),
                })
            return text_result({
                "query": query,
                "signatures": signatures,
                "count": len(signatures),
            })

        elif name == "lincs_l1000_compounds":
            query = arguments["query"]
            max_results = arguments.get("max_results", 20)

            resp = await lincs.get("/v1/compounds", params={"search": query, "limit": max_results})
            data = resp.json()
            items = data if isinstance(data, list) else data.get("data", data.get("compounds", []))
            compounds = []
            for c in items[:max_results]:
                compounds.append({
                    "pert_id": c.get("pert_id", c.get("id", "")),
                    "pert_name": c.get("pert_iname", c.get("pert_name", c.get("name", ""))),
                    "moa": c.get("moa", c.get("mechanism_of_action", "")),
                    "target": c.get("target", ""),
                    "inchi_key": c.get("inchi_key", c.get("canonical_smiles", "")),
                    "compound_aliases": c.get("compound_aliases", c.get("alt_name", "")),
                    "status": c.get("status", c.get("clinical_phase", "")),
                })
            return text_result({
                "query": query,
                "compounds": compounds,
                "count": len(compounds),
            })

        return error_result(f"Unknown tool: {name}")
    except Exception as exc:
        return error_result(f"{name} failed: {exc}")


if __name__ == "__main__":
    start(server)
