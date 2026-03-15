"""Disease/Phenotype MCP Server — OMIM, HPO, DisGeNET, Monarch Initiative, OpenTargets, ClinPGx.

Covers ~12 tools for disease-phenotype associations, gene-disease mappings,
pharmacogenomics annotations, and ontology-driven phenotype queries.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

from base_server import APIClient, Server, error_result, make_tool, start, text_result

server = Server("disease-phenotype")

# ---------------------------------------------------------------------------
# API Clients
# ---------------------------------------------------------------------------

DISGENET_API_KEY = os.getenv("DISGENET_API_KEY", "")
OMIM_API_KEY = os.getenv("OMIM_API_KEY", "")

opentargets = APIClient(
    base_url="https://api.platform.opentargets.org/api/v4",
    headers={"Content-Type": "application/json"},
    rate_limit=10,
)

disgenet = APIClient(
    base_url="https://www.disgenet.org/api",
    headers={
        "Authorization": f"Bearer {DISGENET_API_KEY}",
        "Accept": "application/json",
    },
    rate_limit=5,
)

monarch = APIClient(
    base_url="https://api.monarchinitiative.org/v3/api",
    headers={"Accept": "application/json"},
    rate_limit=10,
)

hpo = APIClient(
    base_url="https://ontology.jax.org/api/hp",
    headers={"Accept": "application/json"},
    rate_limit=10,
)

omim = APIClient(
    base_url="https://api.omim.org/api",
    headers={"Accept": "application/json"},
    rate_limit=3,
)

# ClinPGx is queried via PharmGKB's public REST API
clinpgx = APIClient(
    base_url="https://api.pharmgkb.org/v1/data",
    headers={"Accept": "application/json"},
    rate_limit=5,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list:
    return [
        make_tool(
            "opentargets_target",
            "Search Open Targets for target (gene/protein) information by Ensembl gene ID or search term.",
            {
                "query": {
                    "type": "string",
                    "description": "Ensembl gene ID (e.g. 'ENSG00000141510') or free-text search term",
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
            "opentargets_disease",
            "Search Open Targets for disease information by EFO ID or search term.",
            {
                "query": {
                    "type": "string",
                    "description": "EFO disease ID (e.g. 'EFO_0000616') or free-text search term",
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
            "opentargets_association",
            "Get target-disease associations from Open Targets. Requires at least one of target or disease.",
            {
                "target_id": {
                    "type": "string",
                    "description": "Ensembl gene ID for the target (e.g. 'ENSG00000141510')",
                },
                "disease_id": {
                    "type": "string",
                    "description": "EFO disease ID (e.g. 'EFO_0000616')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return",
                    "default": 25,
                },
            },
            required=[],
        ),
        make_tool(
            "opentargets_drug",
            "Search Open Targets for drug/compound information by name or ChEMBL ID.",
            {
                "query": {
                    "type": "string",
                    "description": "Drug name or ChEMBL ID (e.g. 'imatinib' or 'CHEMBL941')",
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
            "disgenet_gene_disease",
            "Get gene-disease associations from DisGeNET by gene symbol or NCBI gene ID.",
            {
                "gene": {
                    "type": "string",
                    "description": "Gene symbol (e.g. 'TP53') or NCBI gene ID (e.g. '7157')",
                },
                "source": {
                    "type": "string",
                    "description": "Source filter: ALL, CURATED, INFERRED, ANIMAL_MODELS",
                    "default": "ALL",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return",
                    "default": 25,
                },
            },
            required=["gene"],
        ),
        make_tool(
            "disgenet_variant_disease",
            "Get variant-disease associations from DisGeNET by rsID.",
            {
                "variant": {
                    "type": "string",
                    "description": "dbSNP rsID (e.g. 'rs1042522')",
                },
                "source": {
                    "type": "string",
                    "description": "Source filter: ALL, CURATED, INFERRED",
                    "default": "ALL",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return",
                    "default": 25,
                },
            },
            required=["variant"],
        ),
        make_tool(
            "monarch_disease",
            "Search Monarch Initiative for disease information by name or ID.",
            {
                "query": {
                    "type": "string",
                    "description": "Disease name or MONDO/OMIM/DOID identifier",
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
            "monarch_gene_phenotype",
            "Get gene-phenotype associations from Monarch Initiative.",
            {
                "gene_id": {
                    "type": "string",
                    "description": "Gene identifier — HGNC ID (e.g. 'HGNC:11998') or NCBIGene ID (e.g. 'NCBIGene:7157')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return",
                    "default": 25,
                },
            },
            required=["gene_id"],
        ),
        make_tool(
            "hpo_term_search",
            "Search HPO (Human Phenotype Ontology) for phenotype terms by name or keyword.",
            {
                "query": {
                    "type": "string",
                    "description": "Phenotype term or keyword (e.g. 'seizure', 'cardiomyopathy')",
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
            "hpo_gene_to_phenotype",
            "Get phenotypes (HPO terms) associated with a gene from HPO.",
            {
                "gene_symbol": {
                    "type": "string",
                    "description": "Gene symbol (e.g. 'TP53', 'BRCA1')",
                },
            },
            required=["gene_symbol"],
        ),
        make_tool(
            "omim_search",
            "Search OMIM for gene-disease records by keyword or MIM number.",
            {
                "query": {
                    "type": "string",
                    "description": "Search query (gene name, disease name, or MIM number)",
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
            "clinpgx_search",
            "Search ClinPGx / PharmGKB for pharmacogenomics annotations (drug-gene interactions, clinical annotations).",
            {
                "query": {
                    "type": "string",
                    "description": "Drug name, gene symbol, or keyword (e.g. 'warfarin', 'CYP2D6')",
                },
                "resource": {
                    "type": "string",
                    "description": "Resource type: clinicalAnnotation, variantAnnotation, guideline, drugLabel",
                    "default": "clinicalAnnotation",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return",
                    "default": 10,
                },
            },
            required=["query"],
        ),
    ]


# ---------------------------------------------------------------------------
# GraphQL query templates for Open Targets
# ---------------------------------------------------------------------------

_OT_SEARCH_TARGET = """
query SearchTarget($q: String!, $size: Int!) {
  search(queryString: $q, entityNames: ["target"], page: {size: $size, index: 0}) {
    total
    hits {
      id
      name
      description
      entity
    }
  }
}
"""

_OT_TARGET_INFO = """
query TargetInfo($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    biotype
    functionDescriptions
    subcellularLocations {
      location
    }
    pathways {
      pathway
      pathwayId
    }
  }
}
"""

_OT_SEARCH_DISEASE = """
query SearchDisease($q: String!, $size: Int!) {
  search(queryString: $q, entityNames: ["disease"], page: {size: $size, index: 0}) {
    total
    hits {
      id
      name
      description
      entity
    }
  }
}
"""

_OT_DISEASE_INFO = """
query DiseaseInfo($efoId: String!) {
  disease(efoId: $efoId) {
    id
    name
    description
    synonyms {
      relation
      terms
    }
    therapeuticAreas {
      id
      name
    }
  }
}
"""

_OT_ASSOCIATIONS_TARGET = """
query AssociationsForTarget($ensemblId: String!, $size: Int!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    associatedDiseases(page: {size: $size, index: 0}) {
      count
      rows {
        disease {
          id
          name
        }
        score
        datatypeScores {
          id
          score
        }
      }
    }
  }
}
"""

_OT_ASSOCIATIONS_DISEASE = """
query AssociationsForDisease($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: {size: $size, index: 0}) {
      count
      rows {
        target {
          id
          approvedSymbol
          approvedName
        }
        score
        datatypeScores {
          id
          score
        }
      }
    }
  }
}
"""

_OT_SEARCH_DRUG = """
query SearchDrug($q: String!, $size: Int!) {
  search(queryString: $q, entityNames: ["drug"], page: {size: $size, index: 0}) {
    total
    hits {
      id
      name
      description
      entity
    }
  }
}
"""

_OT_DRUG_INFO = """
query DrugInfo($chemblId: String!) {
  drug(chemblId: $chemblId) {
    id
    name
    drugType
    maximumClinicalTrialPhase
    hasBeenWithdrawn
    mechanismsOfAction {
      rows {
        mechanismOfAction
        targets {
          id
          approvedSymbol
        }
      }
    }
    indications {
      count
      rows {
        disease {
          id
          name
        }
        maxPhaseForIndication
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    try:
        # =================================================================
        # Open Targets
        # =================================================================

        if name == "opentargets_target":
            query = arguments["query"]
            max_results = arguments.get("max_results", 10)

            # If it looks like an Ensembl ID, fetch directly
            if query.startswith("ENSG"):
                resp = await opentargets.post(
                    "/graphql",
                    json={
                        "query": _OT_TARGET_INFO,
                        "variables": {"ensemblId": query},
                    },
                )
                data = resp.json().get("data", {}).get("target")
                if not data:
                    return text_result({"target": None, "message": f"No target found for {query}"})
                return text_result({
                    "target": {
                        "id": data.get("id", ""),
                        "symbol": data.get("approvedSymbol", ""),
                        "name": data.get("approvedName", ""),
                        "biotype": data.get("biotype", ""),
                        "function_descriptions": data.get("functionDescriptions", []),
                        "subcellular_locations": [
                            loc.get("location", "") for loc in (data.get("subcellularLocations") or [])
                        ],
                        "pathways": [
                            {"name": p.get("pathway", ""), "id": p.get("pathwayId", "")}
                            for p in (data.get("pathways") or [])[:20]
                        ],
                    }
                })

            # Free-text search
            resp = await opentargets.post(
                "/graphql",
                json={
                    "query": _OT_SEARCH_TARGET,
                    "variables": {"q": query, "size": max_results},
                },
            )
            search = resp.json().get("data", {}).get("search", {})
            hits = search.get("hits", [])
            targets = [
                {"id": h.get("id", ""), "name": h.get("name", ""), "description": h.get("description", "")}
                for h in hits
            ]
            return text_result({"targets": targets, "total": search.get("total", 0)})

        elif name == "opentargets_disease":
            query = arguments["query"]
            max_results = arguments.get("max_results", 10)

            # Direct lookup by EFO ID
            if query.startswith("EFO_") or query.startswith("MONDO_") or query.startswith("OTAR_"):
                resp = await opentargets.post(
                    "/graphql",
                    json={
                        "query": _OT_DISEASE_INFO,
                        "variables": {"efoId": query},
                    },
                )
                data = resp.json().get("data", {}).get("disease")
                if not data:
                    return text_result({"disease": None, "message": f"No disease found for {query}"})
                return text_result({
                    "disease": {
                        "id": data.get("id", ""),
                        "name": data.get("name", ""),
                        "description": data.get("description", ""),
                        "synonyms": [
                            t for s in (data.get("synonyms") or []) for t in (s.get("terms") or [])
                        ][:20],
                        "therapeutic_areas": [
                            {"id": ta.get("id", ""), "name": ta.get("name", "")}
                            for ta in (data.get("therapeuticAreas") or [])
                        ],
                    }
                })

            # Free-text search
            resp = await opentargets.post(
                "/graphql",
                json={
                    "query": _OT_SEARCH_DISEASE,
                    "variables": {"q": query, "size": max_results},
                },
            )
            search = resp.json().get("data", {}).get("search", {})
            hits = search.get("hits", [])
            diseases = [
                {"id": h.get("id", ""), "name": h.get("name", ""), "description": h.get("description", "")}
                for h in hits
            ]
            return text_result({"diseases": diseases, "total": search.get("total", 0)})

        elif name == "opentargets_association":
            target_id = arguments.get("target_id")
            disease_id = arguments.get("disease_id")
            max_results = arguments.get("max_results", 25)

            if not target_id and not disease_id:
                return error_result("At least one of 'target_id' or 'disease_id' is required.")

            if target_id:
                resp = await opentargets.post(
                    "/graphql",
                    json={
                        "query": _OT_ASSOCIATIONS_TARGET,
                        "variables": {"ensemblId": target_id, "size": max_results},
                    },
                )
                data = resp.json().get("data", {}).get("target", {})
                assoc_data = data.get("associatedDiseases", {})
                associations = []
                for row in assoc_data.get("rows", []):
                    disease = row.get("disease", {})
                    associations.append({
                        "disease_id": disease.get("id", ""),
                        "disease_name": disease.get("name", ""),
                        "overall_score": row.get("score"),
                        "datatype_scores": {
                            dt.get("id", ""): dt.get("score")
                            for dt in (row.get("datatypeScores") or [])
                        },
                    })
                return text_result({
                    "target_id": data.get("id", ""),
                    "target_symbol": data.get("approvedSymbol", ""),
                    "associations": associations,
                    "total": assoc_data.get("count", 0),
                })

            # disease_id provided
            resp = await opentargets.post(
                "/graphql",
                json={
                    "query": _OT_ASSOCIATIONS_DISEASE,
                    "variables": {"efoId": disease_id, "size": max_results},
                },
            )
            data = resp.json().get("data", {}).get("disease", {})
            assoc_data = data.get("associatedTargets", {})
            associations = []
            for row in assoc_data.get("rows", []):
                target = row.get("target", {})
                associations.append({
                    "target_id": target.get("id", ""),
                    "target_symbol": target.get("approvedSymbol", ""),
                    "target_name": target.get("approvedName", ""),
                    "overall_score": row.get("score"),
                    "datatype_scores": {
                        dt.get("id", ""): dt.get("score")
                        for dt in (row.get("datatypeScores") or [])
                    },
                })
            return text_result({
                "disease_id": data.get("id", ""),
                "disease_name": data.get("name", ""),
                "associations": associations,
                "total": assoc_data.get("count", 0),
            })

        elif name == "opentargets_drug":
            query = arguments["query"]
            max_results = arguments.get("max_results", 10)

            # Direct lookup by ChEMBL ID
            if query.upper().startswith("CHEMBL"):
                resp = await opentargets.post(
                    "/graphql",
                    json={
                        "query": _OT_DRUG_INFO,
                        "variables": {"chemblId": query},
                    },
                )
                data = resp.json().get("data", {}).get("drug")
                if not data:
                    return text_result({"drug": None, "message": f"No drug found for {query}"})
                moa_rows = (data.get("mechanismsOfAction") or {}).get("rows") or []
                indications = (data.get("indications") or {}).get("rows") or []
                return text_result({
                    "drug": {
                        "id": data.get("id", ""),
                        "name": data.get("name", ""),
                        "drug_type": data.get("drugType", ""),
                        "max_clinical_trial_phase": data.get("maximumClinicalTrialPhase"),
                        "withdrawn": data.get("hasBeenWithdrawn", False),
                        "mechanisms_of_action": [
                            {
                                "mechanism": m.get("mechanismOfAction", ""),
                                "targets": [
                                    {"id": t.get("id", ""), "symbol": t.get("approvedSymbol", "")}
                                    for t in (m.get("targets") or [])
                                ],
                            }
                            for m in moa_rows[:10]
                        ],
                        "indications": [
                            {
                                "disease_id": ind.get("disease", {}).get("id", ""),
                                "disease_name": ind.get("disease", {}).get("name", ""),
                                "max_phase": ind.get("maxPhaseForIndication"),
                            }
                            for ind in indications[:20]
                        ],
                    }
                })

            # Free-text search
            resp = await opentargets.post(
                "/graphql",
                json={
                    "query": _OT_SEARCH_DRUG,
                    "variables": {"q": query, "size": max_results},
                },
            )
            search = resp.json().get("data", {}).get("search", {})
            hits = search.get("hits", [])
            drugs = [
                {"id": h.get("id", ""), "name": h.get("name", ""), "description": h.get("description", "")}
                for h in hits
            ]
            return text_result({"drugs": drugs, "total": search.get("total", 0)})

        # =================================================================
        # DisGeNET
        # =================================================================

        elif name == "disgenet_gene_disease":
            gene = arguments["gene"]
            source = arguments.get("source", "ALL")
            max_results = arguments.get("max_results", 25)

            # Determine if numeric ID or symbol
            if gene.isdigit():
                endpoint = f"/gda/gene/{gene}"
            else:
                endpoint = f"/gda/gene/{gene}"

            resp = await disgenet.get(endpoint, params={
                "source": source,
                "format": "json",
            })
            data = resp.json()
            if isinstance(data, dict) and data.get("status_code"):
                return error_result(f"DisGeNET error: {data.get('message', 'Unknown error')}")

            results = data if isinstance(data, list) else []
            associations = []
            for item in results[:max_results]:
                associations.append({
                    "gene_symbol": item.get("gene_symbol", ""),
                    "gene_id": item.get("geneid"),
                    "disease_name": item.get("disease_name", ""),
                    "disease_id": item.get("diseaseid", ""),
                    "score": item.get("score"),
                    "ei": item.get("ei"),
                    "el": item.get("el"),
                    "source": item.get("source", ""),
                    "pmid_count": item.get("pmid", 0),
                })
            return text_result({
                "gene": gene,
                "associations": associations,
                "count": len(associations),
            })

        elif name == "disgenet_variant_disease":
            variant = arguments["variant"]
            source = arguments.get("source", "ALL")
            max_results = arguments.get("max_results", 25)

            resp = await disgenet.get(f"/vda/variant/{variant}", params={
                "source": source,
                "format": "json",
            })
            data = resp.json()
            if isinstance(data, dict) and data.get("status_code"):
                return error_result(f"DisGeNET error: {data.get('message', 'Unknown error')}")

            results = data if isinstance(data, list) else []
            associations = []
            for item in results[:max_results]:
                associations.append({
                    "variant_id": item.get("variantid", ""),
                    "disease_name": item.get("disease_name", ""),
                    "disease_id": item.get("diseaseid", ""),
                    "score": item.get("score"),
                    "source": item.get("source", ""),
                    "pmid_count": item.get("pmid", 0),
                })
            return text_result({
                "variant": variant,
                "associations": associations,
                "count": len(associations),
            })

        # =================================================================
        # Monarch Initiative
        # =================================================================

        elif name == "monarch_disease":
            query = arguments["query"]
            max_results = arguments.get("max_results", 10)

            resp = await monarch.get("/search", params={
                "q": query,
                "category": "biolink:Disease",
                "limit": max_results,
            })
            data = resp.json()
            items = data.get("items", [])
            diseases = []
            for item in items:
                diseases.append({
                    "id": item.get("id", ""),
                    "name": item.get("name", ""),
                    "description": item.get("description", ""),
                    "category": item.get("category", ""),
                    "xrefs": item.get("xref", [])[:10],
                })
            return text_result({
                "diseases": diseases,
                "total": data.get("total", len(diseases)),
            })

        elif name == "monarch_gene_phenotype":
            gene_id = arguments["gene_id"]
            max_results = arguments.get("max_results", 25)

            resp = await monarch.get(f"/association", params={
                "subject": gene_id,
                "category": "biolink:GeneToPhenotypicFeatureAssociation",
                "limit": max_results,
            })
            data = resp.json()
            items = data.get("items", [])
            associations = []
            for item in items:
                obj = item.get("object_label", "") or item.get("object", "")
                associations.append({
                    "phenotype_id": item.get("object", ""),
                    "phenotype_name": item.get("object_label", ""),
                    "subject_id": item.get("subject", ""),
                    "subject_label": item.get("subject_label", ""),
                    "relation": item.get("predicate", ""),
                    "sources": item.get("publications", [])[:5],
                })
            return text_result({
                "gene_id": gene_id,
                "phenotypes": associations,
                "count": len(associations),
                "total": data.get("total", len(associations)),
            })

        # =================================================================
        # HPO (Human Phenotype Ontology)
        # =================================================================

        elif name == "hpo_term_search":
            query = arguments["query"]
            max_results = arguments.get("max_results", 10)

            resp = await hpo.get("/search", params={
                "q": query,
                "max": max_results,
            })
            data = resp.json()
            terms_list = data.get("terms", [])
            terms = []
            for term in terms_list:
                terms.append({
                    "id": term.get("id", ""),
                    "name": term.get("name", ""),
                    "definition": term.get("definition", ""),
                    "synonyms": term.get("synonyms", [])[:5],
                    "is_obsolete": term.get("isObsolete", False),
                })
            return text_result({
                "terms": terms,
                "count": len(terms),
            })

        elif name == "hpo_gene_to_phenotype":
            gene_symbol = arguments["gene_symbol"]

            resp = await hpo.get(f"/gene/{gene_symbol}")
            data = resp.json()

            # The HPO API returns associations for a gene
            associations = data.get("associations", data.get("diseaseAssoc", []))
            if isinstance(data, dict) and "termAssoc" in data:
                associations = data["termAssoc"]

            phenotypes = []
            if isinstance(associations, list):
                for assoc in associations:
                    phenotypes.append({
                        "hpo_id": assoc.get("ontologyId", assoc.get("hpoId", "")),
                        "hpo_name": assoc.get("name", assoc.get("hpoName", "")),
                        "frequency": assoc.get("frequency", ""),
                    })
            elif isinstance(data, dict):
                # Fallback: try to parse the response as a direct gene info block
                for assoc in data.get("phenotypes", []):
                    phenotypes.append({
                        "hpo_id": assoc.get("id", ""),
                        "hpo_name": assoc.get("name", ""),
                        "frequency": assoc.get("frequency", ""),
                    })

            return text_result({
                "gene_symbol": gene_symbol,
                "phenotypes": phenotypes,
                "count": len(phenotypes),
            })

        # =================================================================
        # OMIM
        # =================================================================

        elif name == "omim_search":
            query = arguments["query"]
            max_results = arguments.get("max_results", 10)

            if not OMIM_API_KEY:
                return error_result("OMIM_API_KEY environment variable is not set.")

            resp = await omim.get("/entry/search", params={
                "search": query,
                "start": 0,
                "limit": max_results,
                "format": "json",
                "apiKey": OMIM_API_KEY,
                "include": "geneMap",
            })
            data = resp.json()
            entries_block = data.get("omim", {}).get("searchResponse", {})
            entry_list = entries_block.get("entryList", [])
            results = []
            for entry_wrapper in entry_list:
                entry = entry_wrapper.get("entry", {})
                gene_map = entry.get("geneMap", {})
                results.append({
                    "mim_number": entry.get("mimNumber"),
                    "title": entry.get("titles", {}).get("preferredTitle", ""),
                    "status": entry.get("status", ""),
                    "gene_symbols": gene_map.get("geneSymbols", ""),
                    "gene_name": gene_map.get("geneName", ""),
                    "chromosome": gene_map.get("chromosome", ""),
                    "cyto_location": gene_map.get("cytoLocation", ""),
                    "phenotypes": [
                        {
                            "phenotype": pm.get("phenotype", ""),
                            "mim_number": pm.get("phenotypeMimNumber"),
                            "inheritance": pm.get("phenotypeInheritance", ""),
                        }
                        for pm in (gene_map.get("phenotypeMapList") or [])
                    ],
                })
            return text_result({
                "query": query,
                "results": results,
                "total": entries_block.get("totalResults", len(results)),
            })

        # =================================================================
        # ClinPGx / PharmGKB
        # =================================================================

        elif name == "clinpgx_search":
            query = arguments["query"]
            resource = arguments.get("resource", "clinicalAnnotation")
            max_results = arguments.get("max_results", 10)

            resp = await clinpgx.get(f"/{resource}", params={
                "view": "max",
                "offset": 0,
                "limit": max_results,
            })
            data = resp.json()
            items = data.get("data", [])

            # Filter results that match the query in any relevant field
            query_lower = query.lower()
            matched = []
            for item in items:
                text_blob = str(item).lower()
                if query_lower in text_blob:
                    matched.append(item)

            # If no filtering match, just return all items (the API may not support keyword search)
            if not matched:
                matched = items

            annotations = []
            for item in matched[:max_results]:
                ann = {
                    "id": item.get("id", ""),
                    "name": item.get("name", ""),
                }
                # Extract common fields based on resource type
                if resource == "clinicalAnnotation":
                    ann.update({
                        "genes": [g.get("symbol", "") for g in (item.get("relatedGenes") or [])],
                        "drugs": [d.get("name", "") for d in (item.get("relatedChemicals") or [])],
                        "phenotypes": [p.get("name", "") for p in (item.get("relatedDiseases") or [])],
                        "level_of_evidence": item.get("evidenceLevelOfEvidence", ""),
                    })
                elif resource == "guideline":
                    ann.update({
                        "source": item.get("source", ""),
                        "summary": item.get("summaryMarkdown", item.get("textHtml", ""))[:500],
                    })
                elif resource == "drugLabel":
                    ann.update({
                        "source": item.get("source", ""),
                        "genes": [g.get("symbol", "") for g in (item.get("relatedGenes") or [])],
                        "drugs": [d.get("name", "") for d in (item.get("relatedChemicals") or [])],
                    })
                else:
                    ann.update({"raw": {k: v for k, v in item.items() if k not in ("id", "name")}})
                annotations.append(ann)

            return text_result({
                "query": query,
                "resource": resource,
                "annotations": annotations,
                "count": len(annotations),
            })

        return error_result(f"Unknown tool: {name}")
    except Exception as exc:
        return error_result(f"{name} failed: {exc}")


if __name__ == "__main__":
    start(server)
