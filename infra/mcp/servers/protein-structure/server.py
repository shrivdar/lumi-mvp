"""Protein Structure MCP Server — PDB/RCSB, AlphaFold DB, InterPro, EMDB, Foldseek.

Covers structural biology data access: crystal structures, cryo-EM maps, predicted
structures, domain annotations, and structural similarity search.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

from base_server import APIClient, Server, error_result, make_tool, start, text_result

server = Server("protein-structure")

# ---------------------------------------------------------------------------
# API Clients
# ---------------------------------------------------------------------------

pdb_search = APIClient(base_url="https://search.rcsb.org/rcsbsearch/v2", rate_limit=5)
pdb_data = APIClient(base_url="https://data.rcsb.org/rest/v1", rate_limit=5)
alphafold = APIClient(base_url="https://alphafold.ebi.ac.uk/api", rate_limit=5)
interpro = APIClient(
    base_url="https://www.ebi.ac.uk/interpro/api",
    headers={"Accept": "application/json"},
    rate_limit=5,
)
emdb = APIClient(
    base_url="https://www.ebi.ac.uk/emdb/api",
    headers={"Accept": "application/json"},
    rate_limit=5,
)
foldseek = APIClient(base_url="https://search.foldseek.com/api", rate_limit=2, timeout=120)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@server.list_tools()
async def list_tools() -> list:
    return [
        make_tool(
            "pdb_search",
            "Search RCSB PDB for structures by keyword, sequence, or PDB ID. "
            "Returns matching entries with basic metadata.",
            {
                "query": {
                    "type": "string",
                    "description": "Search query (keyword, protein name, organism, etc.)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 10,
                },
            },
            required=["query"],
        ),
        make_tool(
            "pdb_entry",
            "Get detailed PDB entry information including resolution, method, authors, "
            "citation, polymer entities, and assembly details.",
            {
                "pdb_id": {
                    "type": "string",
                    "description": "4-character PDB ID (e.g. '1ABC', '7KFB')",
                },
            },
            required=["pdb_id"],
        ),
        make_tool(
            "pdb_ligands",
            "Get ligand/small molecule information for a PDB entry including chemical "
            "name, formula, InChI, and binding site details.",
            {
                "pdb_id": {
                    "type": "string",
                    "description": "4-character PDB ID",
                },
            },
            required=["pdb_id"],
        ),
        make_tool(
            "pdb_sequence_search",
            "Search PDB by protein sequence using RCSB sequence search (BLAST-like). "
            "Finds structures with similar sequences.",
            {
                "sequence": {
                    "type": "string",
                    "description": "Amino acid sequence in one-letter code",
                },
                "identity_cutoff": {
                    "type": "number",
                    "description": "Minimum sequence identity (0.0 to 1.0)",
                    "default": 0.9,
                },
                "evalue_cutoff": {
                    "type": "number",
                    "description": "E-value cutoff for sequence search",
                    "default": 0.1,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 10,
                },
            },
            required=["sequence"],
        ),
        make_tool(
            "alphafold_prediction",
            "Get AlphaFold predicted structure for a UniProt accession. Returns model "
            "URL, confidence scores (pLDDT), and metadata.",
            {
                "uniprot_id": {
                    "type": "string",
                    "description": "UniProt accession ID (e.g. 'P04637' for human TP53)",
                },
            },
            required=["uniprot_id"],
        ),
        make_tool(
            "alphafold_pae",
            "Get the predicted aligned error (PAE) matrix for an AlphaFold model. "
            "PAE indicates confidence in relative domain positions.",
            {
                "uniprot_id": {
                    "type": "string",
                    "description": "UniProt accession ID",
                },
            },
            required=["uniprot_id"],
        ),
        make_tool(
            "interpro_search",
            "Search InterPro for protein domain/family annotations by keyword or "
            "accession. Integrates Pfam, SMART, CDD, PROSITE, and more.",
            {
                "query": {
                    "type": "string",
                    "description": "Search term (domain name, keyword, or InterPro accession)",
                },
                "type_filter": {
                    "type": "string",
                    "description": "Filter by entry type: family, domain, homologous_superfamily, repeat, site",
                    "default": "",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results",
                    "default": 10,
                },
            },
            required=["query"],
        ),
        make_tool(
            "interpro_entry",
            "Get detailed InterPro entry with member database cross-references (Pfam, "
            "SMART, etc.), GO terms, and contributing signatures.",
            {
                "accession": {
                    "type": "string",
                    "description": "InterPro accession (e.g. 'IPR000504') or member DB accession (e.g. 'PF00076')",
                },
            },
            required=["accession"],
        ),
        make_tool(
            "emdb_search",
            "Search the Electron Microscopy Data Bank for cryo-EM maps and tomograms. "
            "Returns map metadata, resolution, and sample information.",
            {
                "query": {
                    "type": "string",
                    "description": "Search query (protein name, complex, organism, etc.)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results",
                    "default": 10,
                },
            },
            required=["query"],
        ),
        make_tool(
            "foldseek_search",
            "Search for structural similarity using the Foldseek API. Accepts a PDB ID "
            "or structure and finds structurally similar proteins across PDB, AlphaFold DB, "
            "and ESMAtlas.",
            {
                "pdb_id": {
                    "type": "string",
                    "description": "PDB ID to use as query structure (e.g. '1ABC'). "
                    "Provide either pdb_id or structure, not both.",
                    "default": "",
                },
                "structure": {
                    "type": "string",
                    "description": "PDB-format structure string to use as query. "
                    "Provide either pdb_id or structure, not both.",
                    "default": "",
                },
                "databases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Databases to search: 'pdb100', 'afdb50', 'afdb-swissprot', "
                    "'afdb-proteome', 'esmatlas30', 'cath50', 'mgnify_esm30'",
                    "default": ["pdb100"],
                },
                "mode": {
                    "type": "string",
                    "description": "Search mode: '3diaa' (fast, 3Di alphabet) or 'tmalign' (slower, TM-align)",
                    "default": "3diaa",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results per database",
                    "default": 10,
                },
            },
            required=[],
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    try:
        # ---------------------------------------------------------------
        # pdb_search
        # ---------------------------------------------------------------
        if name == "pdb_search":
            query_text = arguments["query"]
            max_results = arguments.get("max_results", 10)
            payload = {
                "query": {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {"value": query_text},
                },
                "return_type": "entry",
                "request_options": {
                    "paginate": {"start": 0, "rows": max_results},
                    "results_content_type": ["experimental"],
                    "sort": [{"sort_by": "score", "direction": "desc"}],
                },
            }
            resp = await pdb_search.post("/query", json=payload)
            data = resp.json()
            results = []
            for hit in data.get("result_set", []):
                pdb_id = hit.get("identifier", "")
                results.append({
                    "pdb_id": pdb_id,
                    "score": hit.get("score", 0),
                })
            # Fetch summaries for the hits
            if results:
                ids = [r["pdb_id"] for r in results]
                for entry in results:
                    try:
                        detail = await pdb_data.get(f"/core/entry/{entry['pdb_id']}")
                        d = detail.json()
                        entry["title"] = d.get("struct", {}).get("title", "")
                        entry["method"] = (
                            d.get("exptl", [{}])[0].get("method", "") if d.get("exptl") else ""
                        )
                        entry["resolution"] = (
                            d.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0]
                        )
                        entry["deposit_date"] = d.get("rcsb_accession_info", {}).get(
                            "deposit_date", ""
                        )
                    except Exception:
                        pass
            return text_result({"entries": results, "total_count": data.get("total_count", 0)})

        # ---------------------------------------------------------------
        # pdb_entry
        # ---------------------------------------------------------------
        elif name == "pdb_entry":
            pdb_id = arguments["pdb_id"].upper().strip()
            resp = await pdb_data.get(f"/core/entry/{pdb_id}")
            d = resp.json()

            # Gather polymer entities
            entities = []
            entity_ids = d.get("rcsb_entry_container_identifiers", {}).get(
                "polymer_entity_ids", []
            )
            for eid in entity_ids:
                try:
                    eresp = await pdb_data.get(f"/core/polymer_entity/{pdb_id}/{eid}")
                    ed = eresp.json()
                    entities.append({
                        "entity_id": eid,
                        "description": ed.get("rcsb_polymer_entity", {}).get(
                            "pdbx_description", ""
                        ),
                        "type": ed.get("entity_poly", {}).get("type", ""),
                        "sequence": ed.get("entity_poly", {}).get(
                            "pdbx_seq_one_letter_code_can", ""
                        ),
                        "organism": (
                            ed.get("rcsb_entity_source_organism", [{}])[0].get(
                                "ncbi_scientific_name", ""
                            )
                            if ed.get("rcsb_entity_source_organism")
                            else ""
                        ),
                    })
                except Exception:
                    entities.append({"entity_id": eid})

            entry = {
                "pdb_id": pdb_id,
                "title": d.get("struct", {}).get("title", ""),
                "method": (
                    d.get("exptl", [{}])[0].get("method", "") if d.get("exptl") else ""
                ),
                "resolution": d.get("rcsb_entry_info", {}).get(
                    "resolution_combined", [None]
                )[0],
                "deposit_date": d.get("rcsb_accession_info", {}).get("deposit_date", ""),
                "release_date": d.get("rcsb_accession_info", {}).get(
                    "initial_release_date", ""
                ),
                "authors": d.get("audit_author", []),
                "citation": d.get("rcsb_primary_citation", {}),
                "keywords": d.get("struct_keywords", {}).get("pdbx_keywords", ""),
                "assembly_count": d.get("rcsb_entry_info", {}).get(
                    "assembly_count", 0
                ),
                "polymer_entity_count": d.get("rcsb_entry_info", {}).get(
                    "polymer_entity_count", 0
                ),
                "entities": entities,
            }
            return text_result({"entry": entry})

        # ---------------------------------------------------------------
        # pdb_ligands
        # ---------------------------------------------------------------
        elif name == "pdb_ligands":
            pdb_id = arguments["pdb_id"].upper().strip()
            resp = await pdb_data.get(f"/core/entry/{pdb_id}")
            d = resp.json()

            nonpoly_ids = d.get("rcsb_entry_container_identifiers", {}).get(
                "non_polymer_entity_ids", []
            )
            ligands = []
            for nid in nonpoly_ids:
                try:
                    nresp = await pdb_data.get(
                        f"/core/nonpolymer_entity/{pdb_id}/{nid}"
                    )
                    nd = nresp.json()
                    comp_id = nd.get("pdbx_entity_nonpoly", {}).get("comp_id", "")
                    lig_info = {
                        "entity_id": nid,
                        "comp_id": comp_id,
                        "description": nd.get("rcsb_nonpolymer_entity", {}).get(
                            "pdbx_description", ""
                        ),
                    }
                    # Fetch chemical component details
                    if comp_id:
                        try:
                            cresp = await pdb_data.get(f"/core/chemcomp/{comp_id}")
                            cd = cresp.json()
                            lig_info["name"] = cd.get("chem_comp", {}).get("name", "")
                            lig_info["formula"] = cd.get("chem_comp", {}).get(
                                "formula", ""
                            )
                            lig_info["formula_weight"] = cd.get("chem_comp", {}).get(
                                "formula_weight", ""
                            )
                            lig_info["type"] = cd.get("chem_comp", {}).get("type", "")
                            lig_info["inchi"] = cd.get("rcsb_chem_comp_descriptor", {}).get(
                                "InChI", ""
                            )
                            lig_info["smiles"] = cd.get("rcsb_chem_comp_descriptor", {}).get(
                                "SMILES_stereo", cd.get("rcsb_chem_comp_descriptor", {}).get("SMILES", "")
                            )
                        except Exception:
                            pass
                    ligands.append(lig_info)
                except Exception:
                    ligands.append({"entity_id": nid})
            return text_result({"pdb_id": pdb_id, "ligands": ligands, "count": len(ligands)})

        # ---------------------------------------------------------------
        # pdb_sequence_search
        # ---------------------------------------------------------------
        elif name == "pdb_sequence_search":
            sequence = arguments["sequence"]
            identity_cutoff = arguments.get("identity_cutoff", 0.9)
            evalue_cutoff = arguments.get("evalue_cutoff", 0.1)
            max_results = arguments.get("max_results", 10)
            payload = {
                "query": {
                    "type": "terminal",
                    "service": "sequence",
                    "parameters": {
                        "evalue_cutoff": evalue_cutoff,
                        "identity_cutoff": identity_cutoff,
                        "sequence_type": "protein",
                        "value": sequence,
                    },
                },
                "return_type": "polymer_entity",
                "request_options": {
                    "paginate": {"start": 0, "rows": max_results},
                    "sort": [{"sort_by": "score", "direction": "desc"}],
                },
            }
            resp = await pdb_search.post("/query", json=payload)
            data = resp.json()
            results = []
            for hit in data.get("result_set", []):
                identifier = hit.get("identifier", "")
                services = hit.get("services", [])
                seq_info = {}
                if services:
                    nodes = services[0].get("nodes", [])
                    if nodes:
                        match_ctx = nodes[0].get("match_context", [])
                        if match_ctx:
                            seq_info = {
                                "evalue": match_ctx[0].get("evalue"),
                                "identity": match_ctx[0].get("sequence_identity"),
                                "alignment_length": match_ctx[0].get("alignment_length"),
                            }
                pdb_id = identifier.split("_")[0] if "_" in identifier else identifier
                results.append({
                    "identifier": identifier,
                    "pdb_id": pdb_id,
                    "score": hit.get("score", 0),
                    **seq_info,
                })
            return text_result({"matches": results, "total_count": data.get("total_count", 0)})

        # ---------------------------------------------------------------
        # alphafold_prediction
        # ---------------------------------------------------------------
        elif name == "alphafold_prediction":
            uniprot_id = arguments["uniprot_id"].strip()
            resp = await alphafold.get(f"/prediction/{uniprot_id}")
            data = resp.json()
            # API returns a list; take first entry
            entry = data[0] if isinstance(data, list) and data else data
            return text_result({
                "uniprot_id": uniprot_id,
                "gene": entry.get("gene", ""),
                "organism": entry.get("organismScientificName", ""),
                "tax_id": entry.get("taxId"),
                "model_url": entry.get("cifUrl", ""),
                "pdb_url": entry.get("pdbUrl", ""),
                "bcif_url": entry.get("bcifUrl", ""),
                "pae_image_url": entry.get("paeImageUrl", ""),
                "pae_doc_url": entry.get("paeDocUrl", ""),
                "model_created": entry.get("modelCreatedDate", ""),
                "latest_version": entry.get("latestVersion"),
                "confidence_version": entry.get("confidenceVersion"),
                "confidence_avg_plddt": entry.get("confidenceAvgLocalScore"),
                "sequence_length": entry.get("uniprotEnd"),
                "uniprot_description": entry.get("uniprotDescription", ""),
            })

        # ---------------------------------------------------------------
        # alphafold_pae
        # ---------------------------------------------------------------
        elif name == "alphafold_pae":
            uniprot_id = arguments["uniprot_id"].strip()
            # First get the prediction metadata to find PAE URL
            resp = await alphafold.get(f"/prediction/{uniprot_id}")
            data = resp.json()
            entry = data[0] if isinstance(data, list) and data else data
            pae_url = entry.get("paeDocUrl", "")
            if not pae_url:
                return error_result(f"No PAE data available for {uniprot_id}")
            # Fetch the PAE JSON directly (absolute URL)
            pae_client = APIClient(base_url="", rate_limit=5)
            try:
                pae_resp = await pae_client.get(pae_url)
                pae_data = pae_resp.json()
            finally:
                await pae_client.close()
            # PAE JSON is a list with one element containing predicted_aligned_error matrix
            pae_entry = pae_data[0] if isinstance(pae_data, list) and pae_data else pae_data
            # Summarize instead of returning the full matrix (can be huge)
            residue_count = len(pae_entry.get("predicted_aligned_error", [[]])[0]) if pae_entry.get("predicted_aligned_error") else 0
            max_pae = pae_entry.get("max_predicted_aligned_error", None)
            return text_result({
                "uniprot_id": uniprot_id,
                "pae_url": pae_url,
                "residue_count": residue_count,
                "max_predicted_aligned_error": max_pae,
                "note": "Full PAE matrix available at pae_url. Matrix is NxN where N=residue_count.",
            })

        # ---------------------------------------------------------------
        # interpro_search
        # ---------------------------------------------------------------
        elif name == "interpro_search":
            query = arguments["query"]
            type_filter = arguments.get("type_filter", "")
            max_results = arguments.get("max_results", 10)
            params = {"search": query, "page_size": max_results}
            if type_filter:
                params["type"] = type_filter
            resp = await interpro.get("/entry/interpro", params=params)
            data = resp.json()
            results = []
            for item in data.get("results", []):
                meta = item.get("metadata", {})
                results.append({
                    "accession": meta.get("accession", ""),
                    "name": meta.get("name", ""),
                    "type": meta.get("type", ""),
                    "source_database": meta.get("source_database", ""),
                    "member_databases": list(
                        (meta.get("member_databases") or {}).keys()
                    ),
                    "go_terms": [
                        {"id": g.get("identifier", ""), "name": g.get("name", "")}
                        for g in (meta.get("go_terms") or [])[:5]
                    ],
                })
            return text_result({
                "entries": results,
                "count": data.get("count", len(results)),
            })

        # ---------------------------------------------------------------
        # interpro_entry
        # ---------------------------------------------------------------
        elif name == "interpro_entry":
            accession = arguments["accession"].strip()
            # Determine if this is an InterPro accession or member DB accession
            if accession.startswith("IPR"):
                resp = await interpro.get(f"/entry/interpro/{accession}")
            elif accession.startswith("PF"):
                resp = await interpro.get(f"/entry/pfam/{accession}")
            elif accession.startswith("SM"):
                resp = await interpro.get(f"/entry/smart/{accession}")
            elif accession.startswith("cd"):
                resp = await interpro.get(f"/entry/cdd/{accession}")
            elif accession.startswith("PS"):
                resp = await interpro.get(f"/entry/prosite/{accession}")
            else:
                resp = await interpro.get(f"/entry/interpro/{accession}")
            data = resp.json()
            meta = data.get("metadata", {})
            entry = {
                "accession": meta.get("accession", ""),
                "name": meta.get("name", ""),
                "type": meta.get("type", ""),
                "source_database": meta.get("source_database", ""),
                "description": (
                    meta.get("description", [{}])[0].get("text", "")
                    if meta.get("description")
                    else ""
                ),
                "member_databases": {},
                "go_terms": [
                    {
                        "id": g.get("identifier", ""),
                        "name": g.get("name", ""),
                        "category": g.get("category", {}).get("name", ""),
                    }
                    for g in (meta.get("go_terms") or [])
                ],
                "literature": [
                    {
                        "pmid": ref.get("PMID"),
                        "title": ref.get("title", ""),
                        "authors": ref.get("authors", []),
                    }
                    for ref in list((meta.get("literature") or {}).values())[:5]
                ],
                "protein_count": data.get("counters", {}).get("proteins", 0),
                "structure_count": data.get("counters", {}).get("structures", 0),
            }
            # Extract member database signatures
            for db_name, sigs in (meta.get("member_databases") or {}).items():
                entry["member_databases"][db_name] = [
                    {"accession": acc, "name": info.get("name", "")}
                    for acc, info in (sigs or {}).items()
                ]
            # Cross-references to InterPro if this is a member DB entry
            if meta.get("integrated"):
                entry["integrated_into"] = meta["integrated"]
            return text_result({"entry": entry})

        # ---------------------------------------------------------------
        # emdb_search
        # ---------------------------------------------------------------
        elif name == "emdb_search":
            query = arguments["query"]
            max_results = arguments.get("max_results", 10)
            resp = await emdb.get(
                "/search/current_status:REL AND "
                + query.replace(" ", " AND "),
                params={"rows": max_results, "wt": "json"},
            )
            data = resp.json()
            results = []
            docs = data.get("response", {}).get("docs", data if isinstance(data, list) else [])
            for doc in docs[:max_results]:
                results.append({
                    "emdb_id": doc.get("emdb_id", doc.get("accessCode", "")),
                    "title": doc.get("title", ""),
                    "resolution": doc.get("resolution", ""),
                    "method": doc.get("method", ""),
                    "sample_name": doc.get("sample_name", ""),
                    "organism": doc.get("sample_organism", ""),
                    "authors": doc.get("authors", ""),
                    "release_date": doc.get("release_date", ""),
                    "fitted_pdbs": doc.get("fitted_pdb_ids", []),
                })
            return text_result({"entries": results, "count": len(results)})

        # ---------------------------------------------------------------
        # foldseek_search
        # ---------------------------------------------------------------
        elif name == "foldseek_search":
            pdb_id = arguments.get("pdb_id", "").strip()
            structure_text = arguments.get("structure", "").strip()
            databases = arguments.get("databases", ["pdb100"])
            mode = arguments.get("mode", "3diaa")
            max_results = arguments.get("max_results", 10)

            if not pdb_id and not structure_text:
                return error_result(
                    "Provide either 'pdb_id' or 'structure' for Foldseek search."
                )

            # If pdb_id given, download the structure from RCSB
            if pdb_id and not structure_text:
                pdb_id = pdb_id.upper()
                pdb_download = APIClient(base_url="https://files.rcsb.org", rate_limit=5)
                try:
                    dl_resp = await pdb_download.get(f"/download/{pdb_id}.pdb")
                    structure_text = dl_resp.text
                finally:
                    await pdb_download.close()

            # Submit search to Foldseek
            submit_resp = await foldseek.post(
                "/ticket",
                data={
                    "q": structure_text,
                    "database[]": databases,
                    "mode": mode,
                },
            )
            ticket_data = submit_resp.json()
            ticket_id = ticket_data.get("id", "")
            if not ticket_id:
                return error_result(
                    f"Foldseek submission failed: {ticket_data}"
                )

            # Poll for results
            import asyncio

            for _ in range(60):
                status_resp = await foldseek.get(f"/ticket/{ticket_id}")
                status_data = status_resp.json()
                if status_data.get("status") == "COMPLETE":
                    break
                if status_data.get("status") == "ERROR":
                    return error_result(
                        f"Foldseek search failed: {status_data}"
                    )
                await asyncio.sleep(2)
            else:
                return error_result("Foldseek search timed out after 120 seconds.")

            # Fetch results
            result_resp = await foldseek.get(f"/result/download/{ticket_id}")
            raw = result_resp.text

            # Parse TSV results (Foldseek m8 format)
            hits = []
            for line in raw.strip().split("\n"):
                if not line or line.startswith("#"):
                    continue
                cols = line.split("\t")
                if len(cols) < 12:
                    continue
                hits.append({
                    "query": cols[0],
                    "target": cols[1],
                    "identity": float(cols[2]) if cols[2] else 0,
                    "alignment_length": int(cols[3]) if cols[3] else 0,
                    "mismatches": int(cols[4]) if cols[4] else 0,
                    "gap_opens": int(cols[5]) if cols[5] else 0,
                    "query_start": int(cols[6]) if cols[6] else 0,
                    "query_end": int(cols[7]) if cols[7] else 0,
                    "target_start": int(cols[8]) if cols[8] else 0,
                    "target_end": int(cols[9]) if cols[9] else 0,
                    "evalue": float(cols[10]) if cols[10] else 0,
                    "bit_score": float(cols[11]) if cols[11] else 0,
                })
                if len(hits) >= max_results:
                    break

            return text_result({
                "query_pdb_id": pdb_id or "custom_structure",
                "mode": mode,
                "databases": databases,
                "hits": hits,
                "total_hits": len(raw.strip().split("\n")),
            })

        return error_result(f"Unknown tool: {name}")
    except Exception as exc:
        return error_result(f"{name} failed: {exc}")


if __name__ == "__main__":
    start(server)
