"""Drug Discovery MCP Server — PubChem, BindingDB, DDInter, OpenFDA, DailyMed, UniChem.

Covers compound search, property retrieval, bioassay queries, binding affinity lookup,
adverse event reports, drug labeling, drug-drug interactions, and chemical ID conversion.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

from base_server import APIClient, Server, error_result, make_tool, start, text_result

server = Server("drug-discovery")

# ---------------------------------------------------------------------------
# API Clients
# ---------------------------------------------------------------------------

pubchem = APIClient(base_url="https://pubchem.ncbi.nlm.nih.gov/rest/pug", rate_limit=5)
bindingdb = APIClient(base_url="https://bindingdb.org/axis2/services/BDBService", rate_limit=3)
openfda = APIClient(base_url="https://api.fda.gov", rate_limit=4)
dailymed = APIClient(base_url="https://dailymed.nlm.nih.gov/dailymed/services/v2", rate_limit=3)
unichem = APIClient(base_url="https://www.ebi.ac.uk/unichem/rest", rate_limit=5)
ddinter = APIClient(base_url="https://ddinter.scbdd.com/api", rate_limit=2, timeout=45)

OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", "")

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@server.list_tools()
async def list_tools() -> list:
    return [
        make_tool(
            "pubchem_compound_search",
            "Search PubChem for compounds by name, SMILES, or InChI. Returns CIDs and basic identity info.",
            {
                "query": {"type": "string", "description": "Compound name, SMILES string, or InChI key"},
                "search_type": {
                    "type": "string",
                    "description": "Search namespace: 'name', 'smiles', or 'inchi'",
                    "default": "name",
                },
                "max_results": {"type": "integer", "description": "Maximum results to return", "default": 5},
            },
            required=["query"],
        ),
        make_tool(
            "pubchem_compound_properties",
            "Get computed properties for a PubChem compound (MW, LogP, TPSA, HBD/HBA, etc.).",
            {
                "cid": {"type": "string", "description": "PubChem Compound ID (CID)"},
                "properties": {
                    "type": "string",
                    "description": "Comma-separated property names (e.g. 'MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,RotatableBondCount')",
                    "default": "MolecularFormula,MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,ExactMass,MonoisotopicMass,Complexity",
                },
            },
            required=["cid"],
        ),
        make_tool(
            "pubchem_bioassay_search",
            "Search PubChem BioAssay database for assays related to a compound or target.",
            {
                "query": {"type": "string", "description": "Search term — compound name, gene symbol, or assay keyword"},
                "search_type": {
                    "type": "string",
                    "description": "Search by: 'target' (gene/protein name) or 'compound' (compound name/CID)",
                    "default": "target",
                },
                "max_results": {"type": "integer", "description": "Maximum results", "default": 10},
            },
            required=["query"],
        ),
        make_tool(
            "pubchem_compound_targets",
            "Get known biological targets for a PubChem compound by CID. Returns gene targets from bioassays.",
            {
                "cid": {"type": "string", "description": "PubChem Compound ID"},
                "activity_type": {
                    "type": "string",
                    "description": "Filter by activity outcome: 'active', 'inactive', or 'all'",
                    "default": "active",
                },
            },
            required=["cid"],
        ),
        make_tool(
            "bindingdb_search",
            "Search BindingDB for binding affinity data by compound SMILES or monomer (name).",
            {
                "query": {"type": "string", "description": "Compound SMILES string or compound name"},
                "search_type": {
                    "type": "string",
                    "description": "Search type: 'smiles' for structure search or 'name' for compound name",
                    "default": "smiles",
                },
                "similarity": {
                    "type": "number",
                    "description": "Similarity cutoff (0-1) for SMILES-based search",
                    "default": 0.85,
                },
            },
            required=["query"],
        ),
        make_tool(
            "bindingdb_target",
            "Get binding affinity data for a protein target from BindingDB by UniProt accession.",
            {
                "uniprot_id": {"type": "string", "description": "UniProt accession (e.g. 'P00533' for EGFR)"},
                "max_results": {"type": "integer", "description": "Maximum number of binding records", "default": 20},
            },
            required=["uniprot_id"],
        ),
        make_tool(
            "openfda_adverse_events",
            "Search FDA Adverse Event Reporting System (FAERS) for drug safety signals.",
            {
                "drug_name": {"type": "string", "description": "Drug generic or brand name"},
                "reaction": {
                    "type": "string",
                    "description": "Optional adverse reaction term to filter by (MedDRA preferred term)",
                    "default": "",
                },
                "max_results": {"type": "integer", "description": "Maximum results", "default": 10},
            },
            required=["drug_name"],
        ),
        make_tool(
            "openfda_drug_labeling",
            "Search FDA drug labeling (package inserts / SPL) for a drug.",
            {
                "query": {"type": "string", "description": "Drug name or active ingredient"},
                "section": {
                    "type": "string",
                    "description": "Labeling section to retrieve: 'indications_and_usage', 'warnings', 'adverse_reactions', 'drug_interactions', 'dosage_and_administration', or empty for all",
                    "default": "",
                },
                "max_results": {"type": "integer", "description": "Maximum results", "default": 3},
            },
            required=["query"],
        ),
        make_tool(
            "dailymed_search",
            "Search DailyMed for drug labels (SPL documents) by drug name.",
            {
                "drug_name": {"type": "string", "description": "Drug name to search"},
                "max_results": {"type": "integer", "description": "Maximum results", "default": 5},
            },
            required=["drug_name"],
        ),
        make_tool(
            "unichem_convert",
            "Convert between chemical identifiers using EBI UniChem (e.g. InChIKey to ChEMBL, PubChem, DrugBank IDs).",
            {
                "identifier": {"type": "string", "description": "Chemical identifier to convert (e.g. InChIKey, source-specific ID)"},
                "source_type": {
                    "type": "string",
                    "description": "Source database: 'inchikey', 'chembl' (1), 'drugbank' (2), 'pubchem' (22), 'kegg' (6), 'chebi' (7)",
                    "default": "inchikey",
                },
            },
            required=["identifier"],
        ),
        make_tool(
            "ddinter_search",
            "Search DDInter for drug-drug interactions by drug name.",
            {
                "drug_name": {"type": "string", "description": "Drug name to look up interactions for"},
                "max_results": {"type": "integer", "description": "Maximum interactions to return", "default": 10},
            },
            required=["drug_name"],
        ),
    ]


# ---------------------------------------------------------------------------
# Source ID map for UniChem
# ---------------------------------------------------------------------------

_UNICHEM_SRC: dict[str, int] = {
    "chembl": 1,
    "drugbank": 2,
    "pdb": 3,
    "iuphar": 4,
    "kegg": 6,
    "chebi": 7,
    "nih_ncc": 8,
    "zinc": 9,
    "emolecules": 10,
    "atlas": 15,
    "fdasrs": 18,
    "pubchem": 22,
    "lincs": 31,
}


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    try:
        # ------------------------------------------------------------------
        # PubChem: Compound search
        # ------------------------------------------------------------------
        if name == "pubchem_compound_search":
            query = arguments["query"]
            search_type = arguments.get("search_type", "name")
            max_results = arguments.get("max_results", 5)

            namespace = {"name": "name", "smiles": "smiles", "inchi": "inchi"}.get(search_type, "name")
            resp = await pubchem.get(
                f"/compound/{namespace}/{query}/cids/JSON",
            )
            data = resp.json()
            cids = data.get("IdentifierList", {}).get("CID", [])[:max_results]
            if not cids:
                return text_result({"compounds": [], "count": 0})

            # Fetch identity properties for the matched CIDs
            cid_str = ",".join(str(c) for c in cids)
            prop_resp = await pubchem.get(
                f"/compound/cid/{cid_str}/property/IUPACName,MolecularFormula,MolecularWeight,InChIKey,CanonicalSMILES/JSON",
            )
            props = prop_resp.json().get("PropertyTable", {}).get("Properties", [])
            compounds = []
            for p in props:
                compounds.append({
                    "cid": p.get("CID"),
                    "iupac_name": p.get("IUPACName", ""),
                    "molecular_formula": p.get("MolecularFormula", ""),
                    "molecular_weight": p.get("MolecularWeight"),
                    "inchikey": p.get("InChIKey", ""),
                    "canonical_smiles": p.get("CanonicalSMILES", ""),
                })
            return text_result({"compounds": compounds, "count": len(compounds)})

        # ------------------------------------------------------------------
        # PubChem: Compound properties
        # ------------------------------------------------------------------
        elif name == "pubchem_compound_properties":
            cid = arguments["cid"]
            properties = arguments.get(
                "properties",
                "MolecularFormula,MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,ExactMass,MonoisotopicMass,Complexity",
            )
            resp = await pubchem.get(
                f"/compound/cid/{cid}/property/{properties}/JSON",
            )
            data = resp.json()
            props_list = data.get("PropertyTable", {}).get("Properties", [])
            if not props_list:
                return error_result(f"No properties found for CID {cid}")
            result = props_list[0]
            # Add Lipinski rule-of-five assessment
            mw = result.get("MolecularWeight", 0)
            logp = result.get("XLogP", 0)
            hbd = result.get("HBondDonorCount", 0)
            hba = result.get("HBondAcceptorCount", 0)
            lipinski_violations = sum([
                float(mw) > 500 if mw else False,
                float(logp) > 5 if logp else False,
                int(hbd) > 5 if hbd else False,
                int(hba) > 10 if hba else False,
            ])
            result["lipinski_violations"] = lipinski_violations
            result["lipinski_pass"] = lipinski_violations <= 1
            return text_result({"properties": result})

        # ------------------------------------------------------------------
        # PubChem: BioAssay search
        # ------------------------------------------------------------------
        elif name == "pubchem_bioassay_search":
            query = arguments["query"]
            search_type = arguments.get("search_type", "target")
            max_results = arguments.get("max_results", 10)

            if search_type == "compound":
                # First resolve the compound to CID, then find assays
                cid_resp = await pubchem.get(f"/compound/name/{query}/cids/JSON")
                cids = cid_resp.json().get("IdentifierList", {}).get("CID", [])
                if not cids:
                    return text_result({"assays": [], "count": 0})
                resp = await pubchem.get(
                    f"/compound/cid/{cids[0]}/assaysummary/JSON",
                )
            else:
                # Search by target gene/protein name
                resp = await pubchem.get(
                    f"/assay/target/genesymbol/{query}/aids/JSON",
                )
            data = resp.json()

            # Extract AIDs depending on endpoint
            if search_type == "compound":
                tables = data.get("AssaySummaries", {}).get("AssaySummary", [])[:max_results]
                assays = []
                for a in tables:
                    assays.append({
                        "aid": a.get("AID"),
                        "name": a.get("AssayName", ""),
                        "activity_outcome": a.get("ActivityOutcome", ""),
                        "target_name": a.get("TargetName", ""),
                        "target_gi": a.get("TargetGI"),
                    })
                return text_result({"assays": assays, "count": len(assays)})
            else:
                aids = data.get("IdentifierList", {}).get("AID", [])[:max_results]
                if not aids:
                    return text_result({"assays": [], "count": 0})
                # Fetch assay summaries
                aid_str = ",".join(str(a) for a in aids)
                summary_resp = await pubchem.get(
                    f"/assay/aid/{aid_str}/summary/JSON",
                )
                summaries = summary_resp.json().get("AssaySummaries", {}).get("AssaySummary", [])
                assays = []
                for s in summaries:
                    assays.append({
                        "aid": s.get("AID"),
                        "name": s.get("AssayName", ""),
                        "source": s.get("SourceName", ""),
                        "description": s.get("AssayDescription", "")[:300] if s.get("AssayDescription") else "",
                        "bioassay_type": s.get("AssayType", ""),
                        "cid_count_active": s.get("CIDCountActive", 0),
                        "cid_count_tested": s.get("CIDCountTested", 0),
                    })
                return text_result({"assays": assays, "count": len(assays)})

        # ------------------------------------------------------------------
        # PubChem: Compound targets
        # ------------------------------------------------------------------
        elif name == "pubchem_compound_targets":
            cid = arguments["cid"]
            activity_type = arguments.get("activity_type", "active")

            resp = await pubchem.get(
                f"/compound/cid/{cid}/assaysummary/JSON",
            )
            data = resp.json()
            summaries = data.get("AssaySummaries", {}).get("AssaySummary", [])

            targets: dict[str, dict] = {}
            for s in summaries:
                outcome = s.get("ActivityOutcome", "").lower()
                if activity_type != "all" and outcome != activity_type:
                    continue
                target_name = s.get("TargetName", "")
                target_gi = s.get("TargetGI")
                gene_id = s.get("GeneID")
                if not target_name:
                    continue
                key = target_name
                if key not in targets:
                    targets[key] = {
                        "target_name": target_name,
                        "target_gi": target_gi,
                        "gene_id": gene_id,
                        "assay_count": 0,
                        "active_count": 0,
                    }
                targets[key]["assay_count"] += 1
                if outcome == "active":
                    targets[key]["active_count"] += 1

            target_list = sorted(targets.values(), key=lambda x: x["active_count"], reverse=True)
            return text_result({"cid": cid, "targets": target_list[:50], "total_targets": len(target_list)})

        # ------------------------------------------------------------------
        # BindingDB: Search by SMILES or name
        # ------------------------------------------------------------------
        elif name == "bindingdb_search":
            query = arguments["query"]
            search_type = arguments.get("search_type", "smiles")
            similarity = arguments.get("similarity", 0.85)

            if search_type == "smiles":
                resp = await bindingdb.get(
                    "/getLigandsBySmiles",
                    params={"smiles": query, "cutoff": similarity, "response": "json"},
                )
            else:
                resp = await bindingdb.get(
                    "/getLigandsByName",
                    params={"name": query, "response": "json"},
                )

            data = resp.json()
            # BindingDB wraps results in an affinities list
            affinities_raw = data.get("getLigandsBySmiles", data.get("getLigandsByName", {}))
            records = affinities_raw if isinstance(affinities_raw, list) else affinities_raw.get("affinities", [])
            results = []
            for rec in records[:30]:
                results.append({
                    "monomer_id": rec.get("monomerid", ""),
                    "smiles": rec.get("smiles", ""),
                    "target": rec.get("target", ""),
                    "ki": rec.get("ki", ""),
                    "kd": rec.get("kd", ""),
                    "ic50": rec.get("ic50", ""),
                    "ec50": rec.get("ec50", ""),
                    "source": rec.get("source", ""),
                    "doi": rec.get("doi", ""),
                    "uniprot_id": rec.get("uniprot_id", ""),
                })
            return text_result({"bindings": results, "count": len(results)})

        # ------------------------------------------------------------------
        # BindingDB: Target lookup by UniProt
        # ------------------------------------------------------------------
        elif name == "bindingdb_target":
            uniprot_id = arguments["uniprot_id"]
            max_results = arguments.get("max_results", 20)

            resp = await bindingdb.get(
                "/getLigandsByUniprots",
                params={"uniprot": uniprot_id, "response": "json"},
            )
            data = resp.json()
            affinities_raw = data.get("getLigandsByUniprots", {})
            records = affinities_raw if isinstance(affinities_raw, list) else affinities_raw.get("affinities", [])
            results = []
            for rec in records[:max_results]:
                results.append({
                    "monomer_id": rec.get("monomerid", ""),
                    "smiles": rec.get("smiles", ""),
                    "compound_name": rec.get("zinc_id", rec.get("monomerid", "")),
                    "ki": rec.get("ki", ""),
                    "kd": rec.get("kd", ""),
                    "ic50": rec.get("ic50", ""),
                    "ec50": rec.get("ec50", ""),
                    "source": rec.get("source", ""),
                    "doi": rec.get("doi", ""),
                    "pmid": rec.get("pmid", ""),
                })
            return text_result({
                "uniprot_id": uniprot_id,
                "bindings": results,
                "count": len(results),
            })

        # ------------------------------------------------------------------
        # OpenFDA: Adverse events
        # ------------------------------------------------------------------
        elif name == "openfda_adverse_events":
            drug_name = arguments["drug_name"]
            reaction = arguments.get("reaction", "")
            max_results = arguments.get("max_results", 10)

            search_parts = [f'patient.drug.medicinalproduct:"{drug_name}"']
            if reaction:
                search_parts.append(f'patient.reaction.reactionmeddrapt:"{reaction}"')
            search_query = "+AND+".join(search_parts)

            params: dict = {"search": search_query, "limit": max_results}
            if OPENFDA_API_KEY:
                params["api_key"] = OPENFDA_API_KEY

            resp = await openfda.get("/drug/event.json", params=params)
            data = resp.json()
            events = data.get("results", [])
            results = []
            for ev in events:
                drugs = []
                for d in ev.get("patient", {}).get("drug", []):
                    drugs.append({
                        "name": d.get("medicinalproduct", ""),
                        "indication": d.get("drugindication", ""),
                        "characterization": d.get("drugcharacterization", ""),
                    })
                reactions = [
                    r.get("reactionmeddrapt", "")
                    for r in ev.get("patient", {}).get("reaction", [])
                ]
                results.append({
                    "safety_report_id": ev.get("safetyreportid", ""),
                    "receive_date": ev.get("receivedate", ""),
                    "serious": ev.get("serious", ""),
                    "patient_sex": ev.get("patient", {}).get("patientsex", ""),
                    "patient_age": ev.get("patient", {}).get("patientonsetage", ""),
                    "drugs": drugs,
                    "reactions": reactions,
                    "outcome": ev.get("patient", {}).get("patientdeath", {}).get("patientdeathdate", ""),
                })
            meta = data.get("meta", {}).get("results", {})
            return text_result({
                "events": results,
                "count": len(results),
                "total": meta.get("total", len(results)),
            })

        # ------------------------------------------------------------------
        # OpenFDA: Drug labeling
        # ------------------------------------------------------------------
        elif name == "openfda_drug_labeling":
            query = arguments["query"]
            section = arguments.get("section", "")
            max_results = arguments.get("max_results", 3)

            search_query = f'openfda.generic_name:"{query}"+openfda.brand_name:"{query}"'
            params: dict = {"search": search_query, "limit": max_results}
            if OPENFDA_API_KEY:
                params["api_key"] = OPENFDA_API_KEY

            resp = await openfda.get("/drug/label.json", params=params)
            data = resp.json()
            labels = data.get("results", [])
            results = []
            for label in labels:
                entry: dict = {
                    "spl_id": label.get("id", ""),
                    "brand_name": label.get("openfda", {}).get("brand_name", [""])[0],
                    "generic_name": label.get("openfda", {}).get("generic_name", [""])[0],
                    "manufacturer": label.get("openfda", {}).get("manufacturer_name", [""])[0],
                    "route": label.get("openfda", {}).get("route", [""])[0],
                    "product_type": label.get("openfda", {}).get("product_type", [""])[0],
                }
                # Include requested section or all key sections
                sections_map = {
                    "indications_and_usage": "indications_and_usage",
                    "warnings": "warnings",
                    "adverse_reactions": "adverse_reactions",
                    "drug_interactions": "drug_interactions",
                    "dosage_and_administration": "dosage_and_administration",
                    "contraindications": "contraindications",
                    "mechanism_of_action": "mechanism_of_action",
                    "clinical_pharmacology": "clinical_pharmacology",
                    "boxed_warning": "boxed_warning",
                }
                if section and section in sections_map:
                    key = sections_map[section]
                    content = label.get(key, [])
                    entry[key] = content[0][:2000] if content else ""
                else:
                    for sec_name, sec_key in sections_map.items():
                        content = label.get(sec_key, [])
                        if content:
                            entry[sec_name] = content[0][:1000]
                results.append(entry)
            return text_result({"labels": results, "count": len(results)})

        # ------------------------------------------------------------------
        # DailyMed: Search
        # ------------------------------------------------------------------
        elif name == "dailymed_search":
            drug_name = arguments["drug_name"]
            max_results = arguments.get("max_results", 5)

            resp = await dailymed.get(
                "/spls.json",
                params={"drug_name": drug_name, "page": 1, "pagesize": max_results},
            )
            data = resp.json()
            spls = data.get("data", [])
            results = []
            for spl in spls:
                set_id = spl.get("setid", "")
                results.append({
                    "set_id": set_id,
                    "spl_version": spl.get("spl_version", ""),
                    "title": spl.get("title", ""),
                    "published_date": spl.get("published_date", ""),
                    "labeler": spl.get("labeler", ""),
                    "products": spl.get("products", []),
                    "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}" if set_id else "",
                })
            metadata = data.get("metadata", {})
            return text_result({
                "labels": results,
                "count": len(results),
                "total": metadata.get("total_elements", len(results)),
            })

        # ------------------------------------------------------------------
        # UniChem: ID conversion
        # ------------------------------------------------------------------
        elif name == "unichem_convert":
            identifier = arguments["identifier"]
            source_type = arguments.get("source_type", "inchikey")

            if source_type == "inchikey":
                resp = await unichem.get(f"/inchikey/{identifier}")
            else:
                src_id = _UNICHEM_SRC.get(source_type)
                if src_id is None:
                    return error_result(
                        f"Unknown source type: {source_type}. "
                        f"Supported: {', '.join(_UNICHEM_SRC.keys())}"
                    )
                resp = await unichem.get(f"/src_compound_id/{identifier}/{src_id}")

            data = resp.json()
            # Map source IDs back to human-readable names
            src_name_map = {v: k for k, v in _UNICHEM_SRC.items()}
            mappings = []
            if isinstance(data, list):
                for entry in data:
                    src_num = int(entry.get("src_id", 0))
                    mappings.append({
                        "source": src_name_map.get(src_num, f"source_{src_num}"),
                        "source_id": src_num,
                        "compound_id": entry.get("src_compound_id", ""),
                    })
            return text_result({
                "query": identifier,
                "source_type": source_type,
                "mappings": mappings,
                "count": len(mappings),
            })

        # ------------------------------------------------------------------
        # DDInter: Drug-drug interactions
        # ------------------------------------------------------------------
        elif name == "ddinter_search":
            drug_name = arguments["drug_name"]
            max_results = arguments.get("max_results", 10)

            resp = await ddinter.post(
                "/search/drug",
                json={"drugname": drug_name},
            )
            data = resp.json()
            drug_list = data if isinstance(data, list) else data.get("data", data.get("results", []))

            if not drug_list:
                return text_result({"drug": drug_name, "interactions": [], "count": 0})

            # Grab the first matched drug entry
            drug_entry = drug_list[0] if isinstance(drug_list, list) else drug_list
            drug_id = drug_entry.get("id", drug_entry.get("drug_id", ""))

            if not drug_id:
                return text_result({"drug": drug_name, "interactions": [], "count": 0})

            # Fetch interactions for this drug
            inter_resp = await ddinter.get(
                f"/drug/{drug_id}/interactions",
                params={"limit": max_results},
            )
            inter_data = inter_resp.json()
            interactions_raw = inter_data if isinstance(inter_data, list) else inter_data.get("data", inter_data.get("interactions", []))

            interactions = []
            for ix in interactions_raw[:max_results]:
                interactions.append({
                    "interaction_id": ix.get("id", ix.get("interaction_id", "")),
                    "drug_a": ix.get("drug_a", ix.get("drugA", drug_name)),
                    "drug_b": ix.get("drug_b", ix.get("drugB", "")),
                    "level": ix.get("level", ix.get("severity", "")),
                    "description": ix.get("description", ix.get("mechanism", "")),
                })
            return text_result({
                "drug": drug_name,
                "drug_id": drug_id,
                "interactions": interactions,
                "count": len(interactions),
            })

        return error_result(f"Unknown tool: {name}")

    except Exception as exc:
        return error_result(f"{name} failed: {exc}")


if __name__ == "__main__":
    start(server)
