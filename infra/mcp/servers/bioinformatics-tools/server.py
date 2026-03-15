"""Bioinformatics Tools MCP Server — BLAST, Clustal Omega, MUSCLE, InterProScan, HMMER,
AlphaFold, SWISS-MODEL.

Exposes web-accessible bioinformatics analysis tools. All compute-heavy tools use
submit-then-poll async job APIs (NCBI BLAST, EBI Job Dispatcher, etc.).
"""

from __future__ import annotations

import os
import sys
import time
import asyncio
from xml.etree import ElementTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

from base_server import APIClient, Server, error_result, make_tool, start, text_result

server = Server("bioinformatics-tools")

# ---------------------------------------------------------------------------
# API Clients
# ---------------------------------------------------------------------------

NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
SWISSMODEL_API_KEY = os.getenv("SWISSMODEL_API_KEY", "")

ncbi_blast = APIClient(
    base_url="https://blast.ncbi.nlm.nih.gov/blast",
    rate_limit=1,
    timeout=60,
)
ebi_rest = APIClient(
    base_url="https://www.ebi.ac.uk/Tools/services/rest",
    rate_limit=3,
    timeout=60,
)
ebi_hmmer = APIClient(
    base_url="https://www.ebi.ac.uk/Tools/hmmer",
    headers={"Accept": "application/json"},
    rate_limit=2,
    timeout=60,
)
swissmodel = APIClient(
    base_url="https://swissmodel.expasy.org",
    headers={
        "Authorization": f"Token {SWISSMODEL_API_KEY}" if SWISSMODEL_API_KEY else "",
        "Accept": "application/json",
    },
    rate_limit=1,
    timeout=60,
)

# ---------------------------------------------------------------------------
# Helpers — async job polling
# ---------------------------------------------------------------------------

MAX_POLL_SECONDS = 300  # 5 min ceiling for a single poll cycle
POLL_INTERVAL = 5  # seconds between status checks


async def _poll_ebi_job(tool: str, job_id: str) -> dict:
    """Poll an EBI Job Dispatcher job until FINISHED, ERROR, or timeout."""
    deadline = time.monotonic() + MAX_POLL_SECONDS
    while time.monotonic() < deadline:
        resp = await ebi_rest.get(f"/{tool}/status/{job_id}")
        status = resp.text.strip()
        if status == "FINISHED":
            return {"job_id": job_id, "status": "FINISHED"}
        if status in ("ERROR", "FAILURE", "NOT_FOUND"):
            return {"job_id": job_id, "status": status, "error": f"Job {status}"}
        await asyncio.sleep(POLL_INTERVAL)
    return {"job_id": job_id, "status": "TIMEOUT", "error": "Polling exceeded time limit"}


async def _poll_ncbi_blast(rid: str) -> dict:
    """Poll NCBI BLAST for a request ID until READY or timeout."""
    deadline = time.monotonic() + MAX_POLL_SECONDS
    while time.monotonic() < deadline:
        resp = await ncbi_blast.get(
            "/Blast.cgi",
            params={"CMD": "Get", "FORMAT_OBJECT": "SearchInfo", "RID": rid},
        )
        body = resp.text
        if "Status=READY" in body:
            if "ThereAreHits=yes" in body:
                return {"rid": rid, "status": "READY", "has_hits": True}
            return {"rid": rid, "status": "READY", "has_hits": False}
        if "Status=FAILED" in body:
            return {"rid": rid, "status": "FAILED", "error": "BLAST job failed"}
        if "Status=UNKNOWN" in body:
            return {"rid": rid, "status": "UNKNOWN", "error": "Invalid or expired RID"}
        await asyncio.sleep(POLL_INTERVAL)
    return {"rid": rid, "status": "TIMEOUT", "error": "Polling exceeded time limit"}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list:
    return [
        # -- NCBI BLAST --
        make_tool(
            "ncbi_blast_search",
            "Submit a BLAST search to NCBI. Returns a Request ID (RID) for async result retrieval. "
            "Supports blastn, blastp, blastx, tblastn, tblastx.",
            {
                "program": {
                    "type": "string",
                    "description": "BLAST program: blastn, blastp, blastx, tblastn, tblastx",
                    "enum": ["blastn", "blastp", "blastx", "tblastn", "tblastx"],
                },
                "sequence": {
                    "type": "string",
                    "description": "Query sequence (FASTA or raw nucleotide/protein)",
                },
                "database": {
                    "type": "string",
                    "description": "Target database (e.g. 'nt', 'nr', 'swissprot', 'refseq_protein')",
                    "default": "nr",
                },
                "evalue": {
                    "type": "string",
                    "description": "Expect value threshold",
                    "default": "1e-5",
                },
                "max_hits": {
                    "type": "integer",
                    "description": "Maximum number of aligned sequences to return",
                    "default": 50,
                },
                "wait": {
                    "type": "boolean",
                    "description": "If true, poll until results are ready (up to 5 min). If false, return RID immediately.",
                    "default": False,
                },
            },
            required=["program", "sequence"],
        ),
        make_tool(
            "ncbi_blast_results",
            "Retrieve results for a previously submitted NCBI BLAST job by RID.",
            {
                "rid": {
                    "type": "string",
                    "description": "BLAST Request ID returned from ncbi_blast_search",
                },
                "format": {
                    "type": "string",
                    "description": "Output format: JSON, XML, Text, Tabular",
                    "default": "JSON",
                    "enum": ["JSON", "XML", "Text", "Tabular"],
                },
                "max_hits": {
                    "type": "integer",
                    "description": "Maximum hits to return",
                    "default": 50,
                },
            },
            required=["rid"],
        ),

        # -- EBI Clustal Omega --
        make_tool(
            "ebi_clustalo",
            "Submit multiple sequence alignment to EMBL-EBI Clustal Omega. "
            "Input: sequences in FASTA format. Returns job ID for async result retrieval.",
            {
                "sequences": {
                    "type": "string",
                    "description": "Input sequences in FASTA format (multiple sequences)",
                },
                "stype": {
                    "type": "string",
                    "description": "Sequence type: protein or dna",
                    "default": "protein",
                    "enum": ["protein", "dna"],
                },
                "outfmt": {
                    "type": "string",
                    "description": "Output alignment format: clustal_num, fasta, msf, nexus, phylip, selex, stockholm",
                    "default": "clustal_num",
                },
                "wait": {
                    "type": "boolean",
                    "description": "If true, poll until results are ready. If false, return job ID immediately.",
                    "default": False,
                },
            },
            required=["sequences"],
        ),
        make_tool(
            "ebi_clustalo_results",
            "Get results for a completed EMBL-EBI Clustal Omega job.",
            {
                "job_id": {
                    "type": "string",
                    "description": "Clustal Omega job ID",
                },
                "result_type": {
                    "type": "string",
                    "description": "Result type: aln-clustal_num, aln-fasta, pim (percent identity matrix), phylotree",
                    "default": "aln-clustal_num",
                },
            },
            required=["job_id"],
        ),

        # -- EBI MUSCLE --
        make_tool(
            "ebi_muscle",
            "Submit multiple sequence alignment to EMBL-EBI MUSCLE service. "
            "Returns job ID. Use ebi_clustalo_results with the job ID to retrieve output (same dispatcher).",
            {
                "sequences": {
                    "type": "string",
                    "description": "Input sequences in FASTA format",
                },
                "output": {
                    "type": "string",
                    "description": "Output format: clw (ClustalW), fasta, html, msf, phylip",
                    "default": "fasta",
                },
                "wait": {
                    "type": "boolean",
                    "description": "If true, poll until results are ready.",
                    "default": False,
                },
            },
            required=["sequences"],
        ),

        # -- EBI InterProScan --
        make_tool(
            "ebi_interproscan",
            "Submit a protein sequence to EMBL-EBI InterProScan for domain/family annotation. "
            "Returns job ID for async retrieval.",
            {
                "sequence": {
                    "type": "string",
                    "description": "Protein sequence (FASTA or raw amino acid)",
                },
                "applications": {
                    "type": "string",
                    "description": "Comma-separated list of InterPro member databases to run "
                    "(e.g. 'Pfam,SMART,CDD'). Leave empty for all.",
                    "default": "",
                },
                "goterms": {
                    "type": "boolean",
                    "description": "Include Gene Ontology annotations",
                    "default": True,
                },
                "pathways": {
                    "type": "boolean",
                    "description": "Include pathway annotations",
                    "default": True,
                },
                "wait": {
                    "type": "boolean",
                    "description": "If true, poll until results are ready.",
                    "default": False,
                },
            },
            required=["sequence"],
        ),
        make_tool(
            "ebi_interproscan_results",
            "Get results for a completed InterProScan job.",
            {
                "job_id": {
                    "type": "string",
                    "description": "InterProScan job ID",
                },
                "result_type": {
                    "type": "string",
                    "description": "Result type: json, tsv, xml, gff, svg",
                    "default": "json",
                },
            },
            required=["job_id"],
        ),

        # -- EBI HMMER --
        make_tool(
            "ebi_hmmer_search",
            "Search protein families using the HMMER web service (phmmer, hmmscan, hmmsearch, jackhmmer). "
            "Synchronous — returns results directly.",
            {
                "sequence": {
                    "type": "string",
                    "description": "Protein sequence (raw amino acid or FASTA)",
                },
                "search_type": {
                    "type": "string",
                    "description": "Search type: phmmer (seq vs seq DB), hmmscan (seq vs profile DB)",
                    "default": "phmmer",
                    "enum": ["phmmer", "hmmscan"],
                },
                "database": {
                    "type": "string",
                    "description": "Target database. For phmmer: uniprotrefprot, swissprot, pdb, etc. "
                    "For hmmscan: pfam, tigrfam, gene3d, superfamily.",
                    "default": "pdb",
                },
                "max_hits": {
                    "type": "integer",
                    "description": "Max number of hits to return",
                    "default": 20,
                },
            },
            required=["sequence"],
        ),

        # -- AlphaFold --
        make_tool(
            "alphafold_predict",
            "Submit a protein sequence for structure prediction via AlphaFold DB lookup or "
            "queue-based prediction. First checks AlphaFold DB by UniProt accession; if not found, "
            "explains that on-demand prediction requires ColabFold or local setup.",
            {
                "sequence": {
                    "type": "string",
                    "description": "Protein sequence (amino acid) or UniProt accession",
                },
                "uniprot_id": {
                    "type": "string",
                    "description": "UniProt accession to check AlphaFold DB directly (optional)",
                    "default": "",
                },
            },
            required=["sequence"],
        ),

        # -- SWISS-MODEL --
        make_tool(
            "swiss_model",
            "Submit homology modeling to SWISS-MODEL. Returns a project ID for tracking. "
            "Requires SWISSMODEL_API_KEY environment variable.",
            {
                "sequence": {
                    "type": "string",
                    "description": "Target protein sequence (amino acid)",
                },
                "project_title": {
                    "type": "string",
                    "description": "Title for the modeling project",
                    "default": "YOHAS MCP modeling",
                },
                "template_id": {
                    "type": "string",
                    "description": "PDB template ID to use for modeling (optional; SWISS-MODEL auto-selects if omitted)",
                    "default": "",
                },
                "wait": {
                    "type": "boolean",
                    "description": "If true, poll until modeling is complete (up to 5 min).",
                    "default": False,
                },
            },
            required=["sequence"],
        ),
    ]


# ---------------------------------------------------------------------------
# Tool Dispatch
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    try:
        # ==================================================================
        # NCBI BLAST — Submit
        # ==================================================================
        if name == "ncbi_blast_search":
            program = arguments["program"]
            sequence = arguments["sequence"]
            database = arguments.get("database", "nr")
            evalue = arguments.get("evalue", "1e-5")
            max_hits = arguments.get("max_hits", 50)
            wait = arguments.get("wait", False)

            params = {
                "CMD": "Put",
                "PROGRAM": program,
                "DATABASE": database,
                "QUERY": sequence,
                "EXPECT": evalue,
                "HITLIST_SIZE": str(max_hits),
                "FORMAT_TYPE": "JSON2",
            }
            if NCBI_API_KEY:
                params["API_KEY"] = NCBI_API_KEY

            resp = await ncbi_blast.post("/Blast.cgi", data=params)
            body = resp.text

            # Extract RID from QBlastInfo
            rid = ""
            rtoe = ""
            for line in body.splitlines():
                if line.strip().startswith("RID ="):
                    rid = line.split("=")[1].strip()
                elif line.strip().startswith("RTOE ="):
                    rtoe = line.split("=")[1].strip()

            if not rid:
                return error_result(f"BLAST submission failed — could not extract RID. Response: {body[:500]}")

            result = {
                "rid": rid,
                "estimated_seconds": int(rtoe) if rtoe else None,
                "status": "SUBMITTED",
                "program": program,
                "database": database,
            }

            if wait:
                poll_result = await _poll_ncbi_blast(rid)
                result.update(poll_result)
                if poll_result.get("status") == "READY" and poll_result.get("has_hits"):
                    # Fetch JSON results
                    res_resp = await ncbi_blast.get(
                        "/Blast.cgi",
                        params={
                            "CMD": "Get",
                            "RID": rid,
                            "FORMAT_TYPE": "JSON2",
                            "HITLIST_SIZE": str(max_hits),
                        },
                    )
                    try:
                        result["results"] = res_resp.json()
                    except Exception:
                        result["results_text"] = res_resp.text[:5000]

            return text_result(result)

        # ==================================================================
        # NCBI BLAST — Results
        # ==================================================================
        elif name == "ncbi_blast_results":
            rid = arguments["rid"]
            fmt = arguments.get("format", "JSON")
            max_hits = arguments.get("max_hits", 50)

            # Check status first
            poll_result = await _poll_ncbi_blast(rid)
            if poll_result.get("status") != "READY":
                return text_result(poll_result)

            format_map = {
                "JSON": "JSON2",
                "XML": "XML",
                "Text": "Text",
                "Tabular": "Tabular",
            }

            resp = await ncbi_blast.get(
                "/Blast.cgi",
                params={
                    "CMD": "Get",
                    "RID": rid,
                    "FORMAT_TYPE": format_map.get(fmt, "JSON2"),
                    "HITLIST_SIZE": str(max_hits),
                },
            )

            if fmt == "JSON":
                try:
                    data = resp.json()
                    # Extract key fields from BLAST JSON2 format
                    search = data.get("BlastOutput2", [{}])
                    hits = []
                    if search:
                        report = search[0].get("report", {})
                        search_results = report.get("results", {}).get("search", {})
                        for hit in search_results.get("hits", [])[:max_hits]:
                            desc = hit.get("description", [{}])[0] if hit.get("description") else {}
                            hsps = hit.get("hsps", [{}])
                            top_hsp = hsps[0] if hsps else {}
                            hits.append({
                                "accession": desc.get("accession", ""),
                                "title": desc.get("title", ""),
                                "sciname": desc.get("sciname", ""),
                                "evalue": top_hsp.get("evalue"),
                                "bit_score": top_hsp.get("bit_score"),
                                "identity_pct": (
                                    round(top_hsp["identity"] / top_hsp["align_len"] * 100, 1)
                                    if top_hsp.get("identity") and top_hsp.get("align_len")
                                    else None
                                ),
                                "query_from": top_hsp.get("query_from"),
                                "query_to": top_hsp.get("query_to"),
                                "hit_from": top_hsp.get("hit_from"),
                                "hit_to": top_hsp.get("hit_to"),
                            })
                        program = report.get("program", "")
                        db = report.get("search_target", {}).get("db", "")
                        stats = search_results.get("stat", {})
                        return text_result({
                            "rid": rid,
                            "program": program,
                            "database": db,
                            "hits": hits,
                            "total_hits": len(search_results.get("hits", [])),
                            "stats": {
                                "db_num": stats.get("db_num"),
                                "db_len": stats.get("db_len"),
                            },
                        })
                except Exception:
                    pass
            # Fallback: return raw text (truncated)
            return text_result({"rid": rid, "format": fmt, "raw": resp.text[:8000]})

        # ==================================================================
        # EBI Clustal Omega — Submit
        # ==================================================================
        elif name == "ebi_clustalo":
            sequences = arguments["sequences"]
            stype = arguments.get("stype", "protein")
            outfmt = arguments.get("outfmt", "clustal_num")
            wait = arguments.get("wait", False)

            resp = await ebi_rest.post(
                "/clustalo/run",
                data={
                    "email": os.getenv("EBI_EMAIL", "yohas-mcp@example.org"),
                    "sequence": sequences,
                    "stype": stype,
                    "outfmt": outfmt,
                },
            )
            job_id = resp.text.strip()

            result: dict = {"job_id": job_id, "status": "SUBMITTED", "tool": "clustalo"}

            if wait:
                poll = await _poll_ebi_job("clustalo", job_id)
                result.update(poll)
                if poll["status"] == "FINISHED":
                    aln_resp = await ebi_rest.get(f"/clustalo/result/{job_id}/aln-{outfmt}")
                    result["alignment"] = aln_resp.text

            return text_result(result)

        # ==================================================================
        # EBI Clustal Omega — Results
        # ==================================================================
        elif name == "ebi_clustalo_results":
            job_id = arguments["job_id"]
            result_type = arguments.get("result_type", "aln-clustal_num")

            # Check status
            poll = await _poll_ebi_job("clustalo", job_id)
            if poll["status"] != "FINISHED":
                return text_result(poll)

            resp = await ebi_rest.get(f"/clustalo/result/{job_id}/{result_type}")
            return text_result({
                "job_id": job_id,
                "result_type": result_type,
                "data": resp.text,
            })

        # ==================================================================
        # EBI MUSCLE — Submit
        # ==================================================================
        elif name == "ebi_muscle":
            sequences = arguments["sequences"]
            output = arguments.get("output", "fasta")
            wait = arguments.get("wait", False)

            resp = await ebi_rest.post(
                "/muscle/run",
                data={
                    "email": os.getenv("EBI_EMAIL", "yohas-mcp@example.org"),
                    "sequence": sequences,
                    "output": output,
                    "format": "fasta",
                },
            )
            job_id = resp.text.strip()

            result = {"job_id": job_id, "status": "SUBMITTED", "tool": "muscle"}

            if wait:
                poll = await _poll_ebi_job("muscle", job_id)
                result.update(poll)
                if poll["status"] == "FINISHED":
                    aln_resp = await ebi_rest.get(f"/muscle/result/{job_id}/aln-{output}")
                    result["alignment"] = aln_resp.text

            return text_result(result)

        # ==================================================================
        # EBI InterProScan — Submit
        # ==================================================================
        elif name == "ebi_interproscan":
            sequence = arguments["sequence"]
            applications = arguments.get("applications", "")
            goterms = arguments.get("goterms", True)
            pathways = arguments.get("pathways", True)
            wait = arguments.get("wait", False)

            form_data: dict = {
                "email": os.getenv("EBI_EMAIL", "yohas-mcp@example.org"),
                "sequence": sequence,
                "goterms": str(goterms).lower(),
                "pathways": str(pathways).lower(),
                "stype": "p",  # protein
            }
            if applications:
                form_data["appl"] = applications

            resp = await ebi_rest.post("/iprscan5/run", data=form_data)
            job_id = resp.text.strip()

            result = {"job_id": job_id, "status": "SUBMITTED", "tool": "iprscan5"}

            if wait:
                poll = await _poll_ebi_job("iprscan5", job_id)
                result.update(poll)
                if poll["status"] == "FINISHED":
                    json_resp = await ebi_rest.get(f"/iprscan5/result/{job_id}/json")
                    try:
                        iprscan_data = json_resp.json()
                        matches = []
                        for res in iprscan_data.get("results", []):
                            for match in res.get("matches", []):
                                sig = match.get("signature", {})
                                entry = sig.get("entry", {}) or {}
                                locs = match.get("locations", [])
                                matches.append({
                                    "signature_ac": sig.get("accession", ""),
                                    "signature_name": sig.get("name", ""),
                                    "signature_db": sig.get("signatureLibraryRelease", {}).get("library", ""),
                                    "interpro_ac": entry.get("accession", ""),
                                    "interpro_name": entry.get("name", ""),
                                    "interpro_type": entry.get("type", ""),
                                    "locations": [
                                        {"start": loc.get("start"), "end": loc.get("end"), "score": loc.get("score")}
                                        for loc in locs
                                    ],
                                    "go_terms": [
                                        {"id": go.get("identifier", ""), "name": go.get("name", ""), "category": go.get("category", "")}
                                        for go in entry.get("goXRefs", [])
                                    ],
                                    "pathways": [
                                        {"db": pw.get("databaseName", ""), "id": pw.get("id", ""), "name": pw.get("name", "")}
                                        for pw in entry.get("pathwayXRefs", [])
                                    ],
                                })
                        result["matches"] = matches
                        result["match_count"] = len(matches)
                    except Exception:
                        result["raw"] = json_resp.text[:5000]

            return text_result(result)

        # ==================================================================
        # EBI InterProScan — Results
        # ==================================================================
        elif name == "ebi_interproscan_results":
            job_id = arguments["job_id"]
            result_type = arguments.get("result_type", "json")

            poll = await _poll_ebi_job("iprscan5", job_id)
            if poll["status"] != "FINISHED":
                return text_result(poll)

            resp = await ebi_rest.get(f"/iprscan5/result/{job_id}/{result_type}")

            if result_type == "json":
                try:
                    iprscan_data = resp.json()
                    matches = []
                    for res in iprscan_data.get("results", []):
                        for match in res.get("matches", []):
                            sig = match.get("signature", {})
                            entry = sig.get("entry", {}) or {}
                            locs = match.get("locations", [])
                            matches.append({
                                "signature_ac": sig.get("accession", ""),
                                "signature_name": sig.get("name", ""),
                                "signature_db": sig.get("signatureLibraryRelease", {}).get("library", ""),
                                "interpro_ac": entry.get("accession", ""),
                                "interpro_name": entry.get("name", ""),
                                "interpro_type": entry.get("type", ""),
                                "locations": [
                                    {"start": loc.get("start"), "end": loc.get("end"), "score": loc.get("score")}
                                    for loc in locs
                                ],
                                "go_terms": [
                                    {"id": go.get("identifier", ""), "name": go.get("name", ""), "category": go.get("category", "")}
                                    for go in entry.get("goXRefs", [])
                                ],
                                "pathways": [
                                    {"db": pw.get("databaseName", ""), "id": pw.get("id", ""), "name": pw.get("name", "")}
                                    for pw in entry.get("pathwayXRefs", [])
                                ],
                            })
                    return text_result({
                        "job_id": job_id,
                        "matches": matches,
                        "match_count": len(matches),
                    })
                except Exception:
                    pass

            return text_result({"job_id": job_id, "result_type": result_type, "data": resp.text[:8000]})

        # ==================================================================
        # EBI HMMER — Synchronous search
        # ==================================================================
        elif name == "ebi_hmmer_search":
            sequence = arguments["sequence"]
            search_type = arguments.get("search_type", "phmmer")
            database = arguments.get("database", "pdb")
            max_hits = arguments.get("max_hits", 20)

            # HMMER web service accepts form-encoded POST
            resp = await ebi_hmmer.post(
                f"/search/{search_type}",
                data={
                    "seqdb": database,
                    "seq": sequence,
                },
                headers={"Accept": "application/json"},
            )
            data = resp.json()

            hits = []
            for hit in data.get("results", {}).get("hits", [])[:max_hits]:
                domains = []
                for dom in hit.get("domains", []):
                    domains.append({
                        "accession": dom.get("iali", ""),
                        "env_from": dom.get("ienv"),
                        "env_to": dom.get("jenv"),
                        "ali_from": dom.get("iali"),
                        "ali_to": dom.get("jali"),
                        "score": dom.get("bitscore"),
                        "evalue": dom.get("ievalue"),
                    })
                hits.append({
                    "name": hit.get("name", ""),
                    "accession": hit.get("acc", ""),
                    "description": hit.get("desc", ""),
                    "score": hit.get("score"),
                    "evalue": hit.get("evalue"),
                    "pvalue": hit.get("pvalue"),
                    "ndom": hit.get("ndom"),
                    "domains": domains,
                })

            stats = data.get("results", {}).get("stats", {})
            return text_result({
                "search_type": search_type,
                "database": database,
                "hits": hits,
                "total_hits": len(data.get("results", {}).get("hits", [])),
                "stats": {
                    "nhits": stats.get("nhits"),
                    "Z": stats.get("Z"),
                    "elapsed": stats.get("elapsed"),
                },
            })

        # ==================================================================
        # AlphaFold — DB lookup / prediction stub
        # ==================================================================
        elif name == "alphafold_predict":
            sequence = arguments["sequence"]
            uniprot_id = arguments.get("uniprot_id", "")

            alphafold_db = APIClient(
                base_url="https://alphafold.ebi.ac.uk/api",
                rate_limit=5,
                timeout=30,
            )

            # If a UniProt ID is given, try AlphaFold DB first
            if uniprot_id:
                try:
                    resp = await alphafold_db.get(f"/prediction/{uniprot_id}")
                    data = resp.json()
                    await alphafold_db.close()
                    if isinstance(data, list) and data:
                        entry = data[0]
                        return text_result({
                            "source": "alphafold_db",
                            "uniprot_id": uniprot_id,
                            "entry_id": entry.get("entryId", ""),
                            "gene": entry.get("gene", ""),
                            "organism": entry.get("organismScientificName", ""),
                            "pdb_url": entry.get("pdbUrl", ""),
                            "cif_url": entry.get("cifUrl", ""),
                            "pae_image_url": entry.get("paeImageUrl", ""),
                            "model_confidence": entry.get("globalMetricValue"),
                            "latest_version": entry.get("latestVersion"),
                        })
                except Exception:
                    pass

            # Try to look up by sequence — search UniProt first, then AlphaFold DB
            # Use HMMER phmmer to identify the sequence
            try:
                hmmer_resp = await ebi_hmmer.post(
                    "/search/phmmer",
                    data={"seqdb": "swissprot", "seq": sequence},
                    headers={"Accept": "application/json"},
                )
                hmmer_data = hmmer_resp.json()
                top_hits = hmmer_data.get("results", {}).get("hits", [])
                if top_hits:
                    best_acc = top_hits[0].get("acc", "").split(".")[0]
                    if best_acc:
                        try:
                            af_resp = await alphafold_db.get(f"/prediction/{best_acc}")
                            af_data = af_resp.json()
                            if isinstance(af_data, list) and af_data:
                                entry = af_data[0]
                                await alphafold_db.close()
                                return text_result({
                                    "source": "alphafold_db",
                                    "matched_uniprot": best_acc,
                                    "entry_id": entry.get("entryId", ""),
                                    "gene": entry.get("gene", ""),
                                    "organism": entry.get("organismScientificName", ""),
                                    "pdb_url": entry.get("pdbUrl", ""),
                                    "cif_url": entry.get("cifUrl", ""),
                                    "pae_image_url": entry.get("paeImageUrl", ""),
                                    "model_confidence": entry.get("globalMetricValue"),
                                    "latest_version": entry.get("latestVersion"),
                                })
                        except Exception:
                            pass
            except Exception:
                pass

            await alphafold_db.close()
            return text_result({
                "source": "none",
                "message": "No pre-computed AlphaFold structure found. "
                "For de novo prediction, use ColabFold (https://colab.research.google.com/github/sokrypton/ColabFold) "
                "or a local AlphaFold installation. Sequence length: " + str(len(sequence.replace('\n', '').replace('>', ''))) + " residues.",
                "alternatives": [
                    "swiss_model — homology modeling via SWISS-MODEL",
                    "ebi_hmmer_search — find related structures in PDB",
                ],
            })

        # ==================================================================
        # SWISS-MODEL — Homology Modeling
        # ==================================================================
        elif name == "swiss_model":
            if not SWISSMODEL_API_KEY:
                return error_result(
                    "SWISSMODEL_API_KEY not set. Get an API token at https://swissmodel.expasy.org/dashboard"
                )

            sequence = arguments["sequence"]
            project_title = arguments.get("project_title", "YOHAS MCP modeling")
            template_id = arguments.get("template_id", "")
            wait = arguments.get("wait", False)

            payload: dict = {
                "target_sequences": [sequence],
                "project_title": project_title,
            }
            if template_id:
                payload["template_id"] = template_id

            resp = await swissmodel.post("/automodel", json=payload)
            data = resp.json()
            project_id = data.get("project_id", "")

            result = {
                "project_id": project_id,
                "status": data.get("status", "SUBMITTED"),
                "tool": "swiss_model",
            }

            if wait and project_id:
                deadline = time.monotonic() + MAX_POLL_SECONDS
                while time.monotonic() < deadline:
                    status_resp = await swissmodel.get(f"/project/{project_id}/models/summary/")
                    status_data = status_resp.json()
                    project_status = status_data.get("status", "")

                    if project_status == "COMPLETED":
                        models = []
                        for model in status_data.get("models", []):
                            models.append({
                                "model_id": model.get("model_id", ""),
                                "template": model.get("template", ""),
                                "qmean": model.get("qmean_global", ""),
                                "gmqe": model.get("gmqe", ""),
                                "identity": model.get("identity", ""),
                                "coverage": model.get("coverage", ""),
                                "oligo_state": model.get("oligo_state", ""),
                                "coordinates_url": model.get("coordinates_url", ""),
                            })
                        result.update({
                            "status": "COMPLETED",
                            "models": models,
                            "model_count": len(models),
                        })
                        break
                    elif project_status in ("FAILED", "ERROR"):
                        result.update({
                            "status": project_status,
                            "error": status_data.get("error", "Modeling failed"),
                        })
                        break
                    await asyncio.sleep(POLL_INTERVAL)
                else:
                    result["status"] = "TIMEOUT"
                    result["error"] = "Polling exceeded time limit"

            return text_result(result)

        return error_result(f"Unknown tool: {name}")

    except Exception as exc:
        return error_result(f"{name} failed: {exc}")


if __name__ == "__main__":
    start(server)
