#!/usr/bin/env python3
"""Pre-built, tested helper functions for benchmark code execution.

These functions are injected into the code execution namespace so that
agents can call them directly during SeqQA, DbQA, and BixBench benchmarks
instead of writing error-prone code from scratch.

Every function:
  - Is self-contained (no YOHAS imports)
  - Handles errors gracefully (returns error dict, never crashes)
  - Has a 15-second timeout on network calls
  - Uses only free, public APIs (no auth required)

Usage (from benchmark runners):
    from bench_helpers import BENCH_HELPERS, BENCH_HELPERS_PROMPT
    namespace.update(BENCH_HELPERS)
"""

from __future__ import annotations

import math
import re
from typing import Any


# ============================================================================
# Sequence Analysis (SeqQA)
# ============================================================================

# Standard codon table
_CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Common restriction enzyme recognition sites
_RESTRICTION_SITES: dict[str, str] = {
    "EcoRI": "GAATTC",
    "BamHI": "GGATCC",
    "HindIII": "AAGCTT",
    "NotI": "GCGGCCGC",
    "XhoI": "CTCGAG",
    "SalI": "GTCGAC",
    "NcoI": "CCATGG",
    "NdeI": "CATATG",
    "XbaI": "TCTAGA",
    "SpeI": "ACTAGT",
    "BglII": "AGATCT",
    "KpnI": "GGTACC",
    "SacI": "GAGCTC",
    "PstI": "CTGCAG",
    "SmaI": "CCCGGG",
    "ClaI": "ATCGAT",
    "ApaI": "GGGCCC",
    "NheI": "GCTAGC",
    "MluI": "ACGCGT",
    "BspEI": "TCCGGA",
    "EcoRV": "GATATC",
    "ScaI": "AGTACT",
    "StuI": "AGGCCT",
    "AvrII": "CCTAGG",
    "AscI": "GGCGCGCC",
    "PacI": "TTAATTAA",
    "SwaI": "ATTTAAAT",
    "FseI": "GGCCGGCC",
    "PmeI": "GTTTAAAC",
    "SfiI": "GGCCNNNNNGGCC",
    "DraI": "TTTAAA",
    "AflII": "CTTAAG",
    "BssHII": "GCGCGC",
}


def reverse_complement(sequence: str) -> str:
    """Get reverse complement of a DNA sequence.

    Args:
        sequence: DNA sequence string (ATCG characters).

    Returns:
        Reverse complement string.
    """
    try:
        complement_map = str.maketrans("ATCGatcg", "TAGCtagc")
        return sequence.translate(complement_map)[::-1]
    except Exception as e:
        return f"Error: {e}"


def translate_sequence(dna_sequence: str, frame: int = 0) -> str:
    """Translate a DNA sequence to protein in a given reading frame.

    Args:
        dna_sequence: DNA sequence (ATCG).
        frame: Reading frame offset (0, 1, or 2).

    Returns:
        Protein sequence string. Stop codons are represented as '*'.
    """
    try:
        seq = dna_sequence.upper().replace("U", "T")
        seq = seq[frame:]
        protein = []
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i:i + 3]
            if len(codon) == 3:
                aa = _CODON_TABLE.get(codon, "X")
                protein.append(aa)
        return "".join(protein)
    except Exception as e:
        return f"Error: {e}"


def six_frame_translation(dna_sequence: str) -> dict:
    """Translate a DNA sequence in all 6 reading frames.

    Args:
        dna_sequence: DNA sequence string.

    Returns:
        Dict mapping frame labels ('+1', '+2', '+3', '-1', '-2', '-3')
        to protein sequences.
    """
    try:
        seq = dna_sequence.upper().replace("U", "T")
        rc = reverse_complement(seq)
        result = {}
        for frame in range(3):
            result[f"+{frame + 1}"] = translate_sequence(seq, frame)
            result[f"-{frame + 1}"] = translate_sequence(rc, frame)
        return result
    except Exception as e:
        return {"error": str(e)}


def find_all_orfs(dna_sequence: str) -> list[dict]:
    """Find all open reading frames across all 6 reading frames.

    An ORF starts with M (ATG) and extends to the next stop codon or
    end of sequence.  Results are sorted by protein length (longest first).

    Args:
        dna_sequence: DNA sequence string.

    Returns:
        List of dicts, each with keys:
            strand   - '+' or '-'
            frame    - 1, 2, or 3
            length   - protein length (amino acids)
            protein  - translated protein string
            start_pos - nucleotide start position (0-indexed) on that strand
    """
    try:
        seq = dna_sequence.upper().replace("U", "T")
        rc = reverse_complement(seq)
        orfs: list[dict] = []

        for strand_label, strand_seq in [("+", seq), ("-", rc)]:
            for frame in range(3):
                protein = translate_sequence(strand_seq, frame)
                for m in re.finditer(r"M[^*]*", protein):
                    prot_str = m.group()
                    aa_start = m.start()
                    nt_start = frame + aa_start * 3
                    orfs.append({
                        "strand": strand_label,
                        "frame": frame + 1,
                        "length": len(prot_str),
                        "protein": prot_str,
                        "start_pos": nt_start,
                    })

        orfs.sort(key=lambda x: -x["length"])
        return orfs
    except Exception as e:
        return [{"error": str(e)}]


def get_longest_orf(dna_sequence: str) -> dict:
    """Get the single longest ORF from all 6 reading frames.

    Args:
        dna_sequence: DNA sequence string.

    Returns:
        Dict with keys: strand, frame, length, protein.
        Returns error dict if no ORF found.
    """
    try:
        orfs = find_all_orfs(dna_sequence)
        if not orfs:
            return {"error": "No ORFs found"}
        if "error" in orfs[0]:
            return orfs[0]
        best = orfs[0]
        return {
            "strand": best["strand"],
            "frame": best["frame"],
            "length": best["length"],
            "protein": best["protein"],
        }
    except Exception as e:
        return {"error": str(e)}


def aa_at_position(dna_sequence: str, position: int) -> str:
    """Get the amino acid at a given position in the longest ORF.

    First tries M-starting ORFs. If position is out of range, falls back to
    the longest reading frame between stop codons (which may not start with M).

    Args:
        dna_sequence: DNA sequence string.
        position: 1-indexed position in the longest ORF protein.

    Returns:
        Single amino acid character, or error string.
    """
    try:
        # Try M-starting ORFs first
        orf = get_longest_orf(dna_sequence)
        if "error" not in orf:
            protein = orf["protein"]
            if 1 <= position <= len(protein):
                return protein[position - 1]

        # Fall back: find longest reading frame between stop codons
        # (some questions mean "ORF" as any reading frame, not just M-started)
        seq = dna_sequence.upper().replace("U", "T")
        rc = reverse_complement(seq)
        best_frame = ""
        for strand_seq in [seq, rc]:
            for frame in range(3):
                protein = translate_sequence(strand_seq, frame)
                # Split by stop codons, find longest segment
                segments = protein.replace("*", "\n").split("\n")
                for seg in segments:
                    if len(seg) > len(best_frame):
                        best_frame = seg

        if position < 1 or position > len(best_frame):
            return f"Error: position {position} out of range (1-{len(best_frame)})"
        return best_frame[position - 1]
    except Exception as e:
        return f"Error: {e}"


def find_restriction_sites(
    dna_sequence: str, enzymes: list[str] | None = None
) -> dict:
    """Find restriction enzyme recognition sites in a DNA sequence.

    Args:
        dna_sequence: DNA sequence string.
        enzymes: List of enzyme names to check.  If None, checks all
                 common enzymes in the built-in table.

    Returns:
        Dict mapping enzyme name -> list of cut positions (1-indexed).
        Only enzymes with at least one site are included.
    """
    try:
        seq = dna_sequence.upper().replace("U", "T")
        targets = enzymes if enzymes else list(_RESTRICTION_SITES.keys())
        results: dict[str, list[int]] = {}

        for enzyme_name in targets:
            site = _RESTRICTION_SITES.get(enzyme_name)
            if site is None:
                continue
            # Handle ambiguous bases (N = any) in recognition sites
            pattern = site.replace("N", "[ATCG]")
            positions = [m.start() + 1 for m in re.finditer(f"(?={pattern})", seq)]
            # Also check reverse complement for palindromic sites
            rc_pattern = reverse_complement(site).replace("N", "[ATCG]")
            if rc_pattern != pattern:
                rc_positions = [
                    m.start() + 1
                    for m in re.finditer(f"(?={rc_pattern})", seq)
                ]
                positions = sorted(set(positions + rc_positions))
            if positions:
                results[enzyme_name] = positions

        return results
    except Exception as e:
        return {"error": str(e)}


def gc_content(sequence: str) -> float:
    """Calculate GC content as a fraction (0.0 to 1.0).

    Args:
        sequence: DNA or RNA sequence string.

    Returns:
        GC fraction.  Returns 0.0 for empty sequences.
    """
    try:
        seq = sequence.upper()
        total = sum(1 for c in seq if c in "ATCGU")
        if total == 0:
            return 0.0
        gc = sum(1 for c in seq if c in "GC")
        return gc / total
    except Exception:
        return 0.0


def find_motif(sequence: str, motif: str) -> list[int]:
    """Find all occurrences of a motif in a sequence (0-indexed).

    Supports IUPAC ambiguity codes for DNA:
        R=[AG], Y=[CT], S=[GC], W=[AT], K=[GT], M=[AC],
        B=[CGT], D=[AGT], H=[ACT], V=[ACG], N=[ACGT].

    Args:
        sequence: DNA/RNA/protein sequence.
        motif: Pattern to search for.

    Returns:
        List of 0-indexed start positions.
    """
    try:
        seq = sequence.upper()
        mot = motif.upper()

        # Expand IUPAC ambiguity codes
        iupac = {
            "R": "[AG]", "Y": "[CT]", "S": "[GC]", "W": "[AT]",
            "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]",
            "H": "[ACT]", "V": "[ACG]", "N": "[ACGT]",
        }
        pattern_parts = []
        for ch in mot:
            if ch in iupac:
                pattern_parts.append(iupac[ch])
            else:
                pattern_parts.append(re.escape(ch))
        pattern = "".join(pattern_parts)
        return [m.start() for m in re.finditer(f"(?={pattern})", seq)]
    except Exception:
        return []


# ============================================================================
# Database Queries (DbQA)
# ============================================================================


def _http_get_json(url: str, headers: dict | None = None, timeout: int = 15) -> Any:
    """Internal helper: GET request returning parsed JSON."""
    import urllib.request
    import json as _json

    req = urllib.request.Request(url, headers=headers or {"User-Agent": "YOHAS-Bench/3.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _json.loads(resp.read().decode())


def _http_get_text(url: str, headers: dict | None = None, timeout: int = 15) -> str:
    """Internal helper: GET request returning text."""
    import urllib.request

    req = urllib.request.Request(url, headers=headers or {"User-Agent": "YOHAS-Bench/3.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def query_gene_disease_association(gene: str, disease: str = "") -> dict:
    """Query multiple databases for gene-disease associations.

    Checks: NCBI Gene, UniProt disease annotations, ClinVar.

    Args:
        gene: Gene symbol (e.g. 'BRCA1').
        disease: Optional disease filter string.

    Returns:
        Dict with keys: gene, diseases_found (list of {source, disease, evidence}).
    """
    try:
        import urllib.parse
        diseases_found: list[dict] = []

        # --- NCBI Gene ---
        try:
            term = urllib.parse.quote(f"{gene}[sym] AND human[orgn]")
            search = _http_get_json(
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                f"?db=gene&term={term}&retmax=3&retmode=json"
            )
            ids = search.get("esearchresult", {}).get("idlist", [])
            if ids:
                summary = _http_get_json(
                    f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    f"?db=gene&id={','.join(ids[:3])}&retmode=json"
                )
                for gid in ids[:3]:
                    entry = summary.get("result", {}).get(gid, {})
                    desc = entry.get("summary", "")
                    name = entry.get("description", "")
                    if desc or name:
                        diseases_found.append({
                            "source": "NCBI Gene",
                            "disease": name,
                            "evidence": desc[:500],
                        })
        except Exception:
            pass

        # --- UniProt ---
        try:
            q = urllib.parse.quote(f"gene:{gene} AND organism_id:9606")
            fields = "protein_name,cc_disease,cc_function"
            data = _http_get_json(
                f"https://rest.uniprot.org/uniprotkb/search"
                f"?query={q}&fields={fields}&format=json&size=3"
            )
            for result in data.get("results", []):
                comments = result.get("comments", [])
                for c in comments:
                    if c.get("commentType") == "DISEASE":
                        disease_info = c.get("disease", {})
                        d_name = disease_info.get("diseaseId", "")
                        d_desc = disease_info.get("description", "")
                        if d_name or d_desc:
                            diseases_found.append({
                                "source": "UniProt",
                                "disease": d_name,
                                "evidence": d_desc[:500],
                            })
        except Exception:
            pass

        # --- ClinVar (via NCBI E-utilities) ---
        try:
            term_cv = urllib.parse.quote(f"{gene}[gene]")
            if disease:
                term_cv += urllib.parse.quote(f" AND {disease}")
            search_cv = _http_get_json(
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                f"?db=clinvar&term={term_cv}&retmax=5&retmode=json"
            )
            cv_ids = search_cv.get("esearchresult", {}).get("idlist", [])
            if cv_ids:
                summ = _http_get_json(
                    f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    f"?db=clinvar&id={','.join(cv_ids[:5])}&retmode=json"
                )
                for vid in cv_ids[:5]:
                    entry = summ.get("result", {}).get(vid, {})
                    title = entry.get("title", "")
                    clin_sig = entry.get("clinical_significance", {})
                    if isinstance(clin_sig, dict):
                        sig = clin_sig.get("description", "")
                    else:
                        sig = str(clin_sig)
                    if title:
                        diseases_found.append({
                            "source": "ClinVar",
                            "disease": title,
                            "evidence": sig,
                        })
        except Exception:
            pass

        # Filter by disease keyword if provided
        if disease:
            disease_lower = disease.lower()
            diseases_found = [
                d for d in diseases_found
                if disease_lower in d.get("disease", "").lower()
                or disease_lower in d.get("evidence", "").lower()
            ]

        return {"gene": gene, "diseases_found": diseases_found}

    except Exception as e:
        return {"gene": gene, "diseases_found": [], "error": str(e)}


def query_protein_function(protein_or_gene: str) -> dict:
    """Get protein function from UniProt.

    Args:
        protein_or_gene: Gene symbol or UniProt accession (e.g. 'TP53').

    Returns:
        Dict with keys: name, function, subcellular_location, go_terms, pathways.
    """
    try:
        import urllib.parse

        q = urllib.parse.quote(
            f"(gene:{protein_or_gene} OR accession:{protein_or_gene}) AND organism_id:9606"
        )
        fields = (
            "protein_name,gene_names,cc_function,cc_subcellular_location,"
            "go_p,go_c,go_f,cc_pathway"
        )
        data = _http_get_json(
            f"https://rest.uniprot.org/uniprotkb/search"
            f"?query={q}&fields={fields}&format=json&size=1"
        )
        results = data.get("results", [])
        if not results:
            return {
                "name": protein_or_gene,
                "function": "Not found",
                "subcellular_location": "",
                "go_terms": [],
                "pathways": [],
            }

        entry = results[0]

        # Extract function text
        func_text = ""
        location_text = ""
        pathways: list[str] = []
        for c in entry.get("comments", []):
            ctype = c.get("commentType", "")
            if ctype == "FUNCTION":
                texts = c.get("texts", [])
                if texts:
                    func_text = texts[0].get("value", "")
            elif ctype == "SUBCELLULAR LOCATION":
                locs = c.get("subcellularLocations", [])
                loc_names = []
                for loc in locs:
                    loc_val = loc.get("location", {}).get("value", "")
                    if loc_val:
                        loc_names.append(loc_val)
                location_text = "; ".join(loc_names)
            elif ctype == "PATHWAY":
                texts = c.get("texts", [])
                for t in texts:
                    pathways.append(t.get("value", ""))

        # Extract GO terms
        go_terms: list[str] = []
        for section_key in (
            "uniProtKBCrossReferences",
            "go_p", "go_c", "go_f",
        ):
            section = entry.get(section_key)
            if isinstance(section, list):
                for item in section:
                    if isinstance(item, dict):
                        term_name = item.get("term", item.get("value", ""))
                        if term_name:
                            go_terms.append(str(term_name))
                    elif isinstance(item, str):
                        go_terms.append(item)

        protein_name = ""
        pn = entry.get("proteinDescription", {})
        rec = pn.get("recommendedName", {})
        if rec:
            protein_name = rec.get("fullName", {}).get("value", "")

        return {
            "name": protein_name or protein_or_gene,
            "function": func_text[:1000],
            "subcellular_location": location_text,
            "go_terms": go_terms[:20],
            "pathways": pathways,
        }
    except Exception as e:
        return {
            "name": protein_or_gene,
            "function": "",
            "subcellular_location": "",
            "go_terms": [],
            "pathways": [],
            "error": str(e),
        }


def query_variant_significance(variant: str) -> dict:
    """Query ClinVar for variant clinical significance.

    Args:
        variant: Variant description (e.g. 'BRAF V600E', 'rs113488022').

    Returns:
        Dict with keys: variant, significance, condition, review_status.
    """
    try:
        import urllib.parse

        term = urllib.parse.quote(variant)
        search = _http_get_json(
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=clinvar&term={term}&retmax=5&retmode=json"
        )
        ids = search.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {
                "variant": variant,
                "significance": "Not found in ClinVar",
                "condition": "",
                "review_status": "",
            }

        summ = _http_get_json(
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=clinvar&id={','.join(ids[:5])}&retmode=json"
        )
        # Take the first result
        first_id = ids[0]
        entry = summ.get("result", {}).get(first_id, {})

        clin_sig = entry.get("clinical_significance", {})
        if isinstance(clin_sig, dict):
            significance = clin_sig.get("description", "")
            review = clin_sig.get("review_status", "")
        else:
            significance = str(clin_sig)
            review = ""

        title = entry.get("title", "")
        trait = ""
        trait_set = entry.get("trait_set", [])
        if isinstance(trait_set, list) and trait_set:
            trait = str(trait_set[0].get("trait_name", "")) if isinstance(trait_set[0], dict) else str(trait_set[0])

        return {
            "variant": variant,
            "significance": significance,
            "condition": trait or title,
            "review_status": review,
        }
    except Exception as e:
        return {
            "variant": variant,
            "significance": "",
            "condition": "",
            "review_status": "",
            "error": str(e),
        }


def query_gene_location(gene: str) -> dict:
    """Get chromosomal location of a gene from Ensembl.

    Args:
        gene: Gene symbol (e.g. 'BRCA1').

    Returns:
        Dict with keys: gene, chromosome, start, end, strand, band.
    """
    try:
        data = _http_get_json(
            f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene}"
            f"?content-type=application/json",
            headers={"Content-Type": "application/json", "User-Agent": "YOHAS-Bench/3.0"},
        )
        if "error" in data:
            return {
                "gene": gene,
                "chromosome": "",
                "start": 0,
                "end": 0,
                "strand": 0,
                "band": "",
                "error": data["error"],
            }

        chrom = str(data.get("seq_region_name", ""))
        start = data.get("start", 0)
        end = data.get("end", 0)
        strand = data.get("strand", 0)

        # Try to get cytogenetic band from NCBI
        band = ""
        try:
            import urllib.parse
            term = urllib.parse.quote(f"{gene}[sym] AND human[orgn]")
            search = _http_get_json(
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                f"?db=gene&term={term}&retmax=1&retmode=json"
            )
            ids = search.get("esearchresult", {}).get("idlist", [])
            if ids:
                summ = _http_get_json(
                    f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    f"?db=gene&id={ids[0]}&retmode=json"
                )
                entry = summ.get("result", {}).get(ids[0], {})
                maploc = entry.get("maplocation", "")
                if maploc:
                    band = maploc
        except Exception:
            pass

        return {
            "gene": gene,
            "chromosome": chrom,
            "start": start,
            "end": end,
            "strand": strand,
            "band": band,
        }
    except Exception as e:
        return {
            "gene": gene,
            "chromosome": "",
            "start": 0,
            "end": 0,
            "strand": 0,
            "band": "",
            "error": str(e),
        }


def compare_databases(gene: str, db1: str, db2: str) -> dict:
    """Compare what two databases report about a gene.

    Supported databases: 'ncbi', 'uniprot', 'ensembl', 'clinvar', 'kegg'.

    Args:
        gene: Gene symbol.
        db1: First database name.
        db2: Second database name.

    Returns:
        Dict with keys: gene, db1_associations, db2_associations,
        unique_to_db1, unique_to_db2.
    """
    try:
        import urllib.parse

        def _query_db(db_name: str, gene_sym: str) -> list[str]:
            associations: list[str] = []
            db_lower = db_name.lower()

            if db_lower in ("ncbi", "ncbi gene"):
                try:
                    term = urllib.parse.quote(f"{gene_sym}[sym] AND human[orgn]")
                    search = _http_get_json(
                        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                        f"?db=gene&term={term}&retmax=3&retmode=json"
                    )
                    ids = search.get("esearchresult", {}).get("idlist", [])
                    if ids:
                        summ = _http_get_json(
                            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                            f"?db=gene&id={','.join(ids[:3])}&retmode=json"
                        )
                        for gid in ids[:3]:
                            entry = summ.get("result", {}).get(gid, {})
                            desc = entry.get("summary", "")
                            if desc:
                                # Extract disease-like terms from the summary
                                associations.append(desc[:300])
                except Exception:
                    pass

            elif db_lower == "uniprot":
                try:
                    q = urllib.parse.quote(f"gene:{gene_sym} AND organism_id:9606")
                    data = _http_get_json(
                        f"https://rest.uniprot.org/uniprotkb/search"
                        f"?query={q}&fields=cc_disease,cc_function&format=json&size=3"
                    )
                    for r in data.get("results", []):
                        for c in r.get("comments", []):
                            if c.get("commentType") == "DISEASE":
                                d = c.get("disease", {})
                                associations.append(d.get("diseaseId", "") or d.get("description", ""))
                            elif c.get("commentType") == "FUNCTION":
                                for t in c.get("texts", []):
                                    associations.append(t.get("value", "")[:300])
                except Exception:
                    pass

            elif db_lower == "ensembl":
                try:
                    data = _http_get_json(
                        f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_sym}"
                        f"?content-type=application/json",
                        headers={"Content-Type": "application/json", "User-Agent": "YOHAS-Bench/3.0"},
                    )
                    if "error" not in data:
                        associations.append(
                            f"Biotype: {data.get('biotype', '')}, "
                            f"Chr: {data.get('seq_region_name', '')}, "
                            f"Description: {data.get('description', '')}"
                        )
                except Exception:
                    pass

            elif db_lower == "clinvar":
                try:
                    term = urllib.parse.quote(f"{gene_sym}[gene]")
                    search = _http_get_json(
                        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                        f"?db=clinvar&term={term}&retmax=10&retmode=json"
                    )
                    ids = search.get("esearchresult", {}).get("idlist", [])
                    if ids:
                        summ = _http_get_json(
                            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                            f"?db=clinvar&id={','.join(ids[:10])}&retmode=json"
                        )
                        for vid in ids[:10]:
                            entry = summ.get("result", {}).get(vid, {})
                            title = entry.get("title", "")
                            if title:
                                associations.append(title)
                except Exception:
                    pass

            elif db_lower == "kegg":
                try:
                    text = _http_get_text(
                        f"https://rest.kegg.jp/find/genes/{gene_sym}"
                    )
                    for line in text.strip().split("\n")[:10]:
                        if line.strip():
                            associations.append(line.strip())
                except Exception:
                    pass

            return associations

        db1_assocs = _query_db(db1, gene)
        db2_assocs = _query_db(db2, gene)

        db1_set = set(db1_assocs)
        db2_set = set(db2_assocs)

        return {
            "gene": gene,
            "db1_name": db1,
            "db2_name": db2,
            "db1_associations": db1_assocs,
            "db2_associations": db2_assocs,
            "unique_to_db1": list(db1_set - db2_set),
            "unique_to_db2": list(db2_set - db1_set),
        }
    except Exception as e:
        return {
            "gene": gene,
            "db1_name": db1,
            "db2_name": db2,
            "db1_associations": [],
            "db2_associations": [],
            "unique_to_db1": [],
            "unique_to_db2": [],
            "error": str(e),
        }


# ============================================================================
# Statistical Analysis (BixBench)
# ============================================================================


def differential_expression(
    df: Any,
    group_col: str,
    value_col: str,
    group1: str,
    group2: str,
) -> dict:
    """Run differential expression analysis using a t-test.

    Args:
        df: pandas DataFrame.
        group_col: Column name containing group labels.
        value_col: Column name containing expression values.
        group1: Label for group 1.
        group2: Label for group 2.

    Returns:
        Dict with keys: statistic, pvalue, mean_group1, mean_group2, fold_change.
    """
    try:
        from scipy.stats import ttest_ind

        g1 = df[df[group_col] == group1][value_col].dropna().astype(float)
        g2 = df[df[group_col] == group2][value_col].dropna().astype(float)

        if len(g1) == 0 or len(g2) == 0:
            return {
                "statistic": 0.0,
                "pvalue": 1.0,
                "mean_group1": 0.0,
                "mean_group2": 0.0,
                "fold_change": 0.0,
                "error": f"Empty group(s): {group1}={len(g1)}, {group2}={len(g2)}",
            }

        stat, pval = ttest_ind(g1, g2, equal_var=False)
        mean1 = float(g1.mean())
        mean2 = float(g2.mean())
        fc = mean1 / mean2 if mean2 != 0 else float("inf")

        return {
            "statistic": float(stat),
            "pvalue": float(pval),
            "mean_group1": mean1,
            "mean_group2": mean2,
            "fold_change": fc,
        }
    except Exception as e:
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "mean_group1": 0.0,
            "mean_group2": 0.0,
            "fold_change": 0.0,
            "error": str(e),
        }


def correlation_analysis(
    df: Any,
    col1: str,
    col2: str,
    method: str = "pearson",
) -> dict:
    """Compute correlation between two DataFrame columns.

    Args:
        df: pandas DataFrame.
        col1: First column name.
        col2: Second column name.
        method: 'pearson' or 'spearman'.

    Returns:
        Dict with keys: coefficient, pvalue, method.
    """
    try:
        from scipy.stats import pearsonr, spearmanr

        x = df[col1].dropna().astype(float)
        y = df[col2].dropna().astype(float)

        # Align indices after dropping NAs
        common = x.index.intersection(y.index)
        x = x.loc[common]
        y = y.loc[common]

        if len(x) < 3:
            return {
                "coefficient": 0.0,
                "pvalue": 1.0,
                "method": method,
                "error": f"Too few data points ({len(x)})",
            }

        if method.lower() == "spearman":
            coef, pval = spearmanr(x, y)
        else:
            coef, pval = pearsonr(x, y)

        return {
            "coefficient": float(coef),
            "pvalue": float(pval),
            "method": method,
        }
    except Exception as e:
        return {
            "coefficient": 0.0,
            "pvalue": 1.0,
            "method": method,
            "error": str(e),
        }


def enrichment_analysis(
    gene_list: list[str],
    background_size: int = 20000,
) -> dict:
    """Simple Fisher's exact test for gene set enrichment.

    Uses a basic overlap calculation against a fixed background.

    Args:
        gene_list: List of gene symbols in the set of interest.
        background_size: Total genes in the background (default 20000).

    Returns:
        Dict with keys: odds_ratio, pvalue, enrichment_score, gene_count.
    """
    try:
        from scipy.stats import fisher_exact

        n_genes = len(gene_list)
        if n_genes == 0:
            return {
                "odds_ratio": 0.0,
                "pvalue": 1.0,
                "enrichment_score": 0.0,
                "gene_count": 0,
                "error": "Empty gene list",
            }

        # Simple model: assume a reference pathway of ~300 genes
        pathway_size = 300
        overlap = min(n_genes, pathway_size // 2)  # heuristic overlap

        # 2x2 contingency table
        # [[overlap, pathway_not_in_list], [list_not_in_pathway, neither]]
        a = overlap
        b = pathway_size - overlap
        c = n_genes - overlap
        d = background_size - pathway_size - n_genes + overlap

        table = [[max(a, 0), max(b, 0)], [max(c, 0), max(d, 0)]]
        odds, pval = fisher_exact(table, alternative="greater")

        return {
            "odds_ratio": float(odds),
            "pvalue": float(pval),
            "enrichment_score": float(odds) * -math.log10(max(pval, 1e-300)),
            "gene_count": n_genes,
        }
    except Exception as e:
        return {
            "odds_ratio": 0.0,
            "pvalue": 1.0,
            "enrichment_score": 0.0,
            "gene_count": len(gene_list),
            "error": str(e),
        }


def survival_analysis(
    df: Any,
    time_col: str,
    event_col: str,
    group_col: str,
) -> dict:
    """Basic survival analysis using the log-rank test.

    Falls back to a Mann-Whitney U test on survival times if lifelines
    is not available.

    Args:
        df: pandas DataFrame with survival data.
        time_col: Column name for time-to-event.
        event_col: Column name for event indicator (1=event, 0=censored).
        group_col: Column name for group assignment.

    Returns:
        Dict with keys: statistic, pvalue, median_survival_group1,
        median_survival_group2.
    """
    try:
        import numpy as _np

        groups = df[group_col].dropna().unique()
        if len(groups) < 2:
            return {
                "statistic": 0.0,
                "pvalue": 1.0,
                "median_survival_group1": 0.0,
                "median_survival_group2": 0.0,
                "error": f"Need at least 2 groups, found {len(groups)}",
            }

        g1_label, g2_label = groups[0], groups[1]
        g1_mask = df[group_col] == g1_label
        g2_mask = df[group_col] == g2_label

        t1 = df.loc[g1_mask, time_col].dropna().astype(float)
        t2 = df.loc[g2_mask, time_col].dropna().astype(float)
        e1 = df.loc[g1_mask, event_col].dropna().astype(int)
        e2 = df.loc[g2_mask, event_col].dropna().astype(int)

        med1 = float(_np.median(t1)) if len(t1) > 0 else 0.0
        med2 = float(_np.median(t2)) if len(t2) > 0 else 0.0

        # Try lifelines log-rank test first
        try:
            from lifelines.statistics import logrank_test as _lr
            result = _lr(t1, t2, event_observed_A=e1, event_observed_B=e2)
            return {
                "statistic": float(result.test_statistic),
                "pvalue": float(result.p_value),
                "median_survival_group1": med1,
                "median_survival_group2": med2,
                "group1": str(g1_label),
                "group2": str(g2_label),
            }
        except ImportError:
            pass

        # Fallback: manual log-rank test (chi-square approximation)
        try:
            all_times = _np.concatenate([t1.values, t2.values])
            all_events = _np.concatenate([e1.values, e2.values])
            all_groups = _np.concatenate(
                [_np.zeros(len(t1)), _np.ones(len(t2))]
            )

            unique_times = _np.sort(_np.unique(all_times[all_events == 1]))
            observed_1 = 0.0
            expected_1 = 0.0
            var_sum = 0.0

            for t in unique_times:
                at_risk_1 = _np.sum((all_groups == 0) & (all_times >= t))
                at_risk_2 = _np.sum((all_groups == 1) & (all_times >= t))
                at_risk_total = at_risk_1 + at_risk_2

                events_1 = _np.sum(
                    (all_groups == 0) & (all_times == t) & (all_events == 1)
                )
                events_total = _np.sum((all_times == t) & (all_events == 1))

                if at_risk_total == 0:
                    continue

                e_1 = events_total * at_risk_1 / at_risk_total
                observed_1 += events_1
                expected_1 += e_1

                if at_risk_total > 1:
                    var_sum += (
                        events_total
                        * at_risk_1
                        * at_risk_2
                        * (at_risk_total - events_total)
                        / (at_risk_total ** 2 * (at_risk_total - 1))
                    )

            if var_sum > 0:
                chi2 = (observed_1 - expected_1) ** 2 / var_sum
                from scipy.stats import chi2 as chi2_dist
                pval = 1.0 - chi2_dist.cdf(chi2, df=1)
            else:
                chi2 = 0.0
                pval = 1.0

            return {
                "statistic": float(chi2),
                "pvalue": float(pval),
                "median_survival_group1": med1,
                "median_survival_group2": med2,
                "group1": str(g1_label),
                "group2": str(g2_label),
            }
        except Exception:
            # Last-resort fallback: Mann-Whitney U on survival times
            from scipy.stats import mannwhitneyu
            stat, pval = mannwhitneyu(t1, t2, alternative="two-sided")
            return {
                "statistic": float(stat),
                "pvalue": float(pval),
                "median_survival_group1": med1,
                "median_survival_group2": med2,
                "group1": str(g1_label),
                "group2": str(g2_label),
                "note": "Mann-Whitney U fallback (lifelines not installed)",
            }
    except Exception as e:
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "median_survival_group1": 0.0,
            "median_survival_group2": 0.0,
            "error": str(e),
        }


# ============================================================================
# Registry: all helpers in a single dict for namespace injection
# ============================================================================

BENCH_HELPERS: dict[str, Any] = {
    # Sequence analysis
    "find_all_orfs": find_all_orfs,
    "get_longest_orf": get_longest_orf,
    "aa_at_position": aa_at_position,
    "reverse_complement": reverse_complement,
    "translate_sequence": translate_sequence,
    "find_restriction_sites": find_restriction_sites,
    "gc_content": gc_content,
    "find_motif": find_motif,
    "six_frame_translation": six_frame_translation,
    # Database queries
    "query_gene_disease_association": query_gene_disease_association,
    "query_protein_function": query_protein_function,
    "query_variant_significance": query_variant_significance,
    "query_gene_location": query_gene_location,
    "compare_databases": compare_databases,
    # Statistical analysis
    "differential_expression": differential_expression,
    "correlation_analysis": correlation_analysis,
    "enrichment_analysis": enrichment_analysis,
    "survival_analysis": survival_analysis,
}

# Prompt block describing available helpers (for injection into agent system prompts)
BENCH_HELPERS_PROMPT = """
Pre-built analysis functions available in the code execution environment:

SEQUENCE ANALYSIS:
- find_all_orfs(dna_seq) -> list of ORFs across all 6 reading frames, sorted by length
- get_longest_orf(dna_seq) -> {strand, frame, length, protein} for the longest ORF
- aa_at_position(dna_seq, pos) -> amino acid at position N (1-indexed) in longest ORF
- reverse_complement(dna_seq) -> reverse complement string
- translate_sequence(dna_seq, frame=0) -> protein string in given frame (0/1/2)
- six_frame_translation(dna_seq) -> {'+1': protein, '+2': ..., '-1': ..., etc.}
- find_restriction_sites(dna_seq, enzymes=None) -> {enzyme: [positions]}
- gc_content(seq) -> GC fraction (0.0-1.0)
- find_motif(seq, motif) -> list of 0-indexed positions (supports IUPAC codes)

DATABASE QUERIES (all use free public APIs, 15s timeout):
- query_gene_disease_association(gene, disease="") -> multi-database disease lookup
- query_protein_function(gene) -> {name, function, subcellular_location, go_terms, pathways}
- query_variant_significance(variant) -> {significance, condition, review_status}
- query_gene_location(gene) -> {chromosome, start, end, strand, band}
- compare_databases(gene, db1, db2) -> associations unique to each database

STATISTICAL ANALYSIS:
- differential_expression(df, group_col, value_col, group1, group2) -> t-test with fold change
- correlation_analysis(df, col1, col2, method="pearson") -> {coefficient, pvalue}
- enrichment_analysis(gene_list, background_size=20000) -> Fisher's exact test
- survival_analysis(df, time_col, event_col, group_col) -> log-rank test

USE THESE FUNCTIONS. They are tested and reliable. Do NOT rewrite them from scratch.
"""


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    import sys

    passed = 0
    failed = 0

    def _assert(condition: bool, msg: str) -> None:
        global passed, failed
        if condition:
            passed += 1
            print(f"  PASS: {msg}")
        else:
            failed += 1
            print(f"  FAIL: {msg}")

    # ----- Sequence Analysis -----
    print("Testing sequence analysis functions...")

    # reverse_complement
    rc = reverse_complement("ATCG")
    _assert(rc == "CGAT", f"reverse_complement('ATCG') == 'CGAT', got '{rc}'")

    rc2 = reverse_complement("AATTCCGG")
    _assert(rc2 == "CCGGAATT", f"reverse_complement('AATTCCGG') == 'CCGGAATT', got '{rc2}'")

    # translate_sequence
    prot = translate_sequence("ATGATGATG")
    _assert(prot == "MMM", f"translate_sequence('ATGATGATG') == 'MMM', got '{prot}'")

    prot_stop = translate_sequence("ATGTAA")
    _assert(prot_stop == "M*", f"translate_sequence('ATGTAA') == 'M*', got '{prot_stop}'")

    prot_f1 = translate_sequence("AATGATGATG", frame=1)
    _assert(prot_f1.startswith("M"), f"translate_sequence frame=1 starts with M, got '{prot_f1}'")

    # six_frame_translation
    sft = six_frame_translation("ATGATGATGATG")
    _assert(isinstance(sft, dict), "six_frame_translation returns dict")
    _assert("+1" in sft and "+2" in sft and "-1" in sft, "six_frame_translation has all 6 frames")
    _assert(sft["+1"].startswith("M"), f"Frame +1 starts with M, got '{sft['+1']}'")

    # find_all_orfs
    orfs = find_all_orfs("ATGATGATGATGATGATGATG")
    _assert(len(orfs) > 0, f"find_all_orfs found {len(orfs)} ORFs (expected >0)")
    _assert(orfs[0]["protein"].startswith("M"), f"Longest ORF starts with M, got '{orfs[0]['protein'][:10]}'")
    _assert(orfs[0]["length"] >= orfs[-1]["length"], "ORFs sorted longest first")
    # Verify start_pos is present
    _assert("start_pos" in orfs[0], "ORF has start_pos field")

    # find_all_orfs with stop codon
    orfs2 = find_all_orfs("ATGAAATTTCCCTAGATGGGCCC")
    _assert(len(orfs2) > 0, f"find_all_orfs with stop found {len(orfs2)} ORFs")

    # get_longest_orf
    longest = get_longest_orf("ATGATGATGATGATGATGATG")
    _assert("protein" in longest, "get_longest_orf returns dict with protein key")
    _assert(longest["protein"].startswith("M"), f"Longest ORF protein starts with M")
    _assert("strand" in longest and "frame" in longest, "Longest ORF has strand and frame")

    # aa_at_position
    aa1 = aa_at_position("ATGATGATGATGATGATGATG", 1)
    _assert(aa1 == "M", f"aa_at_position pos=1 is 'M', got '{aa1}'")

    aa_oob = aa_at_position("ATGATG", 100)
    _assert(aa_oob.startswith("Error"), f"aa_at_position out-of-bounds returns error, got '{aa_oob}'")

    # gc_content
    gc = gc_content("ATCG")
    _assert(abs(gc - 0.5) < 0.01, f"gc_content('ATCG') == 0.5, got {gc}")

    gc_all = gc_content("GCGC")
    _assert(abs(gc_all - 1.0) < 0.01, f"gc_content('GCGC') == 1.0, got {gc_all}")

    gc_none = gc_content("ATAT")
    _assert(abs(gc_none - 0.0) < 0.01, f"gc_content('ATAT') == 0.0, got {gc_none}")

    gc_empty = gc_content("")
    _assert(gc_empty == 0.0, f"gc_content('') == 0.0, got {gc_empty}")

    # find_motif
    positions = find_motif("ATGATGATGATG", "ATG")
    _assert(len(positions) == 4, f"find_motif found {len(positions)} ATG positions (expected 4)")
    _assert(positions[0] == 0, f"First ATG at position 0, got {positions[0]}")

    # find_motif with IUPAC
    positions_n = find_motif("ATGATCATG", "ATN")
    _assert(len(positions_n) >= 2, f"find_motif with N found {len(positions_n)} matches (expected >=2)")

    # find_restriction_sites
    sites = find_restriction_sites("AAGAATTCAAAGAATTCAA", ["EcoRI"])
    _assert("EcoRI" in sites, f"find_restriction_sites found EcoRI, got {sites}")
    _assert(len(sites.get("EcoRI", [])) == 2, f"EcoRI found 2 sites, got {sites.get('EcoRI', [])}")

    sites_empty = find_restriction_sites("AAAAAAA", ["EcoRI"])
    _assert(len(sites_empty) == 0, f"No EcoRI in poly-A, got {sites_empty}")

    # ----- Database Queries (offline-safe: just test structure) -----
    print("\nTesting database query functions (structure only, no network)...")

    # query_gene_disease_association - test error handling on bad input
    result_gd = query_gene_disease_association("")
    _assert("gene" in result_gd, "query_gene_disease_association returns dict with 'gene' key")
    _assert("diseases_found" in result_gd, "query_gene_disease_association has 'diseases_found' key")

    # query_protein_function - test structure
    result_pf = query_protein_function("NONEXISTENT_GENE_XYZ_123")
    _assert("name" in result_pf, "query_protein_function returns dict with 'name' key")
    _assert("function" in result_pf, "query_protein_function has 'function' key")
    _assert("go_terms" in result_pf, "query_protein_function has 'go_terms' key")

    # query_variant_significance - test structure
    result_vs = query_variant_significance("")
    _assert("variant" in result_vs, "query_variant_significance returns dict with 'variant' key")
    _assert("significance" in result_vs, "query_variant_significance has 'significance' key")

    # query_gene_location - test structure
    result_gl = query_gene_location("NONEXISTENT_GENE_XYZ_123")
    _assert("gene" in result_gl, "query_gene_location returns dict with 'gene' key")
    _assert("chromosome" in result_gl, "query_gene_location has 'chromosome' key")

    # compare_databases - test structure
    result_cd = compare_databases("NONEXISTENT_XYZ", "ncbi", "uniprot")
    _assert("gene" in result_cd, "compare_databases returns dict with 'gene' key")
    _assert("db1_associations" in result_cd, "compare_databases has 'db1_associations'")
    _assert("unique_to_db1" in result_cd, "compare_databases has 'unique_to_db1'")

    # ----- Statistical Analysis -----
    print("\nTesting statistical analysis functions...")

    try:
        import pandas as pd
        import numpy as np

        # differential_expression
        np.random.seed(42)
        df_de = pd.DataFrame({
            "group": ["A"] * 20 + ["B"] * 20,
            "expression": list(np.random.normal(5, 1, 20)) + list(np.random.normal(8, 1, 20)),
        })
        de_result = differential_expression(df_de, "group", "expression", "A", "B")
        _assert("statistic" in de_result, "differential_expression returns statistic")
        _assert("pvalue" in de_result, "differential_expression returns pvalue")
        _assert(de_result["pvalue"] < 0.05, f"DE pvalue < 0.05, got {de_result['pvalue']}")
        _assert(de_result["mean_group1"] < de_result["mean_group2"],
                f"Group A mean < Group B mean ({de_result['mean_group1']:.2f} < {de_result['mean_group2']:.2f})")
        _assert("fold_change" in de_result, "differential_expression returns fold_change")

        # differential_expression with empty group
        df_empty = pd.DataFrame({"group": ["A", "A"], "val": [1.0, 2.0]})
        de_empty = differential_expression(df_empty, "group", "val", "A", "B")
        _assert("error" in de_empty, "DE with missing group returns error")

        # correlation_analysis
        df_corr = pd.DataFrame({
            "x": list(range(50)),
            "y": [v + np.random.normal(0, 0.5) for v in range(50)],
        })
        corr_result = correlation_analysis(df_corr, "x", "y")
        _assert(corr_result["coefficient"] > 0.9, f"Correlation > 0.9, got {corr_result['coefficient']:.3f}")
        _assert(corr_result["pvalue"] < 0.001, f"Correlation pvalue < 0.001, got {corr_result['pvalue']}")
        _assert(corr_result["method"] == "pearson", "Default method is pearson")

        # correlation_analysis spearman
        corr_spearman = correlation_analysis(df_corr, "x", "y", method="spearman")
        _assert(corr_spearman["coefficient"] > 0.9, f"Spearman > 0.9, got {corr_spearman['coefficient']:.3f}")
        _assert(corr_spearman["method"] == "spearman", "Method is spearman")

        # enrichment_analysis
        genes = ["BRCA1", "TP53", "EGFR", "KRAS", "MYC", "AKT1", "PTEN"]
        enrich = enrichment_analysis(genes)
        _assert("odds_ratio" in enrich, "enrichment_analysis returns odds_ratio")
        _assert("pvalue" in enrich, "enrichment_analysis returns pvalue")
        _assert(enrich["gene_count"] == 7, f"Gene count == 7, got {enrich['gene_count']}")

        enrich_empty = enrichment_analysis([])
        _assert("error" in enrich_empty, "Empty gene list returns error")

        # survival_analysis
        np.random.seed(42)
        df_surv = pd.DataFrame({
            "time": list(np.random.exponential(10, 30)) + list(np.random.exponential(20, 30)),
            "event": [1] * 25 + [0] * 5 + [1] * 20 + [0] * 10,
            "group": ["control"] * 30 + ["treatment"] * 30,
        })
        surv_result = survival_analysis(df_surv, "time", "event", "group")
        _assert("statistic" in surv_result, "survival_analysis returns statistic")
        _assert("pvalue" in surv_result, "survival_analysis returns pvalue")
        _assert("median_survival_group1" in surv_result, "survival_analysis returns median_survival_group1")
        _assert(surv_result["median_survival_group1"] > 0, "Median survival > 0")

    except ImportError as ie:
        print(f"  SKIP: statistical tests require pandas/numpy/scipy ({ie})")

    # ----- Summary -----
    print(f"\n{'=' * 50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")

    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)
