#!/usr/bin/env python3
"""Deterministic solver for SeqQA benchmark questions.

For questions that are purely computational (e.g., "What AA is at position N
in the longest ORF of sequence X?"), we can compute the answer directly
without calling an LLM. This eliminates LLM errors on deterministic questions
and dramatically improves SeqQA accuracy.

Supported question patterns:
  1. AA at position N in longest ORF
  2. Reverse complement of a sequence
  3. GC content of a sequence
  4. Which restriction enzyme cuts a sequence
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, NamedTuple

logger = logging.getLogger("labbench")

# Amino acid name maps for matching against answer choices
_AA_NAMES_3: dict[str, str] = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "E": "Glu", "Q": "Gln", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
}
_AA_FULL: dict[str, str] = {
    "A": "Alanine", "R": "Arginine", "N": "Asparagine", "D": "Aspartic acid",
    "C": "Cysteine", "E": "Glutamic acid", "Q": "Glutamine", "G": "Glycine",
    "H": "Histidine", "I": "Isoleucine", "L": "Leucine", "K": "Lysine",
    "M": "Methionine", "F": "Phenylalanine", "P": "Proline", "S": "Serine",
    "T": "Threonine", "W": "Tryptophan", "Y": "Tyrosine", "V": "Valine",
}


def _match_aa_to_choice(aa: str, letter: str, text: str) -> bool:
    """Check if a single amino acid matches a choice text (any format)."""
    text_clean = text.strip()
    text_lower = text_clean.lower()
    aa_upper = aa.upper()

    # Single letter match
    if text_clean == aa_upper or text_lower == aa_upper.lower():
        return True
    # 3-letter code match
    three = _AA_NAMES_3.get(aa_upper, "")
    if three and text_lower == three.lower():
        return True
    # Full name match
    full = _AA_FULL.get(aa_upper, "")
    if full and text_lower == full.lower():
        return True
    # Partial containment (e.g., "Ala (A)" or "Alanine (Ala)")
    if three and (three in text_clean or three.lower() in text_lower):
        return True
    if full and (full in text_clean or full.lower() in text_lower):
        return True
    # Single-letter AA in text (only if text is short to avoid false matches)
    if len(text_clean) <= 3 and aa_upper in text_clean:
        return True
    return False


def _parse_choice_pairs(choices: list[str]) -> list[tuple[str, str]]:
    """Parse formatted choices like '(A) Ala' into (letter, text) pairs."""
    pairs: list[tuple[str, str]] = []
    for c in choices:
        cm = re.match(r"\(([A-E])\)\s*(.*)", c)
        if cm:
            pairs.append((cm.group(1), cm.group(2)))
    return pairs


def try_deterministic_seqqa(
    question_id: str,
    question_text: str,
    subtask: str,
    bench_subtask: str,
    correct_answer_text: str,
    choices: list[str],
    correct_letter: str,
    refuse_letter: str,
) -> dict[str, Any] | None:
    """Try to answer a SeqQA question deterministically without LLM.

    Returns a dict with result fields if the question can be solved
    computationally, or None if it should fall through to the LLM.

    The returned dict has keys: predicted, predicted_text, is_correct,
    reasoning, latency_ms — ready to be used to construct a ScoredResult.
    """
    from bench_helpers import (
        aa_at_position,
        reverse_complement as rc_helper,
        gc_content as gc_helper,
        find_restriction_sites as frs_helper,
    )

    start = time.monotonic()
    choice_pairs = _parse_choice_pairs(choices)

    def _result(predicted: str, reasoning: str) -> dict[str, Any]:
        idx = ord(predicted) - ord("A")
        predicted_text = choices[idx] if 0 <= idx < len(choices) else ""
        return {
            "predicted": predicted,
            "predicted_text": predicted_text,
            "is_correct": predicted == correct_letter,
            "reasoning": reasoning,
            "latency_ms": int((time.monotonic() - start) * 1000),
        }

    # --- Pattern 1: AA at position N in longest ORF ---
    pos_val = None
    seq_str = None

    # Pattern 1a: "position N in the longest ORF contained within the sequence XXXX"
    pm = re.search(
        r"position\s+(\d+)\s+in\s+the\s+longest\s+ORF.*?(?:sequence\s+)([ATCGU]{10,})",
        question_text,
        re.IGNORECASE | re.DOTALL,
    )
    if pm:
        pos_val = int(pm.group(1))
        seq_str = pm.group(2).upper()
    else:
        # Pattern 1b: "AA encoded at position N ... sequence XXXX"
        pm = re.search(
            r"(?:amino acid|AA)\s+encoded\s+at\s+position\s+(\d+).*?(?:sequence\s+)?([ATCGU]{10,})",
            question_text,
            re.IGNORECASE | re.DOTALL,
        )
        if pm:
            pos_val = int(pm.group(1))
            seq_str = pm.group(2).upper()
        else:
            # Pattern 1c: generic fallback — "position N" + any long DNA + "ORF" mentioned
            pm_pos = re.search(r"position\s+(\d+)", question_text, re.IGNORECASE)
            pm_seq = re.search(r"([ATCGU]{20,})", question_text)
            if pm_pos and pm_seq and "ORF" in question_text.upper():
                pos_val = int(pm_pos.group(1))
                seq_str = pm_seq.group(1).upper()

    if pos_val is not None and seq_str is not None:
        # LAB-Bench defines "longest ORF" as forward-strand only (M-started).
        # We compute forward-strand ORFs directly (faster than find_all_orfs
        # which also processes the reverse complement).
        from bench_helpers import translate_sequence as _translate

        aa = None
        orf_source = ""

        # Strategy 1: forward-strand ORFs only (matches LAB-Bench definition)
        fwd_orfs: list[tuple[int, str]] = []  # (length, protein)
        for frame in range(3):
            protein = _translate(seq_str, frame)
            # Find all M-started ORFs in this frame
            for m_match in re.finditer(r"M[^*]*", protein):
                prot_str = m_match.group()
                fwd_orfs.append((len(prot_str), prot_str))

        # Sort by length descending
        fwd_orfs.sort(key=lambda x: -x[0])

        if fwd_orfs and pos_val <= fwd_orfs[0][0]:
            aa = fwd_orfs[0][1][pos_val - 1]
            orf_source = f"fwd-strand ORF (len={fwd_orfs[0][0]})"

        # Strategy 2: fall back to aa_at_position (all 6 frames + stop-to-stop)
        # Only attempt for shorter sequences to avoid slow reverse_complement
        if aa is None and len(seq_str) < 3000:
            aa_fallback = aa_at_position(seq_str, pos_val)
            if not aa_fallback.startswith("Error"):
                aa = aa_fallback
                orf_source = "aa_at_position fallback (all frames)"

        if aa is not None:
            for letter, text in choice_pairs:
                if letter == refuse_letter:
                    continue
                if _match_aa_to_choice(aa, letter, text):
                    reasoning = (
                        f"[DETERMINISTIC] pos={pos_val} in {orf_source} -> {aa} "
                        f"({_AA_NAMES_3.get(aa, '?')}/{_AA_FULL.get(aa, '?')}). "
                        f"Matched choice ({letter}) {text}"
                    )
                    logger.info(
                        "  SeqQA deterministic: pos=%d -> %s -> (%s) [%s]",
                        pos_val, aa, letter, orf_source,
                    )
                    return _result(letter, reasoning)
            # Computed AA but couldn't match any choice
            logger.warning(
                "  SeqQA deterministic: computed AA=%s (pos=%d, %s) but no choice matched. "
                "Choices: %s",
                aa, pos_val, orf_source,
                [(l, t[:40]) for l, t in choice_pairs],
            )
        else:
            logger.debug(
                "  SeqQA deterministic: could not compute AA at pos=%d (seq_len=%d)",
                pos_val, len(seq_str),
            )

    # --- Pattern 1b: "AA sequence of the longest ORF in [sequence]" ---
    if "sequence of the longest orf" in question_text.lower() or (
        "longest orf" in question_text.lower() and "sequence" in question_text.lower()
        and pos_val is None  # didn't match a position pattern
    ):
        from bench_helpers import translate_sequence as _translate_1b

        pm_seq = re.search(r"([ATCGU]{20,})", question_text)
        if pm_seq:
            orf_seq = pm_seq.group(1).upper()
            # Find longest forward-strand ORF
            best_orf = ""
            for frame in range(3):
                protein = _translate_1b(orf_seq, frame)
                for m_match in re.finditer(r"M[^*]*\*?", protein):
                    prot_str = m_match.group()
                    if len(prot_str) > len(best_orf):
                        best_orf = prot_str

            if best_orf:
                for letter, text in choice_pairs:
                    if letter == refuse_letter:
                        continue
                    text_clean = text.strip()
                    # Match the full protein sequence against the choice
                    if text_clean == best_orf or text_clean.rstrip("*") == best_orf.rstrip("*"):
                        reasoning = (
                            f"[DETERMINISTIC] longest ORF sequence = {best_orf[:40]}... "
                            f"(len={len(best_orf)}). Matched choice ({letter})"
                        )
                        logger.info(
                            "  SeqQA deterministic: ORF sequence -> (%s) [len=%d]",
                            letter, len(best_orf),
                        )
                        return _result(letter, reasoning)

    # --- Pattern 2: reverse complement ---
    # Guard: only attempt regex if the keyword is present (avoids catastrophic backtracking)
    pm = None
    if "reverse complement" in question_text.lower():
        pm = re.search(
            r"reverse complement.*?([ATCGU]{10,})",
            question_text,
            re.IGNORECASE | re.DOTALL,
        )
        if not pm:
            pm = re.search(
                r"([ATCGU]{10,}).*?reverse complement",
                question_text,
                re.IGNORECASE | re.DOTALL,
            )
    if pm:
        seq_str = pm.group(1).upper()
        rc = rc_helper(seq_str)
        if not rc.startswith("Error"):
            for letter, text in choice_pairs:
                if letter == refuse_letter:
                    continue
                text_clean = text.strip().upper().replace("U", "T")
                rc_clean = rc.upper()
                if text_clean == rc_clean or rc_clean in text_clean:
                    reasoning = (
                        f"[DETERMINISTIC] reverse_complement(seq) = {rc[:50]}... "
                        f"Matched choice ({letter})"
                    )
                    logger.info("  SeqQA deterministic: reverse complement -> (%s)", letter)
                    return _result(letter, reasoning)

    # --- Pattern 3: GC content / percent GC ---
    pm = None
    text_lower_gc = question_text.lower()
    if "gc content" in text_lower_gc or "percent gc" in text_lower_gc or "gc percent" in text_lower_gc:
        # Just extract the DNA sequence directly
        pm = re.search(r"([ATCGU]{20,})", question_text)
    if pm:
        seq_str = pm.group(1).upper()
        gc = gc_helper(seq_str)
        gc_pct = round(gc * 100)  # Round to nearest integer for matching
        # Find the closest matching choice
        best_match = None
        best_dist = float("inf")
        for letter, text in choice_pairs:
            if letter == refuse_letter:
                continue
            text_clean = text.strip().replace("%", "").strip()
            try:
                val = float(text_clean)
                # Match as fraction (0-1) or percentage (0-100)
                dist_frac = abs(val - gc)
                dist_pct = abs(val - gc * 100)
                dist = min(dist_frac, dist_pct)
                if dist < best_dist:
                    best_dist = dist
                    best_match = letter
            except ValueError:
                continue
        if best_match is not None and best_dist < 2.0:
            reasoning = (
                f"[DETERMINISTIC] gc_content(seq) = {gc:.4f} ({gc*100:.1f}%). "
                f"Closest choice ({best_match}), dist={best_dist:.2f}"
            )
            logger.info("  SeqQA deterministic: GC=%.1f%% -> (%s)", gc * 100, best_match)
            return _result(best_match, reasoning)

    # --- Pattern 4: restriction enzyme ---
    # Guard: only attempt if enzyme/restriction keywords are present
    text_lower = question_text.lower()
    if ("restriction" in text_lower or "enzyme" in text_lower) and (
        "cut" in text_lower or "cleave" in text_lower or "digest" in text_lower
    ):
        # Extract the DNA sequence independently (avoids catastrophic backtracking)
        pm_seq = re.search(r"([ATCGU]{20,})", question_text)
        if pm_seq:
            seq_str = pm_seq.group(1).upper()
            sites = frs_helper(seq_str)
            if isinstance(sites, dict) and "error" not in sites:
                found_enzymes = set(sites.keys())
                for letter, text in choice_pairs:
                    if letter == refuse_letter:
                        continue
                    text_clean = text.strip()
                    if text_clean in found_enzymes:
                        reasoning = (
                            f"[DETERMINISTIC] find_restriction_sites(seq) found: "
                            f"{', '.join(sorted(found_enzymes))}. "
                            f"Matched choice ({letter}) {text_clean}"
                        )
                        logger.info(
                            "  SeqQA deterministic: restriction enzyme %s -> (%s)",
                            text_clean, letter,
                        )
                        return _result(letter, reasoning)

    # NOTE: ORF count (numlen), RE fragment count, and RE fragment length
    # patterns were tested but produce too many wrong answers due to
    # ambiguous definitions in the benchmark (e.g., ORF counting method,
    # which enzymes to use for digestion). These question types are left
    # to the LLM fallback for now.

    # No deterministic pattern matched
    return None
