#!/usr/bin/env python3
"""Standalone LAB-Bench benchmark pipeline for YOHAS 3.0.

Downloads (if needed), loads, and evaluates YOHAS on LAB-Bench subtasks:
  - DbQA:   520 MCQs testing ability to query biological databases
  - SeqQA:  MCQs testing DNA/protein sequence manipulation
  - LitQA2: MCQs testing literature recall and reasoning (RAG)

Competitor to beat: STELLA at 54% on DbQA.  Target: Biomni Lab at 78%.

Usage:
    python scripts/run_labbench.py --limit 5
    python scripts/run_labbench.py --limit 50 --mode zero-shot
    python scripts/run_labbench.py --mode agentic --trials 3 --limit 10
    python scripts/run_labbench.py                           # all DbQA questions
    python scripts/run_labbench.py --resume                  # resume from checkpoint
    python scripts/run_labbench.py --dry-run                 # no LLM calls, test pipeline
    python scripts/run_labbench.py --subtask seqqa --limit 3 --mode agentic
    python scripts/run_labbench.py --subtask litqa2 --limit 5 --mode agentic
    python scripts/run_labbench.py --subtask all --limit 10
    python scripts/run_labbench.py --subtask dbqa --replicas 3 --limit 5

Requires: ANTHROPIC_API_KEY in .env or environment.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import logging
import os
import random
import re
import signal
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

from benchmark_strategy import (
    BenchmarkStrategyTracker,
    QuestionOutcome,
    detect_databases_from_text,
    detect_question_type,
)
from bench_helpers import BENCH_HELPERS, BENCH_HELPERS_PROMPT

# ---------------------------------------------------------------------------
# Path setup — must happen before any backend imports
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks" / "labbench"
RESULTS_DIR = PROJECT_ROOT / "data" / "benchmarks" / "labbench_results"

sys.path.insert(0, str(BACKEND_DIR))

# Load .env before importing core modules
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    from dotenv import load_dotenv

    load_dotenv(_env_path, override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("labbench")

# ---------------------------------------------------------------------------
# Valid subtask identifiers
# ---------------------------------------------------------------------------

VALID_SUBTASKS = ("dbqa", "seqqa", "litqa2", "all")

# ---------------------------------------------------------------------------
# Model ID mapping — explicit model IDs so we don't depend on settings
# ---------------------------------------------------------------------------

MODEL_ID_MAP: dict[str, str] = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-20250514",
    "haiku": "claude-haiku-4-5-20251001",
}

# Map CLI subtask name -> HuggingFace config name
HF_CONFIG_MAP: dict[str, str] = {
    "dbqa": "DbQA",
    "seqqa": "SeqQA",
    "litqa2": "LitQA2",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class BenchQuestion(NamedTuple):
    """A single benchmark question parsed from the JSONL dataset."""

    id: str
    question: str
    correct_answer: str  # the ideal answer text
    distractors: list[str]  # wrong answer texts
    subtask: str
    context: str
    bench_subtask: str  # top-level subtask: dbqa, seqqa, litqa2


class ScoredResult(NamedTuple):
    """Result of evaluating a single question."""

    question_id: str
    subtask: str
    predicted: str  # letter chosen
    predicted_text: str  # full text of chosen answer
    correct_answer: str  # letter of correct answer
    correct_text: str  # full text of correct answer
    is_correct: bool
    is_refused: bool  # chose "Insufficient information"
    reasoning: str
    tokens_used: int
    latency_ms: int
    error: str | None
    bench_subtask: str  # top-level subtask: dbqa, seqqa, litqa2


# ---------------------------------------------------------------------------
# Dataset download + loading
# ---------------------------------------------------------------------------

REFUSE_CHOICE = "Insufficient information to answer the question"


def _download_subtask_dataset(subtask_key: str) -> Path:
    """Download a LAB-Bench subtask dataset from HuggingFace if not already present.

    Args:
        subtask_key: one of 'dbqa', 'seqqa', 'litqa2'

    Returns path to the JSONL file.
    """
    hf_config = HF_CONFIG_MAP[subtask_key]
    filename = f"{subtask_key}.jsonl"

    # Check legacy path for dbqa
    if subtask_key == "dbqa":
        legacy_path = PROJECT_ROOT / "data" / "benchmarks" / "lab_bench" / "dbqa.jsonl"
        if legacy_path.exists():
            logger.info("Using existing %s dataset at %s", hf_config, legacy_path)
            return legacy_path

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = DATA_DIR / filename

    if jsonl_path.exists():
        logger.info("Using existing %s dataset at %s", hf_config, jsonl_path)
        return jsonl_path

    logger.info("Downloading LAB-Bench %s from HuggingFace...", hf_config)

    try:
        from datasets import load_dataset

        ds = load_dataset("futurehouse/lab-bench", hf_config, split="train")
        with open(jsonl_path, "w") as f:
            for row in ds:
                record = {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": row["ideal"],
                    "choices": row["distractors"],
                    "bench_subtask": subtask_key,
                    "metadata": {
                        "canary": row.get("canary", ""),
                        "source": row.get("source"),
                        "subtask": row.get("subtask", ""),
                    },
                }
                f.write(json.dumps(record) + "\n")
        logger.info("Downloaded %d %s questions to %s", len(ds), hf_config, jsonl_path)
        return jsonl_path

    except ImportError:
        # Try downloading parquet directly
        logger.info("datasets library not available; downloading parquet directly...")
        import urllib.request

        parquet_url = (
            f"https://huggingface.co/datasets/futurehouse/lab-bench/"
            f"resolve/refs%2Fconvert%2Fparquet/{hf_config}/train/0000.parquet"
        )
        parquet_path = DATA_DIR / f"{subtask_key}.parquet"
        urllib.request.urlretrieve(parquet_url, parquet_path)

        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        with open(jsonl_path, "w") as f:
            for _, row in df.iterrows():
                record = {
                    "id": str(row["id"]),
                    "question": row["question"],
                    "answer": row["ideal"],
                    "choices": list(row["distractors"]),
                    "bench_subtask": subtask_key,
                    "metadata": {
                        "canary": row.get("canary", ""),
                        "source": row.get("source"),
                        "subtask": row.get("subtask", ""),
                    },
                }
                f.write(json.dumps(record) + "\n")

        logger.info("Downloaded %d %s questions to %s", len(df), hf_config, jsonl_path)
        return jsonl_path


def download_dbqa_dataset() -> Path:
    """Download DbQA dataset from HuggingFace if not already present.

    Returns path to the JSONL file.  (Kept for backward-compatibility.)
    """
    return _download_subtask_dataset("dbqa")


def download_seqqa_dataset() -> Path:
    """Download SeqQA dataset from HuggingFace if not already present."""
    return _download_subtask_dataset("seqqa")


def download_litqa2_dataset() -> Path:
    """Download LitQA2 dataset from HuggingFace if not already present."""
    return _download_subtask_dataset("litqa2")


def _load_questions_from_jsonl(
    path: Path,
    bench_subtask: str,
    limit: int | None = None,
) -> list[BenchQuestion]:
    """Load questions from a JSONL file."""
    questions: list[BenchQuestion] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            questions.append(
                BenchQuestion(
                    id=row["id"],
                    question=row["question"],
                    correct_answer=row["answer"],
                    distractors=row.get("choices", []),
                    subtask=row.get("metadata", {}).get("subtask", "unknown"),
                    context=row.get("context", ""),
                    bench_subtask=row.get("bench_subtask", bench_subtask),
                )
            )
    return questions


def load_dbqa_questions(path: Path, limit: int | None = None) -> list[BenchQuestion]:
    """Load DbQA questions from JSONL file."""
    return _load_questions_from_jsonl(path, "dbqa", limit)


def load_subtask_questions(
    subtask_key: str,
    limit: int | None = None,
) -> list[BenchQuestion]:
    """Download (if needed) and load questions for a given subtask.

    Args:
        subtask_key: one of 'dbqa', 'seqqa', 'litqa2'
        limit: max questions to load

    Returns list of BenchQuestion.
    """
    path = _download_subtask_dataset(subtask_key)
    return _load_questions_from_jsonl(path, subtask_key, limit)


def load_all_subtask_questions(
    subtask_keys: list[str],
    limit: int | None = None,
) -> list[BenchQuestion]:
    """Load questions for multiple subtasks.

    If limit is set, it applies *per subtask* (so --limit 5 --subtask all
    loads 5 per subtask = 15 total).
    """
    all_questions: list[BenchQuestion] = []
    for key in subtask_keys:
        qs = load_subtask_questions(key, limit=limit)
        logger.info("Loaded %d %s questions", len(qs), key.upper())
        all_questions.extend(qs)
    return all_questions


def build_choices(question: BenchQuestion, seed: int | None = None) -> tuple[list[str], str, str]:
    """Build shuffled multiple-choice list from correct + distractors + refuse.

    Returns:
        (formatted_choices, correct_letter, refuse_letter)
    """
    rng = random.Random(seed if seed is not None else hash(question.id))
    raw = [question.correct_answer] + list(question.distractors) + [REFUSE_CHOICE]
    rng.shuffle(raw)

    formatted = []
    correct_letter = ""
    refuse_letter = ""
    for i, text in enumerate(raw):
        letter = chr(65 + i)  # A, B, C, ...
        formatted.append(f"({letter}) {text}")
        if text == question.correct_answer:
            correct_letter = letter
        if text == REFUSE_CHOICE:
            refuse_letter = letter

    return formatted, correct_letter, refuse_letter


# ---------------------------------------------------------------------------
# Agent: builds prompt and calls LLM
# ---------------------------------------------------------------------------

# Category-specific hints for DbQA subtasks
SUBTASK_HINTS: dict[str, str] = {
    "dga_task": (
        "This question is about disease-gene associations (DGA). It asks which gene is in "
        "DisGeNET but NOT in OMIM for a given disease.\n"
        "KEY REASONING STRATEGY:\n"
        "- OMIM contains well-established Mendelian disease genes with strong evidence.\n"
        "- DisGeNET aggregates from many sources (GWAS, literature mining, animal models) and "
        "contains many more associations, including weaker/indirect ones.\n"
        "- The CORRECT answer is typically a gene with weaker/indirect evidence (found by GWAS, "
        "literature mining, or animal models) that DisGeNET includes but OMIM does not.\n"
        "- Well-known disease genes (like MNX1 for Currarino) are usually in BOTH databases "
        "and are therefore DISTRACTORS, not correct answers.\n"
        "- Genes that seem unrelated or have indirect connections to the disease are more "
        "likely to be the correct answer (in DisGeNET only).\n"
        "- If you recognize a gene as a classic/canonical gene for the disease, it is likely "
        "a distractor (in both DisGeNET AND OMIM)."
    ),
    "gene_location_task": (
        "This question is about gene chromosomal locations. Key databases: "
        "NCBI Gene (official gene symbol, chromosome, map location like 17q21.31), "
        "Ensembl (precise coordinates, GRCh38). Remember: cytogenetic bands (p/q arms), "
        "gene symbols may have aliases, and some genes span multiple bands."
    ),
    "mirna_targets_task": (
        "This question is about microRNA-target interactions. Key database: "
        "miRTarBase (experimentally validated interactions — reporter assay, western blot, "
        "qRT-PCR, microarray, sequencing). Pay attention to: miRNA naming conventions "
        "(hsa-miR-XXX-Xp/Xs), validation methods, and whether the interaction is "
        "strong (reporter assay) or functional (expression change)."
    ),
    "mouse_tumor_gene_sets": (
        "This question involves mouse tumor gene sets from MSigDB. Key collection: "
        "M (mouse) gene sets mapped from human orthologs. Consider tumor type, "
        "up/down regulation, and the specific study the gene set comes from."
    ),
    "oncogenic_signatures_task": (
        "This question is about oncogenic gene signatures from MSigDB C6 collection. "
        "These are gene sets representing signatures of cellular pathways often "
        "dysregulated in cancer (e.g., KRAS.600_UP.V1_UP = genes upregulated by "
        "oncogenic KRAS). Pay attention to: the specific oncogene/pathway, "
        "direction (UP/DN), version, and cell type."
    ),
    "tfbs_GTRD_task": (
        "This question is about transcription factor binding sites from the GTRD "
        "(Gene Transcription Regulation Database). GTRD aggregates ChIP-seq data "
        "to identify TF binding sites. Pay attention to: the specific TF, target gene, "
        "cell line/tissue, and whether the binding is proximal (promoter) or distal (enhancer)."
    ),
    "variant_from_sequence_task": (
        "This question asks about genetic variants. Key databases: ClinVar (clinical "
        "significance: pathogenic, likely pathogenic, VUS, likely benign, benign), "
        "gnomAD (population allele frequencies), dbSNP (rs IDs). Pay attention to: "
        "the specific variant notation (HGVS), review status (stars in ClinVar), "
        "and classification criteria."
    ),
    "variant_multi_sequence_task": (
        "This question involves variants across multiple sequences. Use knowledge of "
        "ClinVar, gnomAD, and dbSNP. Consider: variant consequences (missense, nonsense, "
        "frameshift), position in the protein, and clinical significance across databases."
    ),
    "vax_response_task": (
        "This question is about vaccine response data and immunology. Consider "
        "ImmPort, GEO/ArrayExpress expression data, and immune-related gene sets. "
        "Pay attention to: specific vaccine types, immune cell markers, cytokine responses, "
        "and time-course expression patterns."
    ),
    "viral_ppi_task": (
        "This question is about viral protein-protein interactions. Key databases: "
        "IntAct (MI-score, interaction detection methods), BioGRID (genetic and physical "
        "interactions), VirHostNet. Pay attention to: the specific viral protein, "
        "host interactor, detection method, and whether the interaction is direct (binary) "
        "or indirect (co-complex)."
    ),
}


def _get_subtask_hint(question: BenchQuestion) -> str:
    """Get a hint for the subtask, matching on prefix."""
    # For DbQA questions, check the narrow subtask field
    for key, hint in SUBTASK_HINTS.items():
        if key in question.subtask:
            return hint
    # Fallback hints by bench_subtask
    if question.bench_subtask == "seqqa":
        return (
            "This is a sequence manipulation question. Use BioPython to work "
            "with DNA/RNA/protein sequences. Think carefully about reverse "
            "complements, translations, restriction sites, and codon tables."
        )
    if question.bench_subtask == "litqa2":
        return (
            "This is a literature-based question. Think about which published "
            "studies are relevant and what their key findings were."
        )
    return (
        "Use your knowledge of biological databases to determine the answer. "
        "Consider which database would be the authoritative source for this type of information."
    )


def build_prompt(
    question: BenchQuestion,
    choices: list[str],
    strategy_injection: str = "",
) -> str:
    """Build the LLM prompt for a benchmark question."""
    parts = []

    # Inject learned strategy before the question if available
    if strategy_injection:
        parts.append(strategy_injection)

    if question.context:
        parts.append(f"Context:\n{question.context}")

    parts.append(f"Question: {question.question}")
    parts.append("Choices:\n" + "\n".join(f"  {c}" for c in choices))

    hint = _get_subtask_hint(question)
    parts.append(f"\nDatabase hint: {hint}")

    parts.append(
        "\nFormat your response as:\n"
        "Database(s) involved: [identify the database(s) this question is about]\n"
        "Key facts: [relevant knowledge about the specific entities mentioned]\n"
        "Reasoning: [step-by-step analysis of each option]\n"
        "Answer: X\n\n"
        "Rules:\n"
        "1. Identify the specific database(s) and recall their data types, schema, and conventions.\n"
        "2. For each choice, assess whether it is consistent with what the database would contain.\n"
        "3. Eliminate options that contradict known database entries or biological facts.\n"
        "4. If two options seem plausible, pick the one more consistent with the database's known content.\n"
        "5. \"Insufficient information\" is almost never correct. Only choose it if you are CERTAIN "
        "the databases cannot answer the question. When in doubt, commit to a specific answer.\n"
        "6. State your final answer on the LAST line in exactly this format: Answer: X\n"
        "   where X is a single letter (A, B, C, D, or E)."
    )

    return "\n\n".join(parts)


SYSTEM_PROMPT = (
    "You are an expert biomedical researcher with deep knowledge of biological databases including:\n"
    "- DisGeNET: gene-disease associations (GDA scores, evidence types, source databases)\n"
    "- OMIM: Mendelian disease-gene relationships (phenotype MIM numbers, inheritance)\n"
    "- UniProt: protein function, subcellular location, tissue expression, domains\n"
    "- KEGG: metabolic and signaling pathways, drug targets, disease pathways\n"
    "- ClinVar: variant clinical significance (pathogenic, benign, VUS)\n"
    "- gnomAD: population allele frequencies, constraint metrics (pLI, Z-scores)\n"
    "- IntAct/BioGRID: protein-protein interactions (binary, co-complex)\n"
    "- NCBI Gene: gene summaries, orthologs, RefSeq, chromosomal locations\n"
    "- Ensembl: genome annotation, gene coordinates, regulatory elements\n"
    "- MSigDB: gene sets and signatures (hallmark, C2-CGP, C6-oncogenic, C7-immunologic)\n"
    "- miRTarBase: experimentally validated miRNA-target interactions\n"
    "- GTRD: transcription factor binding sites from ChIP-seq experiments\n"
    "- dbSNP: SNP identifiers, allele frequencies, functional annotations\n"
    "- Reactome: curated biological pathways and reactions\n"
    "- ChEMBL: bioactivity data for drug-like molecules\n"
    "- COSMIC: somatic mutations in cancer\n\n"
    "For each question:\n"
    "1. Identify which database(s) the question is about\n"
    "2. Recall what you know about the specific entries, genes, proteins, or variants mentioned\n"
    "3. Reason carefully about the answer, considering edge cases and database-specific conventions\n"
    "4. Choose the most likely correct answer\n\n"
    "IMPORTANT RULES:\n"
    "1. Most questions have a definitive correct answer. \"Insufficient information\" is rarely "
    "correct (<10% of the time). Prefer a specific answer over refusing.\n"
    "2. DISTRACTOR AWARENESS: Questions about \"which entry is in database X but NOT database Y\" "
    "use well-known/canonical entries as distractors (they appear in BOTH databases). The correct "
    "answer is often a less obvious entry that only appears in the more comprehensive database.\n"
    "3. When a gene/protein seems like the obvious answer because it's strongly associated with "
    "the condition, it is likely a distractor. Look for the less obvious option.\n"
    "4. Read the database hint carefully — it contains reasoning strategies specific to the question type."
)


def extract_answer_letter(response: str) -> str:
    """Extract the answer letter from LLM response."""
    # Try "Answer: X" pattern first (most reliable)
    patterns = [
        r"Answer:\s*\(?([A-Ea-e])\)?",
        r"(?:answer|choice|option)\s*(?:is|:)\s*\(?([A-Ea-e])\)?",
        r"^\s*\(?([A-Ea-e])\)\s*$",
        r"\b([A-Ea-e])\b\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()

    # Fallback: last standalone capital letter A-E in the response
    matches = re.findall(r"\b([A-E])\b", response)
    if matches:
        return matches[-1]

    logger.warning("Could not extract answer letter from response (len=%d)", len(response))
    return ""


# ---------------------------------------------------------------------------
# Subtask-specific agentic system prompts
# ---------------------------------------------------------------------------

AGENTIC_SYSTEM_PROMPT_DBQA = (
    "You are an expert biomedical researcher answering a multiple-choice question "
    "about biological databases. You have deep knowledge of databases including: "
    "DisGeNET, OMIM, ClinVar, UniProt, KEGG, Reactome, ChEMBL, MSigDB, GTRD, "
    "miRTarBase, dbSNP, gnomAD, IntAct, BioGRID, NCBI Gene, and Ensembl.\n\n"
    "STRATEGY — follow this approach:\n"
    "1. FIRST: Think through the problem using your training knowledge. Identify which "
    "database(s) the question is about and what you know about the specific entities.\n"
    "2. THEN: If you are confident (>70%), just give your answer. Do NOT write code.\n"
    "3. ONLY IF UNCERTAIN: Write Python code to query FREE, PUBLIC APIs to verify. "
    "Wrap code in <execute> tags.\n\n"
    "FREE APIs you CAN query (no authentication needed):\n"
    "- UniProt REST: https://rest.uniprot.org/uniprotkb/search?query=...&format=json\n"
    "- NCBI E-utilities: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&term=...\n"
    "- NCBI Gene: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gene&id=...\n"
    "- KEGG REST: https://rest.kegg.jp/get/hsa:GENEID or https://rest.kegg.jp/find/pathway/...\n"
    "- Ensembl REST: https://rest.ensembl.org/lookup/symbol/homo_sapiens/GENE?content-type=application/json\n"
    "- MyGene.info: https://mygene.info/v3/query?q=GENE&fields=genomic_pos,symbol,name\n\n"
    "APIs that REQUIRE authentication (DO NOT try these — they will fail):\n"
    "- DisGeNET API (requires API key)\n"
    "- OMIM API (requires API key)\n"
    "- BioGRID REST (requires access key)\n"
    "For DisGeNET, OMIM, and similar gated databases, rely on your training knowledge instead.\n\n"
    "Available libraries: pandas, numpy, scipy, requests, json, re, math, "
    "collections, itertools, Bio (biopython).\n\n"
    "Rules:\n"
    "1. Prefer answering from knowledge over running code. Code is a fallback, not the default.\n"
    "2. If a code query fails or times out, DO NOT keep retrying — just answer from knowledge.\n"
    "3. \"Insufficient information\" is almost never correct (<10% of questions). Commit to an answer.\n"
    "4. When you are confident in your answer, state it on the LAST line as:\n"
    "   Answer: X\n"
    "   where X is a single letter (A, B, C, D, or E).\n"
    "5. Keep your reasoning concise — do not over-deliberate."
)

AGENTIC_SYSTEM_PROMPT_SEQQA = (
    "You are a molecular biology expert solving a sequence analysis question.\n\n"
    "RULES:\n"
    "1. You MUST write Python code to solve this. Do NOT answer from memory or mental math.\n"
    "2. Use BioPython for ALL sequence operations:\n"
    "   - from Bio.Seq import Seq\n"
    "   - from Bio.SeqUtils import molecular_weight, gc_fraction\n"
    "   - from Bio.Restriction import *\n"
    "   - from Bio.Data.CodonTable import standard_dna_table\n"
    "3. ALWAYS print your intermediate results so you can verify them.\n"
    "4. After computing the answer, state it clearly.\n"
    "5. To execute code, wrap it in <execute> tags.\n\n"
    "Available functions (pre-imported in the execution environment):\n"
    "```python\n"
    "from Bio.Seq import Seq\n"
    "from Bio import SeqIO\n"
    "from Bio.SeqUtils import molecular_weight, gc_fraction\n"
    "from Bio.Restriction import RestrictionBatch, Analysis, AllEnzymes\n"
    "from Bio.Data.CodonTable import standard_dna_table\n"
    "import re, json, math, collections, itertools\n"
    "```\n\n"
    "Common operations:\n"
    "- Reverse complement: str(Seq('ATCG').reverse_complement())\n"
    "- Translate: str(Seq('ATGATGATG').translate())\n"
    "- Find ORFs across all 6 reading frames:\n"
    "    import re\n"
    "    from Bio.Seq import Seq\n"
    "    def find_all_orfs(dna_seq):\n"
    "        s = Seq(dna_seq)\n"
    "        orfs = []\n"
    "        for strand, seq in [('+', s), ('-', s.reverse_complement())]:\n"
    "            for frame in range(3):\n"
    "                protein = str(seq[frame:].translate())\n"
    "                for m in re.finditer(r'M[^*]+', protein):\n"
    "                    orfs.append({'strand': strand, 'frame': frame+1, 'length': len(m.group()), 'protein': m.group()})\n"
    "        return sorted(orfs, key=lambda x: -x['length'])\n"
    "    # Position 1 = first AA (M), position 10 = orfs[0]['protein'][9]\n"
    "- Restriction sites: Analysis(RestrictionBatch(first=[]), seq).full()\n"
    "- GC content: gc_fraction('ATCGATCG')\n"
    "- Molecular weight: molecular_weight('ATCG', 'DNA')\n"
    "- Complement: str(Seq('ATCG').complement())\n"
    "- Transcribe: str(Seq('ATCG').transcribe())\n"
    "- Back-transcribe: str(Seq('AUCG').back_transcribe())\n\n"
    "WORKFLOW:\n"
    "1. Read the question carefully and identify what sequence operation is needed.\n"
    "2. Write Python code using BioPython to compute the answer. Print all results.\n"
    "3. Compare the computed result against each answer choice.\n"
    "4. State your final answer as: Answer: X\n\n"
    "IMPORTANT: You MUST execute at least one code block before giving your answer.\n"
    "Sequence operations done in your head are error-prone. Always use code.\n"
    "Only choose 'Insufficient information' if you truly cannot determine the answer."
)

AGENTIC_SYSTEM_PROMPT_LITQA2 = (
    "You are a biomedical literature expert answering a multiple-choice question "
    "that requires knowledge of published scientific papers and their findings.\n\n"
    "You can reason step-by-step AND write Python code to search for papers. "
    "To execute code, wrap it in <execute> tags:\n\n"
    "<execute>\n"
    "import requests\n"
    "r = requests.get(\n"
    "    'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',\n"
    "    params={'db': 'pubmed', 'term': 'BRCA1 breast cancer therapy', 'retmode': 'json', 'retmax': 5}\n"
    ")\n"
    "data = r.json()\n"
    "ids = data['esearchresult']['idlist']\n"
    "print('PubMed IDs:', ids)\n"
    "</execute>\n\n"
    "Search PubMed and Semantic Scholar for relevant papers. Use:\n"
    "- PubMed E-utilities: `requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=...')`\n"
    "- PubMed fetch: `requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=...&rettype=abstract')`\n"
    "- Semantic Scholar: `requests.get('https://api.semanticscholar.org/graph/v1/paper/search?query=...')`\n"
    "Read abstracts to find the answer. Extract specific claims from papers.\n\n"
    "Available libraries: pandas, numpy, scipy, requests, json, re, math, "
    "collections, itertools, Bio (biopython).\n\n"
    "Rules:\n"
    "1. Reason step-by-step before answering.\n"
    "2. Use code to search for relevant papers when the question references "
    "   specific findings, authors, or studies.\n"
    "3. You can execute multiple code blocks across turns.\n"
    "4. When you are confident in your answer, state it on the LAST line as:\n"
    "   Answer: X\n"
    "   where X is a single letter (A, B, C, D, or E).\n"
    "5. Prefer evidence from paper abstracts over guessing.\n"
    "6. Only choose 'Insufficient information' if you truly cannot find the answer."
)

# Legacy alias — kept for backward-compatibility with any external callers
AGENTIC_SYSTEM_PROMPT = AGENTIC_SYSTEM_PROMPT_DBQA


def _get_agentic_system_prompt(bench_subtask: str) -> str:
    """Return the appropriate agentic system prompt for a bench subtask.

    All prompts get the DB_HELPERS_PROMPT_BLOCK appended so agents know
    about the pre-built query functions available in the code namespace.
    """
    if bench_subtask == "seqqa":
        return AGENTIC_SYSTEM_PROMPT_SEQQA + DB_HELPERS_PROMPT_BLOCK
    if bench_subtask == "litqa2":
        return AGENTIC_SYSTEM_PROMPT_LITQA2 + DB_HELPERS_PROMPT_BLOCK
    return AGENTIC_SYSTEM_PROMPT_DBQA + DB_HELPERS_PROMPT_BLOCK


# ---------------------------------------------------------------------------
# Pre-built database query helpers (injected into agent code namespace)
# ---------------------------------------------------------------------------


def query_uniprot(
    query: str,
    fields: str = "accession,id,gene_names,protein_name,organism_name,length,go_p,go_c,go_f,cc_subcellular_location",
    limit: int = 5,
) -> list[dict]:
    """Query UniProt REST API. Returns list of protein entries."""
    import requests as _req

    url = (
        f"https://rest.uniprot.org/uniprotkb/search"
        f"?query={query}&fields={fields}&format=json&size={limit}"
    )
    r = _req.get(url, timeout=15)
    r.raise_for_status()
    return r.json().get("results", [])


def query_ncbi_gene(query: str, limit: int = 5) -> list[dict]:
    """Query NCBI Gene via E-utilities. Returns gene summaries."""
    import requests as _req

    search_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=gene&term={query}&retmax={limit}&retmode=json"
    )
    r = _req.get(search_url, timeout=15)
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    fetch_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        f"?db=gene&id={','.join(ids)}&retmode=json"
    )
    r = _req.get(fetch_url, timeout=15)
    result = r.json().get("result", {})
    return [result[gid] for gid in ids if gid in result]


def query_kegg(pathway_or_gene: str) -> dict:
    """Query KEGG REST API for pathway or gene info."""
    import requests as _req

    url = f"https://rest.kegg.jp/get/{pathway_or_gene}"
    r = _req.get(url, timeout=15)
    if r.status_code == 200:
        return {"text": r.text[:3000]}
    return {"error": f"KEGG returned {r.status_code}"}


def query_ncbi_clinvar(query: str, limit: int = 5) -> list[dict]:
    """Query ClinVar via E-utilities."""
    import requests as _req

    search_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=clinvar&term={query}&retmax={limit}&retmode=json"
    )
    r = _req.get(search_url, timeout=15)
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    fetch_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        f"?db=clinvar&id={','.join(ids)}&retmode=json"
    )
    r = _req.get(fetch_url, timeout=15)
    result = r.json().get("result", {})
    return [result[vid] for vid in ids if vid in result]


def query_pubmed(query: str, limit: int = 5) -> list[dict]:
    """Search PubMed and return article summaries."""
    import requests as _req

    search_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&term={query}&retmax={limit}&retmode=json"
    )
    r = _req.get(search_url, timeout=15)
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    fetch_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={','.join(ids)}&rettype=abstract&retmode=text"
    )
    r = _req.get(fetch_url, timeout=15)
    return [{"abstracts": r.text[:3000]}]


def query_ensembl_gene(gene_symbol: str, species: str = "human") -> dict:
    """Get gene info from Ensembl REST API."""
    import requests as _req

    url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_symbol}?expand=1"
    r = _req.get(url, headers={"Content-Type": "application/json"}, timeout=15)
    if r.status_code == 200:
        return r.json()
    return {"error": f"Ensembl returned {r.status_code}"}


def query_string_interactions(
    protein: str, species: int = 9606, limit: int = 10,
) -> list[dict]:
    """Get protein-protein interactions from STRING-db."""
    import requests as _req

    url = (
        f"https://string-db.org/api/json/network"
        f"?identifiers={protein}&species={species}&limit={limit}"
    )
    r = _req.get(url, timeout=15)
    if r.status_code == 200:
        return r.json()
    return []


# All pre-built helpers in a dict for easy namespace injection
DB_QUERY_HELPERS: dict[str, Any] = {
    "query_uniprot": query_uniprot,
    "query_ncbi_gene": query_ncbi_gene,
    "query_kegg": query_kegg,
    "query_ncbi_clinvar": query_ncbi_clinvar,
    "query_pubmed": query_pubmed,
    "query_ensembl_gene": query_ensembl_gene,
    "query_string_interactions": query_string_interactions,
}

# Description block injected into agentic system prompts
DB_HELPERS_PROMPT_BLOCK = (
    "\n\nYou have pre-built database query functions available in the code "
    "execution environment:\n"
    "- query_uniprot(\"BRCA1\") -> protein entries with GO terms, subcellular location\n"
    "- query_ncbi_gene(\"TP53\") -> gene summaries from NCBI\n"
    "- query_kegg(\"hsa:7157\") -> KEGG pathway/gene info\n"
    "- query_ncbi_clinvar(\"BRAF V600E\") -> ClinVar variant classifications\n"
    "- query_pubmed(\"EGFR resistance NSCLC\") -> PubMed abstracts\n"
    "- query_ensembl_gene(\"KRAS\") -> Ensembl gene info with coordinates\n"
    "- query_string_interactions(\"TP53\") -> STRING protein interactions\n\n"
    "USE THESE FUNCTIONS. They work reliably. Do NOT try to construct API URLs yourself."
) + BENCH_HELPERS_PROMPT


# ---------------------------------------------------------------------------
# SeqQA helpers: BioPython pre-imports & sequence extraction
# ---------------------------------------------------------------------------


def _inject_biopython_namespace(namespace: dict[str, Any]) -> None:
    """Pre-import BioPython modules into the execution namespace for SeqQA.

    Handles import failures gracefully — each module is tried independently
    so a missing sub-package does not block the rest.
    """
    try:
        from Bio.Seq import Seq
        namespace["Seq"] = Seq
    except ImportError:
        pass
    try:
        from Bio import SeqIO
        namespace["SeqIO"] = SeqIO
    except ImportError:
        pass
    try:
        from Bio.SeqUtils import molecular_weight, GC123
        namespace["molecular_weight"] = molecular_weight
        namespace["GC123"] = GC123
        # BioPython >=1.80: GC renamed to gc_fraction
        try:
            from Bio.SeqUtils import GC  # type: ignore[attr-defined]
            namespace["GC"] = GC
        except ImportError:
            from Bio.SeqUtils import gc_fraction
            namespace["GC"] = gc_fraction
            namespace["gc_fraction"] = gc_fraction
    except ImportError:
        pass
    try:
        from Bio.Restriction import RestrictionBatch, Analysis, AllEnzymes
        namespace["RestrictionBatch"] = RestrictionBatch
        namespace["Analysis"] = Analysis
        namespace["AllEnzymes"] = AllEnzymes
    except ImportError:
        pass
    try:
        from Bio.Data.CodonTable import standard_dna_table
        namespace["standard_dna_table"] = standard_dna_table
    except ImportError:
        pass
    try:
        from Bio.SeqUtils import MeltingTemp
        namespace["MeltingTemp"] = MeltingTemp
    except ImportError:
        pass


def _extract_sequences_from_text(text: str) -> dict[str, list[str]]:
    """Extract DNA/RNA and protein sequences from question text.

    Returns dict with 'dna' and 'protein' keys, each a list of sequences
    found (10+ characters long to avoid false positives).
    """
    # DNA/RNA: runs of ATCGU (case-insensitive)
    dna_seqs = re.findall(r'[ATCGUatcgu]{10,}', text)
    # De-duplicate preserving order, normalize to uppercase
    seen: set[str] = set()
    unique_dna: list[str] = []
    for s in dna_seqs:
        s_upper = s.upper()
        if s_upper not in seen:
            seen.add(s_upper)
            unique_dna.append(s_upper)

    # Protein: runs of standard amino acid single-letter codes (20 AA)
    # Exclude sequences that look like DNA (only ATCG)
    protein_seqs = re.findall(r'[ACDEFGHIKLMNPQRSTVWY]{10,}', text)
    unique_protein: list[str] = []
    seen_prot: set[str] = set()
    for s in protein_seqs:
        if s not in seen_prot and not re.fullmatch(r'[ATCG]+', s):
            seen_prot.add(s)
            unique_protein.append(s)

    return {"dna": unique_dna, "protein": unique_protein}


def _build_seqqa_sequence_context(question_text: str) -> str:
    """Build a context string with pre-extracted sequences from the question.

    Injected into the first-turn prompt for SeqQA so the agent has them
    available as named variables in code.
    """
    seqs = _extract_sequences_from_text(question_text)
    parts: list[str] = []
    if seqs["dna"]:
        parts.append("Detected DNA/RNA sequence(s) in the question:")
        for i, s in enumerate(seqs["dna"]):
            var_name = f"seq{i + 1}" if len(seqs["dna"]) > 1 else "seq"
            parts.append(f'  {var_name} = "{s}"  ({len(s)} nt)')
    if seqs["protein"]:
        parts.append("Detected protein sequence(s) in the question:")
        for i, s in enumerate(seqs["protein"]):
            var_name = f"prot{i + 1}" if len(seqs["protein"]) > 1 else "prot"
            parts.append(f'  {var_name} = "{s}"  ({len(s)} aa)')
    if parts:
        parts.append(
            "\nUse these sequences directly in your code. "
            "Copy them exactly — do NOT retype sequences by hand."
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Agentic mode: multi-turn reasoning with code execution
# ---------------------------------------------------------------------------


def execute_code_safely(
    code: str, timeout: int = 30, *, seqqa_mode: bool = False,
) -> str:
    """Execute Python code in a sandboxed namespace with timeout.

    Args:
        code: Python code string to execute.
        timeout: Max seconds before killing execution.
        seqqa_mode: If True, pre-import all BioPython modules into the
            namespace so agents can use them without explicit imports.

    Returns stdout output or error message, truncated to 5000 chars.
    """
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]
    try:
        import pandas as pd
    except ImportError:
        pd = None  # type: ignore[assignment]
    try:
        import requests as req_lib
    except ImportError:
        req_lib = None  # type: ignore[assignment]
    try:
        import scipy.stats as scipy_stats
    except ImportError:
        scipy_stats = None  # type: ignore[assignment]

    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "json": json,
        "re": re,
        "math": __import__("math"),
        "collections": __import__("collections"),
        "itertools": __import__("itertools"),
    }
    if np is not None:
        namespace["np"] = np
        namespace["numpy"] = np
    if pd is not None:
        namespace["pd"] = pd
        namespace["pandas"] = pd
    if req_lib is not None:
        namespace["requests"] = req_lib
    if scipy_stats is not None:
        namespace["stats"] = scipy_stats
        namespace["scipy"] = __import__("scipy")

    # Always make Bio available for SeqQA
    try:
        import Bio
        namespace["Bio"] = Bio
    except ImportError:
        pass

    # SeqQA mode: pre-import all BioPython modules into namespace
    if seqqa_mode:
        _inject_biopython_namespace(namespace)

    # Inject pre-built database query helpers
    namespace.update(DB_QUERY_HELPERS)

    # Inject bench_helpers (sequence, DB, stats functions)
    namespace.update(BENCH_HELPERS)

    f = io.StringIO()

    # Use signal-based timeout on Unix; skip on Windows
    use_signal = hasattr(signal, "SIGALRM")

    def _handler(signum: int, frame: Any) -> None:
        raise TimeoutError("Code execution timed out")

    if use_signal:
        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout)

    try:
        with contextlib.redirect_stdout(f):
            exec(code, namespace)  # noqa: S102
        output = f.getvalue()
        return output[:5000] if output else "(no output)"
    except TimeoutError:
        return "Error: TimeoutError: Code execution timed out"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    finally:
        if use_signal:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def extract_code_block(response: str) -> str | None:
    """Extract code between <execute> tags."""
    match = re.search(r"<execute>\s*\n?(.*?)</execute>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def has_final_answer(response: str) -> bool:
    """Check if the response contains a final answer."""
    return bool(re.search(r"Answer:\s*\(?[A-Ea-e]\)?", response, re.IGNORECASE))


def build_agentic_turn_prompt(
    question: BenchQuestion,
    choices: list[str],
    history: list[dict[str, str]],
    turn: int,
    hint: str = "",
    strategy_injection: str = "",
) -> str:
    """Build the prompt for a single turn in the agentic loop."""
    is_seqqa = question.bench_subtask == "seqqa"
    parts: list[str] = []

    if turn == 0:
        # Inject learned strategy at the start of the first turn
        if strategy_injection:
            parts.append(strategy_injection)

        # First turn: present the full question
        if question.context:
            parts.append(f"Context:\n{question.context}")
        parts.append(f"Question: {question.question}")
        parts.append("Choices:\n" + "\n".join(f"  {c}" for c in choices))

        subtask_hint = _get_subtask_hint(question)
        parts.append(f"\nDatabase hint: {subtask_hint}")

        # SeqQA: inject pre-extracted sequences
        if is_seqqa:
            seq_context = _build_seqqa_sequence_context(question.question)
            if seq_context:
                parts.append(f"\n{seq_context}")

        if hint:
            parts.append(f"\nAdditional guidance: {hint}")

        if is_seqqa:
            parts.append(
                "\nYou MUST write Python code using BioPython to solve this. "
                "Wrap your code in <execute> tags. Do NOT attempt to compute "
                "sequence operations in your head — use code for ALL of them. "
                "After seeing the code output, compare it against the answer "
                "choices and state your final answer as: Answer: X"
            )
        else:
            parts.append(
                "\nThink step-by-step about the answer. If you are confident from your "
                "knowledge alone, just give the answer immediately — no need to write code. "
                "Only use <execute> tags for code if you are genuinely uncertain and a FREE "
                "public API can help verify. State your final answer as: Answer: X"
            )
    else:
        # Subsequent turns: show history and ask to continue
        parts.append("Here is the conversation so far:\n")
        for entry in history:
            if entry["role"] == "code":
                parts.append(f"[Code executed]:\n```python\n{entry['code']}\n```")
                parts.append(f"[Output]:\n{entry['output']}")
            elif entry["role"] == "think":
                parts.append(f"[Your reasoning]:\n{entry['content']}")
            elif entry["role"] == "assistant":
                parts.append(f"[Your previous response]:\n{entry['content']}")

        if is_seqqa:
            parts.append(
                "\nContinue your analysis. Use the code output to determine "
                "the correct answer by matching against the choices. "
                "If code had an error, fix and re-run it. "
                "You MUST give a final answer now. State: Answer: X"
            )
        else:
            parts.append(
                "\nContinue your analysis. If the code output helps, use it to determine "
                "the answer. If code failed or returned unhelpful results, answer from your "
                "knowledge — do NOT keep retrying failed API calls. You MUST give a final "
                "answer now. State: Answer: X"
            )

    return "\n\n".join(parts)


# Category-specific retry hints for multi-trial mode
RETRY_HINTS: dict[str, str] = {
    "dga_task": "Try querying a different database (OMIM, DisGeNET, OpenTargets)",
    "gene_location_task": "Verify with Ensembl or NCBI Gene REST API",
    "mirna_targets_task": "Check miRTarBase or TargetScan databases",
    "mouse_tumor_gene_sets": "Look at MSigDB mouse gene sets or MGI",
    "oncogenic_signatures_task": "Check MSigDB oncogenic signatures collection (C6)",
    "tfbs_GTRD_task": "Query the GTRD database for ChIP-seq binding site data",
    "variant_from_sequence_task": "Look at ClinVar or gnomAD for variant data",
    "variant_multi_sequence_task": "Use sequence alignment and variant databases",
    "vax_response_task": "Check immunology databases and ImmPort",
    "viral_ppi_task": "Query IntAct or VirHostNet for viral PPIs",
}

# Retry hints keyed by bench_subtask
BENCH_SUBTASK_RETRY_HINTS: dict[str, str] = {
    "seqqa": (
        "Re-check your sequence operations. Try using BioPython code to verify "
        "reverse complements, translations, or restriction sites."
    ),
    "litqa2": (
        "Try searching with different keywords or a different database "
        "(PubMed vs Semantic Scholar). Look at the abstract more carefully."
    ),
}


def _get_retry_hint(question: BenchQuestion) -> str:
    """Get a retry-specific hint for the subtask."""
    for key, hint in RETRY_HINTS.items():
        if key in question.subtask:
            return hint
    # Fall back to bench_subtask-level hint
    if question.bench_subtask in BENCH_SUBTASK_RETRY_HINTS:
        return BENCH_SUBTASK_RETRY_HINTS[question.bench_subtask]
    return "Try a different analytical approach or database query."


def _seqqa_verify_answer(
    code: str, stated_answer: str, choices: list[str], predicted_letter: str,
) -> tuple[bool, str]:
    """Re-execute the agent's final code block and verify output matches answer.

    Returns (verified, detail_message).
    """
    try:
        output = execute_code_safely(code, timeout=15, seqqa_mode=True)
        if output.startswith("Error:"):
            return False, f"Verification code error: {output}"
        # Check if the stated answer text appears somewhere in the code output
        choice_text = ""
        if predicted_letter:
            idx = ord(predicted_letter) - ord("A")
            if 0 <= idx < len(choices):
                # Strip the "(X) " prefix to get raw answer text
                choice_text = re.sub(r"^\([A-E]\)\s*", "", choices[idx])
        if choice_text and choice_text.strip() in output:
            return True, f"Verified: answer text found in code output"
        # Also check if any significant part of the output matches the choice
        # (for numeric answers, single-word answers, etc.)
        output_stripped = output.strip()
        if output_stripped and choice_text:
            # Check if key parts of the choice appear in output
            # For short answers (< 50 chars), check direct containment
            if len(choice_text) < 50 and choice_text.strip().lower() in output.lower():
                return True, f"Verified: answer found in output (case-insensitive)"
        return False, f"Unverified: output={output_stripped[:200]}, choice={choice_text[:100]}"
    except Exception as e:
        return False, f"Verification exception: {e}"


async def evaluate_question_agentic(
    question: BenchQuestion,
    llm: Any,
    *,
    model: str = "",
    max_turns: int = 8,
    timeout_seconds: int = 300,
    trial_hint: str = "",
    temperature: float | None = None,
    strategy_injection: str = "",
) -> tuple[ScoredResult, str]:
    """Evaluate using multi-turn agentic reasoning with code execution.

    For SeqQA questions, this enforces code execution before accepting an
    answer, pre-imports BioPython modules, and runs a verification step.

    Returns (ScoredResult, reasoning_summary) where reasoning_summary can be
    used as context for retry hints in multi-trial mode.
    """
    start = time.monotonic()
    choices, correct_letter, refuse_letter = build_choices(question)
    system_prompt = _get_agentic_system_prompt(question.bench_subtask)

    is_seqqa = question.bench_subtask == "seqqa"
    is_verbose = logger.isEnabledFor(logging.DEBUG)

    history: list[dict[str, str]] = []
    total_tokens = 0
    full_reasoning: list[str] = []
    code_executed = False  # Track whether any code was run (for SeqQA enforcement)
    last_code_block: str | None = None  # For verification

    # Build extra kwargs for temperature if provided
    extra_kwargs: dict[str, Any] = {}
    if temperature is not None:
        extra_kwargs["temperature"] = temperature

    def _make_result(
        predicted: str, reasoning: str, error: str | None = None,
    ) -> ScoredResult:
        return ScoredResult(
            question_id=question.id,
            subtask=question.subtask,
            predicted=predicted,
            predicted_text=_get_choice_text(choices, predicted),
            correct_answer=correct_letter,
            correct_text=question.correct_answer,
            is_correct=predicted == correct_letter,
            is_refused=predicted == refuse_letter,
            reasoning=reasoning,
            tokens_used=total_tokens,
            latency_ms=int((time.monotonic() - start) * 1000),
            error=error,
            bench_subtask=question.bench_subtask,
        )

    try:
        for turn in range(max_turns):
            # Check total timeout
            elapsed = time.monotonic() - start
            if elapsed > timeout_seconds:
                logger.warning("Agentic evaluation timed out at turn %d", turn)
                break

            prompt = build_agentic_turn_prompt(
                question, choices, history, turn, hint=trial_hint,
                strategy_injection=strategy_injection,
            )

            remaining_time = timeout_seconds - elapsed
            resp = await asyncio.wait_for(
                llm.query(
                    prompt,
                    system_prompt=system_prompt,
                    max_tokens=4096,
                    model=model or None,
                    **extra_kwargs,
                ),
                timeout=max(remaining_time, 10),
            )

            response_text = resp.text
            total_tokens += resp.call_tokens
            full_reasoning.append(f"[Turn {turn + 1}]: {response_text}")

            # Check for code execution
            code = extract_code_block(response_text)
            if code:
                if is_verbose:
                    logger.debug(
                        "Turn %d: executing code (%d chars):\n%s",
                        turn, len(code), code,
                    )
                else:
                    logger.debug("Turn %d: executing code (%d chars)", turn, len(code))
                output = execute_code_safely(
                    code, timeout=30, seqqa_mode=is_seqqa,
                )
                history.append({"role": "code", "code": code, "output": output})
                full_reasoning.append(f"[Code output]: {output}")
                code_executed = True
                last_code_block = code

                # If response also has a final answer after code, extract it
                if has_final_answer(response_text):
                    predicted = extract_answer_letter(response_text)
                    reasoning = "\n".join(full_reasoning)

                    # SeqQA verification step
                    if is_seqqa and last_code_block:
                        verified, detail = _seqqa_verify_answer(
                            last_code_block, predicted, choices, predicted,
                        )
                        full_reasoning.append(f"[Verification]: {detail}")
                        reasoning = "\n".join(full_reasoning)
                        if is_verbose:
                            logger.debug("SeqQA verification: %s", detail)

                    result = _make_result(predicted, reasoning)
                    return result, reasoning

            elif has_final_answer(response_text):
                predicted = extract_answer_letter(response_text)

                # SeqQA enforcement: if no code was executed, redirect
                if is_seqqa and not code_executed and turn < max_turns - 1:
                    logger.debug(
                        "SeqQA: answer without code at turn %d — redirecting",
                        turn,
                    )
                    history.append({"role": "think", "content": response_text})
                    history.append({
                        "role": "think",
                        "content": (
                            "STOP. You gave an answer without running any code. "
                            "For sequence questions you MUST use BioPython code "
                            "to compute the answer. Write a <execute> block now."
                        ),
                    })
                    full_reasoning.append(
                        "[SeqQA redirect]: forced code execution — "
                        "answer without code not accepted"
                    )
                    continue

                reasoning = "\n".join(full_reasoning)

                # SeqQA verification step
                if is_seqqa and last_code_block:
                    verified, detail = _seqqa_verify_answer(
                        last_code_block, predicted, choices, predicted,
                    )
                    full_reasoning.append(f"[Verification]: {detail}")
                    reasoning = "\n".join(full_reasoning)
                    if is_verbose:
                        logger.debug("SeqQA verification: %s", detail)

                result = _make_result(predicted, reasoning)
                return result, reasoning

            else:
                # Thinking step — no code, no final answer
                history.append({"role": "think", "content": response_text})

        # Exhausted turns without a final answer — try to extract from last response
        all_text = "\n".join(full_reasoning)
        predicted = extract_answer_letter(all_text)
        reasoning = all_text

        result = _make_result(
            predicted, reasoning,
            error="max_turns_exhausted" if not predicted else None,
        )
        return result, reasoning

    except TimeoutError:
        reasoning = "\n".join(full_reasoning) if full_reasoning else ""
        return _make_result("", reasoning, error="Timeout"), reasoning

    except Exception as exc:
        reasoning = "\n".join(full_reasoning) if full_reasoning else ""
        return _make_result("", reasoning, error=traceback.format_exc()), reasoning


async def verify_answer(
    question: BenchQuestion,
    llm: Any,
    original_result: ScoredResult,
    choices: list[str],
    correct_letter: str,
    refuse_letter: str,
    *,
    model: str = "",
    timeout_seconds: int = 300,
) -> ScoredResult:
    """Run a second-pass verification of the original answer.

    Asks the LLM to verify from a different angle. If the verification
    disagrees, the verification answer is used instead.
    """
    start = time.monotonic()

    original_letter = original_result.predicted
    original_text = original_result.predicted_text

    verify_prompt = (
        "You previously answered a multiple-choice biology question.\n\n"
        f"Question: {question.question}\n\n"
        "Choices:\n" + "\n".join(f"  {c}" for c in choices) + "\n\n"
        f"Your previous answer was: ({original_letter}) {original_text}\n\n"
        "VERIFY this is correct by checking from a different angle. "
        "Consider alternative interpretations, edge cases, and whether "
        "a different choice might actually be correct.\n\n"
        "If your original answer is correct, confirm it. If you find "
        "it is wrong, provide the corrected answer.\n\n"
        "State your final answer on the LAST line as: Answer: X"
    )

    system_prompt = _get_agentic_system_prompt(question.bench_subtask)

    try:
        resp = await asyncio.wait_for(
            llm.query(
                verify_prompt,
                system_prompt=system_prompt,
                max_tokens=2048,
                model=model or None,
            ),
            timeout=timeout_seconds,
        )

        verify_text = resp.text
        verify_tokens = resp.call_tokens
        verified_letter = extract_answer_letter(verify_text)

        if verified_letter and verified_letter != original_letter:
            logger.info(
                "  Verification CHANGED answer: %s -> %s",
                original_letter, verified_letter,
            )
            is_correct = verified_letter == correct_letter
            is_refused = verified_letter == refuse_letter
            return ScoredResult(
                question_id=question.id,
                subtask=question.subtask,
                predicted=verified_letter,
                predicted_text=_get_choice_text(choices, verified_letter),
                correct_answer=correct_letter,
                correct_text=question.correct_answer,
                is_correct=is_correct,
                is_refused=is_refused,
                reasoning=(
                    original_result.reasoning
                    + f"\n\n[VERIFICATION]: Changed {original_letter} -> {verified_letter}\n"
                    + verify_text
                ),
                tokens_used=original_result.tokens_used + verify_tokens,
                latency_ms=original_result.latency_ms + int((time.monotonic() - start) * 1000),
                error=None,
                bench_subtask=question.bench_subtask,
            )
        else:
            logger.info("  Verification CONFIRMED answer: %s", original_letter)
            return ScoredResult(
                question_id=original_result.question_id,
                subtask=original_result.subtask,
                predicted=original_result.predicted,
                predicted_text=original_result.predicted_text,
                correct_answer=original_result.correct_answer,
                correct_text=original_result.correct_text,
                is_correct=original_result.is_correct,
                is_refused=original_result.is_refused,
                reasoning=original_result.reasoning + "\n\n[VERIFICATION]: Confirmed",
                tokens_used=original_result.tokens_used + verify_tokens,
                latency_ms=original_result.latency_ms + int((time.monotonic() - start) * 1000),
                error=None,
                bench_subtask=original_result.bench_subtask,
            )

    except Exception as exc:
        logger.warning("Verification failed: %s \u2014 keeping original answer", exc)
        return original_result

async def evaluate_question_multitrial(
    question: BenchQuestion,
    llm: Any,
    *,
    model: str = "",
    max_trials: int = 3,
    agentic: bool = True,
    max_turns: int = 8,
    timeout_seconds: int = 300,
    strategy_injection: str = "",
) -> tuple[ScoredResult, list[dict[str, Any]]]:
    """Evaluate with multiple trials, injecting hints from previous attempts.

    Returns (best_result, trial_details).
    """
    trial_details: list[dict[str, Any]] = []
    best_result: ScoredResult | None = None
    prev_reasoning = ""
    prev_predicted = ""

    for trial in range(max_trials):
        trial_hint = ""
        if trial > 0 and prev_predicted:
            # Build hint from previous trial
            category_hint = _get_retry_hint(question)
            trial_hint = (
                f"On your previous attempt, you chose ({prev_predicted}) but "
                f"the correct answer was different. Your reasoning was:\n"
                f"{prev_reasoning[:500]}\n\n"
                f"Consider alternative approaches, especially: {category_hint}"
            )

        if agentic:
            result, reasoning = await evaluate_question_agentic(
                question,
                llm,
                model=model,
                max_turns=max_turns,
                timeout_seconds=timeout_seconds,
                trial_hint=trial_hint,
                strategy_injection=strategy_injection,
            )
        else:
            # Zero-shot with hint injection for trial > 0
            result = await evaluate_question_live(
                question, llm, model=model, timeout_seconds=timeout_seconds,
                trial_hint=trial_hint,
                strategy_injection=strategy_injection,
            )
            reasoning = result.reasoning

        trial_details.append({
            "trial": trial + 1,
            "predicted": result.predicted,
            "is_correct": result.is_correct,
            "tokens_used": result.tokens_used,
            "latency_ms": result.latency_ms,
            "error": result.error,
        })

        # Early exit: correct answer found
        if result.is_correct:
            logger.info(
                "  Trial %d/%d: CORRECT (early exit)", trial + 1, max_trials
            )
            return result, trial_details

        logger.info(
            "  Trial %d/%d: WRONG (predicted %s, correct %s)",
            trial + 1, max_trials, result.predicted, result.correct_answer,
        )

        prev_predicted = result.predicted
        prev_reasoning = reasoning

        # Keep the best result (prefer one with an answer over no answer)
        if best_result is None or (not best_result.predicted and result.predicted):
            best_result = result
        elif result.predicted and not best_result.is_correct:
            # Use the latest trial as best (it has more information)
            best_result = result

    # Aggregate tokens and latency across trials
    total_tokens = sum(t["tokens_used"] for t in trial_details)
    total_latency = sum(t["latency_ms"] for t in trial_details)
    assert best_result is not None

    # Return with aggregated stats
    aggregated = ScoredResult(
        question_id=best_result.question_id,
        subtask=best_result.subtask,
        predicted=best_result.predicted,
        predicted_text=best_result.predicted_text,
        correct_answer=best_result.correct_answer,
        correct_text=best_result.correct_text,
        is_correct=best_result.is_correct,
        is_refused=best_result.is_refused,
        reasoning=best_result.reasoning,
        tokens_used=total_tokens,
        latency_ms=total_latency,
        error=best_result.error,
        bench_subtask=best_result.bench_subtask,
    )
    return aggregated, trial_details


# ---------------------------------------------------------------------------
# Majority voting (--replicas N)
# ---------------------------------------------------------------------------


async def evaluate_with_voting(
    question: BenchQuestion,
    llm: Any,
    *,
    n_replicas: int = 3,
    model: str = "",
    mode: str = "zero-shot",
    max_turns: int = 8,
    timeout_seconds: int = 300,
    temperature: float = 0.3,
) -> ScoredResult:
    """Run N replicas per question and take the majority-voted answer.

    Each replica uses temperature > 0 for diversity. The final answer is
    the most common predicted letter across replicas.
    """
    start = time.monotonic()
    choices, correct_letter, refuse_letter = build_choices(question)

    answers: list[str] = []
    total_tokens = 0
    all_reasoning: list[str] = []

    for i in range(n_replicas):
        if mode == "agentic":
            result, reasoning = await evaluate_question_agentic(
                question,
                llm,
                model=model,
                max_turns=max_turns,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
            )
        else:
            result = await evaluate_question_live(
                question,
                llm,
                model=model,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
            )
            reasoning = result.reasoning

        if result.predicted:
            answers.append(result.predicted)
        total_tokens += result.tokens_used
        all_reasoning.append(f"[Replica {i + 1}]: predicted={result.predicted}")

    # Majority vote
    if answers:
        majority_answer = Counter(answers).most_common(1)[0][0]
    else:
        majority_answer = ""

    vote_summary = dict(Counter(answers))
    all_reasoning.append(f"[Vote tally]: {vote_summary} -> majority={majority_answer}")

    is_correct = majority_answer == correct_letter
    is_refused = majority_answer == refuse_letter

    return ScoredResult(
        question_id=question.id,
        subtask=question.subtask,
        predicted=majority_answer,
        predicted_text=_get_choice_text(choices, majority_answer),
        correct_answer=correct_letter,
        correct_text=question.correct_answer,
        is_correct=is_correct,
        is_refused=is_refused,
        reasoning="\n".join(all_reasoning),
        tokens_used=total_tokens,
        latency_ms=int((time.monotonic() - start) * 1000),
        error=None,
        bench_subtask=question.bench_subtask,
    )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


async def evaluate_question_live(
    question: BenchQuestion,
    llm: Any,
    *,
    model: str = "",
    timeout_seconds: int = 300,
    trial_hint: str = "",
    temperature: float | None = None,
    strategy_injection: str = "",
) -> ScoredResult:
    """Evaluate a single question using the LLM (zero-shot)."""
    start = time.monotonic()
    choices, correct_letter, refuse_letter = build_choices(question)
    prompt = build_prompt(question, choices, strategy_injection=strategy_injection)
    if trial_hint:
        prompt += f"\n\nAdditional guidance from a previous attempt:\n{trial_hint}"

    extra_kwargs: dict[str, Any] = {}
    if temperature is not None:
        extra_kwargs["temperature"] = temperature

    try:
        resp = await asyncio.wait_for(
            llm.query(
                prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=2048,
                model=model or None,
                **extra_kwargs,
            ),
            timeout=timeout_seconds,
        )
        response_text = resp.text
        tokens = resp.call_tokens

        predicted = extract_answer_letter(response_text)
        is_correct = predicted == correct_letter
        is_refused = predicted == refuse_letter

        return ScoredResult(
            question_id=question.id,
            subtask=question.subtask,
            predicted=predicted,
            predicted_text=_get_choice_text(choices, predicted),
            correct_answer=correct_letter,
            correct_text=question.correct_answer,
            is_correct=is_correct,
            is_refused=is_refused,
            reasoning=response_text,
            tokens_used=tokens,
            latency_ms=int((time.monotonic() - start) * 1000),
            error=None,
            bench_subtask=question.bench_subtask,
        )

    except TimeoutError:
        return ScoredResult(
            question_id=question.id,
            subtask=question.subtask,
            predicted="",
            predicted_text="",
            correct_answer=correct_letter,
            correct_text=question.correct_answer,
            is_correct=False,
            is_refused=False,
            reasoning="",
            tokens_used=0,
            latency_ms=int((time.monotonic() - start) * 1000),
            error="Timeout",
            bench_subtask=question.bench_subtask,
        )
    except Exception as exc:
        return ScoredResult(
            question_id=question.id,
            subtask=question.subtask,
            predicted="",
            predicted_text="",
            correct_answer=correct_letter,
            correct_text=question.correct_answer,
            is_correct=False,
            is_refused=False,
            reasoning="",
            tokens_used=0,
            latency_ms=int((time.monotonic() - start) * 1000),
            error=traceback.format_exc(),
            bench_subtask=question.bench_subtask,
        )


def evaluate_question_dry(question: BenchQuestion) -> ScoredResult:
    """Dry-run evaluation — no LLM, random answer for pipeline testing."""
    start = time.monotonic()
    choices, correct_letter, refuse_letter = build_choices(question)
    rng = random.Random(hash(question.id) + 42)
    letters = [chr(65 + i) for i in range(len(choices))]
    predicted = rng.choice(letters)

    return ScoredResult(
        question_id=question.id,
        subtask=question.subtask,
        predicted=predicted,
        predicted_text=_get_choice_text(choices, predicted),
        correct_answer=correct_letter,
        correct_text=question.correct_answer,
        is_correct=predicted == correct_letter,
        is_refused=predicted == refuse_letter,
        reasoning="[dry-run: random choice]",
        tokens_used=0,
        latency_ms=int((time.monotonic() - start) * 1000),
        error=None,
        bench_subtask=question.bench_subtask,
    )


def _get_choice_text(choices: list[str], letter: str) -> str:
    """Get the full text of a choice by its letter."""
    if not letter:
        return ""
    idx = ord(letter) - ord("A")
    if 0 <= idx < len(choices):
        return choices[idx]
    return ""


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(results: list[ScoredResult]) -> dict[str, Any]:
    """Compute accuracy, precision, coverage, and per-subtask breakdown."""
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "precision": 0.0, "coverage": 0.0, "total": 0}

    correct = sum(1 for r in results if r.is_correct)
    refused = sum(1 for r in results if r.is_refused)
    confident = total - refused  # answered (not refused)
    errors = sum(1 for r in results if r.error)

    accuracy = correct / total
    precision = correct / confident if confident > 0 else 0.0
    coverage = confident / total

    total_tokens = sum(r.tokens_used for r in results)
    total_latency = sum(r.latency_ms for r in results)

    # Per-subtask breakdown (narrow subtask field)
    subtask_results: dict[str, list[ScoredResult]] = {}
    for r in results:
        subtask_results.setdefault(r.subtask, []).append(r)

    per_subtask = {}
    for subtask, sub_results in sorted(subtask_results.items()):
        sub_total = len(sub_results)
        sub_correct = sum(1 for r in sub_results if r.is_correct)
        sub_refused = sum(1 for r in sub_results if r.is_refused)
        sub_confident = sub_total - sub_refused
        per_subtask[subtask] = {
            "total": sub_total,
            "correct": sub_correct,
            "accuracy": sub_correct / sub_total if sub_total > 0 else 0.0,
            "precision": sub_correct / sub_confident if sub_confident > 0 else 0.0,
            "coverage": sub_confident / sub_total if sub_total > 0 else 0.0,
        }

    # Per bench_subtask breakdown (top-level: dbqa, seqqa, litqa2)
    bench_subtask_results: dict[str, list[ScoredResult]] = {}
    for r in results:
        bench_subtask_results.setdefault(r.bench_subtask, []).append(r)

    per_bench_subtask = {}
    for bs, sub_results in sorted(bench_subtask_results.items()):
        sub_total = len(sub_results)
        sub_correct = sum(1 for r in sub_results if r.is_correct)
        sub_refused = sum(1 for r in sub_results if r.is_refused)
        sub_confident = sub_total - sub_refused
        per_bench_subtask[bs] = {
            "total": sub_total,
            "correct": sub_correct,
            "accuracy": sub_correct / sub_total if sub_total > 0 else 0.0,
            "precision": sub_correct / sub_confident if sub_confident > 0 else 0.0,
            "coverage": sub_confident / sub_total if sub_total > 0 else 0.0,
        }

    return {
        "accuracy": accuracy,
        "precision": precision,
        "coverage": coverage,
        "total": total,
        "correct": correct,
        "refused": refused,
        "errors": errors,
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens / total if total > 0 else 0,
        "total_latency_ms": total_latency,
        "avg_latency_ms": total_latency / total if total > 0 else 0,
        "per_subtask": per_subtask,
        "per_bench_subtask": per_bench_subtask,
    }


# ---------------------------------------------------------------------------
# Checkpoint / crash recovery
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Saves intermediate results for crash recovery."""

    def __init__(self, output_dir: Path, run_id: str) -> None:
        self.output_dir = output_dir
        self.run_id = run_id
        self.checkpoint_path = output_dir / f"{run_id}_checkpoint.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_result(
        self,
        result: ScoredResult,
        trial_details: list[dict[str, Any]] | None = None,
    ) -> None:
        """Append a single result to the checkpoint file."""
        record: dict[str, Any] = {
            "question_id": result.question_id,
            "subtask": result.subtask,
            "bench_subtask": result.bench_subtask,
            "predicted": result.predicted,
            "predicted_text": result.predicted_text,
            "correct_answer": result.correct_answer,
            "correct_text": result.correct_text,
            "is_correct": result.is_correct,
            "is_refused": result.is_refused,
            "tokens_used": result.tokens_used,
            "latency_ms": result.latency_ms,
            "error": result.error,
        }
        if trial_details:
            record["trial_details"] = trial_details
        with open(self.checkpoint_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def load_completed(self) -> tuple[set[str], list[ScoredResult]]:
        """Load previously completed question IDs and results from checkpoint."""
        completed: set[str] = set()
        results: list[ScoredResult] = []
        if not self.checkpoint_path.exists():
            return completed, results

        for line in self.checkpoint_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            qid = record["question_id"]
            completed.add(qid)
            results.append(
                ScoredResult(
                    question_id=qid,
                    subtask=record.get("subtask", ""),
                    predicted=record.get("predicted", ""),
                    predicted_text=record.get("predicted_text", ""),
                    correct_answer=record.get("correct_answer", ""),
                    correct_text=record.get("correct_text", ""),
                    is_correct=record.get("is_correct", False),
                    is_refused=record.get("is_refused", False),
                    reasoning="",  # not saved in checkpoint
                    tokens_used=record.get("tokens_used", 0),
                    latency_ms=record.get("latency_ms", 0),
                    error=record.get("error"),
                    bench_subtask=record.get("bench_subtask", "dbqa"),
                )
            )
        logger.info("Loaded %d completed results from checkpoint", len(completed))
        return completed, results


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------


async def run_evaluation(
    *,
    limit: int | None = None,
    mode: str = "zero-shot",
    model_tier: str = "sonnet",
    dry_run: bool = False,
    resume: bool = False,
    timeout_seconds: int = 300,
    concurrency: int = 1,
    trials: int = 1,
    max_turns: int = 8,
    subtask: str = "dbqa",
    replicas: int = 1,
    verify: bool = False,
) -> dict[str, Any]:
    """Run the full LAB-Bench evaluation pipeline.

    Args:
        limit: Max questions to evaluate per subtask (None = all).
        mode: Evaluation mode — "zero-shot" or "agentic".
        model_tier: Model to use — "sonnet", "opus", "haiku".
        dry_run: Skip LLM calls, random answers.
        resume: Resume from last checkpoint.
        timeout_seconds: Per-question timeout.
        concurrency: Max concurrent LLM calls.
        trials: Number of trials per question (1-5). Multi-trial injects hints.
        max_turns: Max turns for agentic mode (default 8).
        subtask: Which subtask(s) to evaluate — "dbqa", "seqqa", "litqa2", "all".
        replicas: Number of replicas for majority voting (1 = disabled).
        verify: If True, run a second-pass verification on each answer.

    Returns:
        Full results dict with metrics and per-question results.
    """
    # 1. Determine which subtasks to run
    if subtask == "all":
        subtask_keys = ["dbqa", "seqqa", "litqa2"]
    else:
        subtask_keys = [subtask]

    # 2. Download / locate dataset and load questions
    questions = load_all_subtask_questions(subtask_keys, limit=limit)
    logger.info(
        "Loaded %d total questions across subtask(s): %s",
        len(questions),
        ", ".join(k.upper() for k in subtask_keys),
    )

    # 3. Set up checkpoint
    subtask_label = subtask if subtask != "all" else "all"
    run_id = f"{subtask_label}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = CheckpointManager(RESULTS_DIR, run_id)

    completed_ids: set[str] = set()
    results: list[ScoredResult] = []

    if resume:
        # Find the latest checkpoint
        checkpoints = sorted(RESULTS_DIR.glob("*_checkpoint.jsonl"), reverse=True)
        if checkpoints:
            checkpoint.checkpoint_path = checkpoints[0]
            checkpoint.run_id = checkpoints[0].stem.replace("_checkpoint", "")
            run_id = checkpoint.run_id
            completed_ids, results = checkpoint.load_completed()

    # 4. Set up LLM
    llm = None
    model_name = ""
    if not dry_run:
        from core.llm import LLMClient

        llm = LLMClient()
        # Resolve model tier to explicit model ID
        model_name = MODEL_ID_MAP.get(model_tier, model_tier)
        logger.info(
            "Using model: %s (mode: %s, trials: %d, replicas: %d, subtask: %s)",
            model_name, mode, trials, replicas,
            ", ".join(k.upper() for k in subtask_keys),
        )

    # 4b. Set up strategy tracker for within-run learning
    strategy_persist_path = RESULTS_DIR / f"{run_id}_strategy.json"
    strategy_tracker = BenchmarkStrategyTracker(
        persist_path=strategy_persist_path,
        rebuild_interval=10,
    )
    # Seed tracker with outcomes from resumed checkpoint results
    for prev_r in results:
        strategy_tracker.record(QuestionOutcome(
            subtask=prev_r.subtask or prev_r.bench_subtask,
            question_type=prev_r.subtask or "unknown",
            predicted=prev_r.predicted,
            correct=prev_r.correct_answer,
            is_correct=prev_r.is_correct,
            reasoning_summary="(resumed from checkpoint)",
            code_executed=False,
        ))

    # 5. Evaluate
    total = len(questions)
    remaining = [q for q in questions if q.id not in completed_ids]
    correct_so_far = sum(1 for r in results if r.is_correct)

    if remaining:
        logger.info(
            "Evaluating %d questions (%d already completed, %d remaining)",
            total, total - len(remaining), len(remaining),
        )
    else:
        logger.info("All %d questions already completed", total)

    start_time = time.monotonic()
    sem = asyncio.Semaphore(concurrency)

    use_yohas = mode == "yohas"
    use_agentic = mode == "agentic"
    use_codeact = mode == "codeact"
    use_multitrial = trials > 1
    use_voting = replicas > 1

    # Lazy-import YOHAS evaluator only when needed
    if use_yohas:
        from yohas_bench_eval import evaluate_mcq_with_yohas

    # Lazy-import CodeAct evaluator only when needed
    if use_codeact:
        from codeact_loop import codeact_evaluate

    for idx, question in enumerate(remaining):
        question_num = len(results) + 1

        # Print which subtask is being evaluated
        if idx == 0 or (idx > 0 and remaining[idx].bench_subtask != remaining[idx - 1].bench_subtask):
            logger.info(
                "--- Evaluating subtask: %s ---", question.bench_subtask.upper()
            )

        async with sem:
            trial_details: list[dict[str, Any]] | None = None

            # Get strategy injection for this question
            current_strategy = strategy_tracker.get_strategy_injection()

            # Check if strategy tracker recommends escalation/de-escalation
            q_subtask = question.subtask or question.bench_subtask
            effective_agentic = use_agentic
            if not dry_run and strategy_tracker.get_outcome_count() >= 10:
                if strategy_tracker.should_escalate(q_subtask) and not use_agentic:
                    logger.info(
                        "Strategy escalation: switching to agentic for subtask %s (acc < 30%%)",
                        q_subtask,
                    )
                    effective_agentic = True
                elif strategy_tracker.should_use_zero_shot(q_subtask) and use_agentic:
                    logger.info(
                        "Strategy de-escalation: using zero-shot for subtask %s (acc > 60%%)",
                        q_subtask,
                    )
                    effective_agentic = False

            if dry_run:
                result = evaluate_question_dry(question)
            elif use_codeact:
                # CodeAct plan-code-execute-observe loop
                choices_ca, correct_letter_ca, refuse_letter_ca = build_choices(question)
                choices_block = "\n".join(choices_ca)
                formatted_question = (
                    f"{question.question}\n\n"
                    f"Choose the best answer:\n{choices_block}\n\n"
                    f"After your analysis, provide your final answer letter "
                    f"(A, B, C, ...) in <answer> tags like: <answer>B</answer>"
                )
                ca_system = _get_agentic_system_prompt(question.bench_subtask)
                if current_strategy:
                    ca_system += current_strategy

                ca_result = await codeact_evaluate(
                    question=formatted_question,
                    context=question.context or "",
                    llm=llm,
                    model=model_name,
                    execute_fn=lambda code: execute_code_safely(
                        code, timeout=30,
                        seqqa_mode=(question.bench_subtask == "seqqa"),
                    ),
                    max_steps=max_turns,
                    timeout_seconds=timeout_seconds,
                    system_prompt=ca_system,
                    helper_functions_doc=DB_HELPERS_PROMPT_BLOCK,
                )

                # Extract letter from the CodeAct answer
                predicted_letter = extract_answer_letter(ca_result.answer)
                if not predicted_letter:
                    # Try to find a letter directly in the answer string
                    letter_match = re.search(r"\b([A-H])\b", ca_result.answer)
                    predicted_letter = letter_match.group(1) if letter_match else ""

                result = ScoredResult(
                    question_id=question.id,
                    subtask=question.subtask,
                    predicted=predicted_letter,
                    predicted_text=_get_choice_text(choices_ca, predicted_letter),
                    correct_answer=correct_letter_ca,
                    correct_text=question.correct_answer,
                    is_correct=predicted_letter == correct_letter_ca,
                    is_refused=predicted_letter == refuse_letter_ca,
                    reasoning=f"CodeAct loop: {ca_result.code_executions} code executions, "
                              f"{ca_result.code_errors} errors, {len(ca_result.steps)} steps",
                    tokens_used=ca_result.total_tokens,
                    latency_ms=ca_result.total_duration_ms,
                    error=None if predicted_letter else "no_answer",
                    bench_subtask=question.bench_subtask,
                )
            elif use_yohas:
                # Full YOHAS 4-phase pipeline (hypothesize -> investigate -> synthesize -> falsify)
                choices_list, correct_letter_y, refuse_letter_y = build_choices(question)
                yohas_result = await evaluate_mcq_with_yohas(
                    question_text=question.question,
                    correct_answer_text=question.correct_answer,
                    distractor_texts=list(question.distractors),
                    correct_letter=correct_letter_y,
                    choices_formatted=choices_list,
                    llm=llm,
                    model=model_name or None,
                    subtask=question.subtask or question.bench_subtask,
                    enable_falsification=True,
                    timeout_seconds=timeout_seconds,
                )
                # Convert YOHASBenchResult -> ScoredResult
                result = ScoredResult(
                    question_id=question.id,
                    subtask=question.subtask,
                    predicted=yohas_result.predicted,
                    predicted_text=_get_choice_text(choices_list, yohas_result.predicted),
                    correct_answer=correct_letter_y,
                    correct_text=question.correct_answer,
                    is_correct=yohas_result.predicted == correct_letter_y,
                    is_refused=yohas_result.predicted == refuse_letter_y,
                    reasoning=yohas_result.reasoning,
                    tokens_used=yohas_result.tokens_used,
                    latency_ms=yohas_result.duration_ms,
                    error=None if yohas_result.predicted else "no_answer",
                    bench_subtask=question.bench_subtask,
                )
            elif use_voting:
                # Majority voting mode
                result = await evaluate_with_voting(
                    question,
                    llm,
                    n_replicas=replicas,
                    model=model_name,
                    mode=mode,
                    max_turns=max_turns,
                    timeout_seconds=timeout_seconds,
                )
            elif use_multitrial:
                result, trial_details = await evaluate_question_multitrial(
                    question,
                    llm,
                    model=model_name,
                    max_trials=trials,
                    agentic=effective_agentic,
                    max_turns=max_turns,
                    timeout_seconds=timeout_seconds,
                    strategy_injection=current_strategy,
                )
            elif effective_agentic:
                result, _reasoning = await evaluate_question_agentic(
                    question,
                    llm,
                    model=model_name,
                    max_turns=max_turns,
                    timeout_seconds=timeout_seconds,
                    strategy_injection=current_strategy,
                )
            else:
                result = await evaluate_question_live(
                    question,
                    llm,
                    model=model_name,
                    timeout_seconds=timeout_seconds,
                    strategy_injection=current_strategy,
                )

        # Two-pass verification (--verify)
        if verify and not dry_run and result.predicted and not result.error:
            choices_v, correct_v, refuse_v = build_choices(question)
            result = await verify_answer(
                question,
                llm,
                result,
                choices_v,
                correct_v,
                refuse_v,
                model=model_name,
                timeout_seconds=timeout_seconds,
            )

        results.append(result)
        checkpoint.save_result(result, trial_details=trial_details)

        # Record outcome for strategy learning
        reasoning_text = getattr(result, "reasoning", "") or ""
        strategy_tracker.record(QuestionOutcome(
            subtask=result.subtask or result.bench_subtask,
            question_type=detect_question_type(question.question, result.subtask),
            predicted=result.predicted,
            correct=result.correct_answer,
            is_correct=result.is_correct,
            reasoning_summary=reasoning_text[:300],
            tools_used=[],
            databases_queried=detect_databases_from_text(reasoning_text),
            code_executed="[Code" in reasoning_text or "<execute>" in reasoning_text,
        ))

        if result.is_correct:
            correct_so_far += 1

        status = "CORRECT" if result.is_correct else "WRONG"
        if result.error:
            status = "ERROR"
        elif result.is_refused:
            status += " (refused)"

        trial_info = ""
        if trial_details:
            n_trials_used = len(trial_details)
            trial_info = f" [{n_trials_used} trial(s)]"
        if use_voting:
            trial_info = f" [{replicas} replicas]"

        running_acc = correct_so_far / question_num * 100

        # Compare against baselines (DbQA-specific, but shown for all)
        stella_delta = running_acc - 54.0
        biomni_delta = running_acc - 78.0
        baseline_str = (
            f"vs STELLA {stella_delta:+.1f}pp, vs Biomni {biomni_delta:+.1f}pp"
        )

        logger.info(
            "Question %d/%d [%s]: %s [%s] (acc: %.1f%% | %s)%s [%dms, %d tok]",
            question_num,
            total,
            question.bench_subtask.upper(),
            question.id[:12] + "...",
            status,
            running_acc,
            baseline_str,
            trial_info,
            result.latency_ms,
            result.tokens_used,
        )

    elapsed = time.monotonic() - start_time

    # 6. Compute final metrics
    metrics = compute_metrics(results)

    # 7. Build full results
    benchmark_label = (
        f"LAB-Bench {', '.join(k.upper() for k in subtask_keys)}"
    )
    full_results = {
        "benchmark": benchmark_label,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "mode": mode,
            "model_tier": model_tier,
            "model": model_name,
            "dry_run": dry_run,
            "limit": limit,
            "timeout_seconds": timeout_seconds,
            "total_questions": total,
            "trials": trials,
            "replicas": replicas,
            "subtask": subtask,
            "subtask_keys": subtask_keys,
            "max_turns": max_turns if mode == "agentic" else None,
            "verify": verify,
        },
        "metrics": metrics,
        "strategy_evolution": {
            "final_strategy": strategy_tracker.strategy_text,
            "subtask_accuracy": strategy_tracker.subtask_accuracy,
            "total_outcomes_tracked": strategy_tracker.get_outcome_count(),
        },
        "competitor_baselines": {
            "STELLA": 0.54,
            "Biomni_Lab": 0.78,
            "note": "STELLA 54% (LAB-Bench paper), Biomni Lab 78% (target) — DbQA only",
        },
        "delta_vs_stella": metrics["accuracy"] - 0.54,
        "delta_vs_biomni": metrics["accuracy"] - 0.78,
        "elapsed_seconds": round(elapsed, 1),
        "results": [
            {
                "question_id": r.question_id,
                "subtask": r.subtask,
                "bench_subtask": r.bench_subtask,
                "predicted": r.predicted,
                "predicted_text": r.predicted_text,
                "correct_answer": r.correct_answer,
                "correct_text": r.correct_text,
                "is_correct": r.is_correct,
                "is_refused": r.is_refused,
                "tokens_used": r.tokens_used,
                "latency_ms": r.latency_ms,
                "error": r.error,
            }
            for r in results
        ],
    }

    # 8. Save results
    results_path = RESULTS_DIR / "labbench_results.json"
    results_path.write_text(json.dumps(full_results, indent=2, default=str))
    logger.info("Results saved to %s", results_path)

    # Also save to the canonical path requested by the task spec
    canonical_path = PROJECT_ROOT / "data" / "benchmarks" / "labbench_results.json"
    canonical_path.write_text(json.dumps(full_results, indent=2, default=str))
    logger.info("Results also saved to %s", canonical_path)

    # Also save a timestamped copy
    timestamped_path = RESULTS_DIR / f"{run_id}_results.json"
    timestamped_path.write_text(json.dumps(full_results, indent=2, default=str))

    # 9. Print summary
    print("\n" + "=" * 70)
    print(f"LAB-Bench Results — {', '.join(k.upper() for k in subtask_keys)}")
    print("=" * 70)
    print(f"  Mode:          {mode}")
    print(f"  Model:         {model_name or 'dry-run'}")
    print(f"  Trials:        {trials}")
    print(f"  Replicas:      {replicas}")
    print(f"  Verify:        {verify}")
    if mode == "agentic":
        print(f"  Max turns:     {max_turns}")
    print(f"  Questions:     {metrics['total']}")
    print(f"  Correct:       {metrics['correct']}")
    print(f"  Accuracy:      {metrics['accuracy']:.1%}")
    print(f"  Precision:     {metrics['precision']:.1%}  (excl. refused)")
    print(f"  Coverage:      {metrics['coverage']:.1%}  (answered / total)")
    print(f"  Refused:       {metrics['refused']}")
    print(f"  Errors:        {metrics['errors']}")
    print(f"  Tokens:        {metrics['total_tokens']:,} total, {metrics['avg_tokens']:,.0f} avg")
    print(f"  Latency:       {metrics['avg_latency_ms']:,.0f}ms avg")
    print(f"  Elapsed:       {elapsed:.1f}s")
    print(f"  ---")
    print(f"  STELLA (54%):  {full_results['delta_vs_stella']:+.1%}")
    print(f"  Biomni (78%):  {full_results['delta_vs_biomni']:+.1%}")
    print()

    # Per bench_subtask breakdown
    if metrics.get("per_bench_subtask") and len(metrics["per_bench_subtask"]) > 1:
        print("Per-subtask accuracy (top-level):")
        print(f"  {'Subtask':<12s} {'N':>4s} {'Acc':>7s} {'Prec':>7s} {'Cov':>7s}")
        print(f"  {'-' * 12} {'-' * 4} {'-' * 7} {'-' * 7} {'-' * 7}")
        for bs, sm in sorted(metrics["per_bench_subtask"].items()):
            print(
                f"  {bs.upper():<12s} {sm['total']:>4d} "
                f"{sm['accuracy']:>6.1%} {sm['precision']:>6.1%} {sm['coverage']:>6.1%}"
            )
        print()

    if metrics["per_subtask"]:
        print("Per-subtask breakdown (narrow):")
        print(f"  {'Subtask':<45s} {'N':>4s} {'Acc':>7s} {'Prec':>7s} {'Cov':>7s}")
        print(f"  {'-' * 45} {'-' * 4} {'-' * 7} {'-' * 7} {'-' * 7}")
        for st, sm in sorted(metrics["per_subtask"].items()):
            print(
                f"  {st:<45s} {sm['total']:>4d} "
                f"{sm['accuracy']:>6.1%} {sm['precision']:>6.1%} {sm['coverage']:>6.1%}"
            )
        print()

    print(f"Results: {results_path}")
    print("=" * 70)

    return full_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LAB-Bench benchmark for YOHAS 3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_labbench.py --limit 5\n"
            "  python scripts/run_labbench.py --limit 50 --model sonnet\n"
            "  python scripts/run_labbench.py --mode agentic --trials 3 --limit 10\n"
            "  python scripts/run_labbench.py --mode agentic --model opus --limit 5\n"
            "  python scripts/run_labbench.py --dry-run --limit 10\n"
            "  python scripts/run_labbench.py --resume\n"
            "  python scripts/run_labbench.py --subtask seqqa --limit 3 --mode agentic\n"
            "  python scripts/run_labbench.py --subtask litqa2 --limit 5 --mode agentic\n"
            "  python scripts/run_labbench.py --subtask all --limit 10 --dry-run\n"
            "  python scripts/run_labbench.py --subtask dbqa --replicas 3 --limit 5\n"
            "  python scripts/run_labbench.py --subtask dbqa --mode codeact --model sonnet --limit 1\n"
            "  python scripts/run_labbench.py --subtask dbqa --mode yohas --model opus --limit 5\n"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max questions to evaluate per subtask (default: all)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="zero-shot",
        choices=["zero-shot", "agentic", "yohas", "codeact"],
        help="Evaluation mode: zero-shot (single LLM call), agentic (multi-turn with code execution), yohas (full YOHAS 4-phase pipeline), or codeact (plan-code-execute-observe loop)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sonnet",
        choices=["sonnet", "opus", "haiku"],
        help="Model tier to use (default: sonnet — Claude Sonnet for reasoning)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        choices=range(1, 6),
        metavar="N",
        help="Number of trials per question (1-5). Trial 2+ injects hints from prior attempts.",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of replicas for majority voting (default: 1 = disabled). "
            "Runs each question N times with temperature=0.3 and takes the "
            "majority-voted answer."
        ),
    )
    parser.add_argument(
        "--subtask",
        type=str,
        default="dbqa",
        choices=list(VALID_SUBTASKS),
        help=(
            "Which LAB-Bench subtask(s) to evaluate. "
            "'dbqa' = database QA (default), 'seqqa' = sequence manipulation, "
            "'litqa2' = literature recall, 'all' = all three."
        ),
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=4,
        help="Max turns per trial in agentic mode (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM calls; random answers for pipeline testing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-question timeout in seconds (default: 300 = 5 min)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Max concurrent LLM calls (default: 1)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable two-pass verification: after the agent answers, a second pass checks the answer from a different angle.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.replicas > 1 and args.trials > 1:
        parser.error("--replicas and --trials are mutually exclusive. Use one or the other.")

    results = asyncio.run(
        run_evaluation(
            limit=args.limit,
            mode=args.mode,
            model_tier=args.model,
            dry_run=args.dry_run,
            resume=args.resume,
            timeout_seconds=args.timeout,
            concurrency=args.concurrency,
            trials=args.trials,
            max_turns=args.max_turns,
            subtask=args.subtask,
            replicas=args.replicas,
            verify=args.verify,
        )
    )

    # Exit with non-zero if accuracy below STELLA baseline
    if results["metrics"]["accuracy"] < 0.54 and not args.dry_run:
        logger.warning("Accuracy %.1f%% is below STELLA baseline of 54%%", results["metrics"]["accuracy"] * 100)


if __name__ == "__main__":
    main()
