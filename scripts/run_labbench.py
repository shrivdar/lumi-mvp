#!/usr/bin/env python3
"""Standalone LAB-Bench DbQA benchmark pipeline for YOHAS 3.0.

Downloads (if needed), loads, and evaluates YOHAS on the LAB-Bench DbQA subset
(520 multiple-choice questions testing ability to query biological databases).

Competitor to beat: STELLA at 54% on DbQA.

Usage:
    python scripts/run_labbench.py --limit 5
    python scripts/run_labbench.py --limit 50 --mode zero-shot
    python scripts/run_labbench.py                           # all 520 questions
    python scripts/run_labbench.py --resume                  # resume from checkpoint
    python scripts/run_labbench.py --dry-run                 # no LLM calls, test pipeline

Requires: ANTHROPIC_API_KEY in .env or environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

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
# Data classes
# ---------------------------------------------------------------------------


class DbQAQuestion(NamedTuple):
    """A single DbQA question parsed from the JSONL dataset."""

    id: str
    question: str
    correct_answer: str  # the ideal answer text
    distractors: list[str]  # wrong answer texts
    subtask: str
    context: str


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


# ---------------------------------------------------------------------------
# Dataset download + loading
# ---------------------------------------------------------------------------

REFUSE_CHOICE = "Insufficient information to answer the question"


def download_dbqa_dataset() -> Path:
    """Download DbQA dataset from HuggingFace if not already present.

    Returns path to the JSONL file.
    """
    # Check if we already have the converted JSONL in the lab_bench dir
    legacy_path = PROJECT_ROOT / "data" / "benchmarks" / "lab_bench" / "dbqa.jsonl"
    if legacy_path.exists():
        logger.info("Using existing DbQA dataset at %s", legacy_path)
        return legacy_path

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = DATA_DIR / "dbqa.jsonl"

    if jsonl_path.exists():
        logger.info("Using existing DbQA dataset at %s", jsonl_path)
        return jsonl_path

    logger.info("Downloading LAB-Bench DbQA from HuggingFace...")

    try:
        from datasets import load_dataset

        ds = load_dataset("futurehouse/lab-bench", "DbQA", split="train")
        with open(jsonl_path, "w") as f:
            for row in ds:
                record = {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": row["ideal"],
                    "choices": row["distractors"],
                    "metadata": {
                        "canary": row.get("canary", ""),
                        "source": row.get("source"),
                        "subtask": row.get("subtask", ""),
                    },
                }
                f.write(json.dumps(record) + "\n")
        logger.info("Downloaded %d DbQA questions to %s", len(ds), jsonl_path)
        return jsonl_path

    except ImportError:
        # Try downloading parquet directly
        logger.info("datasets library not available; downloading parquet directly...")
        import urllib.request

        parquet_url = (
            "https://huggingface.co/datasets/futurehouse/lab-bench/"
            "resolve/refs%2Fconvert%2Fparquet/DbQA/train/0000.parquet"
        )
        parquet_path = DATA_DIR / "dbqa.parquet"
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
                    "metadata": {
                        "canary": row.get("canary", ""),
                        "source": row.get("source"),
                        "subtask": row.get("subtask", ""),
                    },
                }
                f.write(json.dumps(record) + "\n")

        logger.info("Downloaded %d DbQA questions to %s", len(df), jsonl_path)
        return jsonl_path


def load_dbqa_questions(path: Path, limit: int | None = None) -> list[DbQAQuestion]:
    """Load DbQA questions from JSONL file."""
    questions: list[DbQAQuestion] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            questions.append(
                DbQAQuestion(
                    id=row["id"],
                    question=row["question"],
                    correct_answer=row["answer"],
                    distractors=row.get("choices", []),
                    subtask=row.get("metadata", {}).get("subtask", "unknown"),
                    context=row.get("context", ""),
                )
            )
    return questions


def build_choices(question: DbQAQuestion, seed: int | None = None) -> tuple[list[str], str, str]:
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
        "This question is about disease-gene associations. Consider querying "
        "databases like DisGeNET, OMIM, and ClinVar to find gene-disease relationships."
    ),
    "gene_location_task": (
        "This question is about gene chromosomal locations. Use NCBI Gene, Ensembl, "
        "or UCSC Genome Browser data to determine genomic coordinates."
    ),
    "mirna_targets_task": (
        "This question is about microRNA targets. Consider databases like "
        "miRTarBase, TargetScan, or miRDB."
    ),
    "mouse_tumor_gene_sets": (
        "This question involves mouse tumor gene sets. Consider MSigDB, MGI, "
        "or cancer-related gene set databases."
    ),
    "oncogenic_signatures_task": (
        "This question is about oncogenic gene signatures. Use MSigDB oncogenic "
        "signatures or cancer genomics databases like COSMIC."
    ),
    "tfbs_GTRD_task": (
        "This question is about transcription factor binding sites from the GTRD "
        "database. Consider ChIP-seq data and TFBS databases."
    ),
    "variant_from_sequence_task": (
        "This question asks about genetic variants. Use dbSNP, ClinVar, "
        "or gnomAD to identify variants from sequence data."
    ),
    "variant_multi_sequence_task": (
        "This question involves variants across multiple sequences. Use "
        "variant databases and sequence alignment tools."
    ),
    "vax_response_task": (
        "This question is about vaccine response data. Consider immunology "
        "databases and vaccine-related gene expression studies."
    ),
    "viral_ppi_task": (
        "This question is about viral protein-protein interactions. Use "
        "IntAct, BioGRID, or virus-host interaction databases."
    ),
}


def _get_subtask_hint(subtask: str) -> str:
    """Get a hint for the subtask, matching on prefix."""
    for key, hint in SUBTASK_HINTS.items():
        if key in subtask:
            return hint
    return "Use biological databases to look up the answer."


def build_prompt(question: DbQAQuestion, choices: list[str]) -> str:
    """Build the LLM prompt for a DbQA question."""
    parts = []

    if question.context:
        parts.append(f"Context:\n{question.context}")

    parts.append(f"Question: {question.question}")
    parts.append("Choices:\n" + "\n".join(f"  {c}" for c in choices))

    hint = _get_subtask_hint(question.subtask)
    parts.append(f"\nHint: {hint}")

    parts.append(
        "\nInstructions:\n"
        "1. Reason step-by-step about which answer is correct.\n"
        "2. Consider what biological databases would contain this information.\n"
        "3. Think about which options are plausible distractors vs the correct answer.\n"
        "4. Only choose 'Insufficient information' if you truly cannot determine the answer.\n"
        "5. State your final answer on the LAST line in exactly this format: Answer: X\n"
        "   where X is a single letter (A, B, C, D, or E)."
    )

    return "\n\n".join(parts)


SYSTEM_PROMPT = (
    "You are a biomedical database expert taking a multiple-choice exam about "
    "biological databases. You have deep knowledge of databases including: "
    "DisGeNET, OMIM, ClinVar, UniProt, KEGG, Reactome, ChEMBL, MSigDB, GTRD, "
    "miRTarBase, dbSNP, gnomAD, IntAct, BioGRID, NCBI Gene, and Ensembl.\n\n"
    "For each question, reason carefully through the options. Use your knowledge "
    "of what information each database contains and how they differ. "
    "Select the single best answer."
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
# Evaluation loop
# ---------------------------------------------------------------------------


async def evaluate_question_live(
    question: DbQAQuestion,
    llm: Any,
    *,
    model: str = "",
    timeout_seconds: int = 300,
) -> ScoredResult:
    """Evaluate a single DbQA question using the LLM."""
    start = time.monotonic()
    choices, correct_letter, refuse_letter = build_choices(question)
    prompt = build_prompt(question, choices)

    try:
        resp = await asyncio.wait_for(
            llm.query(
                prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=2048,
                model=model or None,
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
        )


def evaluate_question_dry(question: DbQAQuestion) -> ScoredResult:
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

    # Per-subtask breakdown
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

    def save_result(self, result: ScoredResult) -> None:
        """Append a single result to the checkpoint file."""
        record = {
            "question_id": result.question_id,
            "subtask": result.subtask,
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
    mode: str = "sonnet",
    dry_run: bool = False,
    resume: bool = False,
    timeout_seconds: int = 300,
    concurrency: int = 1,
) -> dict[str, Any]:
    """Run the full LAB-Bench DbQA evaluation pipeline.

    Args:
        limit: Max questions to evaluate (None = all 520).
        mode: Model to use — "sonnet", "opus", "haiku".
        dry_run: Skip LLM calls, random answers.
        resume: Resume from last checkpoint.
        timeout_seconds: Per-question timeout.
        concurrency: Max concurrent LLM calls.

    Returns:
        Full results dict with metrics and per-question results.
    """
    # 1. Download / locate dataset
    dataset_path = download_dbqa_dataset()
    questions = load_dbqa_questions(dataset_path, limit=limit)
    logger.info("Loaded %d DbQA questions", len(questions))

    # 2. Set up checkpoint
    run_id = f"dbqa_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
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

    # 3. Set up LLM
    llm = None
    model_name = ""
    if not dry_run:
        from core.config import settings
        from core.llm import LLMClient

        llm = LLMClient()
        if mode == "sonnet":
            model_name = settings.llm_fast_model
        elif mode == "opus":
            model_name = settings.llm_model
        elif mode == "haiku":
            model_name = settings.llm_cheap_model
        else:
            model_name = mode  # allow direct model name
        logger.info("Using model: %s", model_name)

    # 4. Evaluate
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

    for idx, question in enumerate(remaining):
        question_num = len(results) + 1

        async with sem:
            if dry_run:
                result = evaluate_question_dry(question)
            else:
                result = await evaluate_question_live(
                    question,
                    llm,
                    model=model_name,
                    timeout_seconds=timeout_seconds,
                )

        results.append(result)
        checkpoint.save_result(result)

        if result.is_correct:
            correct_so_far += 1

        status = "CORRECT" if result.is_correct else "WRONG"
        if result.error:
            status = "ERROR"
        elif result.is_refused:
            status += " (refused)"

        running_acc = correct_so_far / question_num * 100
        logger.info(
            "Question %d/%d: %s [%s] (running accuracy: %.1f%%) [%dms, %d tok]",
            question_num,
            total,
            question.id[:12] + "...",
            status,
            running_acc,
            result.latency_ms,
            result.tokens_used,
        )

    elapsed = time.monotonic() - start_time

    # 5. Compute final metrics
    metrics = compute_metrics(results)

    # 6. Build full results
    full_results = {
        "benchmark": "LAB-Bench DbQA",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "mode": mode,
            "model": model_name,
            "dry_run": dry_run,
            "limit": limit,
            "timeout_seconds": timeout_seconds,
            "total_questions": total,
        },
        "metrics": metrics,
        "competitor_baseline": {
            "STELLA": 0.54,
            "note": "STELLA DbQA accuracy from LAB-Bench paper",
        },
        "delta_vs_stella": metrics["accuracy"] - 0.54,
        "elapsed_seconds": round(elapsed, 1),
        "results": [
            {
                "question_id": r.question_id,
                "subtask": r.subtask,
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

    # 7. Save results
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

    # 8. Print summary
    print("\n" + "=" * 70)
    print("LAB-Bench DbQA Results")
    print("=" * 70)
    print(f"  Model:         {model_name or 'dry-run'}")
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
    print(f"  STELLA target: 54.0%")
    print(f"  Delta:         {full_results['delta_vs_stella']:+.1%}")
    print()

    if metrics["per_subtask"]:
        print("Per-subtask breakdown:")
        print(f"  {'Subtask':<45s} {'N':>4s} {'Acc':>7s} {'Prec':>7s} {'Cov':>7s}")
        print(f"  {'-' * 45} {'-' * 4} {'-' * 7} {'-' * 7} {'-' * 7}")
        for subtask, sm in sorted(metrics["per_subtask"].items()):
            print(
                f"  {subtask:<45s} {sm['total']:>4d} "
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
        description="Run LAB-Bench DbQA benchmark for YOHAS 3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_labbench.py --limit 5\n"
            "  python scripts/run_labbench.py --limit 50 --mode sonnet\n"
            "  python scripts/run_labbench.py --dry-run --limit 10\n"
            "  python scripts/run_labbench.py --resume\n"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max questions to evaluate (default: all 520)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sonnet",
        choices=["sonnet", "opus", "haiku"],
        help="Model tier to use (default: sonnet — Claude Sonnet for reasoning)",
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
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    results = asyncio.run(
        run_evaluation(
            limit=args.limit,
            mode=args.mode,
            dry_run=args.dry_run,
            resume=args.resume,
            timeout_seconds=args.timeout,
            concurrency=args.concurrency,
        )
    )

    # Exit with non-zero if accuracy below STELLA baseline
    if results["metrics"]["accuracy"] < 0.54 and not args.dry_run:
        logger.warning("Accuracy %.1f%% is below STELLA baseline of 54%%", results["metrics"]["accuracy"] * 100)


if __name__ == "__main__":
    main()
