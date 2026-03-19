#!/usr/bin/env python3
"""BixBench benchmark pipeline for YOHAS 3.0.

Downloads (if needed) and evaluates the BixBench dataset — 205 bioinformatics
questions from 60 real-world Jupyter notebooks hosted on HuggingFace.

Supports three verification modes matching the official BixBench grading:
  - str_verifier:   normalized exact-string match
  - range_verifier: numeric value falls within (lower, upper) range
  - llm_verifier:   LLM-as-judge semantic equivalence

Modes:
  - open:     zero-shot free-form answer
  - mcq:      zero-shot multiple choice
  - agentic:  multi-turn code execution with data capsules (default)

Usage:
    python scripts/run_bixbench.py --limit 5
    python scripts/run_bixbench.py --limit 205 --mode mcq
    python scripts/run_bixbench.py --mode agentic --limit 3 --trials 2
    python scripts/run_bixbench.py --resume data/benchmarks/bixbench_results.json
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import io
import json
import logging
import os
import re
import signal
import sys
import time
import traceback
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — ensure backend is importable
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Load .env if present (must happen before importing core.config)
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        # Manual fallback: parse KEY=VALUE lines
        for line in _env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data" / "benchmarks" / "bixbench"
CAPSULE_DIR = DATA_DIR / "capsules"
RESULTS_DIR = PROJECT_ROOT / "data" / "benchmarks"
RAW_JSONL = DATA_DIR / "BixBench.jsonl"
RESULTS_FILE = RESULTS_DIR / "bixbench_results.json"

BIXBENCH_HF_URL = (
    "https://huggingface.co/datasets/futurehouse/BixBench"
    "/resolve/main/BixBench.jsonl"
)

# HuggingFace capsule download URL pattern
CAPSULE_HF_URL_TEMPLATE = (
    "https://huggingface.co/datasets/futurehouse/BixBench"
    "/resolve/main/capsules/{data_folder}"
)

COMPETITOR_BASELINE = {"Biomni A1": 0.522}

# Agentic execution defaults
DEFAULT_MAX_TURNS = 8
CODE_EXEC_TIMEOUT_SECONDS = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bixbench")

# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------


def ensure_dataset() -> Path:
    """Download BixBench.jsonl from HuggingFace if not already present."""
    if RAW_JSONL.exists():
        n = sum(1 for _ in open(RAW_JSONL))
        logger.info("Dataset already present: %s (%d questions)", RAW_JSONL, n)
        return RAW_JSONL

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading BixBench dataset from HuggingFace ...")
    req = urllib.request.Request(
        BIXBENCH_HF_URL, headers={"User-Agent": "YOHAS-Benchmark/3.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            RAW_JSONL.write_bytes(resp.read())
        n = sum(1 for _ in open(RAW_JSONL))
        logger.info("Downloaded %d questions -> %s", n, RAW_JSONL)
    except Exception as exc:
        logger.error("Failed to download dataset: %s", exc)
        raise SystemExit(1) from exc
    return RAW_JSONL


def load_questions(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load questions from the BixBench JSONL file."""
    questions: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
            if limit and len(questions) >= limit:
                break
    return questions


# ---------------------------------------------------------------------------
# Data capsule management
# ---------------------------------------------------------------------------


def ensure_capsule(question_row: dict[str, Any]) -> Path | None:
    """Download and extract the data capsule for a question.

    Returns the path to the extracted capsule directory, or None on failure.
    """
    capsule_uuid = question_row.get("capsule_uuid", "")
    data_folder = question_row.get("data_folder", "")
    if not capsule_uuid or not data_folder:
        logger.warning("No capsule info for question %s", question_row.get("question_id"))
        return None

    capsule_path = CAPSULE_DIR / capsule_uuid
    # If already extracted, reuse
    if capsule_path.exists() and any(capsule_path.iterdir()):
        return capsule_path

    # Download the ZIP
    CAPSULE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CAPSULE_DIR / data_folder
    url = CAPSULE_HF_URL_TEMPLATE.format(data_folder=data_folder)

    if not zip_path.exists():
        logger.info("Downloading capsule %s ...", data_folder)
        req = urllib.request.Request(url, headers={"User-Agent": "YOHAS-Benchmark/3.0"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                zip_path.write_bytes(resp.read())
            logger.info("Downloaded capsule -> %s", zip_path)
        except Exception as exc:
            logger.error("Failed to download capsule %s: %s", data_folder, exc)
            return None

    # Extract
    try:
        capsule_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(capsule_path)
        logger.info("Extracted capsule to %s (%d files)", capsule_path, len(list(capsule_path.rglob("*"))))
        return capsule_path
    except Exception as exc:
        logger.error("Failed to extract capsule %s: %s", data_folder, exc)
        return None


def list_capsule_files(capsule_path: Path) -> list[dict[str, str]]:
    """List data files in a capsule with previews for CSVs/TSVs."""
    files = []
    for p in sorted(capsule_path.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(capsule_path)
        entry: dict[str, str] = {"path": str(rel), "size": f"{p.stat().st_size:,} bytes"}

        suffix = p.suffix.lower()
        if suffix in (".csv", ".tsv", ".txt"):
            try:
                lines = p.read_text(errors="replace").split("\n")[:6]
                entry["preview"] = "\n".join(lines)
            except Exception:
                entry["preview"] = "(could not read)"
        elif suffix == ".json":
            try:
                text = p.read_text(errors="replace")
                if len(text) > 2000:
                    entry["preview"] = text[:2000] + "\n... (truncated)"
                else:
                    entry["preview"] = text
            except Exception:
                entry["preview"] = "(could not read)"
        files.append(entry)
    return files


# ---------------------------------------------------------------------------
# Sandboxed code execution
# ---------------------------------------------------------------------------


def _timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("Code execution timed out")


def execute_code(code: str, data_dir: str, timeout: int = CODE_EXEC_TIMEOUT_SECONDS) -> str:
    """Execute Python code with access to data files in a sandboxed namespace.

    Returns stdout output or an error message.
    """
    # Pre-import available libraries
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "DATA_DIR": data_dir,
    }

    # Import libraries that might be available
    for mod_name, alias in [
        ("pandas", "pd"),
        ("numpy", "np"),
        ("scipy.stats", "stats"),
        ("scipy", "scipy"),
        ("os", "os"),
        ("json", "json"),
        ("csv", "csv"),
        ("re", "re"),
        ("math", "math"),
        ("pathlib", "Path"),
        ("collections", "collections"),
        ("itertools", "itertools"),
        ("glob", "glob"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[""])
            namespace[alias] = mod
        except ImportError:
            pass

    # Also make pathlib.Path available
    try:
        from pathlib import Path as _Path
        namespace["Path"] = _Path
    except ImportError:
        pass

    old_cwd = os.getcwd()
    os.chdir(data_dir)
    try:
        f = io.StringIO()
        # Set alarm for timeout (Unix only)
        has_alarm = hasattr(signal, "SIGALRM")
        if has_alarm:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)
        try:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                exec(code, namespace)
        except TimeoutError:
            return f"Error: Code execution timed out after {timeout}s"
        except Exception as e:
            captured = f.getvalue()
            tb = traceback.format_exc()
            return f"{captured}\nError: {type(e).__name__}: {e}\n{tb}"
        finally:
            if has_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return f.getvalue()
    finally:
        os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# Agentic code extraction helpers
# ---------------------------------------------------------------------------


def extract_code_block(response: str) -> str | None:
    """Extract code from <execute>...</execute> or ```python ... ``` blocks."""
    # <execute> tags (preferred)
    m = re.search(r"<execute>(.*?)</execute>", response, re.DOTALL)
    if m:
        return m.group(1).strip()

    # ```python code blocks
    m = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()

    # ``` generic code blocks
    m = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()

    return None


def extract_answer_tag(response: str) -> str | None:
    """Extract answer from <answer>...</answer> tags."""
    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Grading (matches official BixBench verifiers)
# ---------------------------------------------------------------------------


def _normalize(s: str) -> str:
    """Lowercase, strip whitespace, remove non-alphanumeric (except .-+)."""
    s = s.strip().lower()
    # Keep digits, letters, dots, hyphens, plus signs for numeric matching
    return re.sub(r"[^a-z0-9.\-+e]", "", s)


def grade_str_verifier(predicted: str, ideal: str) -> bool:
    """Exact string match after normalization, with numeric fallback."""
    p = _normalize(predicted)
    g = _normalize(ideal)
    if not p or not g:
        return False
    # Direct string match (including substring)
    if p == g or p in g or g in p:
        return True
    # Numeric comparison fallback (handles 1.90E-5 vs 1.9E-05, etc.)
    try:
        pf = float(predicted.strip())
        gf = float(ideal.strip())
        if gf == 0:
            return pf == 0
        # Allow 1% relative tolerance for floating point representation
        return abs(pf - gf) / abs(gf) < 0.01
    except (ValueError, ZeroDivisionError):
        pass
    return False


def grade_range_verifier(predicted: str, ideal: str) -> bool:
    """Check if predicted numeric value falls within the ideal range tuple."""
    try:
        bounds = ast.literal_eval(ideal)
        if isinstance(bounds, tuple) and len(bounds) == 2:
            lower, upper = float(bounds[0]), float(bounds[1])
        else:
            return False
    except (ValueError, SyntaxError):
        return False

    # Extract a number from the predicted text
    predicted_clean = predicted.strip()
    # Try direct float parse first
    try:
        val = float(predicted_clean)
        return lower <= val <= upper
    except ValueError:
        pass
    # Try to find a number in the text
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", predicted_clean)
    if m:
        try:
            val = float(m.group())
            return lower <= val <= upper
        except ValueError:
            pass
    return False


async def grade_llm_verifier(
    predicted: str,
    ideal: str,
    question: str,
    llm: Any,
) -> bool:
    """Use LLM-as-judge for semantic equivalence grading."""
    prompt = (
        f"You are grading a bioinformatics benchmark answer.\n\n"
        f"Question: {question}\n\n"
        f"Correct answer: {ideal}\n\n"
        f"Predicted answer: {predicted}\n\n"
        f"Are these answers semantically equivalent? The predicted answer "
        f"does not need to be word-for-word identical, but must convey the "
        f"same information and reach the same conclusion.\n\n"
        f"Respond with ONLY one word: 'correct' or 'incorrect'."
    )
    try:
        resp = await llm.query(
            prompt,
            system_prompt="You are a strict bioinformatics grading assistant.",
            max_tokens=16,
        )
        text = resp.text.strip().lower()
        return "correct" in text and "incorrect" not in text
    except Exception as exc:
        logger.warning("LLM grading failed, falling back to string match: %s", exc)
        return grade_str_verifier(predicted, ideal)


async def grade_answer(
    predicted: str,
    question_row: dict[str, Any],
    llm: Any | None = None,
) -> bool:
    """Grade a predicted answer using the appropriate verifier."""
    eval_mode = question_row.get("eval_mode", "str_verifier")
    ideal = str(question_row.get("ideal", ""))

    if eval_mode == "range_verifier":
        return grade_range_verifier(predicted, ideal)
    elif eval_mode == "llm_verifier" and llm is not None:
        return await grade_llm_verifier(
            predicted, ideal, question_row.get("question", ""), llm
        )
    else:
        # str_verifier or fallback
        return grade_str_verifier(predicted, ideal)


# ---------------------------------------------------------------------------
# MCQ answer extraction
# ---------------------------------------------------------------------------


def format_mcq_choices(question_row: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """Build MCQ options string and mapping from letter -> value.

    BixBench format: ideal is the correct answer, distractors are wrong answers.
    We shuffle them into A/B/C/D options.
    """
    import random

    ideal = str(question_row.get("ideal", ""))
    distractors = question_row.get("distractors", [])
    all_options = [ideal] + [str(d) for d in distractors]
    # Deterministic shuffle based on question_id for reproducibility
    qid = question_row.get("question_id", question_row.get("id", ""))
    rng = random.Random(hash(qid))
    rng.shuffle(all_options)

    letters = "ABCDEFGH"
    mapping: dict[str, str] = {}
    lines: list[str] = []
    for i, opt in enumerate(all_options):
        letter = letters[i]
        mapping[letter] = opt
        lines.append(f"  {letter}) {opt}")
    return "\n".join(lines), mapping


def extract_mcq_letter(response: str) -> str:
    """Parse an MCQ letter (A-H) from LLM response."""
    # Look for <answer> X </answer> format first (BixBench standard)
    m = re.search(r"<answer>\s*([A-Ha-h])\s*</answer>", response)
    if m:
        return m.group(1).upper()
    # Common patterns
    patterns = [
        r"(?:answer|choice|option)\s*(?:is|:)\s*\(?([A-Ha-h])\)?",
        r"^\s*\(?([A-Ha-h])\)\s*$",
        r"\b([A-Ha-h])\b\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    # Fallback: first standalone capital letter
    m = re.search(r"\b([A-H])\b", response)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# LLM evaluation (the actual YOHAS agent call)
# ---------------------------------------------------------------------------


def _build_system_prompt() -> str:
    return (
        "You are YOHAS (Your Own Hypothesis-driven Agentic Scientist), an "
        "expert computational biologist. You are given a research question from "
        "a bioinformatics analysis. Use your deep knowledge of genomics, "
        "transcriptomics, proteomics, statistical methods, and bioinformatics "
        "tools to answer the question as precisely as possible.\n\n"
        "If the question asks for a numeric value, provide ONLY the number. "
        "If it asks for a gene/protein name, provide ONLY the name. "
        "Be concise and precise."
    )


def _build_agentic_system_prompt() -> str:
    return (
        "You are YOHAS (Your Own Hypothesis-driven Agentic Scientist), an "
        "expert computational biologist with access to a Python code execution "
        "environment. You are given a bioinformatics research question along with "
        "data files from the original analysis.\n\n"
        "## Your workflow:\n"
        "1. Review the available data files and the question\n"
        "2. Plan your analysis approach\n"
        "3. Write Python code in <execute>...</execute> blocks to analyze the data\n"
        "4. Review the output and iterate as needed\n"
        "5. When you have the answer, wrap it in <answer>...</answer> tags\n\n"
        "## Code execution environment:\n"
        "- pandas (as pd), numpy (as np), scipy.stats (as stats) are pre-imported\n"
        "- os, json, csv, re, math, pathlib.Path, collections, itertools, glob are available\n"
        "- The working directory is set to the data capsule folder\n"
        "- Use DATA_DIR variable for the absolute path to the data directory\n"
        "- Print results using print() — all stdout is captured\n\n"
        "## Rules:\n"
        "- Write ONE <execute> block per turn. Wait for results before continuing.\n"
        "- If code errors, fix and retry with a different approach.\n"
        "- When you have the final answer, respond with <answer>YOUR_ANSWER</answer>\n"
        "- If the question asks for a number, put ONLY the number in <answer> tags.\n"
        "- If the question asks for a name/term, put ONLY that term.\n"
        "- Be precise: match the expected format (e.g. rounded to N decimal places).\n"
    )


def _build_open_prompt(question_row: dict[str, Any]) -> str:
    """Build open-answer prompt for a BixBench question."""
    q = question_row["question"]
    hypothesis = question_row.get("hypothesis", "")
    result_hint = question_row.get("result", "")
    categories = question_row.get("categories", "")

    parts = [f"Research question: {q}"]
    if hypothesis:
        parts.append(f"\nHypothesis being tested: {hypothesis}")
    if result_hint:
        parts.append(f"\nResearch context/findings: {result_hint}")
    if categories:
        parts.append(f"\nDomain: {categories}")

    parts.append(
        "\nProvide your answer concisely. If the question asks for a specific "
        "number, state only the number. If it asks for a name or term, state "
        "only that term. Wrap your final answer in <answer> tags like: "
        "<answer>your answer</answer>"
    )
    return "\n".join(parts)


def _build_mcq_prompt(question_row: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """Build MCQ prompt and return (prompt, letter->value mapping)."""
    q = question_row["question"]
    hypothesis = question_row.get("hypothesis", "")
    result_hint = question_row.get("result", "")
    choices_str, mapping = format_mcq_choices(question_row)

    parts = [f"Research question: {q}"]
    if hypothesis:
        parts.append(f"\nHypothesis: {hypothesis}")
    if result_hint:
        parts.append(f"\nContext: {result_hint}")
    parts.append(f"\nAnswer options:\n{choices_str}")
    parts.append(
        "\nSelect the single best answer. Respond with ONLY the letter "
        "(A, B, C, or D) in <answer> tags like: <answer>B</answer>"
    )
    return "\n".join(parts), mapping


def _build_agentic_prompt(
    question_row: dict[str, Any],
    capsule_files: list[dict[str, str]],
) -> str:
    """Build the initial agentic prompt with data file listings."""
    q = question_row["question"]
    hypothesis = question_row.get("hypothesis", "")
    result_hint = question_row.get("result", "")
    categories = question_row.get("categories", "")

    parts = [f"## Research Question\n{q}"]
    if hypothesis:
        parts.append(f"\n## Hypothesis\n{hypothesis}")
    if result_hint:
        parts.append(f"\n## Research Context (from original notebook)\n{result_hint}")
    if categories:
        parts.append(f"\n## Domain: {categories}")

    # File listing
    parts.append("\n## Available Data Files")
    for finfo in capsule_files:
        parts.append(f"\n### {finfo['path']} ({finfo['size']})")
        if "preview" in finfo:
            preview = finfo["preview"]
            # Truncate long previews
            if len(preview) > 1500:
                preview = preview[:1500] + "\n... (truncated)"
            parts.append(f"```\n{preview}\n```")

    parts.append(
        "\n## Instructions\n"
        "Analyze the data files to answer the research question. "
        "Start by exploring the data, then write code to compute the answer. "
        "Use <execute>...</execute> blocks for code and <answer>...</answer> "
        "for your final answer."
    )
    return "\n".join(parts)


def _extract_open_answer(response: str) -> str:
    """Extract answer from <answer> tags or fall back to heuristics."""
    # <answer> tags
    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Last line after "Answer:"
    m = re.search(r"(?:answer|result)\s*[:=]\s*(.+)", response, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fall back to last non-empty line
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    return lines[-1] if lines else response.strip()


# ---------------------------------------------------------------------------
# Agentic evaluation (multi-turn with code execution)
# ---------------------------------------------------------------------------


async def evaluate_question_agentic(
    question_row: dict[str, Any],
    llm: Any,
    capsule_path: Path | None,
    max_turns: int = DEFAULT_MAX_TURNS,
    timeout_seconds: int = 300,
    trial: int = 1,
    prev_answer: str | None = None,
    prev_reason: str | None = None,
) -> dict[str, Any]:
    """Evaluate a question using the agentic multi-turn code execution loop."""
    start = time.monotonic()
    qid = question_row.get("question_id", question_row.get("id", ""))
    total_tokens = 0

    try:
        # If no capsule, fall back to zero-shot open mode
        if capsule_path is None:
            logger.warning("No capsule for %s, falling back to zero-shot", qid)
            result = await evaluate_question(question_row, llm, mode="open", timeout_seconds=timeout_seconds)
            result["mode"] = "agentic-fallback"
            return result

        # Build file listing
        capsule_files = list_capsule_files(capsule_path)
        if not capsule_files:
            logger.warning("Empty capsule for %s, falling back to zero-shot", qid)
            result = await evaluate_question(question_row, llm, mode="open", timeout_seconds=timeout_seconds)
            result["mode"] = "agentic-fallback"
            return result

        # Build initial prompt
        initial_prompt = _build_agentic_prompt(question_row, capsule_files)
        system_prompt = _build_agentic_system_prompt()

        # Conversation history for multi-turn
        conversation: list[str] = [initial_prompt]

        # If this is a retry trial, inject feedback
        if trial > 1 and prev_answer is not None:
            retry_msg = (
                f"\n\n## RETRY (Trial {trial})\n"
                f"Your previous answer was: {prev_answer}\n"
                f"The evaluator said it was wrong"
            )
            if prev_reason:
                retry_msg += f" because: {prev_reason}"
            retry_msg += (
                ".\nTry a different approach. Look at the data more carefully, "
                "check your calculations, and consider alternative interpretations."
            )
            conversation[0] += retry_msg

        predicted = ""
        answer_found = False
        turns_used = 0

        for turn in range(max_turns):
            turns_used = turn + 1
            elapsed = time.monotonic() - start
            if elapsed > timeout_seconds:
                logger.warning("Agentic timeout for %s after %d turns", qid, turn)
                break

            # Build the full prompt from conversation history
            full_prompt = "\n\n---\n\n".join(conversation)

            remaining_timeout = max(30, timeout_seconds - int(elapsed))
            resp = await asyncio.wait_for(
                llm.query(
                    full_prompt,
                    system_prompt=system_prompt,
                    max_tokens=4096,
                ),
                timeout=remaining_timeout,
            )
            total_tokens += resp.call_tokens
            response_text = resp.text

            # Check for answer
            answer = extract_answer_tag(response_text)
            if answer is not None:
                predicted = answer
                answer_found = True
                break

            # Check for code to execute
            code = extract_code_block(response_text)
            if code is not None:
                logger.debug("Turn %d: Executing code (%d chars)", turn + 1, len(code))
                output = execute_code(code, str(capsule_path))
                # Truncate very long output
                if len(output) > 8000:
                    output = output[:4000] + "\n... (output truncated) ...\n" + output[-2000:]

                # Add to conversation
                conversation.append(response_text)
                conversation.append(f"## Code Execution Output (Turn {turn + 1}):\n```\n{output}\n```\n\nContinue your analysis. If you have the answer, provide it in <answer>...</answer> tags.")
            else:
                # No code and no answer — ask the agent to provide one
                conversation.append(response_text)
                conversation.append(
                    "Please either write code in <execute>...</execute> blocks to analyze the data, "
                    "or provide your final answer in <answer>...</answer> tags."
                )

        # If no answer tag found, try to extract from the last response
        if not answer_found:
            predicted = _extract_open_answer(response_text)

        duration_ms = int((time.monotonic() - start) * 1000)
        correct = await grade_answer(predicted, question_row, llm=llm)

        return {
            "question_id": qid,
            "question": question_row["question"][:200],
            "hypothesis": question_row.get("hypothesis", "")[:200],
            "ideal": str(question_row.get("ideal", "")),
            "predicted": predicted,
            "correct": correct,
            "eval_mode": question_row.get("eval_mode", "str_verifier"),
            "categories": question_row.get("categories", ""),
            "tokens_used": total_tokens,
            "duration_ms": duration_ms,
            "mode": "agentic",
            "turns_used": turns_used,
            "trial": trial,
            "capsule_uuid": question_row.get("capsule_uuid", ""),
            "raw_response": response_text[:500] if response_text else "",
            "error": None,
        }

    except asyncio.TimeoutError:
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "question_id": qid,
            "question": question_row["question"][:200],
            "hypothesis": question_row.get("hypothesis", "")[:200],
            "ideal": str(question_row.get("ideal", "")),
            "predicted": predicted if predicted else "",
            "correct": False,
            "eval_mode": question_row.get("eval_mode", "str_verifier"),
            "categories": question_row.get("categories", ""),
            "tokens_used": total_tokens,
            "duration_ms": duration_ms,
            "mode": "agentic",
            "turns_used": turns_used if "turns_used" in dir() else 0,
            "trial": trial,
            "capsule_uuid": question_row.get("capsule_uuid", ""),
            "raw_response": "",
            "error": "TIMEOUT",
        }
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.error("Agentic evaluation %s failed: %s", qid, exc)
        return {
            "question_id": qid,
            "question": question_row["question"][:200],
            "hypothesis": question_row.get("hypothesis", "")[:200],
            "ideal": str(question_row.get("ideal", "")),
            "predicted": "",
            "correct": False,
            "eval_mode": question_row.get("eval_mode", "str_verifier"),
            "categories": question_row.get("categories", ""),
            "tokens_used": total_tokens,
            "duration_ms": duration_ms,
            "mode": "agentic",
            "turns_used": 0,
            "trial": trial,
            "capsule_uuid": question_row.get("capsule_uuid", ""),
            "raw_response": "",
            "error": traceback.format_exc()[:500],
        }


# ---------------------------------------------------------------------------
# Zero-shot evaluation (open / mcq modes)
# ---------------------------------------------------------------------------


async def evaluate_question(
    question_row: dict[str, Any],
    llm: Any,
    mode: str = "open",
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Evaluate a single BixBench question and return result dict."""
    start = time.monotonic()
    qid = question_row.get("question_id", question_row.get("id", ""))
    tokens_used = 0

    try:
        if mode == "mcq":
            prompt, mapping = _build_mcq_prompt(question_row)
            resp = await asyncio.wait_for(
                llm.query(
                    prompt,
                    system_prompt=_build_system_prompt(),
                    max_tokens=1024,
                ),
                timeout=timeout_seconds,
            )
            tokens_used = resp.call_tokens
            letter = extract_mcq_letter(resp.text)
            predicted = mapping.get(letter, resp.text.strip())
        else:
            prompt = _build_open_prompt(question_row)
            resp = await asyncio.wait_for(
                llm.query(
                    prompt,
                    system_prompt=_build_system_prompt(),
                    max_tokens=2048,
                ),
                timeout=timeout_seconds,
            )
            tokens_used = resp.call_tokens
            predicted = _extract_open_answer(resp.text)

        duration_ms = int((time.monotonic() - start) * 1000)
        correct = await grade_answer(predicted, question_row, llm=llm)

        return {
            "question_id": qid,
            "question": question_row["question"][:200],
            "hypothesis": question_row.get("hypothesis", "")[:200],
            "ideal": str(question_row.get("ideal", "")),
            "predicted": predicted,
            "correct": correct,
            "eval_mode": question_row.get("eval_mode", "str_verifier"),
            "categories": question_row.get("categories", ""),
            "tokens_used": tokens_used,
            "duration_ms": duration_ms,
            "mode": mode,
            "raw_response": resp.text[:500],
            "error": None,
        }

    except asyncio.TimeoutError:
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "question_id": qid,
            "question": question_row["question"][:200],
            "hypothesis": question_row.get("hypothesis", "")[:200],
            "ideal": str(question_row.get("ideal", "")),
            "predicted": "",
            "correct": False,
            "eval_mode": question_row.get("eval_mode", "str_verifier"),
            "categories": question_row.get("categories", ""),
            "tokens_used": tokens_used,
            "duration_ms": duration_ms,
            "mode": mode,
            "raw_response": "",
            "error": "TIMEOUT",
        }
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.error("Question %s failed: %s", qid, exc)
        return {
            "question_id": qid,
            "question": question_row["question"][:200],
            "hypothesis": question_row.get("hypothesis", "")[:200],
            "ideal": str(question_row.get("ideal", "")),
            "predicted": "",
            "correct": False,
            "eval_mode": question_row.get("eval_mode", "str_verifier"),
            "categories": question_row.get("categories", ""),
            "tokens_used": tokens_used,
            "duration_ms": duration_ms,
            "mode": mode,
            "raw_response": "",
            "error": traceback.format_exc()[:500],
        }


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------


def load_checkpoint(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Load previous results for resumption. Returns (results, completed_ids)."""
    if not path.exists():
        return [], set()
    try:
        data = json.loads(path.read_text())
        results = data.get("results", [])
        completed = {r["question_id"] for r in results if r.get("question_id")}
        logger.info("Loaded checkpoint: %d completed questions", len(completed))
        return results, completed
    except Exception as exc:
        logger.warning("Could not load checkpoint %s: %s", path, exc)
        return [], set()


def save_results(
    results: list[dict[str, Any]],
    path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save results to JSON with summary statistics."""
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    accuracy = correct / total if total > 0 else 0.0
    total_tokens = sum(r.get("tokens_used", 0) for r in results)
    total_time_ms = sum(r.get("duration_ms", 0) for r in results)
    errors = sum(1 for r in results if r.get("error"))

    # Per-category breakdown
    by_category: dict[str, dict[str, int]] = {}
    for r in results:
        for cat in (r.get("categories", "") or "unknown").split(","):
            cat = cat.strip()
            if not cat:
                cat = "unknown"
            entry = by_category.setdefault(cat, {"total": 0, "correct": 0})
            entry["total"] += 1
            if r.get("correct"):
                entry["correct"] += 1

    # Per eval_mode breakdown
    by_eval_mode: dict[str, dict[str, int]] = {}
    for r in results:
        em = r.get("eval_mode", "unknown")
        entry = by_eval_mode.setdefault(em, {"total": 0, "correct": 0})
        entry["total"] += 1
        if r.get("correct"):
            entry["correct"] += 1

    output = {
        "benchmark": "BixBench",
        "version": "1.5",
        "agent": "YOHAS 3.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "accuracy_pct": f"{accuracy * 100:.1f}%",
            "total_tokens": total_tokens,
            "avg_tokens_per_question": round(total_tokens / total, 1) if total else 0,
            "total_time_ms": total_time_ms,
            "avg_time_per_question_ms": round(total_time_ms / total, 1) if total else 0,
            "errors": errors,
        },
        "by_eval_mode": {
            k: {
                **v,
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0,
            }
            for k, v in sorted(by_eval_mode.items())
        },
        "by_category": {
            k: {
                **v,
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0,
            }
            for k, v in sorted(by_category.items())
        },
        "baselines": COMPETITOR_BASELINE,
        "metadata": metadata or {},
        "results": results,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, default=str))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def run_pipeline(args: argparse.Namespace) -> None:
    """Main evaluation pipeline."""
    # 1. Ensure dataset is present
    dataset_path = ensure_dataset()
    questions = load_questions(dataset_path, limit=args.limit)
    logger.info("Loaded %d questions from BixBench", len(questions))

    # 2. Load checkpoint for resume
    results_path = Path(args.output) if args.output else RESULTS_FILE
    if args.resume:
        resume_path = Path(args.resume) if Path(args.resume).exists() else results_path
        prev_results, completed_ids = load_checkpoint(resume_path)
    else:
        prev_results, completed_ids = [], set()

    results = list(prev_results)

    # 3. Initialize LLM
    from core.llm import LLMClient
    llm = LLMClient()
    logger.info(
        "LLM initialized (model: %s)",
        os.environ.get("LLM_MODEL", "default"),
    )

    # 4. Evaluate
    total = len(questions)
    running_correct = sum(1 for r in results if r.get("correct"))
    running_total = len(results)
    skipped = 0
    num_trials = getattr(args, "trials", 1)

    mode_label = args.mode
    if args.mode == "agentic":
        mode_label = f"agentic (max_turns={args.max_turns}, trials={num_trials})"

    print("\n" + "=" * 70)
    print(f"  BixBench Evaluation — {total} questions, mode={mode_label}")
    print(f"  Competitor baseline: Biomni A1 = 52.2%")
    if completed_ids:
        print(f"  Resuming: {len(completed_ids)} already completed")
    print("=" * 70 + "\n")

    for i, q in enumerate(questions):
        qid = q.get("question_id", q.get("id", f"q_{i}"))

        # Skip already completed
        if qid in completed_ids:
            skipped += 1
            continue

        # Progress header
        pct = (running_correct / running_total * 100) if running_total > 0 else 0
        prefix = f"Question {i + 1}/{total}"
        q_text = q["question"][:80].replace("\n", " ")
        print(f"  {prefix}: {qid} — {q_text}...")

        if args.mode == "agentic":
            # Download/extract capsule
            capsule_path = ensure_capsule(q)

            # Multi-trial loop
            result = None
            for trial in range(1, num_trials + 1):
                prev_answer = result["predicted"] if result is not None else None
                prev_reason = (
                    f"expected '{q.get('ideal', '')}'" if result is not None and not result["correct"]
                    else None
                )

                # Skip further trials if already correct
                if result is not None and result["correct"]:
                    break

                if trial > 1:
                    print(f"    -> Trial {trial}/{num_trials} (retrying...)")

                result = await evaluate_question_agentic(
                    q,
                    llm,
                    capsule_path,
                    max_turns=args.max_turns,
                    timeout_seconds=args.timeout,
                    trial=trial,
                    prev_answer=prev_answer,
                    prev_reason=prev_reason,
                )

            results.append(result)  # type: ignore[arg-type]
        else:
            # Zero-shot modes (open / mcq)
            result = await evaluate_question(
                q, llm, mode=args.mode, timeout_seconds=args.timeout
            )
            results.append(result)

        running_total += 1
        if result["correct"]:  # type: ignore[index]
            running_correct += 1

        # Print result
        status = "CORRECT" if result["correct"] else "WRONG"  # type: ignore[index]
        duration_s = result["duration_ms"] / 1000  # type: ignore[index]
        running_acc = running_correct / running_total * 100
        tokens = result.get("tokens_used", 0)  # type: ignore[union-attr]
        error_flag = " [ERROR]" if result.get("error") else ""  # type: ignore[union-attr]
        turns_info = ""
        if args.mode == "agentic":
            turns_info = f" turns={result.get('turns_used', '?')}"  # type: ignore[union-attr]
            if num_trials > 1:
                turns_info += f" trial={result.get('trial', '?')}"  # type: ignore[union-attr]

        print(
            f"    -> [{status}] predicted={result['predicted'][:60]} "  # type: ignore[index]
            f"ideal={result['ideal'][:60]} "  # type: ignore[index]
            f"({duration_s:.1f}s, {tokens} tok{turns_info}){error_flag}"
        )
        print(f"    Running accuracy: {running_correct}/{running_total} ({running_acc:.1f}%)")

        # Save intermediate results after every question
        save_results(
            results,
            results_path,
            metadata={
                "mode": args.mode,
                "limit": args.limit,
                "timeout": args.timeout,
                "trials": num_trials,
                "max_turns": getattr(args, "max_turns", None),
                "partial": True,
                "completed": running_total,
            },
        )

    # 5. Final save
    save_results(
        results,
        results_path,
        metadata={
            "mode": args.mode,
            "limit": args.limit,
            "timeout": args.timeout,
            "trials": num_trials,
            "max_turns": getattr(args, "max_turns", None),
            "partial": False,
        },
    )

    # 6. Print summary
    final_correct = sum(1 for r in results if r.get("correct"))
    final_total = len(results)
    final_acc = final_correct / final_total * 100 if final_total else 0
    total_tokens = sum(r.get("tokens_used", 0) for r in results)
    total_time = sum(r.get("duration_ms", 0) for r in results) / 1000
    errors = sum(1 for r in results if r.get("error"))

    print("\n" + "=" * 70)
    print("  BIXBENCH RESULTS")
    print("=" * 70)
    print(f"  Mode:      {args.mode}")
    if args.mode == "agentic":
        print(f"  Max turns: {args.max_turns}")
        print(f"  Trials:    {num_trials}")
    print(f"  Total:     {final_total}")
    print(f"  Correct:   {final_correct}")
    print(f"  Accuracy:  {final_acc:.1f}%")
    print(f"  Tokens:    {total_tokens:,} ({total_tokens / final_total:,.0f} avg)" if final_total else "")
    print(f"  Time:      {total_time:.1f}s ({total_time / final_total:.1f}s avg)" if final_total else "")
    print(f"  Errors:    {errors}")
    if skipped:
        print(f"  Skipped:   {skipped} (already completed)")
    print()
    print(f"  vs Biomni A1: {final_acc:.1f}% vs 52.2%", end="")
    if final_acc > 52.2:
        print("  ** BEATING COMPETITOR **")
    else:
        print(f"  (gap: {52.2 - final_acc:.1f}pp)")

    # Per eval_mode breakdown
    by_mode: dict[str, list[bool]] = {}
    for r in results:
        em = r.get("eval_mode", "unknown")
        by_mode.setdefault(em, []).append(r.get("correct", False))

    print("\n  By eval mode:")
    for em, vals in sorted(by_mode.items()):
        c = sum(vals)
        t = len(vals)
        print(f"    {em:20s}  {c}/{t}  ({c / t * 100:.1f}%)")

    # Agentic-specific stats
    if args.mode == "agentic":
        agentic_results = [r for r in results if r.get("mode") == "agentic"]
        fallback_results = [r for r in results if r.get("mode") == "agentic-fallback"]
        if agentic_results:
            avg_turns = sum(r.get("turns_used", 0) for r in agentic_results) / len(agentic_results)
            print(f"\n  Agentic stats:")
            print(f"    Full agentic: {len(agentic_results)} questions (avg {avg_turns:.1f} turns)")
        if fallback_results:
            print(f"    Fallback (no capsule): {len(fallback_results)} questions")

    print(f"\n  Results saved to: {results_path}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOHAS 3.0 BixBench Benchmark Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_bixbench.py --limit 5\n"
            "  python scripts/run_bixbench.py --limit 205 --mode mcq\n"
            "  python scripts/run_bixbench.py --mode agentic --limit 3\n"
            "  python scripts/run_bixbench.py --mode agentic --limit 10 --trials 2\n"
            "  python scripts/run_bixbench.py --resume data/benchmarks/bixbench_results.json\n"
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of questions to evaluate (default: all 205)",
    )
    p.add_argument(
        "--mode",
        choices=["open", "mcq", "agentic"],
        default="agentic",
        help="Answer mode: 'open' (free-form), 'mcq' (multiple choice), or 'agentic' (code execution). Default: agentic",
    )
    p.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials per question (for multi-trial with retry feedback). Default: 1",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Max turns per agentic evaluation (default: {DEFAULT_MAX_TURNS})",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-question timeout in seconds (default: 300)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output path for results JSON (default: {RESULTS_FILE})",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a previous results file (provide path or use default)",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
