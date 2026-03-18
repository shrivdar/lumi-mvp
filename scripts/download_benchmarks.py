#!/usr/bin/env python3
"""Download BixBench + LAB-Bench benchmark datasets.

Downloads the question JSONL files from HuggingFace and converts them
to the format expected by our benchmark adapters.

Usage:
    python scripts/download_benchmarks.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks"

# HuggingFace raw file URLs
BIXBENCH_URL = (
    "https://huggingface.co/datasets/futurehouse/BixBench/resolve/main/BixBench.jsonl"
)
LABBENCH_BASE = (
    "https://huggingface.co/datasets/futurehouse/lab-bench/resolve/main"
)


def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL to dest path."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "YOHAS-Benchmark/3.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            dest.write_bytes(resp.read())
        print(f"  Saved to {dest} ({dest.stat().st_size:,} bytes)")
        return True
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return False


def convert_bixbench(raw_path: Path, out_path: Path) -> int:
    """Convert BixBench.jsonl to our adapter format.

    Raw format (v2, flattened): one question per line with fields like
    id, capsule_id, question, answer, category, etc.

    Our adapter expects:
    {"id": str, "question": str, "context": str, "answer": str,
     "category": str, "data_files": [...]}
    """
    if out_path.exists():
        count = sum(1 for _ in open(out_path))
        print(f"  Already converted: {out_path} ({count} instances)")
        return count

    instances = []
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # BixBench stores the actual answer in "ideal" field;
            # the top-level "answer" is a boolean (True = answerable)
            actual_answer = row.get("ideal", row.get("answer", row.get("ground_truth", "")))
            instance = {
                "id": row.get("id", row.get("question_id", f"bix_{len(instances)}")),
                "question": row.get("question", ""),
                "context": row.get("context", row.get("hypothesis", "")),
                "answer": str(actual_answer),
                "category": row.get("category", row.get("capsule_category", "bioinformatics")),
                "data_files": row.get("data_files", []),
                "metadata": {
                    k: v for k, v in row.items()
                    if k not in {"id", "question_id", "question", "context",
                                 "hypothesis", "answer", "ground_truth",
                                 "category", "capsule_category", "data_files"}
                },
            }
            instances.append(instance)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for inst in instances:
            f.write(json.dumps(inst) + "\n")
    print(f"  Converted {len(instances)} instances -> {out_path}")
    return len(instances)


def convert_labbench_parquet(parquet_path: Path, out_path: Path, subset: str) -> int:
    """Convert LAB-Bench parquet to our adapter JSONL format."""
    if out_path.exists():
        count = sum(1 for _ in open(out_path))
        print(f"  Already converted: {out_path} ({count} instances)")
        return count

    try:
        import pandas as pd
    except ImportError:
        print("  WARNING: pandas not installed, cannot convert parquet files")
        return 0

    import numpy as np

    df = pd.read_parquet(parquet_path)
    instances = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        # LAB-Bench typically has: question, ideal (answer), distractors
        answer = row_dict.get("ideal", row_dict.get("answer", ""))
        distractors = row_dict.get("distractors", [])
        # Convert numpy arrays to plain lists
        if hasattr(distractors, "tolist"):
            distractors = distractors.tolist()
        elif isinstance(distractors, str):
            distractors = [s.strip().strip("'\"") for s in distractors.strip("[]").split("' '")]
            distractors = [d for d in distractors if d]
        instance = {
            "id": row_dict.get("id", f"{subset}_{len(instances)}"),
            "question": row_dict.get("question", ""),
            "context": row_dict.get("context", ""),
            "answer": str(answer),
            "choices": distractors,
            "metadata": {
                k: (v.tolist() if isinstance(v, np.ndarray) else
                    v if not isinstance(v, float) or v == v else None)
                for k, v in row_dict.items()
                if k not in {"id", "question", "context", "ideal", "answer", "distractors"}
            },
        }
        instances.append(instance)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for inst in instances:
            f.write(json.dumps(inst, default=str) + "\n")
    print(f"  Converted {len(instances)} instances -> {out_path}")
    return len(instances)


def download_labbench_subset(subset: str, filename: str, out_name: str) -> int:
    """Download and convert a LAB-Bench subset."""
    # Try parquet first (HuggingFace datasets format)
    parquet_url = f"{LABBENCH_BASE}/{subset}/train-00000-of-00001.parquet"
    parquet_dest = DATA_DIR / "lab_bench" / f"{out_name}.parquet"
    jsonl_dest = DATA_DIR / "lab_bench" / f"{out_name}.jsonl"

    if jsonl_dest.exists():
        count = sum(1 for _ in open(jsonl_dest))
        print(f"  Already exists: {jsonl_dest} ({count} instances)")
        return count

    if download_file(parquet_url, parquet_dest):
        count = convert_labbench_parquet(parquet_dest, jsonl_dest, out_name)
        if count > 0:
            return count

    # Fallback: try JSONL directly
    jsonl_url = f"{LABBENCH_BASE}/{subset}/{filename}"
    raw_dest = DATA_DIR / "lab_bench" / f"{out_name}_raw.jsonl"
    if download_file(jsonl_url, raw_dest):
        # Copy as-is (format may already match)
        import shutil
        shutil.copy(raw_dest, jsonl_dest)
        count = sum(1 for _ in open(jsonl_dest))
        print(f"  Copied {count} instances -> {jsonl_dest}")
        return count

    print(f"  WARNING: Could not download {subset}")
    return 0


def main() -> None:
    print("=" * 60)
    print("YOHAS 3.0 — Benchmark Dataset Downloader")
    print("=" * 60)

    total = 0

    # 1. BixBench
    print("\n[1/4] BixBench (bioinformatics questions)")
    raw_path = DATA_DIR / "bixbench" / "BixBench.jsonl"
    converted_path = DATA_DIR / "bixbench" / "instances.jsonl"
    if download_file(BIXBENCH_URL, raw_path):
        count = convert_bixbench(raw_path, converted_path)
        total += count
    else:
        print("  FAILED to download BixBench")

    # 2. LAB-Bench DbQA
    print("\n[2/4] LAB-Bench DbQA (database QA)")
    count = download_labbench_subset("DbQA", "test.jsonl", "dbqa")
    total += count

    # 3. LAB-Bench SeqQA
    print("\n[3/4] LAB-Bench SeqQA (sequence QA)")
    count = download_labbench_subset("SeqQA", "test.jsonl", "seqqa")
    total += count

    # 4. LAB-Bench LitQA2
    print("\n[4/4] LAB-Bench LitQA2 (literature QA)")
    count = download_labbench_subset("LitQA2", "test.jsonl", "litqa2")
    total += count

    print(f"\n{'=' * 60}")
    print(f"Done. Total instances: {total}")
    print(f"Data directory: {DATA_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
