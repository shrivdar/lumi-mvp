#!/usr/bin/env python3
"""Scan trajectory JSONL files and remove mock data.

Mock trajectories are identified by:
  - reward == 1.0  AND
  - empty turns list  AND
  - wall_time_ms == 0

Usage:
    python scripts/cleanup_mock_trajectories.py              # dry-run (report only)
    python scripts/cleanup_mock_trajectories.py --delete     # delete mock files
    python scripts/cleanup_mock_trajectories.py --dir path/  # custom directory
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"

# Default directories to scan
DEFAULT_DIRS = [
    BACKEND_DIR / "data" / "trajectories",
    Path(__file__).resolve().parent.parent / "data" / "trajectories",
]


def is_mock_trajectory(record: dict) -> bool:
    """Check if a trajectory record is mock/synthetic data."""
    reward = record.get("reward", 0.0)
    turns = record.get("turns", [])
    wall_time = record.get("wall_time_ms", -1)

    return (
        reward == 1.0
        and len(turns) == 0
        and wall_time == 0
    )


def scan_file(path: Path) -> dict:
    """Scan a single JSONL file and report mock vs real records."""
    total = 0
    mock_count = 0
    real_count = 0

    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    total += 1
                    if is_mock_trajectory(record):
                        mock_count += 1
                    else:
                        real_count += 1
                except json.JSONDecodeError:
                    continue
    except OSError:
        return {"path": str(path), "error": "Could not read file"}

    return {
        "path": str(path),
        "total": total,
        "mock": mock_count,
        "real": real_count,
        "all_mock": mock_count == total and total > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean up mock trajectory files")
    parser.add_argument("--dir", type=str, nargs="*", help="Directories to scan")
    parser.add_argument("--delete", action="store_true", help="Delete files that are entirely mock")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-file details")
    args = parser.parse_args()

    dirs = [Path(d) for d in args.dir] if args.dir else DEFAULT_DIRS

    all_results: list[dict] = []
    files_scanned = 0
    mock_files = 0
    mixed_files = 0
    real_files = 0
    total_records = 0
    total_mock = 0

    for scan_dir in dirs:
        if not scan_dir.exists():
            print(f"  Directory not found: {scan_dir}")
            continue

        print(f"Scanning: {scan_dir}")
        jsonl_files = sorted(scan_dir.glob("*.jsonl"))

        if not jsonl_files:
            print("  No JSONL files found")
            continue

        for path in jsonl_files:
            result = scan_file(path)
            all_results.append(result)
            files_scanned += 1
            total_records += result.get("total", 0)
            total_mock += result.get("mock", 0)

            if result.get("all_mock"):
                mock_files += 1
                if args.verbose:
                    print(f"  MOCK  {path.name} ({result['total']} records)")
            elif result.get("mock", 0) > 0:
                mixed_files += 1
                if args.verbose:
                    print(f"  MIXED {path.name} ({result['mock']}/{result['total']} mock)")
            else:
                real_files += 1
                if args.verbose:
                    print(f"  REAL  {path.name} ({result['total']} records)")

    # Summary
    print(f"\n{'=' * 50}")
    print("TRAJECTORY SCAN SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Files scanned:    {files_scanned}")
    print(f"  All-mock files:   {mock_files}")
    print(f"  Mixed files:      {mixed_files}")
    print(f"  Real files:       {real_files}")
    print(f"  Total records:    {total_records}")
    print(f"  Mock records:     {total_mock}")
    print(f"  Real records:     {total_records - total_mock}")
    print(f"{'=' * 50}")

    if args.delete and mock_files > 0:
        print(f"\nDeleting {mock_files} all-mock files...")
        deleted = 0
        for result in all_results:
            if result.get("all_mock"):
                path = Path(result["path"])
                try:
                    path.unlink()
                    deleted += 1
                    print(f"  Deleted: {path.name}")
                except OSError as e:
                    print(f"  Failed to delete {path.name}: {e}")
        print(f"Deleted {deleted} files.")
    elif mock_files > 0 and not args.delete:
        print(f"\nRun with --delete to remove {mock_files} all-mock files.")


if __name__ == "__main__":
    main()
