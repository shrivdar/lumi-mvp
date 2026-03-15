"""Benchmark dataset adapters — load instances from each benchmark suite.

Each adapter converts the benchmark's native format into BenchmarkInstance objects.
Datasets are expected to live in data/benchmarks/<suite_name>/ as JSONL files.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from benchmarks.models import BenchmarkInstance, BenchmarkSuite

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmarks"


class BaseAdapter(ABC):
    """Base class for benchmark dataset adapters."""

    suite: BenchmarkSuite

    @abstractmethod
    def load_instances(self, limit: int | None = None) -> list[BenchmarkInstance]:
        """Load benchmark instances from disk."""
        ...

    def _read_jsonl(self, path: Path, limit: int | None = None) -> list[dict[str, Any]]:
        """Read a JSONL file and return list of dicts."""
        if not path.exists():
            logger.warning("Dataset file not found: %s — generating synthetic instances", path)
            return []
        rows: list[dict[str, Any]] = []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


# ---------------------------------------------------------------------------
# Biomni-Eval1 Adapter
# ---------------------------------------------------------------------------


class BiomniEval1Adapter(BaseAdapter):
    """Adapter for Biomni-Eval1 (433 biomedical QA instances).

    Expected format: data/benchmarks/biomni_eval1/instances.jsonl
    Each line: {"id": str, "question": str, "context": str, "answer": str,
                "category": str, "choices": [...]}
    """

    suite = BenchmarkSuite.BIOMNI_EVAL1

    def load_instances(self, limit: int | None = None) -> list[BenchmarkInstance]:
        path = DATA_DIR / "biomni_eval1" / "instances.jsonl"
        rows = self._read_jsonl(path, limit)

        if not rows:
            return self._generate_synthetic(limit or 433)

        instances = []
        for row in rows:
            instances.append(
                BenchmarkInstance(
                    suite=self.suite,
                    instance_id=row["id"],
                    question=row["question"],
                    context=row.get("context", ""),
                    choices=row.get("choices", []),
                    ground_truth=row["answer"],
                    category=row.get("category", "general"),
                    metadata=row.get("metadata", {}),
                )
            )
        return instances

    def _generate_synthetic(self, count: int) -> list[BenchmarkInstance]:
        """Generate synthetic Biomni-Eval1 instances for testing."""
        categories = [
            "gwas_causal_gene",
            "protein_function",
            "drug_target",
            "pathway_analysis",
            "disease_mechanism",
            "gene_expression",
            "clinical_outcome",
            "variant_effect",
        ]
        instances = []
        for i in range(count):
            cat = categories[i % len(categories)]
            instances.append(
                BenchmarkInstance(
                    suite=self.suite,
                    instance_id=f"biomni_{i:04d}",
                    question=_SYNTHETIC_QUESTIONS.get(cat, f"What is the role of gene_{i} in disease_{i}?"),
                    context=f"Synthetic context for {cat} instance {i}.",
                    choices=["A", "B", "C", "D"],
                    ground_truth="A",
                    category=cat,
                    metadata={"synthetic": True, "index": i},
                )
            )
        return instances


# ---------------------------------------------------------------------------
# BixBench Adapter
# ---------------------------------------------------------------------------


class BixBenchAdapter(BaseAdapter):
    """Adapter for BixBench (205 bioinformatics trajectory instances).

    Expected format: data/benchmarks/bixbench/instances.jsonl
    Each line: {"id": str, "question": str, "context": str, "answer": str,
                "category": str, "data_files": [...]}
    """

    suite = BenchmarkSuite.BIXBENCH

    def load_instances(self, limit: int | None = None) -> list[BenchmarkInstance]:
        path = DATA_DIR / "bixbench" / "instances.jsonl"
        rows = self._read_jsonl(path, limit)

        if not rows:
            return self._generate_synthetic(limit or 205)

        instances = []
        for row in rows:
            instances.append(
                BenchmarkInstance(
                    suite=self.suite,
                    instance_id=row["id"],
                    question=row["question"],
                    context=row.get("context", ""),
                    ground_truth=row["answer"],
                    category=row.get("category", "bioinformatics"),
                    metadata={
                        **row.get("metadata", {}),
                        "data_files": row.get("data_files", []),
                    },
                )
            )
        return instances

    def _generate_synthetic(self, count: int) -> list[BenchmarkInstance]:
        categories = [
            "sequence_analysis",
            "structure_prediction",
            "expression_analysis",
            "variant_calling",
            "pathway_enrichment",
        ]
        instances = []
        for i in range(count):
            cat = categories[i % len(categories)]
            instances.append(
                BenchmarkInstance(
                    suite=self.suite,
                    instance_id=f"bixbench_{i:04d}",
                    question=f"Analyze the {cat.replace('_', ' ')} data and determine the primary finding.",
                    context=f"Synthetic bioinformatics context for {cat} instance {i}.",
                    ground_truth="significant_enrichment",
                    category=cat,
                    metadata={"synthetic": True, "index": i},
                )
            )
        return instances


# ---------------------------------------------------------------------------
# LAB-Bench Adapters
# ---------------------------------------------------------------------------


class LABBenchDbQAAdapter(BaseAdapter):
    """Adapter for LAB-Bench Database QA tasks.

    Expected format: data/benchmarks/lab_bench/dbqa.jsonl
    """

    suite = BenchmarkSuite.LAB_BENCH_DBQA

    def load_instances(self, limit: int | None = None) -> list[BenchmarkInstance]:
        path = DATA_DIR / "lab_bench" / "dbqa.jsonl"
        rows = self._read_jsonl(path, limit)

        if not rows:
            return self._generate_synthetic(limit or 100)

        instances = []
        for row in rows:
            instances.append(
                BenchmarkInstance(
                    suite=self.suite,
                    instance_id=row["id"],
                    question=row["question"],
                    context=row.get("context", ""),
                    choices=row.get("choices", []),
                    ground_truth=row["answer"],
                    category="dbqa",
                    metadata=row.get("metadata", {}),
                )
            )
        return instances

    def _generate_synthetic(self, count: int) -> list[BenchmarkInstance]:
        subtasks = [
            "protein_lookup",
            "gene_query",
            "pathway_membership",
            "drug_interaction",
            "variant_annotation",
        ]
        instances = []
        for i in range(count):
            sub = subtasks[i % len(subtasks)]
            instances.append(
                BenchmarkInstance(
                    suite=self.suite,
                    instance_id=f"dbqa_{i:04d}",
                    question=f"Query the database to find {sub.replace('_', ' ')} information for entity_{i}.",
                    context="Database schema with tables for genes, proteins, pathways, drugs.",
                    choices=["Result A", "Result B", "Result C", "Result D"],
                    ground_truth="Result A",
                    category=sub,
                    metadata={"synthetic": True, "index": i},
                )
            )
        return instances


class LABBenchSeqQAAdapter(BaseAdapter):
    """Adapter for LAB-Bench Sequence QA tasks.

    Expected format: data/benchmarks/lab_bench/seqqa.jsonl
    """

    suite = BenchmarkSuite.LAB_BENCH_SEQQA

    def load_instances(self, limit: int | None = None) -> list[BenchmarkInstance]:
        path = DATA_DIR / "lab_bench" / "seqqa.jsonl"
        rows = self._read_jsonl(path, limit)

        if not rows:
            return self._generate_synthetic(limit or 100)

        instances = []
        for row in rows:
            instances.append(
                BenchmarkInstance(
                    suite=self.suite,
                    instance_id=row["id"],
                    question=row["question"],
                    context=row.get("context", ""),
                    choices=row.get("choices", []),
                    ground_truth=row["answer"],
                    category="seqqa",
                    metadata=row.get("metadata", {}),
                )
            )
        return instances

    def _generate_synthetic(self, count: int) -> list[BenchmarkInstance]:
        subtasks = [
            "motif_identification",
            "domain_prediction",
            "homology_search",
            "mutation_effect",
            "secondary_structure",
        ]
        instances = []
        for i in range(count):
            sub = subtasks[i % len(subtasks)]
            instances.append(
                BenchmarkInstance(
                    suite=self.suite,
                    instance_id=f"seqqa_{i:04d}",
                    question=f"Analyze the protein sequence for {sub.replace('_', ' ')} characteristics.",
                    context="Protein sequence: MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
                    choices=["Alpha helix", "Beta sheet", "Coil", "Turn"],
                    ground_truth="Alpha helix",
                    category=sub,
                    metadata={"synthetic": True, "index": i},
                )
            )
        return instances


class LABBenchLitQA2Adapter(BaseAdapter):
    """Adapter for LAB-Bench LitQA2 (literature QA) tasks.

    Expected format: data/benchmarks/lab_bench/litqa2.jsonl
    """

    suite = BenchmarkSuite.LAB_BENCH_LITQA2

    def load_instances(self, limit: int | None = None) -> list[BenchmarkInstance]:
        path = DATA_DIR / "lab_bench" / "litqa2.jsonl"
        rows = self._read_jsonl(path, limit)

        if not rows:
            return self._generate_synthetic(limit or 100)

        instances = []
        for row in rows:
            instances.append(
                BenchmarkInstance(
                    suite=self.suite,
                    instance_id=row["id"],
                    question=row["question"],
                    context=row.get("context", ""),
                    choices=row.get("choices", []),
                    ground_truth=row["answer"],
                    category="litqa2",
                    metadata=row.get("metadata", {}),
                )
            )
        return instances

    def _generate_synthetic(self, count: int) -> list[BenchmarkInstance]:
        topics = [
            "immunotherapy",
            "gene_therapy",
            "protein_engineering",
            "clinical_trials",
            "drug_resistance",
        ]
        instances = []
        for i in range(count):
            topic = topics[i % len(topics)]
            instances.append(
                BenchmarkInstance(
                    suite=self.suite,
                    instance_id=f"litqa2_{i:04d}",
                    question=(
                        f"Based on the literature, what is the mechanism of "
                        f"{topic.replace('_', ' ')} in the described context?"
                    ),
                    context=(
                        f"Recent studies on {topic.replace('_', ' ')} "
                        f"have shown promising results in preclinical models."
                    ),
                    choices=["Mechanism A", "Mechanism B", "Mechanism C", "Mechanism D"],
                    ground_truth="Mechanism A",
                    category=topic,
                    metadata={"synthetic": True, "index": i},
                )
            )
        return instances


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


ADAPTERS: dict[BenchmarkSuite, type[BaseAdapter]] = {
    BenchmarkSuite.BIOMNI_EVAL1: BiomniEval1Adapter,
    BenchmarkSuite.BIXBENCH: BixBenchAdapter,
    BenchmarkSuite.LAB_BENCH_DBQA: LABBenchDbQAAdapter,
    BenchmarkSuite.LAB_BENCH_SEQQA: LABBenchSeqQAAdapter,
    BenchmarkSuite.LAB_BENCH_LITQA2: LABBenchLitQA2Adapter,
}


def get_adapter(suite: BenchmarkSuite) -> BaseAdapter:
    """Get the adapter for a benchmark suite."""
    cls = ADAPTERS.get(suite)
    if cls is None:
        raise ValueError(f"No adapter for suite: {suite}")
    return cls()


# ---------------------------------------------------------------------------
# Synthetic question templates (used when no real dataset is present)
# ---------------------------------------------------------------------------

_SYNTHETIC_QUESTIONS: dict[str, str] = {
    "gwas_causal_gene": (
        "Given GWAS results showing a significant association at locus 17q21, "
        "which gene is most likely causal for breast cancer susceptibility?"
    ),
    "protein_function": "What is the primary molecular function of the protein encoded by EGFR?",
    "drug_target": "Which protein is the primary target of imatinib in chronic myeloid leukemia?",
    "pathway_analysis": "In the MAPK/ERK signaling pathway, which kinase directly phosphorylates ERK1/2?",
    "disease_mechanism": "What is the molecular mechanism by which loss of APC leads to colorectal cancer?",
    "gene_expression": "Which tissue shows the highest expression of the insulin gene (INS)?",
    "clinical_outcome": "In NSCLC patients with EGFR L858R mutation, what is the expected response rate to erlotinib?",
    "variant_effect": "What is the predicted functional impact of the BRCA1 c.5266dupC variant?",
}
