"""Shared infrastructure for YOHAS 3.0 benchmark evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure backend is importable
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root / "backend"))

from core.config import settings  # noqa: E402
from core.llm import LLMClient  # noqa: E402
from core.models import (  # noqa: E402
    AgentType,
    BenchmarkMetric,
    BenchmarkRun,
    KGEdge,
    KGNode,
    NodeType,
    ResearchConfig,
    ResearchSession,
)
from agents import (  # noqa: E402
    ClinicalAnalystAgent,
    DrugHunterAgent,
    ExperimentDesignerAgent,
    GenomicsMapperAgent,
    LiteratureAnalystAgent,
    PathwayAnalystAgent,
    ProteinEngineerAgent,
    ScientificCriticAgent,
    get_template,
)
from orchestrator.research_loop import ResearchOrchestrator  # noqa: E402
from world_model.knowledge_graph import InMemoryKnowledgeGraph  # noqa: E402

logger = logging.getLogger("yohas.benchmarks")

RESULTS_ROOT = _project_root / "results"

# Agent class lookup
AGENT_CLASSES: dict[AgentType, type] = {
    AgentType.LITERATURE_ANALYST: LiteratureAnalystAgent,
    AgentType.PROTEIN_ENGINEER: ProteinEngineerAgent,
    AgentType.GENOMICS_MAPPER: GenomicsMapperAgent,
    AgentType.PATHWAY_ANALYST: PathwayAnalystAgent,
    AgentType.DRUG_HUNTER: DrugHunterAgent,
    AgentType.CLINICAL_ANALYST: ClinicalAnalystAgent,
    AgentType.SCIENTIFIC_CRITIC: ScientificCriticAgent,
    AgentType.EXPERIMENT_DESIGNER: ExperimentDesignerAgent,
}


def _get_benchmark_mode() -> str:
    return os.environ.get("YOHAS_BENCHMARK_MODE", "agentic")


def _get_max_iterations() -> int:
    return int(os.environ.get("YOHAS_MAX_ITERATIONS", "5"))


def _agent_factory(
    agent_type: AgentType,
    llm: LLMClient,
    kg: Any,
    yami: Any = None,
    **kwargs: Any,
) -> Any:
    cls = AGENT_CLASSES[agent_type]
    return cls(template=get_template(agent_type), llm=llm, kg=kg, yami=yami)


class YOHASRunner:
    """Programmatic interface to run YOHAS research sessions without HTTP."""

    def __init__(
        self,
        *,
        max_iterations: int | None = None,
        enable_falsification: bool = True,
        enable_hitl: bool = False,
        agent_types: list[AgentType] | None = None,
    ) -> None:
        self.llm = LLMClient()
        self.max_iterations = max_iterations or _get_max_iterations()
        self.enable_falsification = enable_falsification
        self.enable_hitl = enable_hitl
        self.agent_types = agent_types

    def run_session(self, query: str) -> tuple[ResearchSession, InMemoryKnowledgeGraph]:
        """Run a full YOHAS research session, returning session and KG."""
        kg = InMemoryKnowledgeGraph(graph_id=f"bench-{int(time.time())}")
        orchestrator = ResearchOrchestrator(
            llm=self.llm,
            kg=kg,
            agent_factory=_agent_factory,
        )
        config = ResearchConfig(
            max_mcts_iterations=self.max_iterations,
            enable_falsification=self.enable_falsification,
            enable_hitl=self.enable_hitl,
            agent_types=self.agent_types,
        )
        session = asyncio.run(orchestrator.run(query, config))
        return session, kg

    async def run_session_async(
        self, query: str
    ) -> tuple[ResearchSession, InMemoryKnowledgeGraph]:
        """Async variant for callers already in an event loop."""
        kg = InMemoryKnowledgeGraph(graph_id=f"bench-{int(time.time())}")
        orchestrator = ResearchOrchestrator(
            llm=self.llm,
            kg=kg,
            agent_factory=_agent_factory,
        )
        config = ResearchConfig(
            max_mcts_iterations=self.max_iterations,
            enable_falsification=self.enable_falsification,
            enable_hitl=self.enable_hitl,
            agent_types=self.agent_types,
        )
        session = await orchestrator.run(query, config)
        return session, kg

    def query_llm_with_kg(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        kg: InMemoryKnowledgeGraph | None = None,
    ) -> str:
        """Direct LLM call with optional KG context injection (for zero-shot mode)."""
        kg_context = kg.to_json() if kg else None
        return asyncio.run(
            self.llm.query(
                prompt,
                system_prompt=system_prompt,
                kg_context=kg_context,
            )
        )

    @property
    def token_summary(self) -> dict[str, int]:
        return self.llm.token_summary


class AnswerExtractor:
    """Extracts structured answers from KG nodes/edges and research report."""

    # Node type mappings for specific benchmark task types
    TASK_NODE_TYPES: dict[str, list[NodeType]] = {
        "gwas": [NodeType.GENE, NodeType.BIOMARKER],
        "drug_repurposing": [NodeType.DRUG, NodeType.COMPOUND],
        "gene_disease": [NodeType.GENE, NodeType.DISEASE],
        "protein_function": [NodeType.PROTEIN, NodeType.MECHANISM],
        "pathway": [NodeType.PATHWAY, NodeType.MECHANISM],
        "clinical": [NodeType.CLINICAL_TRIAL, NodeType.DRUG],
        "side_effect": [NodeType.SIDE_EFFECT, NodeType.DRUG],
        "biomarker": [NodeType.BIOMARKER, NodeType.GENE, NodeType.PROTEIN],
        "drug_target": [NodeType.PROTEIN, NodeType.DRUG],
        "gene_interaction": [NodeType.GENE, NodeType.PROTEIN],
    }

    @staticmethod
    def extract_answer_text(
        session: ResearchSession, kg: InMemoryKnowledgeGraph
    ) -> str:
        """Extract a natural-language answer from session results + KG."""
        parts: list[str] = []
        if session.result:
            if session.result.best_hypothesis:
                parts.append(session.result.best_hypothesis.hypothesis)
            if session.result.report_markdown:
                parts.append(session.result.report_markdown)
        if not parts:
            parts.append(kg.to_markdown_summary())
        return "\n\n".join(parts)

    @staticmethod
    def extract_nodes_by_type(
        kg: InMemoryKnowledgeGraph,
        node_types: list[NodeType],
        *,
        min_confidence: float = 0.3,
    ) -> list[KGNode]:
        """Get nodes matching given types, sorted by confidence descending."""
        kg_json = kg.to_json()
        nodes: list[KGNode] = []
        for nd in kg_json.get("nodes", []):
            node = KGNode(**nd) if isinstance(nd, dict) else nd
            if node.type in node_types and node.confidence >= min_confidence:
                nodes.append(node)
        nodes.sort(key=lambda n: n.confidence, reverse=True)
        return nodes

    @classmethod
    def extract_for_task(
        cls,
        task_type: str,
        kg: InMemoryKnowledgeGraph,
        *,
        top_k: int = 10,
    ) -> list[str]:
        """Extract answer entities for a benchmark task type."""
        node_types = cls.TASK_NODE_TYPES.get(task_type, list(NodeType))
        nodes = cls.extract_nodes_by_type(kg, node_types)
        return [n.name for n in nodes[:top_k]]

    @staticmethod
    def extract_top_edges(
        kg: InMemoryKnowledgeGraph, *, top_k: int = 10
    ) -> list[KGEdge]:
        """Get highest-confidence edges from the KG."""
        kg_json = kg.to_json()
        edges: list[KGEdge] = []
        for ed in kg_json.get("edges", []):
            edge = KGEdge(**ed) if isinstance(ed, dict) else ed
            if not edge.falsified:
                edges.append(edge)
        edges.sort(key=lambda e: e.confidence.overall, reverse=True)
        return edges[:top_k]

    @staticmethod
    def extract_multiple_choice(text: str) -> str:
        """Parse a multiple-choice letter (A-E) from LLM response text."""
        # Look for patterns like "Answer: B", "The answer is C", "(D)", etc.
        patterns = [
            r"(?:answer|choice|option)\s*(?:is|:)\s*\(?([A-Ea-e])\)?",
            r"^\s*\(?([A-Ea-e])\)\s*$",
            r"\b([A-Ea-e])\b\s*$",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).upper()
        # Fallback: first standalone capital letter A-E
        m = re.search(r"\b([A-E])\b", text)
        if m:
            return m.group(1)
        # No letter found — return empty string, don't guess
        logger.warning("Could not extract multiple-choice letter from: %s", text[:200])
        return ""


class ResultLogger:
    """Saves benchmark results as JSON and generates markdown summary tables."""

    def __init__(self, benchmark_name: str, output_dir: Path | None = None) -> None:
        self.benchmark_name = benchmark_name
        self.output_dir = output_dir or RESULTS_ROOT / benchmark_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._results: list[dict[str, Any]] = []
        self._run_id = f"{benchmark_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self._checkpoint_path = self.output_dir / f"{self._run_id}_checkpoint.jsonl"

    @property
    def run_id(self) -> str:
        return self._run_id

    def log_instance(self, instance: dict[str, Any]) -> None:
        """Log a single benchmark instance result and append to checkpoint."""
        self._results.append(instance)
        with open(self._checkpoint_path, "a") as f:
            f.write(json.dumps(instance, default=str) + "\n")

    def load_checkpoint(self) -> set[str]:
        """Load completed instance IDs from checkpoint file for resumption."""
        completed: set[str] = set()
        if self._checkpoint_path.exists():
            for line in self._checkpoint_path.read_text().splitlines():
                if line.strip():
                    data = json.loads(line)
                    iid = data.get("instance_id", data.get("task_id", ""))
                    if iid:
                        completed.add(str(iid))
            self._results = [
                json.loads(line)
                for line in self._checkpoint_path.read_text().splitlines()
                if line.strip()
            ]
        return completed

    def save_final(self, extra_metadata: dict[str, Any] | None = None) -> Path:
        """Save complete results JSON and markdown summary."""
        total = len(self._results)
        correct = sum(1 for r in self._results if r.get("correct", False))
        total_tokens = sum(r.get("tokens_used", 0) for r in self._results)

        run = BenchmarkRun(
            benchmark_name=self.benchmark_name,
            total_tasks=total,
            correct_tasks=correct,
            accuracy=correct / total if total > 0 else 0.0,
            metrics=[
                BenchmarkMetric(name="accuracy", value=correct / total if total else 0),
                BenchmarkMetric(name="total_tokens", value=total_tokens),
                BenchmarkMetric(name="instances_completed", value=total),
            ],
            run_config=extra_metadata or {},
        )

        results_path = self.output_dir / f"{self._run_id}_results.json"
        results_path.write_text(
            json.dumps(
                {"run": run.model_dump(mode="json"), "instances": self._results},
                indent=2,
                default=str,
            )
        )

        md = self._generate_markdown(run)
        md_path = self.output_dir / f"{self._run_id}_summary.md"
        md_path.write_text(md)

        logger.info("Results saved to %s", results_path)
        return results_path

    def _generate_markdown(self, run: BenchmarkRun) -> str:
        lines = [
            f"# {self.benchmark_name} Benchmark Results",
            f"",
            f"**Run ID:** {self._run_id}",
            f"**Date:** {run.started_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Total:** {run.total_tasks} | **Correct:** {run.correct_tasks} | "
            f"**Accuracy:** {run.accuracy:.1%}",
            f"",
        ]

        # Per-category breakdown if results have categories
        categories: dict[str, list[dict]] = {}
        for r in self._results:
            cat = r.get("category", r.get("task_type", "default"))
            categories.setdefault(cat, []).append(r)

        if len(categories) > 1:
            lines.append("## Results by Category")
            lines.append("")
            lines.append("| Category | Total | Correct | Accuracy |")
            lines.append("|----------|-------|---------|----------|")
            for cat, items in sorted(categories.items()):
                c = sum(1 for i in items if i.get("correct", False))
                acc = c / len(items) if items else 0
                lines.append(f"| {cat} | {len(items)} | {c} | {acc:.1%} |")
            lines.append("")

        # Token usage
        total_tokens = sum(r.get("tokens_used", 0) for r in self._results)
        lines.append("## Resource Usage")
        lines.append("")
        lines.append(f"- **Total tokens:** {total_tokens:,}")
        if self._results:
            avg = total_tokens / len(self._results)
            lines.append(f"- **Avg tokens/instance:** {avg:,.0f}")
        lines.append("")

        # Errors
        errors = [r for r in self._results if r.get("error")]
        if errors:
            lines.append(f"## Errors ({len(errors)})")
            lines.append("")
            for e in errors[:20]:
                iid = e.get("instance_id", e.get("task_id", "?"))
                lines.append(f"- `{iid}`: {e['error'][:120]}")
            lines.append("")

        return "\n".join(lines)
