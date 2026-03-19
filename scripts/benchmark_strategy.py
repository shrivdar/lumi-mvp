"""Within-run strategy evolution for benchmark runners.

Tracks question outcomes and builds running strategies that get injected
into subsequent LLM prompts. The system learns what works (code execution,
specific databases, answer patterns) and adapts mid-run.

Lightweight: no extra LLM calls. Strategy is built from simple pattern
analysis of outcomes so far.

Usage:
    tracker = BenchmarkStrategyTracker()

    # After each question:
    tracker.record(QuestionOutcome(...))

    # Before each question:
    strategy_text = tracker.get_strategy_injection()
    escalate = tracker.should_escalate(subtask)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("benchmark_strategy")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QuestionOutcome:
    """Record of a single question's outcome for strategy learning."""

    subtask: str
    question_type: str  # detected from question text (e.g., dga_task, gene_location)
    predicted: str
    correct: str
    is_correct: bool
    reasoning_summary: str  # brief summary of the reasoning approach
    tools_used: list[str] = field(default_factory=list)
    databases_queried: list[str] = field(default_factory=list)
    code_executed: bool = False
    question_index: int = 0  # position in the run


# ---------------------------------------------------------------------------
# Strategy analysis (no LLM calls -- pure pattern matching)
# ---------------------------------------------------------------------------


def _group_by(items: list[QuestionOutcome], key: str) -> dict[str, list[QuestionOutcome]]:
    """Group outcomes by a field value."""
    groups: dict[str, list[QuestionOutcome]] = {}
    for item in items:
        val = getattr(item, key, "unknown")
        groups.setdefault(val, []).append(item)
    return groups


def build_running_strategy(outcomes: list[QuestionOutcome]) -> str:
    """Analyze what's working and what's failing.

    Returns a compact strategy string (<500 tokens) for prompt injection.
    """
    if len(outcomes) < 5:
        return ""

    correct = [o for o in outcomes if o.is_correct]
    wrong = [o for o in outcomes if not o.is_correct]

    if not correct and not wrong:
        return ""

    strategy: list[str] = []
    total = len(outcomes)
    acc = len(correct) / total

    # -- Overall accuracy signal --
    strategy.append(
        f"Overall accuracy so far: {len(correct)}/{total} ({acc:.0%})"
    )

    # -- Code execution correlation --
    code_correct = len([o for o in correct if o.code_executed])
    code_wrong = len([o for o in wrong if o.code_executed])
    nocode_correct = len([o for o in correct if not o.code_executed])
    nocode_wrong = len([o for o in wrong if not o.code_executed])

    code_total = code_correct + code_wrong
    nocode_total = nocode_correct + nocode_wrong

    if code_total >= 3 and nocode_total >= 3:
        code_acc = code_correct / code_total
        nocode_acc = nocode_correct / nocode_total
        if code_acc > nocode_acc + 0.1:
            strategy.append(
                f"Code execution improves accuracy ({code_acc:.0%} vs {nocode_acc:.0%} without) "
                "-- always try to verify with code"
            )
        elif nocode_acc > code_acc + 0.1:
            strategy.append(
                f"Direct reasoning outperforms code ({nocode_acc:.0%} vs {code_acc:.0%} with code) "
                "-- prefer reasoning from knowledge when confident"
            )

    # -- Database correlation --
    db_names = ["uniprot", "ncbi", "kegg", "ensembl", "clinvar", "pubmed",
                "disgenet", "omim", "mirtarbase", "gtrd", "msigdb", "string"]
    for db in db_names:
        db_correct = len([o for o in correct if db in str(o.databases_queried).lower()])
        db_wrong = len([o for o in wrong if db in str(o.databases_queried).lower()])
        db_total = db_correct + db_wrong
        if db_total >= 3:
            db_acc = db_correct / db_total
            if db_acc >= 0.7:
                strategy.append(
                    f"Querying {db} tends to lead to correct answers ({db_correct}/{db_total})"
                )
            elif db_acc <= 0.3:
                strategy.append(
                    f"Querying {db} has low success rate ({db_correct}/{db_total}) "
                    "-- results may be misleading, cross-verify"
                )

    # -- Refusal / insufficient info pattern --
    refused = [
        o for o in outcomes
        if "insufficient" in o.predicted.lower()
        or o.predicted.upper() in ("D", "E")
        and "insufficient" in o.reasoning_summary.lower()
    ]
    if len(refused) >= 3:
        refused_wrong = sum(1 for r in refused if not r.is_correct)
        if refused_wrong / len(refused) > 0.7:
            strategy.append(
                f"Choosing 'Insufficient information' is usually wrong "
                f"({refused_wrong}/{len(refused)} incorrect) -- commit to a specific answer"
            )

    # -- Per-subtask accuracy --
    by_subtask = _group_by(outcomes, "subtask")
    subtask_notes: list[str] = []
    for st, items in sorted(by_subtask.items()):
        if len(items) < 3:
            continue
        st_correct = sum(1 for o in items if o.is_correct)
        st_acc = st_correct / len(items)
        if st_acc >= 0.7:
            subtask_notes.append(f"{st}: {st_acc:.0%} accuracy (strong)")
        elif st_acc <= 0.3:
            subtask_notes.append(
                f"{st}: {st_acc:.0%} accuracy (weak -- use more careful analysis)"
            )

    if subtask_notes:
        strategy.append("Per-subtask performance: " + "; ".join(subtask_notes))

    # -- Question type patterns --
    by_qtype = _group_by(outcomes, "question_type")
    for qt, items in sorted(by_qtype.items()):
        if len(items) < 3 or qt == "unknown":
            continue
        qt_correct = sum(1 for o in items if o.is_correct)
        qt_acc = qt_correct / len(items)
        if qt_acc <= 0.3:
            strategy.append(
                f"'{qt}' questions have low accuracy ({qt_correct}/{len(items)}) "
                "-- try a different approach for these"
            )

    # -- Recent trend (last 10 vs prior) --
    if len(outcomes) >= 20:
        recent = outcomes[-10:]
        prior = outcomes[:-10]
        recent_acc = sum(1 for o in recent if o.is_correct) / len(recent)
        prior_acc = sum(1 for o in prior if o.is_correct) / len(prior)
        if recent_acc > prior_acc + 0.15:
            strategy.append(
                f"Improving: recent accuracy {recent_acc:.0%} vs earlier {prior_acc:.0%} "
                "-- current approach is working well"
            )
        elif recent_acc < prior_acc - 0.15:
            strategy.append(
                f"Declining: recent accuracy {recent_acc:.0%} vs earlier {prior_acc:.0%} "
                "-- consider reverting to earlier strategies"
            )

    if not strategy:
        return ""

    return "\n".join(f"- {s}" for s in strategy)


# ---------------------------------------------------------------------------
# Detect question metadata from text/result
# ---------------------------------------------------------------------------


def detect_question_type(question_text: str, subtask: str = "") -> str:
    """Detect question type from the question text.

    Returns a short label like 'dga_task', 'gene_location', 'numeric', etc.
    """
    text_lower = question_text.lower()

    # LAB-Bench subtask types
    if "disgenet" in text_lower and "omim" in text_lower:
        return "dga_task"
    if "chromosom" in text_lower or "cytogenetic" in text_lower or "map location" in text_lower:
        return "gene_location"
    if "mirna" in text_lower or "microrna" in text_lower or "mir-" in text_lower:
        return "mirna_targets"
    if "oncogenic" in text_lower or "signature" in text_lower:
        return "oncogenic_signatures"
    if "transcription factor" in text_lower or "chip-seq" in text_lower or "tfbs" in text_lower:
        return "tfbs"
    if "variant" in text_lower or "mutation" in text_lower or "clinvar" in text_lower:
        return "variant"
    if "interaction" in text_lower and ("viral" in text_lower or "virus" in text_lower):
        return "viral_ppi"
    if "vaccine" in text_lower or "immune response" in text_lower:
        return "vax_response"
    if "sequence" in text_lower and ("reverse complement" in text_lower or "translate" in text_lower):
        return "sequence_manipulation"

    # BixBench types
    if "p-value" in text_lower or "statistical" in text_lower or "significance" in text_lower:
        return "statistical_test"
    if "correlation" in text_lower or "pearson" in text_lower or "spearman" in text_lower:
        return "correlation"
    if "differential" in text_lower and "expression" in text_lower:
        return "diff_expression"
    if "cluster" in text_lower or "pca" in text_lower or "dimensionality" in text_lower:
        return "clustering"
    if "gene" in text_lower and ("name" in text_lower or "symbol" in text_lower):
        return "gene_identification"

    # Fall back to subtask if provided
    if subtask:
        return subtask

    return "unknown"


def detect_databases_from_text(text: str) -> list[str]:
    """Detect which databases were likely queried based on reasoning text."""
    text_lower = text.lower()
    databases = []
    db_keywords = {
        "uniprot": ["uniprot", "uniprotkb"],
        "ncbi": ["ncbi", "entrez", "eutils", "pubmed"],
        "kegg": ["kegg"],
        "ensembl": ["ensembl"],
        "clinvar": ["clinvar"],
        "disgenet": ["disgenet"],
        "omim": ["omim"],
        "mirtarbase": ["mirtarbase"],
        "gtrd": ["gtrd"],
        "msigdb": ["msigdb"],
        "string": ["string-db", "string_db"],
        "pubmed": ["pubmed"],
        "semantic_scholar": ["semantic scholar", "semanticscholar"],
    }
    for db, keywords in db_keywords.items():
        if any(kw in text_lower for kw in keywords):
            databases.append(db)
    return databases


# ---------------------------------------------------------------------------
# Main tracker class
# ---------------------------------------------------------------------------


class BenchmarkStrategyTracker:
    """Tracks question outcomes and builds running strategies for prompt injection.

    Lightweight: no LLM calls. Strategy is rebuilt every `rebuild_interval`
    questions from simple pattern analysis.

    Persists to a JSON file so interrupted runs can resume with learned strategies.
    """

    def __init__(
        self,
        persist_path: Path | None = None,
        rebuild_interval: int = 10,
    ) -> None:
        self.outcomes: list[QuestionOutcome] = []
        self.strategy_text: str = ""
        self.subtask_accuracy: dict[str, float] = {}
        self.persist_path = persist_path
        self.rebuild_interval = rebuild_interval

        # Load persisted state if available
        if persist_path and persist_path.exists():
            self._load_persisted()

    def record(self, outcome: QuestionOutcome) -> None:
        """Record a question outcome and rebuild strategy if needed."""
        outcome.question_index = len(self.outcomes)
        self.outcomes.append(outcome)

        # Rebuild strategy every N questions
        if len(self.outcomes) % self.rebuild_interval == 0:
            self.strategy_text = build_running_strategy(self.outcomes)
            self._update_subtask_accuracy()
            logger.info(
                "Strategy rebuilt after %d questions (strategy length: %d chars)",
                len(self.outcomes),
                len(self.strategy_text),
            )
            if self.strategy_text:
                logger.info("Current strategy:\n%s", self.strategy_text)

        # Persist after every question
        if self.persist_path:
            self._save_persisted()

    def get_strategy_injection(self) -> str:
        """Get the strategy text for injection into a prompt.

        Returns empty string if no strategy has been built yet.
        The injection is kept under ~500 tokens.
        """
        if not self.strategy_text:
            return ""

        injection = (
            "\n\nBased on performance patterns observed so far in this benchmark run:\n"
            f"{self.strategy_text}\n"
            "Apply these insights to improve your accuracy on this question.\n"
        )

        # Hard cap at ~2000 chars (~500 tokens) to stay within budget
        if len(injection) > 2000:
            injection = injection[:1950] + "\n...\n"

        return injection

    def should_escalate(self, subtask: str) -> bool:
        """Should this subtask use more compute (agentic mode)?

        Returns True if subtask accuracy is below 30% after enough samples.
        """
        acc = self.subtask_accuracy.get(subtask)
        if acc is None:
            return False  # not enough data
        return acc < 0.3

    def should_use_zero_shot(self, subtask: str) -> bool:
        """Should this subtask use less compute (zero-shot mode)?

        Returns True if subtask accuracy is above 60% after enough samples.
        """
        acc = self.subtask_accuracy.get(subtask)
        if acc is None:
            return False
        return acc > 0.6

    def get_subtask_accuracy(self, subtask: str) -> float | None:
        """Get current accuracy for a subtask, or None if insufficient data."""
        return self.subtask_accuracy.get(subtask)

    def get_outcome_count(self) -> int:
        """Return total number of recorded outcomes."""
        return len(self.outcomes)

    def _update_subtask_accuracy(self) -> None:
        """Update per-subtask accuracy from all recorded outcomes."""
        by_subtask = _group_by(self.outcomes, "subtask")
        self.subtask_accuracy = {}
        for st, items in by_subtask.items():
            if len(items) >= 5:  # need enough samples
                self.subtask_accuracy[st] = (
                    sum(1 for o in items if o.is_correct) / len(items)
                )

    def _save_persisted(self) -> None:
        """Save state to JSON file for resumption."""
        if not self.persist_path:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "outcomes": [asdict(o) for o in self.outcomes],
                "strategy_text": self.strategy_text,
                "subtask_accuracy": self.subtask_accuracy,
            }
            self.persist_path.write_text(json.dumps(data, indent=2, default=str))
        except Exception as exc:
            logger.warning("Failed to persist strategy state: %s", exc)

    def _load_persisted(self) -> None:
        """Load state from persisted JSON file."""
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            data = json.loads(self.persist_path.read_text())
            self.outcomes = [
                QuestionOutcome(**o) for o in data.get("outcomes", [])
            ]
            self.strategy_text = data.get("strategy_text", "")
            self.subtask_accuracy = data.get("subtask_accuracy", {})
            logger.info(
                "Loaded %d outcomes from persisted strategy state at %s",
                len(self.outcomes),
                self.persist_path,
            )
        except Exception as exc:
            logger.warning("Failed to load persisted strategy state: %s", exc)
            self.outcomes = []
            self.strategy_text = ""
            self.subtask_accuracy = {}
