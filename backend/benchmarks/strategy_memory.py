"""Strategy memory — extracts and stores reusable research strategy templates.

After each research run completes, `extract_template()` distills the approach
(agent composition, tool usage, hypothesis structure) into a named template.
On subsequent trials, `get_hint()` produces a concise strategy summary that
can be injected into the system prompt to guide the agent toward proven patterns.

Multi-trial protocol:
  Trial 1: baseline (no hints)
  Trial 2: inject strategy summary from trial 1
  Trial 3: inject summaries from trials 1 + 2
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StrategyTemplate(BaseModel):
    """A reusable research strategy template extracted from a completed run."""

    name: str
    query_archetype: str = ""  # e.g. "drug_resistance", "pathway_analysis"
    description: str = ""
    agent_types_used: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    hypothesis_count: int = 0
    iterations_used: int = 0
    key_findings_count: int = 0
    score: float = 0.0
    reasoning_summary: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class TrialSummary:
    """Summary of a single trial's results for injection into subsequent trials."""

    trial_number: int
    predicted: str = ""
    score: float = 0.0
    reasoning_trace: str = ""
    tools_used: list[str] = field(default_factory=list)
    tokens_used: int = 0
    key_insights: str = ""


class StrategyMemory:
    """Stores and retrieves research strategy templates for multi-trial injection."""

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self._templates: list[StrategyTemplate] = []
        self._storage_dir = Path(storage_dir) if storage_dir else None
        if self._storage_dir:
            self._storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_templates()

    @property
    def templates(self) -> list[StrategyTemplate]:
        return list(self._templates)

    def extract_template(
        self,
        *,
        name: str,
        query: str,
        result: Any,
        score: float = 0.0,
        tools_used: list[str] | None = None,
        agent_types: list[str] | None = None,
    ) -> StrategyTemplate:
        """Extract a strategy template from a completed research run.

        Args:
            name: Human-readable name for the template.
            query: The original research query.
            result: The ResearchResult or InstanceResult from the run.
            score: Evaluation score (0.0–1.0).
            tools_used: Tools that were used during the run.
            agent_types: Agent types that participated.

        Returns:
            The extracted StrategyTemplate.
        """
        # Classify the query archetype
        archetype = self._classify_archetype(query)

        # Extract reasoning summary from result
        reasoning = ""
        key_findings_count = 0
        iterations = 0
        if hasattr(result, "report_markdown") and result.report_markdown:
            reasoning = self._summarize_reasoning(result.report_markdown)
        elif hasattr(result, "reasoning_trace") and result.reasoning_trace:
            reasoning = self._summarize_reasoning(result.reasoning_trace)

        if hasattr(result, "key_findings"):
            key_findings_count = len(result.key_findings) if result.key_findings else 0
        if hasattr(result, "total_iterations"):
            iterations = result.total_iterations

        template = StrategyTemplate(
            name=name,
            query_archetype=archetype,
            description=f"Strategy for: {query[:100]}",
            agent_types_used=agent_types or [],
            tools_used=tools_used or [],
            hypothesis_count=0,
            iterations_used=iterations,
            key_findings_count=key_findings_count,
            score=score,
            reasoning_summary=reasoning,
        )

        self._templates.append(template)

        if self._storage_dir:
            self._save_template(template)

        logger.info("Extracted strategy template: %s (archetype=%s, score=%.2f)", name, archetype, score)
        return template

    def get_hint(self, trial_summaries: list[TrialSummary]) -> str:
        """Build a hint string from prior trial summaries for injection into system prompt.

        Args:
            trial_summaries: Summaries of all prior trials.

        Returns:
            A formatted hint string for the system prompt.
        """
        if not trial_summaries:
            return ""

        parts = ["## Prior Trial Results\n"]
        parts.append("Use these prior attempts to improve your answer. "
                      "Build on what worked and avoid repeating mistakes.\n")

        for ts in trial_summaries:
            parts.append(f"### Trial {ts.trial_number}")
            if ts.predicted:
                parts.append(f"- **Answer**: {ts.predicted[:200]}")
            parts.append(f"- **Score**: {ts.score:.2f}")
            if ts.tools_used:
                parts.append(f"- **Tools used**: {', '.join(ts.tools_used[:10])}")
            if ts.key_insights:
                parts.append(f"- **Key insights**: {ts.key_insights[:500]}")
            elif ts.reasoning_trace:
                # Extract first meaningful paragraph
                summary = self._summarize_reasoning(ts.reasoning_trace)
                if summary:
                    parts.append(f"- **Reasoning summary**: {summary[:500]}")
            parts.append("")

        # Add strategy templates if available
        relevant = self._find_relevant_templates(trial_summaries)
        if relevant:
            parts.append("### Proven Strategies")
            for t in relevant[:3]:
                parts.append(f"- **{t.name}** (score={t.score:.2f}): {t.reasoning_summary[:200]}")
                if t.tools_used:
                    parts.append(f"  Tools: {', '.join(t.tools_used[:5])}")
            parts.append("")

        return "\n".join(parts)

    def get_best_template(self, archetype: str | None = None) -> StrategyTemplate | None:
        """Return the highest-scoring template, optionally filtered by archetype."""
        candidates = self._templates
        if archetype:
            candidates = [t for t in candidates if t.query_archetype == archetype]
        if not candidates:
            return None
        return max(candidates, key=lambda t: t.score)

    # ------------------------------------------------------------------
    # Archetype classification
    # ------------------------------------------------------------------

    _ARCHETYPE_KEYWORDS: dict[str, list[str]] = {
        "drug_resistance": ["resistance", "TKI", "inhibitor resistance", "PARP inhibitor"],
        "drug_design": ["PROTAC", "degrader", "ADC", "payload", "conjugate", "agonist"],
        "pathway_analysis": ["pathway", "signaling", "checkpoint", "SIRPα", "STING", "IL-17"],
        "gene_editing": ["CRISPR", "base editing", "off-target", "RNAi", "siRNA"],
        "immunotherapy": ["CAR-T", "bispecific", "PD-1", "CD47", "TIGIT", "immune"],
        "cancer_biology": ["ferroptosis", "synthetic lethality", "epigenetic", "AML"],
        "protein_science": ["tau", "propagation", "misfolding", "aggregation"],
        "target_discovery": ["KRAS", "EGFR", "PCSK9", "GLP-1", "target"],
        "microbiome": ["microbiome", "gut", "commensal"],
    }

    def _classify_archetype(self, query: str) -> str:
        query_lower = query.lower()
        best_archetype = "general"
        best_count = 0
        for archetype, keywords in self._ARCHETYPE_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw.lower() in query_lower)
            if count > best_count:
                best_count = count
                best_archetype = archetype
        return best_archetype

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize_reasoning(text: str, max_length: int = 300) -> str:
        """Extract first meaningful paragraph from reasoning text."""
        if not text:
            return ""
        # Skip markdown headers and blank lines
        lines = text.strip().split("\n")
        paragraphs: list[str] = []
        current: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
            elif stripped.startswith("#"):
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
            else:
                current.append(stripped)
        if current:
            paragraphs.append(" ".join(current))

        # Return first non-trivial paragraph
        for p in paragraphs:
            if len(p) > 20:
                return p[:max_length]
        return text[:max_length]

    def _find_relevant_templates(self, trial_summaries: list[TrialSummary]) -> list[StrategyTemplate]:
        """Find templates relevant to the current trial context."""
        if not self._templates:
            return []
        # Return top templates by score
        return sorted(self._templates, key=lambda t: t.score, reverse=True)[:3]

    def _save_template(self, template: StrategyTemplate) -> None:
        """Persist template to storage directory."""
        if not self._storage_dir:
            return
        path = self._storage_dir / f"{template.name.replace(' ', '_').lower()}.json"
        path.write_text(template.model_dump_json(indent=2))

    def _load_templates(self) -> None:
        """Load templates from storage directory."""
        if not self._storage_dir or not self._storage_dir.exists():
            return
        for path in self._storage_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                self._templates.append(StrategyTemplate.model_validate(data))
            except Exception:
                logger.warning("Failed to load strategy template: %s", path)
