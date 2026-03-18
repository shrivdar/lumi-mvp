"""Strategy Memory — cross-run learning for the orchestrator.

After each research session, extracts a strategy template capturing what
worked: which agent types were effective, tool usage patterns, MCTS branch
strategies, falsification insights, and code patterns.

Templates are stored in Postgres and loaded at session start. The top-K
most relevant templates (by query similarity) are injected into the
orchestrator's system prompt to guide task decomposition and swarm
composition.

Supports multi-trial protocol: trial 1 runs baseline, trial 2 injects
hints from trial 1, trial 3 injects both previous attempts.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

MAX_STRATEGY_INJECTION_TOKENS = 2000  # ~chars to inject into orchestrator prompt
MAX_TEMPLATES_PER_QUERY = 5


@dataclass
class StrategyTemplate:
    """A reusable strategy extracted from a completed research session."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_category: str = ""  # e.g. "target_evaluation", "mechanism_investigation"
    query_text: str = ""  # original research query
    description: str = ""  # human-readable summary of what worked

    # What worked
    effective_agent_types: list[str] = field(default_factory=list)
    effective_tool_sequence: list[str] = field(default_factory=list)
    effective_code_patterns: list[str] = field(default_factory=list)
    mcts_insight: str = ""  # e.g. "branch early on mechanism vs resistance"
    falsification_insight: str = ""  # e.g. "check counter-evidence in clinical trials"

    # Scoring
    reward_score: float = 0.0
    usage_count: int = 0
    success_rate: float = 0.0

    # Metadata
    source_session_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_category": self.task_category,
            "query_text": self.query_text,
            "description": self.description,
            "effective_agent_types": self.effective_agent_types,
            "effective_tool_sequence": self.effective_tool_sequence,
            "effective_code_patterns": self.effective_code_patterns,
            "mcts_insight": self.mcts_insight,
            "falsification_insight": self.falsification_insight,
            "reward_score": self.reward_score,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "source_session_id": self.source_session_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyTemplate:
        created = data.get("created_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        elif not isinstance(created, datetime):
            created = datetime.now(timezone.utc)
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            task_category=data.get("task_category", ""),
            query_text=data.get("query_text", ""),
            description=data.get("description", ""),
            effective_agent_types=data.get("effective_agent_types", []),
            effective_tool_sequence=data.get("effective_tool_sequence", []),
            effective_code_patterns=data.get("effective_code_patterns", []),
            mcts_insight=data.get("mcts_insight", ""),
            falsification_insight=data.get("falsification_insight", ""),
            reward_score=data.get("reward_score", 0.0),
            usage_count=data.get("usage_count", 0),
            success_rate=data.get("success_rate", 0.0),
            source_session_id=data.get("source_session_id", ""),
            created_at=created,
        )


class StrategyMemory:
    """In-memory strategy template library with optional Postgres persistence."""

    def __init__(self) -> None:
        self._templates: dict[str, StrategyTemplate] = {}

    @property
    def count(self) -> int:
        return len(self._templates)

    def add_template(self, template: StrategyTemplate) -> None:
        self._templates[template.id] = template
        logger.info(
            "strategy_template_added",
            extra={
                "template_id": template.id,
                "category": template.task_category,
                "reward": template.reward_score,
            },
        )

    def get_template(self, template_id: str) -> StrategyTemplate | None:
        return self._templates.get(template_id)

    def list_templates(self, min_reward: float = 0.0) -> list[StrategyTemplate]:
        return sorted(
            [t for t in self._templates.values() if t.reward_score >= min_reward],
            key=lambda t: t.reward_score,
            reverse=True,
        )

    def retrieve_relevant(
        self,
        query: str,
        top_k: int = MAX_TEMPLATES_PER_QUERY,
    ) -> list[StrategyTemplate]:
        """Retrieve the most relevant strategy templates for a query.

        Uses keyword overlap scoring (fast, no LLM call required).
        """
        query_words = set(query.lower().split())
        scored: list[tuple[float, StrategyTemplate]] = []
        for template in self._templates.values():
            if template.reward_score < 0.3:
                continue
            template_words = set(template.query_text.lower().split())
            template_words |= set(template.description.lower().split())
            # Split snake_case agent type names into individual words
            for t in template.effective_agent_types:
                template_words |= set(t.lower().replace("_", " ").split())
            overlap = len(query_words & template_words)
            if overlap == 0:
                continue
            # Jaccard-like score weighted by reward
            score = (overlap / max(len(query_words | template_words), 1)) * template.reward_score
            scored.append((score, template))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_k]]

    def format_for_injection(
        self,
        templates: list[StrategyTemplate],
        max_chars: int = MAX_STRATEGY_INJECTION_TOKENS,
    ) -> str:
        """Format templates for injection into the orchestrator system prompt."""
        if not templates:
            return ""

        lines = [
            "## Strategy Templates from Previous Sessions",
            "The following strategies worked well for similar queries:\n",
        ]
        total_chars = sum(len(l) for l in lines)

        for i, t in enumerate(templates, 1):
            block = [
                f"### Strategy {i}: {t.description or t.task_category}",
                f"- **Query**: {t.query_text[:100]}",
                f"- **Effective agents**: {', '.join(t.effective_agent_types[:5])}",
                f"- **Key tools**: {', '.join(t.effective_tool_sequence[:5])}",
            ]
            if t.mcts_insight:
                block.append(f"- **MCTS insight**: {t.mcts_insight[:150]}")
            if t.falsification_insight:
                block.append(f"- **Falsification insight**: {t.falsification_insight[:150]}")
            if t.effective_code_patterns:
                block.append(f"- **Code patterns**: {', '.join(t.effective_code_patterns[:3])}")
            block.append(f"- **Reward**: {t.reward_score:.2f}")
            block.append("")

            block_text = "\n".join(block)
            if total_chars + len(block_text) > max_chars:
                break
            lines.append(block_text)
            total_chars += len(block_text)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Extraction — build a template from a completed session
    # ------------------------------------------------------------------

    @staticmethod
    def extract_template(
        query: str,
        session_id: str,
        agent_results: list[dict[str, Any]],
        hypothesis_tree: dict[str, Any] | None = None,
        reward_score: float = 0.0,
        category: str = "",
    ) -> StrategyTemplate:
        """Extract a strategy template from completed session data.

        Args:
            query: Original research query.
            session_id: Session ID.
            agent_results: List of AgentResult dicts.
            hypothesis_tree: Serialized hypothesis tree (optional).
            reward_score: Composite reward score for the session.
            category: Task category (auto-inferred if empty).
        """
        # Identify successful agents
        successful = [r for r in agent_results if r.get("success", False)]
        failed = [r for r in agent_results if not r.get("success", False)]

        effective_types = list({
            r.get("agent_type", "unknown") for r in successful
        })

        # Extract tool usage patterns from successful agents
        tool_counts: dict[str, int] = {}
        for r in successful:
            for turn in r.get("turns", []):
                for tc in turn.get("tool_calls", []):
                    tool_name = tc.get("tool_name", "")
                    if tool_name:
                        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        effective_tools = sorted(tool_counts, key=tool_counts.get, reverse=True)[:10]

        # Extract code patterns (from python_repl tool calls)
        code_patterns: list[str] = []
        for r in successful:
            if len(code_patterns) >= 5:
                break
            for turn in r.get("turns", []):
                if len(code_patterns) >= 5:
                    break
                for tc in turn.get("tool_calls", []):
                    if len(code_patterns) >= 5:
                        break
                    if tc.get("tool_name") == "python_repl":
                        code = tc.get("arguments", {}).get("code", "")
                        for line in code.split("\n"):
                            line = line.strip()
                            if line.startswith("import ") or line.startswith("from "):
                                if line not in code_patterns:
                                    code_patterns.append(line)
                                    if len(code_patterns) >= 5:
                                        break

        # MCTS insight from hypothesis tree
        mcts_insight = ""
        if hypothesis_tree:
            nodes = hypothesis_tree.get("nodes", [])
            if nodes:
                best = max(nodes, key=lambda n: n.get("score", 0), default=None)
                if best:
                    mcts_insight = (
                        f"Best hypothesis scored {best.get('score', 0):.2f}: "
                        f"{best.get('hypothesis', '')[:100]}"
                    )

        # Falsification insight
        falsification_insight = ""
        total_falsified = 0
        total_edges = 0
        for r in agent_results:
            fr = r.get("falsification_results", [])
            total_falsified += sum(1 for f in fr if f.get("counter_evidence_found"))
            total_edges += len(r.get("edges_added", []))
        if total_edges > 0:
            ratio = total_falsified / total_edges
            if ratio > 0.3:
                falsification_insight = f"High falsification rate ({ratio:.0%}) — strengthen evidence before asserting"
            elif ratio > 0:
                falsification_insight = f"Some counter-evidence found ({total_falsified}/{total_edges} edges falsified)"

        # Auto-categorize
        if not category:
            q_lower = query.lower()
            if any(w in q_lower for w in ["target", "inhibitor", "agonist", "antagonist"]):
                category = "target_evaluation"
            elif any(w in q_lower for w in ["mechanism", "pathway", "signal"]):
                category = "mechanism_investigation"
            elif any(w in q_lower for w in ["drug", "compound", "therapeutic"]):
                category = "drug_analysis"
            elif any(w in q_lower for w in ["safety", "toxicity", "adverse"]):
                category = "safety_assessment"
            elif any(w in q_lower for w in ["resistance", "mutation"]):
                category = "resistance_analysis"
            else:
                category = "general_research"

        # Build description
        desc_parts = []
        if effective_types:
            desc_parts.append(f"Used {', '.join(effective_types[:3])}")
        if len(successful) > 0 and len(failed) > 0:
            desc_parts.append(f"{len(successful)}/{len(agent_results)} agents succeeded")
        if mcts_insight:
            desc_parts.append("with MCTS hypothesis exploration")
        description = "; ".join(desc_parts) if desc_parts else f"{category} strategy"

        return StrategyTemplate(
            task_category=category,
            query_text=query,
            description=description,
            effective_agent_types=effective_types,
            effective_tool_sequence=effective_tools,
            effective_code_patterns=code_patterns,
            mcts_insight=mcts_insight,
            falsification_insight=falsification_insight,
            reward_score=reward_score,
            source_session_id=session_id,
        )

    # ------------------------------------------------------------------
    # Multi-trial protocol
    # ------------------------------------------------------------------

    def build_trial_hint(
        self,
        previous_results: list[dict[str, Any]],
        trial_number: int,
    ) -> str:
        """Build a hint string from previous trial results for multi-trial protocol.

        Args:
            previous_results: List of result dicts from previous trials.
            trial_number: Current trial number (2 = has 1 previous, 3 = has 2 previous).

        Returns:
            Hint string to inject into the orchestrator prompt.
        """
        if not previous_results:
            return ""

        lines = [
            f"## Hints from Previous Trial{'s' if len(previous_results) > 1 else ''}\n"
        ]

        for i, result in enumerate(previous_results, 1):
            lines.append(f"### Trial {i} Result")
            answer = result.get("answer", result.get("final_answer", ""))
            if answer:
                lines.append(f"- **Answer**: {str(answer)[:200]}")
            score = result.get("score", result.get("reward", None))
            if score is not None:
                lines.append(f"- **Score**: {score}")

            # What worked
            successful_agents = [
                r.get("agent_type", "?")
                for r in result.get("agent_results", [])
                if r.get("success")
            ]
            if successful_agents:
                lines.append(f"- **Successful agents**: {', '.join(successful_agents[:5])}")

            # What failed
            failed_agents = [
                r.get("agent_type", "?")
                for r in result.get("agent_results", [])
                if not r.get("success")
            ]
            if failed_agents:
                lines.append(f"- **Failed agents**: {', '.join(failed_agents[:5])}")

            # Key findings
            findings = result.get("key_findings", [])
            if findings:
                lines.append(f"- **Key findings**: {'; '.join(str(f)[:100] for f in findings[:3])}")

            lines.append("")

        lines.append(
            "Use these previous results to improve your approach. "
            "Try different strategies for aspects that failed. "
            "Build on what worked."
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence (JSON file-based, upgradeable to Postgres)
    # ------------------------------------------------------------------

    def save_to_file(self, path: str) -> None:
        """Save all templates to a JSON file."""
        data = [t.to_dict() for t in self._templates.values()]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("strategy_memory_saved", extra={"path": path, "count": len(data)})

    def load_from_file(self, path: str) -> int:
        """Load templates from a JSON file. Returns count loaded."""
        try:
            with open(path) as f:
                data = json.load(f)
            for item in data:
                template = StrategyTemplate.from_dict(item)
                self._templates[template.id] = template
            logger.info("strategy_memory_loaded", extra={"path": path, "count": len(data)})
            return len(data)
        except FileNotFoundError:
            logger.info("strategy_memory_file_not_found", extra={"path": path})
            return 0
        except Exception as exc:
            logger.warning("strategy_memory_load_error", extra={"path": path, "error": str(exc)})
            return 0
