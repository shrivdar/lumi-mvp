"""Experiment Designer agent — reasoning-only agent that designs experiments."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgentImpl
from core.models import AgentTask, AgentType, KGEdge


class ExperimentDesignerAgent(BaseAgentImpl):
    """Reasoning-only agent — designs experiments to resolve KG uncertainties.

    No external tools. Analyzes KG state and proposes the highest-value
    experiment to resolve the biggest uncertainty.
    """

    agent_type = AgentType.EXPERIMENT_DESIGNER

    async def _investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._multi_turn_investigate(
            task,
            kg_context,
            investigation_focus=(
                "You are an experiment designer. Analyze KG uncertainties to design "
                "the most informative experiment.\n"
                "1. Use kg_get_weakest_edges to find the highest-uncertainty edges.\n"
                "2. Use kg_get_orphan_nodes to find unconnected nodes.\n"
                "3. Design a single experiment to resolve the biggest knowledge gap.\n"
                "4. Output an EXPERIMENT entity with these properties:\n"
                '   - experiment_type: "in_vitro|in_vivo|in_silico|clinical|observational"\n'
                "   - hypothesis: specific testable hypothesis\n"
                "   - rationale: why this resolves the biggest uncertainty\n"
                "   - expected_outcome_positive / expected_outcome_negative\n"
                "   - methods: list of techniques\n"
                "   - materials: list of required materials\n"
                "   - timeline_weeks: estimated duration\n"
                "   - success_criteria: measurable endpoint\n"
                "   - information_gain_estimate: 0.0-1.0\n"
                "   - feasibility_score: 0.0-1.0\n"
                "5. Also output ASSOCIATED_WITH relationships linking the experiment "
                "to the KG nodes it addresses.\n"
                "You propose experiments — you do NOT assert biological claims."
            ),
        )

    async def falsify(self, edges: list[KGEdge]) -> list:
        """Experiment designer does not perform falsification — it proposes, doesn't assert."""
        return []
