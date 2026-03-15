"""Scientific Critic agent — systematic falsification of KG edges.

This agent evaluates KG edges for validity by searching for counter-evidence.
It can only modify confidence scores, add EVIDENCE_AGAINST edges, and add
PUBLICATION nodes for counter-evidence papers.
"""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgentImpl
from core.models import AgentTask, AgentType


class ScientificCriticAgent(BaseAgentImpl):
    """Systematically falsifies KG edges — the skeptic in the swarm.

    Unlike other agents, the critic does NOT add new biological claims.
    It can only:
    - Modify confidence scores on existing edges (via kg_update_edge_confidence)
    - Add EVIDENCE_AGAINST edges
    - Add PUBLICATION nodes for counter-evidence papers
    """

    agent_type = AgentType.SCIENTIFIC_CRITIC

    async def _investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._multi_turn_investigate(
            task,
            kg_context,
            investigation_focus=(
                "You are a scientific critic. Your job is to CHALLENGE existing KG edges, "
                "NOT add new biological claims.\n"
                "1. Use kg_get_recent_edges and kg_get_weakest_edges to find edges to evaluate.\n"
                "2. For each suspicious or weak edge, formulate what would disprove it.\n"
                "3. Search PubMed/Semantic Scholar for counter-evidence.\n"
                "4. If counter-evidence is found, use kg_update_edge_confidence to lower confidence.\n"
                "5. Only output PUBLICATION nodes (for counter-evidence papers) and EVIDENCE_AGAINST edges.\n"
                "6. If no counter-evidence found, slightly increase confidence (survived falsification).\n"
                "Be thorough but fair — distinguish genuine refutation from weak counter-evidence."
            ),
        )
