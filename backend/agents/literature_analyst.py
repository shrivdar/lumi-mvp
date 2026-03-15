"""Literature Analyst agent — extracts biological claims from publications."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgentImpl
from core.models import AgentTask, AgentType


class LiteratureAnalystAgent(BaseAgentImpl):
    """Searches PubMed and Semantic Scholar, extracts biological relationships as KG edges."""

    agent_type = AgentType.LITERATURE_ANALYST

    async def _investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._multi_turn_investigate(
            task,
            kg_context,
            investigation_focus=(
                "Search PubMed and Semantic Scholar for papers relevant to the research question. "
                "Extract biological entities (genes, proteins, diseases, pathways, drugs, biomarkers) "
                "and their relationships from paper abstracts. Every claim must cite a PMID or DOI. "
                "Assess evidence quality based on journal, citation count, and study design."
            ),
        )
