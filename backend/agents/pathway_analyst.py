"""Pathway Analyst agent — deep pathway analysis and signaling cascade mapping."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgentImpl
from core.models import AgentTask, AgentType


class PathwayAnalystAgent(BaseAgentImpl):
    """Deep pathway analysis — KEGG + Reactome for signaling cascades and cross-talk."""

    agent_type = AgentType.PATHWAY_ANALYST

    async def _investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._multi_turn_investigate(
            task,
            kg_context,
            investigation_focus=(
                "Retrieve detailed pathway information from KEGG and Reactome. "
                "Map signaling cascades and enzymatic reaction chains. "
                "Identify pathway cross-talk and shared components between pathways. "
                "Create PATHWAY nodes with KEGG/Reactome IDs and GENE nodes for members. "
                "Build MEMBER_OF, UPSTREAM_OF, DOWNSTREAM_OF, and REGULATES edges."
            ),
        )
