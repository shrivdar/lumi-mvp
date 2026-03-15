"""Genomics Mapper agent — maps genes to pathways and expression patterns."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgentImpl
from core.models import AgentTask, AgentType


class GenomicsMapperAgent(BaseAgentImpl):
    """Maps genes to pathways and expression patterns via MyGene and KEGG."""

    agent_type = AgentType.GENOMICS_MAPPER

    async def _investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._multi_turn_investigate(
            task,
            kg_context,
            investigation_focus=(
                "Identify relevant genes, then look up gene info from MyGene and "
                "map them to KEGG pathways. Create GENE nodes with Entrez/Ensembl IDs "
                "and PATHWAY nodes with KEGG IDs. Identify regulatory relationships "
                "(upregulates, downregulates, activates, inhibits) between genes."
            ),
        )
