"""Protein Engineer agent — structural biology via UniProt + ESM/Yami."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgentImpl
from core.models import AgentTask, AgentType


class ProteinEngineerAgent(BaseAgentImpl):
    """Fetches protein data from UniProt, predicts structure via ESMFold/Yami."""

    agent_type = AgentType.PROTEIN_ENGINEER

    async def _investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._multi_turn_investigate(
            task,
            kg_context,
            investigation_focus=(
                "Look up protein sequences and annotations from UniProt. "
                "Predict protein structure using Yami (ESMFold) for sequences ≤400 residues. "
                "Identify functional domains, active sites, and binding interfaces. "
                "Map protein-protein interactions. Include pLDDT scores as confidence metrics. "
                "Create PROTEIN and STRUCTURE nodes with UniProt accessions as external_ids."
            ),
        )
