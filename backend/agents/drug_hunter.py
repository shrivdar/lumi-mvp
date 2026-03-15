"""Drug Hunter agent — finds drugs/compounds targeting KG entities."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgentImpl
from core.models import AgentTask, AgentType


class DrugHunterAgent(BaseAgentImpl):
    """Finds drugs and compounds targeting proteins/genes, writes DRUG nodes."""

    agent_type = AgentType.DRUG_HUNTER

    async def _investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._multi_turn_investigate(
            task,
            kg_context,
            investigation_focus=(
                "Search ChEMBL for compounds targeting relevant proteins/genes. "
                "Search ClinicalTrials.gov for trial data. Create DRUG nodes "
                "(max_phase ≥ 1) or COMPOUND nodes (preclinical) with ChEMBL IDs. "
                "Create CLINICAL_TRIAL nodes with NCT IDs. Build TARGETS, TREATS, "
                "and SIDE_EFFECT_OF edges with binding affinity and mechanism data."
            ),
        )
