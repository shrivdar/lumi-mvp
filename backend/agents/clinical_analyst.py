"""Clinical Analyst agent — clinical trial search, outcome analysis, failure analysis."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgentImpl
from core.models import AgentTask, AgentType


class ClinicalAnalystAgent(BaseAgentImpl):
    """Searches clinical trials, reports outcomes and failure analyses."""

    agent_type = AgentType.CLINICAL_ANALYST

    async def _investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._multi_turn_investigate(
            task,
            kg_context,
            investigation_focus=(
                "Search ClinicalTrials.gov and PubMed for relevant clinical trials. "
                "Analyze trial design, endpoints, and outcomes. Perform failure analysis "
                "on terminated or withdrawn trials. Create CLINICAL_TRIAL nodes with "
                "NCT IDs, phase, status, and enrollment. Build TREATS, EVIDENCE_FOR, "
                "and EVIDENCE_AGAINST edges linking trials to conditions."
            ),
        )
