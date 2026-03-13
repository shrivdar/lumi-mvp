"""ClinicalTrials.gov v2 API client — trial search and detail retrieval."""

from __future__ import annotations

from typing import Any

from core.constants import CACHE_TTL_CLINICALTRIALS, RATE_LIMIT_CLINICALTRIALS
from integrations.base_tool import BaseTool

CT_BASE = "https://clinicaltrials.gov/api/v2"


class ClinicalTrialsTool(BaseTool):
    tool_id = "clinicaltrials"
    name = "clinicaltrials_search"
    description = (
        "Search ClinicalTrials.gov for clinical trials with status, phases, interventions, and outcomes."
    )
    category = "clinical"
    rate_limit = RATE_LIMIT_CLINICALTRIALS
    cache_ttl = CACHE_TTL_CLINICALTRIALS

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "search")
        if action == "search":
            return await self._search(
                query=kwargs["query"],
                max_results=kwargs.get("max_results", 20),
                status=kwargs.get("status"),
                phase=kwargs.get("phase"),
            )
        elif action == "study":
            return await self._get_study(nct_id=kwargs["nct_id"])
        raise ValueError(f"Unknown ClinicalTrials action: {action}")

    async def _search(
        self,
        query: str,
        max_results: int = 20,
        status: str | None = None,
        phase: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "query.term": query,
            "pageSize": min(max_results, 100),
            "format": "json",
        }
        if status:
            params["filter.overallStatus"] = status
        if phase:
            params["filter.phase"] = phase

        resp = await self._http.get(f"{CT_BASE}/studies", params=params)
        resp.raise_for_status()
        data = resp.json()
        studies = [self._normalize(s) for s in data.get("studies", [])]
        return {
            "trials": studies,
            "total_count": data.get("totalCount", len(studies)),
            "query": query,
        }

    async def _get_study(self, nct_id: str) -> dict[str, Any]:
        resp = await self._http.get(f"{CT_BASE}/studies/{nct_id}", params={"format": "json"})
        resp.raise_for_status()
        return {"study": self._normalize(resp.json())}

    @staticmethod
    def _normalize(study: dict[str, Any]) -> dict[str, Any]:
        proto = study.get("protocolSection", {})
        id_mod = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design_mod = proto.get("designModule", {})
        arms_mod = proto.get("armsInterventionsModule", {})
        cond_mod = proto.get("conditionsModule", {})
        desc_mod = proto.get("descriptionModule", {})
        outcomes_mod = proto.get("outcomesModule", {})
        enroll_mod = design_mod.get("enrollmentInfo", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
        eligibility_mod = proto.get("eligibilityModule", {})

        # Interventions
        interventions: list[dict[str, str]] = []
        for intervention in arms_mod.get("interventions", []):
            interventions.append({
                "name": intervention.get("name", ""),
                "type": intervention.get("type", ""),
                "description": intervention.get("description", ""),
            })

        # Primary outcomes
        primary_outcomes: list[dict[str, str]] = []
        for outcome in outcomes_mod.get("primaryOutcomes", []):
            primary_outcomes.append({
                "measure": outcome.get("measure", ""),
                "description": outcome.get("description", ""),
                "time_frame": outcome.get("timeFrame", ""),
            })

        phases = design_mod.get("phases", [])
        lead_sponsor = sponsor_mod.get("leadSponsor", {})

        return {
            "nct_id": id_mod.get("nctId", ""),
            "title": id_mod.get("officialTitle", id_mod.get("briefTitle", "")),
            "brief_title": id_mod.get("briefTitle", ""),
            "status": status_mod.get("overallStatus", ""),
            "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
            "completion_date": status_mod.get("completionDateStruct", {}).get("date", ""),
            "phases": phases,
            "study_type": design_mod.get("studyType", ""),
            "enrollment": enroll_mod.get("count"),
            "enrollment_type": enroll_mod.get("type", ""),
            "conditions": cond_mod.get("conditions", []),
            "interventions": interventions,
            "primary_outcomes": primary_outcomes,
            "brief_summary": desc_mod.get("briefSummary", ""),
            "sponsor": lead_sponsor.get("name", ""),
            "eligibility_criteria": eligibility_mod.get("eligibilityCriteria", ""),
            "min_age": eligibility_mod.get("minimumAge", ""),
            "max_age": eligibility_mod.get("maximumAge", ""),
            "sex": eligibility_mod.get("sex", ""),
        }
