"""Biosecurity screener — LLM-based dual-use screening for research outputs.

Screens every ResearchResult before delivery to the user. Uses a cheap model
(Haiku) for cost efficiency and logs all screening decisions for audit.
"""

from __future__ import annotations

from typing import Any

import anthropic
import structlog

from core.audit import AuditLogger
from core.config import settings
from core.models import (
    BiosecurityCategory,
    ResearchResult,
    ScreeningResult,
    ScreeningTier,
)

logger = structlog.get_logger(__name__)

_SCREENING_SYSTEM_PROMPT = """\
You are a biosecurity screening system. Your job is to evaluate biomedical \
research outputs for dual-use concerns that could pose biosecurity risks.

Evaluate the research output against these categories:
- pathogen_enhancement: Research that could enhance pathogen transmissibility, \
virulence, or host range
- toxin_synthesis: Information enabling synthesis of biological toxins
- weapons_potential: Research with direct applicability to biological weapons
- gain_of_function: Gain-of-function research on pathogens of pandemic potential
- dual_use_concern: Other dual-use research that could be misused

Assign a tier:
- CLEAR: No biosecurity concerns. Proceed normally.
- WARNING: Mild dual-use potential but legitimate research. Add a disclaimer.
- BLOCKED: Significant biosecurity risk. Do not present to user.

Respond with a JSON object:
{
  "tier": "CLEAR" | "WARNING" | "BLOCKED",
  "flagged_categories": ["category1", ...],
  "reasoning": "Brief explanation of the decision",
  "disclaimer": "Disclaimer text if tier is WARNING, empty string otherwise"
}
"""


class BiosecurityScreener:
    """Screen research outputs for dual-use/biosecurity concerns.

    Uses a cheap model (Haiku by default) to evaluate research results
    and assign a screening tier. All decisions are audit-logged.
    """

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.biosecurity_screening_model
        self._audit = AuditLogger("biosecurity")

    async def screen(self, result: ResearchResult) -> ScreeningResult:
        """Screen a ResearchResult for biosecurity concerns.

        Returns a ScreeningResult that should be attached to the result
        before delivery to the user.
        """
        summary = self._build_summary(result)

        self._audit.log(
            "screening_started",
            research_id=result.research_id,
            model=self._model,
        )

        try:
            screening = await self._call_llm(summary, result.research_id)
        except Exception as exc:
            # On screener failure, default to WARNING — fail safe
            logger.error("biosecurity_screening_failed", error=str(exc))
            screening = ScreeningResult(
                research_id=result.research_id,
                tier=ScreeningTier.WARNING,
                flagged_categories=[],
                reasoning=f"Screening failed ({exc}); defaulting to WARNING",
                disclaimer=(
                    "Biosecurity screening could not be completed. "
                    "Please review this output carefully for dual-use concerns."
                ),
            )

        self._audit.log(
            "screening_completed",
            research_id=result.research_id,
            tier=screening.tier,
            flagged_categories=[str(c) for c in screening.flagged_categories],
            reasoning=screening.reasoning,
        )

        return screening

    async def _call_llm(self, summary: str, research_id: str) -> ScreeningResult:
        """Send the research summary to the screening model and parse the response."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=_SCREENING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": summary}],
        )

        text = response.content[0].text if response.content else ""

        parsed = self._parse_response(text)

        return ScreeningResult(
            research_id=research_id,
            tier=ScreeningTier(parsed["tier"]),
            flagged_categories=[
                BiosecurityCategory(c) for c in parsed.get("flagged_categories", [])
            ],
            reasoning=parsed.get("reasoning", ""),
            disclaimer=parsed.get("disclaimer", ""),
        )

    @staticmethod
    def _parse_response(text: str) -> dict[str, Any]:
        """Extract JSON from the LLM response."""
        from core.llm import LLMClient

        return LLMClient.parse_json(text)  # type: ignore[return-value]

    @staticmethod
    def _build_summary(result: ResearchResult) -> str:
        """Build a text summary of the research result for screening."""
        parts: list[str] = []

        parts.append(f"Research ID: {result.research_id}")
        parts.append(f"Best hypothesis: {result.best_hypothesis.hypothesis}")

        if result.hypothesis_ranking:
            parts.append("Top hypotheses:")
            for h in result.hypothesis_ranking[:5]:
                parts.append(f"  - {h.hypothesis}")

        if result.key_findings:
            parts.append("Key findings (high-confidence edges):")
            for edge in result.key_findings[:10]:
                desc = edge.properties.get("description", edge.relation)
                parts.append(f"  - {desc} (confidence: {edge.confidence.overall:.2f})")

        if result.recommended_experiments:
            parts.append("Recommended experiments:")
            for exp in result.recommended_experiments[:5]:
                parts.append(f"  - {exp[:200]}")

        if result.report_markdown:
            # Include first 2000 chars of the report
            parts.append(f"Report excerpt:\n{result.report_markdown[:2000]}")

        return "\n".join(parts)
