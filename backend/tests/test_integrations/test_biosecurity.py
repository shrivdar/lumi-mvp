"""Tests for the biosecurity screener."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.models import (
    BiosecurityCategory,
    HypothesisNode,
    ResearchResult,
    ScreeningResult,
    ScreeningTier,
)
from integrations.biosecurity import BiosecurityScreener


def _make_result(**overrides) -> ResearchResult:
    """Create a minimal ResearchResult for testing."""
    defaults = {
        "research_id": "test-research-1",
        "best_hypothesis": HypothesisNode(hypothesis="Test hypothesis"),
        "hypothesis_ranking": [],
        "key_findings": [],
    }
    defaults.update(overrides)
    return ResearchResult(**defaults)


class TestBiosecurityScreener:
    """Tests for BiosecurityScreener."""

    @pytest.mark.asyncio
    async def test_clear_result(self):
        """Safe research should be screened as CLEAR."""
        llm_response = MagicMock()
        llm_response.content = [
            MagicMock(
                text='{"tier": "CLEAR", "flagged_categories": [], '
                '"reasoning": "Standard biomedical research", "disclaimer": ""}'
            )
        ]
        llm_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        with patch("integrations.biosecurity.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = llm_response
            mock_cls.return_value = mock_client

            screener = BiosecurityScreener()
            result = _make_result(
                best_hypothesis=HypothesisNode(
                    hypothesis="BRCA1 mutations are associated with breast cancer"
                ),
            )
            screening = await screener.screen(result)

        assert screening.tier == ScreeningTier.CLEAR
        assert screening.flagged_categories == []
        assert screening.research_id == "test-research-1"

    @pytest.mark.asyncio
    async def test_warning_result(self):
        """Dual-use research should get WARNING tier with disclaimer."""
        llm_response = MagicMock()
        llm_response.content = [
            MagicMock(
                text='{"tier": "WARNING", "flagged_categories": ["dual_use_concern"], '
                '"reasoning": "Research touches on gain-of-function concepts", '
                '"disclaimer": "This research involves topics with dual-use potential."}'
            )
        ]

        with patch("integrations.biosecurity.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = llm_response
            mock_cls.return_value = mock_client

            screener = BiosecurityScreener()
            result = _make_result()
            screening = await screener.screen(result)

        assert screening.tier == ScreeningTier.WARNING
        assert BiosecurityCategory.DUAL_USE_CONCERN in screening.flagged_categories
        assert screening.disclaimer != ""

    @pytest.mark.asyncio
    async def test_blocked_result(self):
        """Dangerous research should be BLOCKED."""
        llm_response = MagicMock()
        llm_response.content = [
            MagicMock(
                text='{"tier": "BLOCKED", '
                '"flagged_categories": ["pathogen_enhancement", "gain_of_function"], '
                '"reasoning": "Research describes pathogen enhancement techniques", '
                '"disclaimer": ""}'
            )
        ]

        with patch("integrations.biosecurity.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = llm_response
            mock_cls.return_value = mock_client

            screener = BiosecurityScreener()
            result = _make_result()
            screening = await screener.screen(result)

        assert screening.tier == ScreeningTier.BLOCKED
        assert BiosecurityCategory.PATHOGEN_ENHANCEMENT in screening.flagged_categories
        assert BiosecurityCategory.GAIN_OF_FUNCTION in screening.flagged_categories

    @pytest.mark.asyncio
    async def test_llm_failure_defaults_to_warning(self):
        """If the screening LLM call fails, default to WARNING (fail safe)."""
        with patch("integrations.biosecurity.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = Exception("API error")
            mock_cls.return_value = mock_client

            screener = BiosecurityScreener()
            result = _make_result()
            screening = await screener.screen(result)

        assert screening.tier == ScreeningTier.WARNING
        assert "failed" in screening.reasoning.lower()
        assert screening.disclaimer != ""

    def test_build_summary(self):
        """Summary builder should include key research details."""
        result = _make_result(
            best_hypothesis=HypothesisNode(
                hypothesis="BRCA1 is a tumor suppressor"
            ),
            recommended_experiments=["Validate with Western blot"],
            report_markdown="# Report\nSome findings here.",
        )

        summary = BiosecurityScreener._build_summary(result)

        assert "test-research-1" in summary
        assert "BRCA1 is a tumor suppressor" in summary
        assert "Western blot" in summary
        assert "# Report" in summary

    def test_screening_result_model(self):
        """ScreeningResult model should serialize correctly."""
        sr = ScreeningResult(
            research_id="r1",
            tier=ScreeningTier.WARNING,
            flagged_categories=[BiosecurityCategory.DUAL_USE_CONCERN],
            reasoning="Some concern",
            disclaimer="Please review carefully.",
        )
        data = sr.model_dump()
        assert data["tier"] == "WARNING"
        assert data["flagged_categories"] == ["dual_use_concern"]
