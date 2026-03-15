"""Tests for orchestrator checkpoint callback integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from orchestrator.research_loop import ResearchOrchestrator


class TestCheckpointCallback:
    def test_orchestrator_accepts_checkpoint_callback(self):
        """Verify the orchestrator constructor accepts a checkpoint_callback."""
        callback = AsyncMock()
        orch = ResearchOrchestrator(
            llm=MagicMock(),
            kg=MagicMock(),
            checkpoint_callback=callback,
        )
        assert orch._checkpoint_callback is callback

    def test_orchestrator_without_callback_is_fine(self):
        """Verify orchestrator works without checkpoint_callback (backward compat)."""
        orch = ResearchOrchestrator(
            llm=MagicMock(),
            kg=MagicMock(),
        )
        assert orch._checkpoint_callback is None
