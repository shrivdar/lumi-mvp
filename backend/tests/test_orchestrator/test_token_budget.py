"""Tests for TokenBudgetManager — allocation, tracking, and enforcement."""

from __future__ import annotations

from core.models import AgentConstraints, ResearchConfig
from orchestrator.token_budget import TokenBudgetManager


class TestTokenBudgetManagerInit:
    def test_init_defaults(self):
        mgr = TokenBudgetManager(session_budget=1_000_000)
        assert mgr.session_budget == 1_000_000
        assert mgr.remaining == 1_000_000
        assert mgr.used == 0
        assert mgr.utilization == 0.0
        assert mgr.is_exhausted() is False

    def test_zero_budget(self):
        mgr = TokenBudgetManager(session_budget=0)
        assert mgr.is_exhausted() is True
        assert mgr.utilization == 1.0


class TestTokenBudgetAllocation:
    def test_allocate_hypothesis_budget(self):
        mgr = TokenBudgetManager(session_budget=1_000_000)
        budget = mgr.allocate_hypothesis_budget("h1", active_hypothesis_count=4)
        assert budget == 250_000  # 1M / 4

    def test_allocate_hypothesis_budget_respects_used(self):
        mgr = TokenBudgetManager(session_budget=1_000_000)
        mgr.allocate_hypothesis_budget("h1", active_hypothesis_count=2)
        mgr.record_usage("h1", "agent-1", 200_000)
        # Remaining is 800K, split across 2 hypotheses = 400K each
        budget = mgr.allocate_hypothesis_budget("h2", active_hypothesis_count=2)
        assert budget == 400_000

    def test_allocate_for_swarm(self):
        mgr = TokenBudgetManager(session_budget=1_000_000)
        mgr.allocate_hypothesis_budget("h1", active_hypothesis_count=1)
        config = ResearchConfig(agent_token_budget=50_000, max_llm_calls_per_agent=20)

        constraints = mgr.allocate_for_swarm("h1", agent_count=5, config=config)
        assert len(constraints) == 5
        for c in constraints:
            assert isinstance(c, AgentConstraints)
            assert c.token_budget <= 50_000
            assert c.max_llm_calls == 20

    def test_allocate_for_swarm_empty(self):
        mgr = TokenBudgetManager(session_budget=1_000_000)
        result = mgr.allocate_for_swarm("h1", agent_count=0, config=ResearchConfig())
        assert result == []

    def test_per_agent_budget_capped_by_config(self):
        mgr = TokenBudgetManager(session_budget=10_000_000)
        mgr.allocate_hypothesis_budget("h1", active_hypothesis_count=1)
        config = ResearchConfig(agent_token_budget=50_000)

        constraints = mgr.allocate_for_swarm("h1", agent_count=3, config=config)
        for c in constraints:
            assert c.token_budget <= 50_000


class TestTokenBudgetTracking:
    def test_record_usage(self):
        mgr = TokenBudgetManager(session_budget=1_000_000)
        mgr.record_usage("h1", "agent-1", 10_000)
        assert mgr.used == 10_000
        assert mgr.remaining == 990_000

    def test_record_multiple_usage(self):
        mgr = TokenBudgetManager(session_budget=100_000)
        mgr.record_usage("h1", "a1", 30_000)
        mgr.record_usage("h1", "a2", 20_000)
        mgr.record_usage("h2", "a3", 10_000)
        assert mgr.used == 60_000
        assert mgr.remaining == 40_000

    def test_exhaustion_detection(self):
        mgr = TokenBudgetManager(session_budget=50_000)
        mgr.record_usage("h1", "a1", 50_000)
        assert mgr.is_exhausted() is True

    def test_check_agent_budget_within(self):
        mgr = TokenBudgetManager(session_budget=100_000)
        mgr.record_usage("h1", "a1", 5_000)
        assert mgr.check_agent_budget("a1", 10_000) is True

    def test_check_agent_budget_exceeded(self):
        mgr = TokenBudgetManager(session_budget=100_000)
        mgr.record_usage("h1", "a1", 15_000)
        assert mgr.check_agent_budget("a1", 10_000) is False


class TestTokenBudgetSummary:
    def test_summary(self):
        mgr = TokenBudgetManager(session_budget=1_000_000, session_id="s1")
        mgr.allocate_hypothesis_budget("h1", active_hypothesis_count=1)
        mgr.record_usage("h1", "a1", 10_000)
        mgr.record_usage("h1", "a2", 20_000)

        summary = mgr.summary()
        assert summary["session_budget"] == 1_000_000
        assert summary["session_used"] == 30_000
        assert summary["session_remaining"] == 970_000
        assert summary["agent_count"] == 2


class TestTokenBudgetEvents:
    def test_drain_events(self):
        mgr = TokenBudgetManager(session_budget=100_000, session_id="s1")
        events = mgr.drain_events()
        assert events == []
