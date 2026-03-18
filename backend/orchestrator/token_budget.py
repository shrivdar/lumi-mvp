"""Token Budget Manager — distributes a global token budget across the hypothesis tree.

Hierarchy: Session → Hypothesis Branch → Swarm → Individual Agent.

The manager tracks usage at each level and enforces limits. When the orchestrator
spawns agents, it calls ``allocate_for_swarm()`` to get per-agent budgets that
fit within the remaining session budget.
"""

from __future__ import annotations

import structlog

from core.models import AgentConstraints, ResearchConfig, ResearchEvent

logger = structlog.get_logger(__name__)


class TokenBudgetManager:
    """Distributes and tracks a global token budget across the research session.

    Budget hierarchy:
      session_budget
        └─ hypothesis_budget  (session_budget / active_hypotheses)
             └─ swarm_budget  (hypothesis_budget / expected_iterations)
                  └─ agent_budget  (swarm_budget / agents_in_swarm)

    Usage is tracked per-hypothesis and per-agent, with hard enforcement at
    the session level and soft warnings at the agent level.
    """

    def __init__(
        self,
        *,
        session_budget: int,
        session_id: str = "",
    ) -> None:
        self.session_budget = session_budget
        self.session_id = session_id

        # Tracking
        self._session_used: int = 0
        self._hypothesis_used: dict[str, int] = {}  # hypothesis_id → tokens used
        self._hypothesis_budgets: dict[str, int] = {}  # hypothesis_id → allocated budget
        self._agent_used: dict[str, int] = {}  # agent_id → tokens used
        self._events: list[ResearchEvent] = []

    # ------------------------------------------------------------------
    # Budget queries
    # ------------------------------------------------------------------

    @property
    def remaining(self) -> int:
        """Tokens remaining in the session budget."""
        return max(0, self.session_budget - self._session_used)

    @property
    def used(self) -> int:
        return self._session_used

    @property
    def utilization(self) -> float:
        """Fraction of session budget consumed (0.0 – 1.0)."""
        if self.session_budget == 0:
            return 1.0
        return self._session_used / self.session_budget

    def is_exhausted(self) -> bool:
        return self._session_used >= self.session_budget

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate_hypothesis_budget(
        self,
        hypothesis_id: str,
        active_hypothesis_count: int,
        remaining_iterations: int = 1,
    ) -> int:
        """Allocate a token budget for a hypothesis branch.

        Splits the remaining session budget evenly among active hypotheses,
        weighted by remaining iterations.

        Returns:
            Token budget for this hypothesis.
        """
        if active_hypothesis_count <= 0:
            active_hypothesis_count = 1

        # Fair share of remaining budget
        per_hypothesis = self.remaining // active_hypothesis_count

        # Already-allocated hypotheses keep their budget unless it exceeds remaining
        existing = self._hypothesis_budgets.get(hypothesis_id, 0)
        existing_used = self._hypothesis_used.get(hypothesis_id, 0)
        existing_remaining = max(0, existing - existing_used)

        budget = max(per_hypothesis, existing_remaining)
        self._hypothesis_budgets[hypothesis_id] = existing_used + budget

        logger.debug(
            "hypothesis_budget_allocated",
            hypothesis_id=hypothesis_id,
            budget=budget,
            session_remaining=self.remaining,
            active_hypotheses=active_hypothesis_count,
        )
        return budget

    def allocate_for_swarm(
        self,
        hypothesis_id: str,
        agent_count: int,
        config: ResearchConfig,
    ) -> list[AgentConstraints]:
        """Allocate per-agent constraints for a swarm under a hypothesis.

        Distributes the hypothesis budget evenly across agents, capped by
        the per-agent budget in config.

        Returns:
            List of AgentConstraints, one per agent in the swarm.
        """
        if agent_count <= 0:
            return []

        # How much budget does this hypothesis have left?
        hyp_budget = self._hypothesis_budgets.get(hypothesis_id, 0)
        hyp_used = self._hypothesis_used.get(hypothesis_id, 0)
        hyp_remaining = max(0, hyp_budget - hyp_used)

        # Also cap by session remaining
        available = min(hyp_remaining, self.remaining)

        # Per-agent share: fair split capped by config and session remaining
        fair_share = available // max(agent_count, 1)
        per_agent = min(fair_share, config.agent_token_budget)
        # Ensure at least a minimum viable budget (5000 tokens),
        # but never exceed fair share or config cap.
        per_agent = max(per_agent, min(5_000, fair_share))
        per_agent = min(per_agent, fair_share, config.agent_token_budget)

        constraints_list = []
        for _ in range(agent_count):
            constraints_list.append(
                AgentConstraints(
                    max_turns=200,
                    token_budget=per_agent,
                    timeout_seconds=300,
                    max_llm_calls=config.max_llm_calls_per_agent,
                )
            )

        logger.debug(
            "swarm_budget_allocated",
            hypothesis_id=hypothesis_id,
            agent_count=agent_count,
            per_agent_budget=per_agent,
            hypothesis_remaining=hyp_remaining,
            session_remaining=self.remaining,
        )

        return constraints_list

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def record_usage(
        self,
        hypothesis_id: str,
        agent_id: str,
        tokens_used: int,
    ) -> None:
        """Record token usage for an agent under a hypothesis."""
        self._session_used += tokens_used
        self._hypothesis_used[hypothesis_id] = (
            self._hypothesis_used.get(hypothesis_id, 0) + tokens_used
        )
        self._agent_used[agent_id] = (
            self._agent_used.get(agent_id, 0) + tokens_used
        )

        # Warn if session budget is getting low
        if self.utilization > 0.9:
            logger.warning(
                "token_budget_nearly_exhausted",
                session_used=self._session_used,
                session_budget=self.session_budget,
                utilization=f"{self.utilization:.1%}",
            )

    def check_agent_budget(self, agent_id: str, budget: int) -> bool:
        """Check if an agent has exceeded its allocated budget.

        Returns True if within budget, False if exceeded.
        """
        used = self._agent_used.get(agent_id, 0)
        return used <= budget

    def enforce_agent_budget(self, agent_id: str, budget: int) -> None:
        """Hard-kill enforcement: raise TokenBudgetExceededError if over budget.

        Called before each LLM call to prevent agents from exceeding their
        allocated token budget.
        """
        from core.exceptions import TokenBudgetExceededError

        used = self._agent_used.get(agent_id, 0)
        if used > budget:
            logger.warning(
                "agent_token_budget_hard_kill",
                agent_id=agent_id,
                tokens_used=used,
                budget=budget,
            )
            raise TokenBudgetExceededError(
                f"Agent {agent_id} exceeded token budget: {used}/{budget}",
                error_code="TOKEN_BUDGET_HARD_KILL",
                details={"agent_id": agent_id, "tokens_used": used, "budget": budget},
            )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a summary of budget usage across the session."""
        return {
            "session_budget": self.session_budget,
            "session_used": self._session_used,
            "session_remaining": self.remaining,
            "utilization": round(self.utilization, 4),
            "hypothesis_budgets": dict(self._hypothesis_budgets),
            "hypothesis_used": dict(self._hypothesis_used),
            "agent_count": len(self._agent_used),
            "top_agents": dict(
                sorted(
                    self._agent_used.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )[:10]
            ),
        }

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, **data) -> None:
        event = ResearchEvent(
            session_id=self.session_id,
            event_type=event_type,
            data=data,
        )
        self._events.append(event)

    def drain_events(self) -> list[ResearchEvent]:
        events = self._events[:]
        self._events.clear()
        return events
