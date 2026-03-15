"""Research Loop — main orchestrator driving the MCTS cycle.

Lifecycle:
1. INITIALIZE — Seed KG, generate initial hypotheses, build tree
2. COMPOSE — Select agents for the swarm
3. MCTS LOOP — Select → Dispatch → Execute → Critique → Evaluate → Backpropagate
4. COMPILE — Extract results, rank hypotheses, generate report
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from core.audit import AuditLogger, set_request_context
from core.exceptions import OrchestrationError
from core.interfaces import KnowledgeGraph, YamiInterface
from core.llm import LLMClient
from core.models import (
    AgentResult,
    AgentTask,
    AgentType,
    HypothesisNode,
    KGEdge,
    ResearchConfig,
    ResearchEvent,
    ResearchResult,
    ResearchSession,
    SessionStatus,
    TaskStatus,
    ToolRegistryEntry,
)
from orchestrator.hypothesis_tree import HypothesisTree
from orchestrator.swarm_composer import SwarmComposer
from orchestrator.uncertainty import UncertaintyAggregator

logger = structlog.get_logger(__name__)


class ResearchOrchestrator:
    """Main orchestrator — runs the MCTS-driven research loop.

    Coordinates the hypothesis tree, swarm composer, agent execution,
    uncertainty aggregation, and HITL triggering.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        kg: KnowledgeGraph,
        yami: YamiInterface | None = None,
        agent_factory: Any = None,  # callable(agent_type, llm, kg, yami, tools) -> BaseAgentImpl
        tool_entries: list[ToolRegistryEntry] | None = None,
        slack_tool: Any = None,  # SlackTool instance for HITL
    ) -> None:
        self.llm = llm
        self.kg = kg
        self.yami = yami
        self.agent_factory = agent_factory
        self.tool_entries = tool_entries or []
        self.slack_tool = slack_tool
        self.audit = AuditLogger("orchestrator")

        self._session: ResearchSession | None = None
        self._tree: HypothesisTree | None = None
        self._composer: SwarmComposer | None = None
        self._uncertainty: UncertaintyAggregator | None = None
        self._all_results: list[AgentResult] = []
        self._events: list[ResearchEvent] = []
        self._total_agents_spawned: int = 0
        self._session_tokens_used: int = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        query: str,
        config: ResearchConfig | None = None,
    ) -> ResearchSession:
        """Run a complete research session.

        This is the top-level entry point. It creates a session, runs the
        MCTS loop, and returns the completed session with results.
        """
        config = config or ResearchConfig()
        session = self._create_session(query, config)
        self._session = session

        set_request_context(research_id=session.id)
        self.audit.log("research_started", session_id=session.id, query=query)

        start_ms = int(time.monotonic() * 1000)

        try:
            # Phase 1: Initialize
            session.status = SessionStatus.RUNNING
            tree, root_hypotheses = await self._initialize(query, config, session.id)
            self._tree = tree

            # Phase 2: MCTS Loop
            await self._mcts_loop(query, config, session)

            # Phase 3: Compile results
            result = self._compile_results(session, start_ms)
            session.result = result
            session.status = SessionStatus.COMPLETED

        except Exception as exc:
            session.status = SessionStatus.FAILED
            self.audit.error("research_failed", session_id=session.id, error=str(exc))
            raise OrchestrationError(
                f"Research session failed: {exc}",
                error_code="RESEARCH_FAILED",
                details={"session_id": session.id},
            ) from exc
        finally:
            elapsed = int(time.monotonic() * 1000) - start_ms
            session.updated_at = session.created_at.__class__.now(session.created_at.tzinfo)
            self.audit.log(
                "research_completed",
                session_id=session.id,
                status=str(session.status),
                duration_ms=elapsed,
            )

        return session

    # ------------------------------------------------------------------
    # Phase 1: Initialize
    # ------------------------------------------------------------------

    async def _initialize(
        self,
        query: str,
        config: ResearchConfig,
        session_id: str,
    ) -> tuple[HypothesisTree, list[HypothesisNode]]:
        """Seed the KG and generate initial hypotheses."""
        self._composer = SwarmComposer(
            llm=self.llm,
            tool_registry_entries=self.tool_entries,
            session_id=session_id,
        )
        self._uncertainty = UncertaintyAggregator(session_id=session_id)

        # Generate initial hypotheses via LLM
        hypotheses_raw = await self._generate_hypotheses(query, config)

        tree = HypothesisTree(
            max_depth=config.max_hypothesis_depth,
            max_breadth=config.max_hypothesis_breadth,
            session_id=session_id,
        )

        # Create root node
        root = tree.set_root(
            hypothesis=f"Investigation of: {query}",
            rationale="Root hypothesis for the research query",
        )

        # Expand root with generated hypotheses
        tree.expand(root.id, hypotheses_raw)

        self._emit(
            "initialization_complete",
            hypothesis_count=len(hypotheses_raw),
            root_id=root.id,
        )

        return tree, tree.get_children(root.id)

    async def _generate_hypotheses(
        self,
        query: str,
        config: ResearchConfig,
    ) -> list[dict[str, str]]:
        """Use LLM to generate competing hypotheses from the research query.

        Per LAB-Bench optimization: generate hypotheses covering different
        reasoning paths, not just variations of the same idea.
        """
        prompt = (
            f"Research query: {query}\n\n"
            f"Generate 3-5 competing hypotheses that could answer this query. "
            f"Each hypothesis should explore a DIFFERENT reasoning path or mechanism. "
            f"Cover diverse angles: molecular, clinical, pathway-level, computational.\n\n"
            f"For each hypothesis, provide:\n"
            f'- "hypothesis": A clear, testable statement\n'
            f'- "rationale": Why this is a plausible direction (2-3 sentences)\n\n'
            f"Return a JSON array of objects."
        )

        try:
            response = await self.llm.query(
                prompt,
                system_prompt=(
                    "You are a senior biomedical researcher generating diverse hypotheses. "
                    "Cover multiple reasoning paths: molecular mechanisms, clinical evidence, "
                    "pathway biology, and computational predictions. Each hypothesis must be "
                    "distinct and testable."
                ),
                research_id=self._session.id if self._session else "",
            )
            parsed = LLMClient.parse_json(response)
            if isinstance(parsed, list):
                return [
                    {"hypothesis": h.get("hypothesis", ""), "rationale": h.get("rationale", "")}
                    for h in parsed
                    if isinstance(h, dict) and h.get("hypothesis")
                ]
        except Exception as exc:
            logger.warning("hypothesis_generation_failed", error=str(exc))

        # Fallback: single generic hypothesis
        return [
            {
                "hypothesis": f"There exist known molecular mechanisms relevant to: {query}",
                "rationale": "Literature-driven investigation of established biology",
            },
            {
                "hypothesis": f"Novel therapeutic targets can be identified for: {query}",
                "rationale": "Drug discovery angle exploring existing compound databases",
            },
            {
                "hypothesis": f"Clinical evidence supports specific interventions for: {query}",
                "rationale": "Clinical trials and outcome data may provide direct answers",
            },
        ]

    # ------------------------------------------------------------------
    # Phase 2: MCTS Loop
    # ------------------------------------------------------------------

    async def _mcts_loop(
        self,
        query: str,
        config: ResearchConfig,
        session: ResearchSession,
    ) -> None:
        """Run the MCTS loop with per-hypothesis parallel swarms.

        Each iteration selects multiple leaf hypotheses, composes a swarm
        for each, and dispatches all agents concurrently behind a shared
        semaphore.  Results are backpropagated as they arrive.
        """
        assert self._tree is not None
        assert self._composer is not None
        assert self._uncertainty is not None

        semaphore = asyncio.Semaphore(config.max_concurrent_agents)

        for iteration in range(config.max_mcts_iterations):
            session.current_iteration = iteration + 1

            # Budget guard — stop if session token budget exhausted
            if self._session_tokens_used >= config.session_token_budget:
                logger.info(
                    "mcts_terminated",
                    reason="session_token_budget_exhausted",
                    tokens_used=self._session_tokens_used,
                )
                self._emit(
                    "mcts_iteration_end",
                    iteration=iteration + 1,
                    should_stop=True,
                    stop_reason="session_token_budget_exhausted",
                )
                break

            # Budget guard — stop if total agent cap reached
            if self._total_agents_spawned >= config.max_total_agents:
                logger.info(
                    "mcts_terminated",
                    reason="max_total_agents_reached",
                    agents_spawned=self._total_agents_spawned,
                )
                self._emit(
                    "mcts_iteration_end",
                    iteration=iteration + 1,
                    should_stop=True,
                    stop_reason="max_total_agents_reached",
                )
                break

            self._emit(
                "mcts_iteration_start",
                iteration=iteration + 1,
                total_visits=self._tree.total_visits,
                node_count=self._tree.node_count,
                total_agents_spawned=self._total_agents_spawned,
                session_tokens_used=self._session_tokens_used,
            )

            # 1. SELECT — pick multiple leaf hypotheses for parallel exploration
            remaining_agent_slots = config.max_total_agents - self._total_agents_spawned
            max_swarms = max(1, remaining_agent_slots // max(config.max_agents_per_swarm, 1))
            leaves = self._tree.select_leaves(max_leaves=max_swarms)

            # 2. COMPOSE + GENERATE per-hypothesis swarms concurrently
            async def _compose_for_hypothesis(
                hypothesis: HypothesisNode,
            ) -> tuple[HypothesisNode, list[AgentTask]]:
                agent_types = await self._composer.compose_swarm(query, hypothesis, config)
                tasks = await self._composer.generate_tasks(
                    query, hypothesis, agent_types, session.id,
                )
                return hypothesis, tasks

            compose_results = await asyncio.gather(
                *[_compose_for_hypothesis(h) for h in leaves],
                return_exceptions=True,
            )

            # 3. DISPATCH all swarms concurrently behind shared semaphore
            all_swarm_coros = []
            hypothesis_task_map: list[tuple[HypothesisNode, list[AgentTask]]] = []
            for result in compose_results:
                if isinstance(result, Exception):
                    logger.warning("swarm_composition_failed", error=str(result))
                    continue
                hypothesis, tasks = result
                # Cap agent spawns to stay within budget
                budget_left = config.max_total_agents - self._total_agents_spawned
                tasks = tasks[:budget_left]
                if not tasks:
                    continue
                hypothesis_task_map.append((hypothesis, tasks))
                all_swarm_coros.append(
                    self._execute_agents_with_semaphore(
                        tasks, hypothesis, config, semaphore,
                    )
                )

            if not all_swarm_coros:
                # No swarms could be composed — fall back to single select
                selected = self._tree.select()
                agent_types = await self._composer.compose_swarm(query, selected, config)
                tasks = await self._composer.generate_tasks(
                    query, selected, agent_types, session.id,
                )
                swarm_results_list = [
                    await self._execute_agents_with_semaphore(
                        tasks, selected, config, semaphore,
                    )
                ]
                hypothesis_task_map = [(selected, tasks)]
            else:
                swarm_results_list = await asyncio.gather(*all_swarm_coros)

            # 4. EVALUATE + BACKPROPAGATE per hypothesis (real-time)
            iteration_info_gains: list[float] = []
            iteration_edges_added = 0

            for (hypothesis, _tasks), results in zip(
                hypothesis_task_map, swarm_results_list, strict=False,
            ):
                self._all_results.extend(results)

                # Track token usage
                for r in results:
                    self._session_tokens_used += r.llm_tokens_used

                info_gain = self._evaluate_iteration(results, hypothesis)
                iteration_info_gains.append(info_gain)

                edges_added = sum(len(r.edges_added) for r in results)
                edges_falsified = sum(
                    1 for r in results for f in r.falsification_results if f.falsified
                )
                contradictions = sum(
                    1 for r in results for e in r.edges_added if e.is_contradiction
                )
                iteration_edges_added += edges_added

                self._tree.backpropagate(
                    hypothesis.id,
                    info_gain,
                    edges_added=edges_added,
                    edges_falsified=edges_falsified,
                    contradictions_found=contradictions,
                )

                # EXPAND — if high info gain and depth allows
                if info_gain > 0.5 and hypothesis.depth < config.max_hypothesis_depth:
                    child_hypotheses = await self._generate_child_hypotheses(
                        query, hypothesis, results,
                    )
                    if child_hypotheses:
                        self._tree.expand(hypothesis.id, child_hypotheses)

                # Update hypothesis confidence
                self._update_hypothesis_confidence(hypothesis)

            # Update session stats
            session.total_nodes = self.kg.node_count()
            session.total_edges = self.kg.edge_count()
            session.total_hypotheses = self._tree.node_count

            # 5. UNCERTAINTY / HITL — aggregate across all swarm results
            all_iter_results = [
                r for (_h, _t), results in zip(
                    hypothesis_task_map, swarm_results_list, strict=False,
                )
                for r in results
            ]
            if all_iter_results:
                agg_uncertainty = self._uncertainty.aggregate(all_iter_results)
                should_hitl, hitl_reason = self._uncertainty.should_trigger_hitl(
                    agg_uncertainty, config,
                )
                if should_hitl:
                    # Use the first hypothesis for context
                    await self._handle_hitl(
                        query, hypothesis_task_map[0][0],
                        agg_uncertainty, hitl_reason, config,
                    )

            # 6. AUTO-PRUNE
            self._tree.auto_prune()

            # 7. CHECK TERMINATION
            avg_info_gain = (
                sum(iteration_info_gains) / len(iteration_info_gains)
                if iteration_info_gains
                else 0.0
            )
            should_stop, stop_reason = self._tree.should_terminate(
                confidence_threshold=config.confidence_threshold,
                max_iterations=config.max_mcts_iterations,
                current_iteration=iteration + 1,
            )

            self._emit(
                "mcts_iteration_end",
                iteration=iteration + 1,
                info_gain=avg_info_gain,
                edges_added=iteration_edges_added,
                swarms_dispatched=len(hypothesis_task_map),
                total_agents_spawned=self._total_agents_spawned,
                session_tokens_used=self._session_tokens_used,
                should_stop=should_stop,
                stop_reason=stop_reason,
            )

            if should_stop:
                logger.info("mcts_terminated", reason=stop_reason, iteration=iteration + 1)
                break

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    async def _execute_agents_with_semaphore(
        self,
        tasks: list[AgentTask],
        hypothesis: HypothesisNode,
        config: ResearchConfig,
        semaphore: asyncio.Semaphore,
    ) -> list[AgentResult]:
        """Execute agent tasks behind a shared semaphore, isolating failures."""

        async def _run_single(task: AgentTask) -> AgentResult | None:
            async with semaphore:
                # Check per-agent token budget will be tracked post-execution
                self._total_agents_spawned += 1

                try:
                    if self.agent_factory is None:
                        raise OrchestrationError("No agent_factory configured")

                    agent = self.agent_factory(
                        agent_type=task.agent_type,
                        llm=self.llm,
                        kg=self.kg,
                        yami=self.yami,
                    )

                    self._emit(
                        "agent_started",
                        agent_id=agent.agent_id,
                        agent_type=str(task.agent_type),
                        hypothesis_id=hypothesis.id,
                        task_id=task.task_id,
                        total_agents_spawned=self._total_agents_spawned,
                    )

                    task.status = TaskStatus.RUNNING
                    result = await agent.execute(task)

                    # Enforce per-agent token budget
                    if result.llm_tokens_used > config.agent_token_budget:
                        logger.warning(
                            "agent_token_budget_exceeded",
                            agent_type=str(task.agent_type),
                            tokens_used=result.llm_tokens_used,
                            budget=config.agent_token_budget,
                        )

                    self._emit(
                        "agent_completed",
                        agent_id=agent.agent_id,
                        agent_type=str(task.agent_type),
                        success=result.success,
                        nodes_added=len(result.nodes_added),
                        edges_added=len(result.edges_added),
                        duration_ms=result.duration_ms,
                        tokens_used=result.llm_tokens_used,
                    )

                    return result

                except Exception as exc:
                    # Isolate agent failures — never let one agent crash the session
                    self.audit.error(
                        "agent_execution_failed",
                        agent_type=str(task.agent_type),
                        task_id=task.task_id,
                        error=str(exc),
                    )
                    self._emit(
                        "agent_failed",
                        agent_type=str(task.agent_type),
                        task_id=task.task_id,
                        error=str(exc),
                    )
                    return AgentResult(
                        task_id=task.task_id,
                        agent_id=task.agent_id or "unknown",
                        agent_type=task.agent_type,
                        hypothesis_id=hypothesis.id,
                        success=False,
                        errors=[str(exc)],
                    )

        # Run agents concurrently (semaphore limits actual parallelism)
        coros = [_run_single(task) for task in tasks]
        raw_results = await asyncio.gather(*coros)
        return [r for r in raw_results if r is not None]

    async def _execute_agents(
        self,
        tasks: list[AgentTask],
        hypothesis: HypothesisNode,
        config: ResearchConfig,
    ) -> list[AgentResult]:
        """Execute agent tasks without external semaphore (backwards compat)."""
        sem = asyncio.Semaphore(config.max_concurrent_agents)
        return await self._execute_agents_with_semaphore(tasks, hypothesis, config, sem)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_iteration(
        self,
        results: list[AgentResult],
        hypothesis: HypothesisNode,
    ) -> float:
        """Compute info gain for this MCTS iteration."""
        edges_added = sum(len(r.edges_added) for r in results)
        edges_falsified = sum(
            1 for r in results for f in r.falsification_results if f.falsified
        )
        contradictions = sum(
            1 for r in results for e in r.edges_added if e.is_contradiction
        )

        # Average evidence quality from new edges
        all_evidence_scores = []
        for r in results:
            for edge in r.edges_added:
                for ev in edge.evidence:
                    all_evidence_scores.append(ev.quality_score)
        avg_evidence_quality = (
            sum(all_evidence_scores) / len(all_evidence_scores)
            if all_evidence_scores
            else 0.5
        )

        # Average confidence delta from falsification
        confidence_deltas = [
            f.confidence_delta for r in results for f in r.falsification_results
        ]
        avg_confidence_delta = (
            sum(confidence_deltas) / len(confidence_deltas)
            if confidence_deltas
            else 0.0
        )

        info_gain = HypothesisTree.compute_info_gain(
            edges_added=edges_added,
            edges_falsified=edges_falsified,
            contradictions_found=contradictions,
            avg_confidence_delta=avg_confidence_delta,
            avg_evidence_quality=avg_evidence_quality,
        )

        # Track supporting/contradicting edges on the hypothesis
        for r in results:
            for edge in r.edges_added:
                if edge.is_contradiction:
                    hypothesis.contradicting_edges.append(edge.id)
                else:
                    hypothesis.supporting_edges.append(edge.id)

        return info_gain

    def _update_hypothesis_confidence(self, hypothesis: HypothesisNode) -> None:
        """Update hypothesis confidence based on supporting/contradicting evidence ratio."""
        supporting = len(hypothesis.supporting_edges)
        contradicting = len(hypothesis.contradicting_edges)
        total = supporting + contradicting

        if total == 0:
            return

        # Confidence = weighted ratio, slightly biased toward caution
        raw_confidence = supporting / total
        hypothesis.confidence = max(0.0, min(1.0, raw_confidence * 0.9 + 0.05))

        # Mark confirmed/refuted based on thresholds
        if hypothesis.confidence >= 0.8 and supporting >= 3:
            self._tree.confirm(hypothesis.id, hypothesis.confidence)  # type: ignore[union-attr]
        elif hypothesis.confidence < 0.2 and contradicting >= 3:
            self._tree.refute(hypothesis.id, "Majority contradicting evidence")  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Child hypothesis generation
    # ------------------------------------------------------------------

    async def _generate_child_hypotheses(
        self,
        query: str,
        parent: HypothesisNode,
        results: list[AgentResult],
    ) -> list[dict[str, str]]:
        """Generate child hypotheses based on what agents found."""
        # Summarize findings
        findings = []
        for r in results:
            if r.summary:
                findings.append(f"[{r.agent_type}] {r.summary}")

        prompt = (
            f"Research query: {query}\n"
            f"Parent hypothesis: {parent.hypothesis}\n\n"
            f"Agent findings:\n" + "\n".join(findings) + "\n\n"
            "Based on these findings, generate 2-3 more specific sub-hypotheses "
            "that dig deeper into the most promising or uncertain aspects. "
            "Each should explore a different angle.\n\n"
            "Return a JSON array of {\"hypothesis\": \"...\", \"rationale\": \"...\"}."
        )

        try:
            response = await self.llm.query(
                prompt,
                system_prompt=(
                    "Generate specific sub-hypotheses that explore different aspects "
                    "of the findings. Focus on resolving contradictions and uncertainty."
                ),
                research_id=self._session.id if self._session else "",
            )
            parsed = LLMClient.parse_json(response)
            if isinstance(parsed, list):
                return [
                    {"hypothesis": h["hypothesis"], "rationale": h.get("rationale", "")}
                    for h in parsed
                    if isinstance(h, dict) and h.get("hypothesis")
                ]
        except Exception as exc:
            logger.warning("child_hypothesis_generation_failed", error=str(exc))

        return []

    # ------------------------------------------------------------------
    # HITL handling
    # ------------------------------------------------------------------

    async def _handle_hitl(
        self,
        query: str,
        hypothesis: HypothesisNode,
        uncertainty: Any,
        reason: str,
        config: ResearchConfig,
    ) -> None:
        """Trigger HITL via Slack and wait for response."""
        assert self._uncertainty is not None
        assert self._session is not None

        self._uncertainty.mark_hitl_triggered()
        self._session.status = SessionStatus.WAITING_HITL

        self._emit(
            "hitl_requested",
            hypothesis_id=hypothesis.id,
            hypothesis=hypothesis.hypothesis,
            uncertainty=uncertainty.composite,
            reason=reason,
        )

        if self.slack_tool is None:
            logger.warning("hitl_no_slack_tool", reason="Slack tool not configured")
            self._session.status = SessionStatus.RUNNING
            return

        # Post to Slack
        message = self._uncertainty.format_hitl_message(
            query, hypothesis.hypothesis, uncertainty, reason,
        )

        try:
            await self.slack_tool.execute(
                action="post_hitl_request",
                channel=config.slack_channel_id or "",
                message=message,
            )

            # Wait for response (with timeout)
            response = await self.slack_tool.execute(
                action="wait_for_response",
                timeout_seconds=config.hitl_timeout_seconds,
            )

            if response.get("received"):
                self._uncertainty.record_hitl_response(response)
                self._emit("hitl_response_received", response=str(response)[:200])

        except Exception as exc:
            logger.warning("hitl_slack_failed", error=str(exc))

        self._session.status = SessionStatus.RUNNING

    # ------------------------------------------------------------------
    # Phase 3: Compile results
    # ------------------------------------------------------------------

    def _compile_results(
        self,
        session: ResearchSession,
        start_ms: int,
    ) -> ResearchResult:
        """Compile the final research result from the MCTS run."""
        assert self._tree is not None

        best = self._tree.get_best_hypothesis()
        ranking = self._tree.get_ranking()

        # Gather key findings (high-confidence edges)
        all_edges_added: list[KGEdge] = []
        for r in self._all_results:
            all_edges_added.extend(r.edges_added)

        key_findings = sorted(
            [e for e in all_edges_added if not e.falsified],
            key=lambda e: e.confidence.overall,
            reverse=True,
        )[:20]

        # Gather contradictions
        contradiction_pairs: list[tuple[KGEdge, KGEdge]] = []
        for edge in all_edges_added:
            if edge.is_contradiction and edge.contradicted_by:
                for contra_id in edge.contradicted_by:
                    contra_edge = next(
                        (e for e in all_edges_added if e.id == contra_id), None
                    )
                    if contra_edge:
                        contradiction_pairs.append((edge, contra_edge))

        # Gather uncertainties
        uncertainties = self._uncertainty._history if self._uncertainty else []

        # Gather experiment recommendations
        recommended_experiments: list[str] = []
        for r in self._all_results:
            if r.agent_type == AgentType.EXPERIMENT_DESIGNER and r.summary:
                recommended_experiments.append(r.summary)

        # Token usage
        total_llm_calls = sum(r.llm_calls for r in self._all_results)
        total_tokens = sum(r.llm_tokens_used for r in self._all_results)

        duration_ms = int(time.monotonic() * 1000) - start_ms

        result = ResearchResult(
            research_id=session.id,
            best_hypothesis=best or HypothesisNode(hypothesis="No hypothesis found"),
            hypothesis_ranking=ranking,
            key_findings=key_findings,
            contradictions=contradiction_pairs,
            uncertainties=uncertainties,
            recommended_experiments=recommended_experiments,
            graph_snapshot=self.kg.to_json(),
            kg_stats={
                "nodes": self.kg.node_count(),
                "edges": self.kg.edge_count(),
            },
            total_duration_ms=duration_ms,
            total_llm_calls=total_llm_calls,
            total_tokens=total_tokens,
        )

        self._emit(
            "research_completed",
            best_hypothesis=best.hypothesis if best else "none",
            key_findings_count=len(key_findings),
            contradiction_count=len(contradiction_pairs),
            total_duration_ms=duration_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _create_session(self, query: str, config: ResearchConfig) -> ResearchSession:
        session = ResearchSession(
            query=query,
            config=config,
            status=SessionStatus.INITIALIZING,
        )
        self._emit("session_created", session_id=session.id, query=query)
        return session

    @property
    def session(self) -> ResearchSession | None:
        return self._session

    @property
    def tree(self) -> HypothesisTree | None:
        return self._tree

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, **data: Any) -> None:
        session_id = self._session.id if self._session else ""
        event = ResearchEvent(
            session_id=session_id,
            event_type=event_type,
            data=data,
        )
        self._events.append(event)

    def drain_events(self) -> list[ResearchEvent]:
        """Drain events from orchestrator + sub-components."""
        events = self._events[:]
        self._events.clear()

        if self._tree:
            events.extend(self._tree.drain_events())
        if self._composer:
            events.extend(self._composer.drain_events())
        if self._uncertainty:
            events.extend(self._uncertainty.drain_events())

        return events
