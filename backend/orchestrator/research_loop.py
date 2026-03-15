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
    AgentSpec,
    AgentTask,
    AgentType,
    HypothesisNode,
    KGEdge,
    ResearchConfig,
    ResearchEvent,
    ResearchResult,
    ResearchSession,
    ScreeningResult,
    ScreeningTier,
    SessionStatus,
    TaskStatus,
    ToolRegistryEntry,
)
from integrations.biosecurity import BiosecurityScreener
from integrations.living_document import LivingDocument
from orchestrator.hypothesis_tree import HypothesisTree
from orchestrator.swarm_composer import SwarmComposer
from orchestrator.token_budget import TokenBudgetManager
from orchestrator.uncertainty import UncertaintyAggregator
from rl.trajectory_collector import TrajectoryCollector

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
        spec_factory: Any = None,  # callable(spec, llm, kg, yami, tools) -> BaseAgentImpl
        tool_entries: list[ToolRegistryEntry] | None = None,
        tool_instances: dict[str, Any] | None = None,  # name → BaseTool for dynamic assignment
        slack_tool: Any = None,  # SlackTool instance for HITL
        checkpoint_callback: Any = None,  # async callable(session_id, iteration, orchestrator) for DB persistence
    ) -> None:
        self.llm = llm
        self.kg = kg
        self.yami = yami
        self.agent_factory = agent_factory
        self.spec_factory = spec_factory
        self.tool_entries = tool_entries or []
        self._tool_instances = tool_instances or {}
        self.slack_tool = slack_tool
        self._checkpoint_callback = checkpoint_callback
        self.audit = AuditLogger("orchestrator")

        # Integration components
        self._living_doc: LivingDocument | None = None
        self._trajectory_collector: TrajectoryCollector | None = None
        self._token_budget: TokenBudgetManager | None = None

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
            # Phase 0: Attach integration components
            self._living_doc = LivingDocument(
                session_id=session.id,
                title=f"Research: {query[:80]}",
            )
            self._living_doc.attach(self.kg)
            self._trajectory_collector = TrajectoryCollector(benchmark_run_id=session.id)

            # Phase 1: Initialize
            session.status = SessionStatus.RUNNING
            tree, root_hypotheses = await self._initialize(query, config, session.id)
            self._tree = tree

            # Phase 2: MCTS Loop
            await self._mcts_loop(query, config, session)

            # Phase 3: Compile results
            result = self._compile_results(session, start_ms)

            # Phase 4: Biosecurity screening
            screening = await self._screen_results(result)
            result.screening = screening

            if screening.tier == ScreeningTier.BLOCKED:
                result.report_markdown = (
                    "**This research output has been blocked by biosecurity screening.**\n\n"
                    f"Reason: {screening.reasoning}"
                )
                result.key_findings = []
                result.recommended_experiments = []

            # Attach living document snapshot to result
            if self._living_doc:
                result.living_document = self._living_doc.render()

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

            # Flush collected trajectories to disk
            if self._trajectory_collector:
                try:
                    self._trajectory_collector.flush()
                except Exception as flush_exc:
                    logger.warning("trajectory_flush_failed", error=str(flush_exc))

            # Detach living document from KG
            if self._living_doc:
                try:
                    self._living_doc.detach()
                except Exception:
                    pass

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
        self._token_budget = TokenBudgetManager(
            session_budget=config.session_token_budget,
            session_id=session_id,
        )

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

            # 2. COMPOSE per-hypothesis swarms concurrently (dynamic specs)
            async def _compose_for_hypothesis(
                hypothesis: HypothesisNode,
            ) -> tuple[HypothesisNode, list[AgentSpec], list[AgentTask]]:
                # Allocate token budget for this hypothesis
                if self._token_budget:
                    active_count = len(leaves)
                    remaining_iters = max(1, config.max_mcts_iterations - iteration)
                    self._token_budget.allocate_hypothesis_budget(
                        hypothesis.id, active_count, remaining_iters,
                    )
                    agent_constraints = self._token_budget.allocate_for_swarm(
                        hypothesis.id, config.max_agents_per_swarm, config,
                    )
                else:
                    agent_constraints = None

                # Dynamic spec composition (the only path)
                specs = await self._composer.compose_swarm_specs(
                    query, hypothesis, config,
                    agent_constraints=agent_constraints,
                )
                # Generate tasks for tracking
                agent_types = [
                    s.agent_type_hint or AgentType.LITERATURE_ANALYST for s in specs
                ]
                tasks = await self._composer.generate_tasks(
                    query, hypothesis, agent_types, session.id,
                )
                return hypothesis, specs, tasks

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
                hypothesis, specs, tasks = result
                # Cap agent spawns to stay within budget
                budget_left = config.max_total_agents - self._total_agents_spawned

                specs = specs[:budget_left]
                if not specs:
                    continue
                # Pair specs with tasks for tracking
                paired_tasks = tasks[:len(specs)]
                hypothesis_task_map.append((hypothesis, paired_tasks))
                all_swarm_coros.append(
                    self._execute_specs_with_semaphore(
                        specs, paired_tasks, hypothesis, config, semaphore,
                    )
                )

            if not all_swarm_coros:
                # No swarms could be composed — fall back to single select via specs
                selected = self._tree.select()
                specs = await self._composer.compose_swarm_specs(
                    query, selected, config,
                )
                agent_types = [
                    s.agent_type_hint or AgentType.LITERATURE_ANALYST for s in specs
                ]
                tasks = await self._composer.generate_tasks(
                    query, selected, agent_types, session.id,
                )
                swarm_results_list = [
                    await self._execute_specs_with_semaphore(
                        specs, tasks, selected, config, semaphore,
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

                # Track token usage (both local and via budget manager)
                for r in results:
                    self._session_tokens_used += r.llm_tokens_used
                    if self._token_budget:
                        self._token_budget.record_usage(
                            hypothesis.id,
                            r.agent_id,
                            r.llm_tokens_used,
                        )

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

            # 8. CHECKPOINT — persist state after each iteration
            if self._checkpoint_callback:
                try:
                    await self._checkpoint_callback(
                        session.id, iteration + 1, self,
                    )
                except Exception as ckpt_exc:
                    logger.warning("checkpoint_failed", error=str(ckpt_exc))

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

                    # Dynamic tool selection: pick tools for this specific task
                    agent_tools: dict[str, Any] = {}
                    if self._composer and self._tool_instances:
                        try:
                            tool_names = await self._composer.select_tools_for_task(
                                task.agent_type, task,
                            )
                            agent_tools = {
                                n: t for n, t in self._tool_instances.items()
                                if n in tool_names
                            }
                        except Exception as exc:
                            logger.warning(
                                "dynamic_tool_selection_failed",
                                agent_type=str(task.agent_type),
                                error=str(exc),
                            )
                    # Always include python_repl if available
                    if "python_repl" in self._tool_instances and "python_repl" not in agent_tools:
                        agent_tools["python_repl"] = self._tool_instances["python_repl"]

                    agent = self.agent_factory(
                        agent_type=task.agent_type,
                        llm=self.llm,
                        kg=self.kg,
                        yami=self.yami,
                        tools=agent_tools,
                    )

                    self._emit(
                        "agent_started",
                        agent_id=agent.agent_id,
                        agent_type=str(task.agent_type),
                        hypothesis_id=hypothesis.id,
                        task_id=task.task_id,
                        total_agents_spawned=self._total_agents_spawned,
                        tools=list(agent_tools.keys()),
                    )

                    task.status = TaskStatus.RUNNING
                    result = await agent.execute(task)

                    # Collect trajectory for RL training
                    if self._trajectory_collector:
                        try:
                            self._trajectory_collector.collect(task, result)
                        except Exception as tc_exc:
                            logger.warning("trajectory_collection_failed", error=str(tc_exc))

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

    async def _execute_specs_with_semaphore(
        self,
        specs: list[AgentSpec],
        tasks: list[AgentTask],
        hypothesis: HypothesisNode,
        config: ResearchConfig,
        semaphore: asyncio.Semaphore,
    ) -> list[AgentResult]:
        """Execute dynamically-generated AgentSpec agents behind a shared semaphore."""

        async def _run_spec(spec: AgentSpec, task: AgentTask) -> AgentResult | None:
            async with semaphore:
                self._total_agents_spawned += 1
                try:
                    from agents.factory import create_agent_from_spec

                    # Resolve tools from spec's tool list
                    agent_tools: dict[str, Any] = {}
                    if spec.tools and self._tool_instances:
                        agent_tools = {
                            n: t for n, t in self._tool_instances.items()
                            if n in spec.tools
                        }
                    # Always include python_repl if available
                    if "python_repl" in self._tool_instances and "python_repl" not in agent_tools:
                        agent_tools["python_repl"] = self._tool_instances["python_repl"]

                    # Use spec_factory if provided (backward compat), else use factory directly
                    if self.spec_factory is not None:
                        agent = self.spec_factory(
                            spec=spec,
                            llm=self.llm,
                            kg=self.kg,
                            yami=self.yami,
                            tools=agent_tools,
                        )
                    else:
                        agent = create_agent_from_spec(
                            spec=spec,
                            llm=self.llm,
                            kg=self.kg,
                            yami=self.yami,
                            tools=agent_tools,
                        )

                    self._emit(
                        "agent_started",
                        agent_id=agent.agent_id,
                        role=spec.role,
                        agent_type_hint=str(spec.agent_type_hint) if spec.agent_type_hint else None,
                        hypothesis_id=hypothesis.id,
                        task_id=task.task_id,
                        total_agents_spawned=self._total_agents_spawned,
                        tools=list(agent_tools.keys()),
                        token_budget=spec.constraints.token_budget,
                    )

                    task.status = TaskStatus.RUNNING
                    result = await agent.execute(task)

                    # Collect trajectory
                    if self._trajectory_collector:
                        try:
                            self._trajectory_collector.collect(task, result)
                        except Exception as tc_exc:
                            logger.warning("trajectory_collection_failed", error=str(tc_exc))

                    # Check token budget
                    if result.llm_tokens_used > spec.constraints.token_budget:
                        logger.warning(
                            "spec_agent_token_budget_exceeded",
                            role=spec.role,
                            tokens_used=result.llm_tokens_used,
                            budget=spec.constraints.token_budget,
                        )

                    self._emit(
                        "agent_completed",
                        agent_id=agent.agent_id,
                        role=spec.role,
                        success=result.success,
                        nodes_added=len(result.nodes_added),
                        edges_added=len(result.edges_added),
                        duration_ms=result.duration_ms,
                        tokens_used=result.llm_tokens_used,
                    )

                    return result

                except Exception as exc:
                    self.audit.error(
                        "spec_agent_execution_failed",
                        role=spec.role,
                        task_id=task.task_id,
                        error=str(exc),
                    )
                    self._emit(
                        "agent_failed",
                        role=spec.role,
                        task_id=task.task_id,
                        error=str(exc),
                    )
                    agent_type = spec.agent_type_hint or AgentType.LITERATURE_ANALYST
                    return AgentResult(
                        task_id=task.task_id,
                        agent_id=task.agent_id or "unknown",
                        agent_type=agent_type,
                        hypothesis_id=hypothesis.id,
                        success=False,
                        errors=[str(exc)],
                    )

        # Pair specs with tasks (use task as tracking context)
        paired = list(zip(specs, tasks))
        coros = [_run_spec(spec, task) for spec, task in paired]
        raw_results = await asyncio.gather(*coros)
        return [r for r in raw_results if r is not None]

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
    # Phase 4: Biosecurity screening
    # ------------------------------------------------------------------

    async def _screen_results(self, result: ResearchResult) -> ScreeningResult:
        """Screen compiled results for dual-use/biosecurity concerns."""
        screener = BiosecurityScreener()
        screening = await screener.screen(result)

        self._emit(
            "biosecurity_screening_completed",
            tier=screening.tier,
            flagged_categories=[str(c) for c in screening.flagged_categories],
        )

        if screening.tier == ScreeningTier.BLOCKED:
            self.audit.warn(
                "biosecurity_blocked",
                research_id=result.research_id,
                reasoning=screening.reasoning,
                flagged_categories=[str(c) for c in screening.flagged_categories],
            )
        elif screening.tier == ScreeningTier.WARNING:
            self.audit.log(
                "biosecurity_warning",
                research_id=result.research_id,
                reasoning=screening.reasoning,
                disclaimer=screening.disclaimer,
            )

        return screening

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
        if self._token_budget:
            events.extend(self._token_budget.drain_events())

        return events

    # ------------------------------------------------------------------
    # Benchmark mode
    # ------------------------------------------------------------------

    async def run_benchmark(
        self,
        question: str,
        *,
        config: ResearchConfig | None = None,
        expected_answer: str = "",
        task_id: str = "",
        benchmark_name: str = "benchmark",
    ) -> dict[str, Any]:
        """Run a single benchmark task through hierarchical task decomposition.

        Instead of the full MCTS loop, benchmark mode:
        1. Parses the benchmark question
        2. Generates a single targeted hypothesis
        3. Spawns a focused swarm to investigate
        4. Collects and returns a structured answer

        This is designed for benchmarks like Biomni-Eval1, BixBench, and LAB-Bench
        where each question needs a direct answer rather than open-ended research.

        Args:
            question: The benchmark question/task to solve.
            config: Research configuration (defaults to benchmark-friendly settings).
            expected_answer: Optional expected answer for scoring.
            task_id: Benchmark task identifier.
            benchmark_name: Name of the benchmark suite.

        Returns:
            Dict with answer, reasoning, metadata, and optional score.
        """
        # Benchmark-friendly config: tighter budget, fewer iterations
        config = config or ResearchConfig(
            max_hypothesis_depth=2,
            max_mcts_iterations=3,
            max_agents=5,
            max_agents_per_swarm=4,
            max_concurrent_agents=10,
            max_total_agents=20,
            session_token_budget=500_000,
            agent_token_budget=50_000,
        )

        session = self._create_session(question, config)
        self._session = session
        set_request_context(research_id=session.id)

        self.audit.log(
            "benchmark_started",
            session_id=session.id,
            task_id=task_id,
            benchmark=benchmark_name,
            question=question[:200],
        )

        start_ms = int(time.monotonic() * 1000)

        try:
            session.status = SessionStatus.RUNNING

            # Initialize components
            self._composer = SwarmComposer(
                llm=self.llm,
                tool_registry_entries=self.tool_entries,
                session_id=session.id,
            )
            self._uncertainty = UncertaintyAggregator(session_id=session.id)
            self._token_budget = TokenBudgetManager(
                session_budget=config.session_token_budget,
                session_id=session.id,
            )

            # Step 1: Decompose the question into a hypothesis
            hypothesis_data = await self._decompose_benchmark_question(question)

            # Step 2: Build a minimal hypothesis tree
            tree = HypothesisTree(
                max_depth=config.max_hypothesis_depth,
                max_breadth=5,
                session_id=session.id,
            )
            root = tree.set_root(
                hypothesis=f"Benchmark: {question[:100]}",
                rationale="Benchmark task root",
            )
            children = tree.expand(root.id, [hypothesis_data])
            self._tree = tree

            if not children:
                raise OrchestrationError("Failed to create benchmark hypothesis")

            target_hypothesis = children[0]

            # Step 3: Compose and execute a targeted swarm
            semaphore = asyncio.Semaphore(config.max_concurrent_agents)

            if self._token_budget:
                self._token_budget.allocate_hypothesis_budget(target_hypothesis.id, 1)
                agent_constraints = self._token_budget.allocate_for_swarm(
                    target_hypothesis.id, config.max_agents_per_swarm, config,
                )
            else:
                agent_constraints = None

            specs = await self._composer.compose_swarm_specs(
                question, target_hypothesis, config,
                agent_constraints=agent_constraints,
            )
            agent_types = [
                s.agent_type_hint or AgentType.LITERATURE_ANALYST for s in specs
            ]
            tasks = await self._composer.generate_tasks(
                question, target_hypothesis, agent_types, session.id,
            )
            results = await self._execute_specs_with_semaphore(
                specs, tasks, target_hypothesis, config, semaphore,
            )

            self._all_results.extend(results)

            # Track token usage
            for r in results:
                self._session_tokens_used += r.llm_tokens_used
                if self._token_budget:
                    self._token_budget.record_usage(
                        target_hypothesis.id, r.agent_id, r.llm_tokens_used,
                    )

            # Step 4: Synthesize answer from agent results
            answer = await self._synthesize_benchmark_answer(question, results)

            duration_ms = int(time.monotonic() * 1000) - start_ms
            session.status = SessionStatus.COMPLETED

            benchmark_result = {
                "task_id": task_id,
                "benchmark": benchmark_name,
                "question": question,
                "answer": answer,
                "expected_answer": expected_answer,
                "session_id": session.id,
                "agents_used": len(results),
                "agents_succeeded": sum(1 for r in results if r.success),
                "total_tokens": self._session_tokens_used,
                "total_llm_calls": sum(r.llm_calls for r in results),
                "duration_ms": duration_ms,
                "edges_added": sum(len(r.edges_added) for r in results),
                "nodes_added": sum(len(r.nodes_added) for r in results),
                "falsification_results": sum(
                    len(r.falsification_results) for r in results
                ),
                "agent_summaries": [
                    {"agent_type": str(r.agent_type), "summary": r.summary[:300]}
                    for r in results if r.success and r.summary
                ],
            }

            self._emit(
                "benchmark_completed",
                task_id=task_id,
                benchmark=benchmark_name,
                duration_ms=duration_ms,
                answer_preview=answer[:200],
            )

            return benchmark_result

        except Exception as exc:
            session.status = SessionStatus.FAILED
            self.audit.error(
                "benchmark_failed",
                task_id=task_id,
                benchmark=benchmark_name,
                error=str(exc),
            )
            return {
                "task_id": task_id,
                "benchmark": benchmark_name,
                "question": question,
                "answer": "",
                "error": str(exc),
                "session_id": session.id,
                "duration_ms": int(time.monotonic() * 1000) - start_ms,
            }

    async def _decompose_benchmark_question(
        self,
        question: str,
    ) -> dict[str, str]:
        """Decompose a benchmark question into a testable hypothesis."""
        prompt = (
            f"Benchmark question: {question}\n\n"
            "Convert this into a testable hypothesis for investigation by biomedical "
            "research agents. The hypothesis should be specific enough to guide tool use "
            "(literature search, protein analysis, pathway analysis, etc.).\n\n"
            'Return JSON: {"hypothesis": "...", "rationale": "..."}'
        )

        try:
            response = await self.llm.query(
                prompt,
                system_prompt=(
                    "Convert benchmark questions into testable hypotheses. "
                    "Be specific and actionable. The hypothesis should guide "
                    "which tools and databases to query."
                ),
                research_id=self._session.id if self._session else "",
            )
            parsed = LLMClient.parse_json(response)
            if isinstance(parsed, dict) and parsed.get("hypothesis"):
                return {
                    "hypothesis": parsed["hypothesis"],
                    "rationale": parsed.get("rationale", ""),
                }
        except Exception as exc:
            logger.warning("benchmark_decomposition_failed", error=str(exc))

        # Fallback: use the question directly
        return {
            "hypothesis": f"The answer to '{question[:200]}' can be determined through evidence-based investigation",
            "rationale": "Direct investigation of the benchmark question",
        }

    async def _synthesize_benchmark_answer(
        self,
        question: str,
        results: list[AgentResult],
    ) -> str:
        """Synthesize a final answer from agent investigation results."""
        # Collect summaries and key findings
        findings = []
        for r in results:
            if r.success and r.summary:
                findings.append(f"[{r.agent_type}] {r.summary}")
            for edge in r.edges_added:
                if edge.confidence.overall >= 0.6 and not edge.falsified:
                    findings.append(
                        f"  Edge: {edge.relation} (confidence={edge.confidence.overall:.2f})"
                    )

        prompt = (
            f"Question: {question}\n\n"
            f"Investigation findings:\n" + "\n".join(findings[:30]) + "\n\n"
            "Based on these findings, provide a concise, direct answer to the question. "
            "Include key evidence and confidence level. Be specific and factual."
        )

        try:
            answer = await self.llm.query(
                prompt,
                system_prompt=(
                    "You are synthesizing research findings into a direct answer. "
                    "Be concise, factual, and cite the most relevant evidence."
                ),
                research_id=self._session.id if self._session else "",
            )
            return answer.strip()
        except Exception as exc:
            logger.warning("benchmark_synthesis_failed", error=str(exc))
            # Fallback: concatenate agent summaries
            return " | ".join(
                r.summary for r in results if r.success and r.summary
            ) or "Unable to synthesize answer"
