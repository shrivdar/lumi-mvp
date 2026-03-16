"""BaseAgent — concrete implementation of the agent execute loop.

Provides the template-method pattern: subclasses override ``_investigate()``
while this class handles KG context, LLM calls, KG writes, falsification,
uncertainty, audit logging, and sub-agent spawning.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any

import structlog

from core.audit import AuditLogger, Timer, set_request_context
from core.constants import MAX_SUB_AGENT_DEPTH, MAX_SUB_AGENTS_PER_PARENT
from core.exceptions import AgentError, TokenBudgetExceededError
from core.interfaces import BaseTool, KnowledgeGraph, YamiInterface
from core.models import (
    AgentConstraints,
    AgentResult,
    AgentSpec,
    AgentTask,
    AgentTemplate,
    AgentTurn,
    AgentType,
    EdgeConfidence,
    EdgeRelationType,
    EvidenceSource,
    EvidenceSourceType,
    FalsificationResult,
    KGEdge,
    KGNode,
    NodeType,
    TurnType,
    UncertaintyVector,
)
from integrations.data_lake import data_lake_context
from know_how.retriever import KnowHowRetriever

logger = structlog.get_logger(__name__)


class BaseAgentImpl:
    """Concrete base agent with full execute loop, LLM integration, and falsification.

    Subclasses must implement ``_investigate(task, kg_context)`` which returns
    a dict with ``nodes``, ``edges``, ``summary``, ``reasoning_trace``, and
    optionally ``recommended_next``.
    """

    agent_type: AgentType = AgentType.LITERATURE_ANALYST  # overridden by subclass

    def __init__(
        self,
        *,
        agent_id: str | None = None,
        template: AgentTemplate | None = None,
        spec: AgentSpec | None = None,
        llm: Any,  # LLMClient
        kg: KnowledgeGraph,
        yami: YamiInterface | None = None,
        tools: dict[str, BaseTool] | None = None,
        audit_logger: AuditLogger | None = None,
        parent_agent_id: str | None = None,
        depth: int = 0,
        trajectory_collector: Any | None = None,  # rl.TrajectoryCollector
    ) -> None:
        if template is None and spec is None:
            raise ValueError("Either template or spec must be provided")

        self.agent_id = agent_id or str(uuid.uuid4())
        self.template = template
        self.spec = spec
        self.llm = llm
        self.kg = kg
        self.yami = yami
        self.tools = tools or {}
        self.audit = audit_logger or AuditLogger("agents")
        self._know_how_retriever = KnowHowRetriever()
        self._current_know_how: str = ""
        self.parent_agent_id = parent_agent_id or (spec.parent_agent_id if spec else None)
        self.depth = depth
        self.trajectory_collector = trajectory_collector

        # Resolve agent_type: spec hint → template type → class default
        if spec and spec.agent_type_hint:
            self.agent_type = spec.agent_type_hint
        elif template:
            self.agent_type = template.agent_type
        # else: keep class-level default

        # Tracking state during execution
        self._nodes_added: list[KGNode] = []
        self._edges_added: list[KGEdge] = []
        self._nodes_updated: list[str] = []
        self._edges_updated: list[str] = []
        self._llm_calls: int = 0
        self._llm_tokens: int = 0
        self._errors: list[str] = []
        self._sub_agent_results: list[AgentResult] = []
        self._sub_agents_spawned: int = 0

    # ------------------------------------------------------------------
    # Spec/template accessors — unified resolution
    # ------------------------------------------------------------------

    @property
    def effective_system_prompt(self) -> str:
        """Return system prompt: spec override → template → empty."""
        if self.spec and self.spec.system_prompt:
            return self.spec.system_prompt
        if self.template:
            return self.template.system_prompt
        return ""

    @property
    def effective_kg_write_permissions(self) -> list:
        """Return KG write permissions: spec override → template → all types."""
        if self.spec and self.spec.kg_write_permissions:
            return self.spec.kg_write_permissions
        if self.template:
            return self.template.kg_write_permissions
        return list(NodeType)

    @property
    def effective_kg_edge_permissions(self) -> list:
        """Return KG edge permissions: spec override → template → all types."""
        if self.spec and self.spec.kg_edge_permissions:
            return self.spec.kg_edge_permissions
        if self.template:
            return self.template.kg_edge_permissions
        return list(EdgeRelationType)

    @property
    def effective_falsification_protocol(self) -> str:
        """Return falsification protocol: spec override → template → empty."""
        if self.spec and self.spec.falsification_protocol:
            return self.spec.falsification_protocol
        if self.template:
            return self.template.falsification_protocol
        return ""

    @property
    def effective_constraints(self) -> AgentConstraints:
        """Return agent constraints from spec, or defaults derived from template."""
        if self.spec:
            return self.spec.constraints
        if self.template:
            return AgentConstraints(
                max_turns=self.template.max_iterations * 2,
                timeout_seconds=self.template.timeout_seconds,
            )
        return AgentConstraints()

    # ------------------------------------------------------------------
    # Main execute loop (template method)
    # ------------------------------------------------------------------

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute the full agent loop:

        1. Set audit context
        2. Query KG for relevant subgraph
        3. Call ``_investigate()`` (subclass hook)
        4. Write nodes/edges to KG
        5. Run self-falsification
        6. Compute uncertainty
        7. Return AgentResult
        """
        set_request_context(
            research_id=task.research_id,
            agent_id=self.agent_id,
        )
        self.audit.log("agent_execute_start", agent_id=self.agent_id, task_id=task.task_id)

        # Reset per-execution state
        self._nodes_added = []
        self._edges_added = []
        self._nodes_updated = []
        self._edges_updated = []
        self._llm_calls = 0
        self._llm_tokens = 0
        self._errors = []
        self._sub_agent_results = []
        self._sub_agents_spawned = 0

        start_ms = int(time.monotonic() * 1000)

        try:
            # 1. Get KG context
            kg_context = self._build_kg_context(task)

            # 1b. Retrieve domain know-how for injection into LLM calls
            self._current_know_how = ""
            try:
                know_how = await self._know_how_retriever.get_context_for_task(
                    task_instruction=task.instruction,
                    agent_type=str(self.agent_type),
                    context=task.context.get("query", ""),
                )
                if know_how:
                    self._current_know_how = know_how
                    self.audit.log(
                        "agent_knowhow_injected",
                        agent_id=self.agent_id,
                        task_id=task.task_id,
                        know_how_len=len(know_how),
                    )
            except Exception as exc:
                # Know-how injection is non-critical — do not fail the agent
                self.audit.error("agent_knowhow_failed", agent_id=self.agent_id, error=str(exc))

            # 2. Investigate (subclass hook)
            investigation = await self._investigate(task, kg_context)

            # 3. Write results to KG
            nodes = investigation.get("nodes", [])
            edges = investigation.get("edges", [])

            for node in nodes:
                self.write_node(node, task.hypothesis_branch)
            for edge in edges:
                self.write_edge(edge, task.hypothesis_branch)

            # 4. Self-falsification
            falsification_results: list[FalsificationResult] = []
            if self._edges_added and self.effective_falsification_protocol:
                try:
                    falsification_results = await self.falsify(self._edges_added)
                except Exception as exc:
                    self._errors.append(f"Falsification error: {exc}")
                    self.audit.error("falsification_failed", agent_id=self.agent_id, error=str(exc))

            # 5. Compute uncertainty
            uncertainty = self.get_uncertainty()

            duration_ms = int(time.monotonic() * 1000) - start_ms

            self.audit.log(
                "agent_execute_complete",
                agent_id=self.agent_id,
                task_id=task.task_id,
                nodes_added=len(self._nodes_added),
                edges_added=len(self._edges_added),
                duration_ms=duration_ms,
            )

            result = AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                hypothesis_id=task.hypothesis_branch or "",
                parent_agent_id=self.parent_agent_id,
                depth=self.depth,
                nodes_added=self._nodes_added,
                edges_added=self._edges_added,
                nodes_updated=self._nodes_updated,
                edges_updated=self._edges_updated,
                falsification_results=falsification_results,
                uncertainty=uncertainty,
                summary=investigation.get("summary", ""),
                reasoning_trace=investigation.get("reasoning_trace", ""),
                recommended_next=investigation.get("recommended_next"),
                sub_agent_results=self._sub_agent_results,
                token_usage=self.llm.token_summary if self.llm else {},
                duration_ms=duration_ms,
                llm_calls=self._llm_calls,
                llm_tokens_used=self._llm_tokens,
                turns=investigation.get("turns", []),
                success=True,
                errors=self._errors,
            )
            self._record_trajectory(task, result)
            return result

        except TokenBudgetExceededError as exc:
            # Hard kill — return partial results, agent is terminated
            duration_ms = int(time.monotonic() * 1000) - start_ms
            self.audit.log(
                "agent_token_budget_hard_kill",
                agent_id=self.agent_id,
                task_id=task.task_id,
                tokens_used=self._llm_tokens,
                budget=exc.details.get("budget", 0),
            )
            result = AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                hypothesis_id=task.hypothesis_branch or "",
                parent_agent_id=self.parent_agent_id,
                depth=self.depth,
                nodes_added=self._nodes_added,
                edges_added=self._edges_added,
                nodes_updated=self._nodes_updated,
                edges_updated=self._edges_updated,
                uncertainty=self.get_uncertainty(),
                summary=f"Agent killed: token budget exceeded ({self._llm_tokens} tokens used)",
                sub_agent_results=self._sub_agent_results,
                duration_ms=duration_ms,
                llm_calls=self._llm_calls,
                llm_tokens_used=self._llm_tokens,
                success=False,
                errors=[*self._errors, f"TOKEN_BUDGET_HARD_KILL: {exc}"],
            )
            self._record_trajectory(task, result)
            return result

        except Exception as exc:
            duration_ms = int(time.monotonic() * 1000) - start_ms
            self.audit.error(
                "agent_execute_failed",
                agent_id=self.agent_id,
                task_id=task.task_id,
                error=str(exc),
            )
            result = AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                hypothesis_id=task.hypothesis_branch or "",
                parent_agent_id=self.parent_agent_id,
                depth=self.depth,
                nodes_added=self._nodes_added,
                edges_added=self._edges_added,
                uncertainty=self.get_uncertainty(),
                summary=f"Agent failed: {exc}",
                sub_agent_results=self._sub_agent_results,
                duration_ms=duration_ms,
                llm_calls=self._llm_calls,
                llm_tokens_used=self._llm_tokens,
                success=False,
                errors=[*self._errors, str(exc)],
            )
            self._record_trajectory(task, result)
            return result

    # ------------------------------------------------------------------
    # Trajectory recording
    # ------------------------------------------------------------------

    def _record_trajectory(self, task: AgentTask, result: AgentResult) -> None:
        """Record the trajectory if a collector is attached. Non-critical."""
        if self.trajectory_collector is None:
            return
        try:
            self.trajectory_collector.collect(task, result)
        except Exception as exc:
            logger.warning("trajectory_record_failed", agent_id=self.agent_id, error=str(exc))

    # ------------------------------------------------------------------
    # Subclass hook — override this
    # ------------------------------------------------------------------

    async def _investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Subclasses implement their investigation logic here.

        Must return a dict with keys:
        - ``nodes``: list[KGNode]
        - ``edges``: list[KGEdge]
        - ``summary``: str
        - ``reasoning_trace``: str
        - ``recommended_next``: str | None

        When running from an AgentSpec (no subclass), the base implementation
        runs the multi-turn loop using the spec's role and instructions.
        """
        if self.spec is not None:
            return await self._multi_turn_investigate(
                task,
                kg_context,
                investigation_focus="",  # spec role/instructions injected automatically
            )
        raise NotImplementedError("Subclasses must implement _investigate()")

    # ------------------------------------------------------------------
    # KG context building
    # ------------------------------------------------------------------

    def _build_kg_context(self, task: AgentTask) -> dict[str, Any]:
        """Build KG context dict from task's kg_context node/edge IDs."""
        context: dict[str, Any] = {"nodes": [], "edges": []}

        # If specific IDs given, build subgraph around first node
        if task.kg_context:
            for node_id in task.kg_context:
                node = self.kg.get_node(node_id)
                if node:
                    subgraph = self.kg.get_subgraph(node_id, hops=1)
                    for n in subgraph.get("nodes", []):
                        if n not in context["nodes"]:
                            context["nodes"].append(n)
                    for e in subgraph.get("edges", []):
                        if e not in context["edges"]:
                            context["edges"].append(e)

        # Add stats
        context["stats"] = {
            "total_nodes": self.kg.node_count(),
            "total_edges": self.kg.edge_count(),
            "avg_confidence": self.kg.avg_confidence(),
        }

        return context

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    async def query_llm(
        self,
        prompt: str,
        *,
        kg_context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Wraps LLM call with system prompt, KG injection, know-how, audit, and token tracking."""
        sys_prompt = system_prompt or self.effective_system_prompt
        # Inject domain know-how if retrieved for this execution
        if self._current_know_how:
            sys_prompt = sys_prompt + "\n\n" + self._current_know_how
        # Inject data lake context so agents know what local datasets are available
        dl_ctx = data_lake_context()
        if dl_ctx:
            sys_prompt = sys_prompt + "\n\n" + dl_ctx

        self.audit.log(
            "agent_llm_call",
            agent_id=self.agent_id,
            prompt_len=len(prompt),
        )

        # Snapshot session-level token counts BEFORE the call so we can
        # compute the per-call delta (LLMClient tracks cumulative totals).
        _tokens_before = 0
        if hasattr(self.llm, "token_summary"):
            _tokens_before = self.llm.token_summary.get("total_tokens", 0)

        with Timer() as t:
            response = await self.llm.query(
                prompt,
                system_prompt=sys_prompt,
                kg_context=kg_context,
                agent_id=self.agent_id,
                **({"max_tokens": max_tokens} if max_tokens else {}),
            )

        self._llm_calls += 1
        if hasattr(self.llm, "token_summary"):
            _tokens_after = self.llm.token_summary.get("total_tokens", 0)
            self._llm_tokens += _tokens_after - _tokens_before

        self.audit.log(
            "agent_llm_result",
            agent_id=self.agent_id,
            duration_ms=t.elapsed_ms,
            response_len=len(response),
        )

        return response

    # ------------------------------------------------------------------
    # Yami helper
    # ------------------------------------------------------------------

    async def query_yami(self, method: str, **kwargs: Any) -> dict[str, Any]:
        """Wraps Yami calls with audit logging and error handling."""
        if self.yami is None:
            raise AgentError(
                "Yami interface not available",
                error_code="YAMI_NOT_AVAILABLE",
            )

        self.audit.log("agent_yami_call", agent_id=self.agent_id, method=method)

        with Timer() as t:
            fn = getattr(self.yami, method)
            result = await fn(**kwargs)

        self.audit.log(
            "agent_yami_result",
            agent_id=self.agent_id,
            method=method,
            duration_ms=t.elapsed_ms,
        )

        return result if isinstance(result, dict) else {"result": result}

    # ------------------------------------------------------------------
    # KG write helpers
    # ------------------------------------------------------------------

    def write_node(self, node: KGNode, hypothesis_branch: str | None = None) -> str:
        """Write node to KG with agent_id and hypothesis_branch set. Audit logged."""
        node.created_by = self.agent_id
        if hypothesis_branch:
            node.hypothesis_branch = hypothesis_branch

        node_id = self.kg.add_node(node)
        self._nodes_added.append(node)

        self.audit.kg_mutation(
            "add_node",
            agent_id=self.agent_id,
            hypothesis_branch=hypothesis_branch or "",
            node_id=node_id,
            node_type=str(node.type),
            node_name=node.name,
        )

        return node_id

    def write_edge(self, edge: KGEdge, hypothesis_branch: str | None = None) -> str:
        """Write edge to KG with agent_id and hypothesis_branch set.

        Auto-triggers contradiction check via KG. Audit logged.
        """
        edge.created_by = self.agent_id
        if hypothesis_branch:
            edge.hypothesis_branch = hypothesis_branch

        edge_id = self.kg.add_edge(edge)
        self._edges_added.append(edge)

        self.audit.kg_mutation(
            "add_edge",
            agent_id=self.agent_id,
            hypothesis_branch=hypothesis_branch or "",
            edge_id=edge_id,
            relation=str(edge.relation),
        )

        return edge_id

    # ------------------------------------------------------------------
    # Falsification
    # ------------------------------------------------------------------

    async def falsify(self, edges: list[KGEdge]) -> list[FalsificationResult]:
        """For each edge, ask LLM for counter-evidence query, search, adjust confidence."""
        results: list[FalsificationResult] = []

        for edge in edges:
            source_node = self.kg.get_node(edge.source_id)
            target_node = self.kg.get_node(edge.target_id)
            source_name = source_node.name if source_node else edge.source_id
            target_name = target_node.name if target_node else edge.target_id

            # Ask LLM what would disprove this edge
            falsification_prompt = (
                f"I have a knowledge graph edge claiming: {source_name} --[{edge.relation}]--> {target_name}\n"
                f"Evidence: {[e.claim or e.title or '' for e in edge.evidence]}\n\n"
                f"1. What would disprove this claim?\n"
                f"2. Provide a precise search query to find counter-evidence.\n"
                f"Respond as JSON: {{\"disproof_criteria\": \"...\", \"search_query\": \"...\"}}"
            )

            try:
                llm_response = await self.query_llm(falsification_prompt)
                parsed = self.llm.parse_json(llm_response)
                search_query = parsed.get("search_query", f"NOT {source_name} {edge.relation} {target_name}")
            except Exception:
                search_query = f"{source_name} {target_name} contradicts disproven negative"

            # Search for counter-evidence using available tools
            tool_result_keys = {"pubmed": "articles", "semantic_scholar": "papers"}
            candidate_papers: list[tuple[str, dict]] = []  # (tool_name, paper)
            for tool_name in ["pubmed", "semantic_scholar"]:
                tool = self.tools.get(tool_name)
                if tool is None:
                    continue
                try:
                    search_results = await tool.execute(action="search", query=search_query, max_results=3)
                    result_key = tool_result_keys.get(tool_name, "results")
                    for paper in search_results.get(result_key, []):
                        candidate_papers.append((tool_name, paper))
                except Exception:
                    continue

            # LLM evaluation: ask whether each candidate actually contradicts the claim
            claim_statement = f"{source_name} {edge.relation} {target_name}"
            counter_evidence: list[EvidenceSource] = []
            for tool_name, paper in candidate_papers:
                abstract = paper.get("abstract") or ""
                title = paper.get("title") or ""
                if not abstract and not title:
                    continue
                try:
                    eval_prompt = (
                        f"Does the following paper provide evidence AGAINST the claim: "
                        f"\"{claim_statement}\"?\n\n"
                        f"Paper title: {title}\n"
                        f"Abstract: {abstract}\n\n"
                        f"Respond as JSON: {{\"contradicts\": true/false, \"reasoning\": \"...\"}}"
                    )
                    eval_response = await self.query_llm(eval_prompt)
                    eval_parsed = self.llm.parse_json(eval_response)
                    is_counter = eval_parsed.get("contradicts", False)
                except Exception:
                    # If LLM eval fails, fall back to treating it as potential counter-evidence
                    is_counter = True

                if is_counter:
                    counter_evidence.append(
                        EvidenceSource(
                            source_type=(
                                EvidenceSourceType.PUBMED
                                if tool_name == "pubmed"
                                else EvidenceSourceType.SEMANTIC_SCHOLAR
                            ),
                            source_id=paper.get("pmid") or paper.get("paper_id", ""),
                            title=title,
                            claim=f"Counter-evidence for {claim_statement}",
                            doi=paper.get("doi"),
                            quality_score=0.5,
                            confidence=0.4,
                            agent_id=self.agent_id,
                        )
                    )

            # Assess result
            original_confidence = edge.confidence.overall
            counter_found = len(counter_evidence) > 0

            if counter_found:
                # Lower confidence
                confidence_delta = -min(0.15, 0.05 * len(counter_evidence))
                revised = max(0.05, original_confidence + confidence_delta)
                falsified = revised < 0.3

                # Update edge in KG
                evidence_source = EvidenceSource(
                    source_type=EvidenceSourceType.AGENT_REASONING,
                    claim=f"Falsification: found {len(counter_evidence)} potential counter-evidence(s)",
                    quality_score=0.5,
                    confidence=revised,
                    agent_id=self.agent_id,
                )
                self.kg.update_edge_confidence(edge.id, revised, evidence_source)
                self._edges_updated.append(edge.id)

                if falsified:
                    self.kg.mark_edge_falsified(edge.id, counter_evidence)
            else:
                # Survived falsification — slight boost
                confidence_delta = 0.02
                revised = min(1.0, original_confidence + confidence_delta)
                evidence_source = EvidenceSource(
                    source_type=EvidenceSourceType.AGENT_REASONING,
                    claim="Survived falsification attempt — no counter-evidence found",
                    quality_score=0.5,
                    confidence=revised,
                    agent_id=self.agent_id,
                )
                self.kg.update_edge_confidence(edge.id, revised, evidence_source)
                self._edges_updated.append(edge.id)
                falsified = False

            result = FalsificationResult(
                edge_id=edge.id,
                agent_id=self.agent_id,
                hypothesis_branch=edge.hypothesis_branch or "",
                original_confidence=original_confidence,
                revised_confidence=revised,
                falsified=falsified,
                search_query=search_query,
                method="llm_directed_search",
                counter_evidence_found=counter_found,
                counter_evidence=counter_evidence,
                reasoning=f"Searched for counter-evidence. Found {len(counter_evidence)} results.",
                confidence_delta=confidence_delta,
            )
            results.append(result)

            self.audit.falsification(
                agent_id=self.agent_id,
                edge_id=edge.id,
                result="falsified" if falsified else ("weakened" if counter_found else "survived"),
                original_confidence=original_confidence,
                revised_confidence=revised,
            )

        return results

    # ------------------------------------------------------------------
    # Uncertainty
    # ------------------------------------------------------------------

    def get_uncertainty(self) -> UncertaintyVector:
        """Compute uncertainty vector from agent state."""
        # Data quality: based on evidence quality of edges we added
        evidence_scores = []
        for edge in self._edges_added:
            for ev in edge.evidence:
                evidence_scores.append(ev.quality_score)
        data_quality = 1.0 - (sum(evidence_scores) / len(evidence_scores)) if evidence_scores else 0.5

        # Conflict: check for contradictions
        contradiction_count = 0
        for edge in self._edges_added:
            contradictions = self.kg.get_contradictions(edge)
            contradiction_count += len(contradictions)
        conflict = min(1.0, contradiction_count * 0.2)

        # Input ambiguity: how many errors occurred
        input_ambiguity = min(1.0, len(self._errors) * 0.2)

        uv = UncertaintyVector(
            input_ambiguity=input_ambiguity,
            data_quality=data_quality,
            reasoning_divergence=0.0,
            model_disagreement=0.0,
            conflict_uncertainty=conflict,
            novelty_uncertainty=0.0,
        )
        uv.compute_composite()
        uv.is_critical = uv.composite > 0.6
        return uv

    # ------------------------------------------------------------------
    # Tool helper
    # ------------------------------------------------------------------

    async def call_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        """Call a tool by name with audit logging and error handling."""
        tool = self.tools.get(tool_name)
        if tool is None:
            raise AgentError(
                f"Tool '{tool_name}' not available",
                error_code="TOOL_NOT_FOUND",
                details={"available": list(self.tools.keys())},
            )

        self.audit.tool_call(tool_name, self.agent_id, **{k: str(v)[:100] for k, v in kwargs.items()})

        with Timer() as t:
            result = await tool.execute(**kwargs)

        self.audit.tool_result(tool_name, self.agent_id, success=True, duration_ms=t.elapsed_ms)
        return result

    # ------------------------------------------------------------------
    # Multi-turn investigation loop
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tag(text: str, tag: str, *, allow_truncated: bool = False) -> str | None:
        """Extract content from an XML tag in an LLM response.

        If *allow_truncated* is True and the opening tag is found but the
        closing tag is missing (e.g. due to max_tokens truncation), return
        everything after the opening tag.
        """
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        if allow_truncated:
            open_tag = f"<{tag}>"
            idx = text.find(open_tag)
            if idx != -1:
                return text[idx + len(open_tag):].strip()
        return None

    def _parse_agent_response(self, response: str) -> tuple[str, str, str | None]:
        """Parse LLM response for action tags.

        Returns ``(action_type, content, think_content)``.
        Priority: answer > tool > execute > think.

        For ``<answer>`` tags we also accept truncated responses (opening tag
        present but closing tag missing due to max-token cutoff).
        """
        think = self._extract_tag(response, "think")
        answer = self._extract_tag(response, "answer", allow_truncated=True)
        if answer:
            return ("answer", answer, think)
        tool = self._extract_tag(response, "tool")
        if tool:
            return ("tool", tool, think)
        execute = self._extract_tag(response, "execute")
        if execute:
            return ("execute", execute, think)
        return ("think", response, think or response)

    async def _execute_tool_action(self, tool_str: str) -> str:
        """Parse and execute a tool call from ``<tool>name:args</tool>``."""
        colon_idx = tool_str.find(":")
        if colon_idx == -1:
            return f"Error: Invalid tool format. Expected 'tool_name:{{...}}', got: {tool_str[:100]}"

        tool_name = tool_str[:colon_idx].strip()
        args_str = tool_str[colon_idx + 1:].strip()

        try:
            kwargs = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON args for tool '{tool_name}': {e}"

        # KG virtual tools
        if tool_name.startswith("kg_"):
            return self._execute_kg_tool(tool_name, kwargs)

        # Yami
        if tool_name == "yami" and self.yami:
            method = kwargs.pop("method", "")
            if not method:
                return "Error: Yami tool requires 'method' argument"
            try:
                result = await self.query_yami(method, **kwargs)
                return json.dumps(result, indent=2, default=str)[:4000]
            except Exception as e:
                return f"Error calling Yami: {e}"

        # Regular tool
        try:
            result = await self.call_tool(tool_name, **kwargs)
            return json.dumps(result, indent=2, default=str)[:4000]
        except Exception as e:
            return f"Error calling tool '{tool_name}': {e}"

    def _execute_kg_tool(self, tool_name: str, kwargs: dict[str, Any]) -> str:
        """Dispatch KG virtual-tool calls (get_recent_edges, etc.)."""

        def _edge_to_dict(e: KGEdge) -> dict[str, Any]:
            src = self.kg.get_node(e.source_id)
            tgt = self.kg.get_node(e.target_id)
            return {
                "id": e.id,
                "source": src.name if src else e.source_id,
                "target": tgt.name if tgt else e.target_id,
                "relation": str(e.relation),
                "confidence": e.confidence.overall,
                "evidence": [ev.claim or ev.title or "" for ev in e.evidence[:2]],
                "falsified": e.falsified,
            }

        if tool_name == "kg_get_recent_edges":
            edges = self.kg.get_recent_edges(n=kwargs.get("n", 20))
            return json.dumps([_edge_to_dict(e) for e in edges], indent=2, default=str)[:4000]

        if tool_name == "kg_get_weakest_edges":
            edges = self.kg.get_weakest_edges(n=kwargs.get("n", 10))
            return json.dumps([_edge_to_dict(e) for e in edges], indent=2, default=str)[:4000]

        if tool_name == "kg_get_orphan_nodes":
            nodes = self.kg.get_orphan_nodes()
            return json.dumps(
                [{"id": n.id, "name": n.name, "type": str(n.type)} for n in nodes[:20]],
                indent=2, default=str,
            )[:4000]

        if tool_name == "kg_update_edge_confidence":
            edge_id = kwargs.get("edge_id", "")
            new_confidence = float(kwargs.get("confidence", 0.5))
            reason = kwargs.get("reason", "")

            edge = self.kg.get_edge(edge_id)
            if not edge:
                return f"Error: Edge '{edge_id}' not found"

            ev = EvidenceSource(
                source_type=EvidenceSourceType.AGENT_REASONING,
                claim=reason,
                quality_score=0.6,
                confidence=new_confidence,
                agent_id=self.agent_id,
            )
            self.kg.update_edge_confidence(edge_id, new_confidence, ev)
            self._edges_updated.append(edge_id)

            if new_confidence < 0.3:
                self.kg.mark_edge_falsified(edge_id, [ev])

            return json.dumps({"status": "updated", "edge_id": edge_id, "new_confidence": new_confidence})

        return f"Error: Unknown KG tool '{tool_name}'"

    async def _execute_code_action(self, code: str) -> str:
        """Execute code via PythonREPLTool (mock-safe for initial dev)."""
        repl = self.tools.get("python_repl")
        if repl:
            try:
                result = await repl.execute(code=code)
                return str(result.get("output", ""))[:4000]
            except Exception as e:
                return f"Error executing code: {e}"
        return f"Code execution not available (PythonREPLTool not configured). Code:\n{code[:500]}"

    def _build_tool_descriptions(self) -> str:
        """Build a description of available tools for multi-turn prompts."""
        descriptions: list[str] = []
        if self.tools:
            for name, tool in self.tools.items():
                desc = getattr(tool, "description", f"{name} tool")
                descriptions.append(f"  - {name}: {desc}")
        else:
            descriptions.append("  (No external tools — reasoning only)")

        # KG virtual tools always available
        descriptions.extend([
            "  - kg_get_recent_edges: Get N most recent KG edges. Args: {\"n\": 20}",
            "  - kg_get_weakest_edges: Get N lowest-confidence KG edges. Args: {\"n\": 10}",
            "  - kg_get_orphan_nodes: Get KG nodes with no connections. Args: {}",
            "  - kg_update_edge_confidence: Update an edge's confidence. "
            "Args: {\"edge_id\": \"...\", \"confidence\": 0.5, \"reason\": \"...\"}",
        ])

        if self.yami:
            descriptions.append(
                "  - yami: Protein structure prediction (ESMFold). "
                "Args: {\"method\": \"predict_structure\", \"sequence\": \"...\"}"
            )

        return "\n".join(descriptions)

    async def _compress_observations(
        self,
        observations: list[str],
        *,
        keep_recent: int = 10,
    ) -> list[str]:
        """Compress older observation history when it grows too large.

        Keeps the last *keep_recent* observations verbatim and summarises
        the rest into a single compressed block via an LLM call.  This
        prevents context-window bloat in long-running investigations.
        """
        if len(observations) <= keep_recent + 1:
            return observations

        old_obs = observations[:-keep_recent]
        recent_obs = observations[-keep_recent:]

        # Estimate token count (~4 chars per token)
        old_text = "\n\n".join(old_obs)
        estimated_tokens = len(old_text) // 4

        # Only compress if the old observations are substantial
        if estimated_tokens < 5_000:
            return observations

        compress_prompt = (
            "Summarise the following agent observation history into a concise "
            "context block.  Preserve key facts, tool results, and findings. "
            "Drop redundant reasoning and failed attempts.  Keep it under "
            "2000 characters.\n\n"
            f"--- OBSERVATIONS ---\n{old_text[:20_000]}\n--- END ---"
        )

        try:
            summary = await self.query_llm(compress_prompt, max_tokens=1024)
            compressed = [f"[COMPRESSED HISTORY — {len(old_obs)} turns]\n{summary}"]
            return compressed + recent_obs
        except Exception:
            # If compression fails, just keep truncated old observations
            return [f"[HISTORY — {len(old_obs)} earlier turns omitted]"] + recent_obs

    async def _multi_turn_investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
        *,
        max_turns: int | None = None,
        token_budget: int = 0,
        investigation_focus: str = "",
    ) -> dict[str, Any]:
        """Multi-turn investigation loop: Plan → Generate → Execute → Observe → Repeat.

        Agents call this from their ``_investigate()`` override, passing
        an ``investigation_focus`` string that steers the LLM.

        Turn budget resolution: explicit ``max_turns`` arg → spec constraints →
        default (20).

        Args:
            max_turns: Maximum number of turns before forcing an answer.
                If None, resolved from spec constraints then default (20).
            token_budget: Per-agent token budget. When exceeded the agent
                is forced to answer on the next turn. 0 means no limit.
            investigation_focus: Steering string for the LLM.
        """
        # Resolve turn budget from spec constraints, then fallback to default
        constraints = self.effective_constraints
        if max_turns is None:
            max_turns = constraints.max_turns

        turns: list[AgentTurn] = []
        observations: list[str] = []

        tool_descriptions = self._build_tool_descriptions()

        allowed_nodes = ", ".join(str(t) for t in self.effective_kg_write_permissions)
        allowed_edges = ", ".join(str(t) for t in self.effective_kg_edge_permissions)

        # Inject spec role/instructions into the investigation focus if from a spec
        if self.spec:
            spec_prefix = f"Agent role: {self.spec.role}\nAgent instructions: {self.spec.instructions}\n\n"
            investigation_focus = spec_prefix + (investigation_focus or "")

        # ---- Phase 1: Planning ----
        plan_prompt = (
            f"Research task: {task.instruction}\n\n"
            + (f"Investigation focus: {investigation_focus}\n\n" if investigation_focus else "")
            + f"KG context: {len(kg_context.get('nodes', []))} nodes, "
            f"{len(kg_context.get('edges', []))} edges\n"
            f"KG stats: {kg_context.get('stats', {})}\n\n"
            f"Available tools:\n{tool_descriptions}\n\n"
            f"Allowed KG node types: {allowed_nodes}\n"
            f"Allowed KG edge types: {allowed_edges}\n\n"
            "Create a numbered plan (3-8 steps) to investigate this. "
            "Wrap your plan in <think> tags."
        )

        plan_response = await self.query_llm(plan_prompt, kg_context=kg_context)
        plan_think = self._extract_tag(plan_response, "think") or plan_response

        turns.append(AgentTurn(
            turn_number=0,
            turn_type=TurnType.THINK,
            input_prompt=plan_prompt[:500],
            raw_response=plan_response[:2000],
            parsed_action=plan_think[:2000],
        ))
        observations.append(f"[PLAN]\n{plan_think}")

        # ---- Phase 2: Multi-turn execution loop ----
        for turn_num in range(1, max_turns + 1):
            # Compress observations if they've grown large
            if len(observations) > 15:
                observations = await self._compress_observations(observations)

            # Token budget enforcement — force answer if budget exhausted
            budget_exhausted = (
                token_budget > 0 and self._llm_tokens >= token_budget
            )

            obs_text = "\n\n".join(observations[-10:])

            # Determine if we should nudge the agent toward answering
            remaining = max_turns - turn_num
            urgency = ""
            if budget_exhausted:
                urgency = (
                    "\n\n⚠️ TOKEN BUDGET EXHAUSTED. "
                    "Provide your <answer> NOW with whatever information you have gathered."
                )
            elif remaining <= 2:
                urgency = (
                    "\n\n⚠️ You are running low on turns. "
                    "Provide your <answer> NOW with whatever information you have gathered."
                )
            elif remaining <= 5:
                urgency = (
                    "\n\nNote: You have only a few turns left. "
                    "Start synthesizing your findings and prepare to answer soon."
                )

            turn_prompt = (
                f"Research task: {task.instruction}\n\n"
                f"Turn {turn_num}/{max_turns}\n\n"
                f"Previous observations:\n{obs_text}\n\n"
                f"Available tools:\n{tool_descriptions}\n\n"
                "Execute the next step. Use ONE of these formats:\n\n"
                "To reason: <think>your reasoning</think>\n"
                "To call a tool: <tool>tool_name:{\"arg\": \"value\"}</tool>\n"
                "To execute code: <execute>python_code</execute>\n"
                "To provide final answer (KEEP CONCISE — max 5-8 entities, 5-8 relationships, "
                "short descriptions):\n"
                "<answer>{\"entities\": [{\"name\": \"...\", "
                "\"type\": \"GENE|PROTEIN|DISEASE|...\", "
                "\"description\": \"brief\", \"confidence\": 0.8, "
                "\"evidence_source\": \"PUBMED\", "
                "\"evidence_id\": \"PMID:...\"}], "
                "\"relationships\": [{\"source\": \"name\", "
                "\"target\": \"name\", "
                "\"relation\": \"ASSOCIATED_WITH|...\", "
                "\"confidence\": 0.8, \"claim\": \"brief claim\", "
                "\"evidence_source\": \"PUBMED\", "
                "\"evidence_id\": \"PMID:...\"}], "
                "\"summary\": \"2-3 sentence summary\", "
                "\"reasoning_trace\": \"brief trace\"}</answer>\n\n"
                "If you have gathered enough information, provide your <answer>."
                + urgency
            )

            # Hard kill: token budget exceeded — raise immediately
            if constraints.token_budget and self._llm_tokens >= constraints.token_budget:
                raise TokenBudgetExceededError(
                    f"Agent {self.agent_id} token budget hard kill: "
                    f"{self._llm_tokens}/{constraints.token_budget}",
                    error_code="TOKEN_BUDGET_HARD_KILL",
                    details={
                        "agent_id": self.agent_id,
                        "tokens_used": self._llm_tokens,
                        "budget": constraints.token_budget,
                    },
                )

            start_ms = int(time.monotonic() * 1000)
            # Use higher token limit for later turns where answer is expected
            answer_max_tokens = 8192 if remaining <= 5 else None
            response = await self.query_llm(
                turn_prompt, kg_context=kg_context, max_tokens=answer_max_tokens,
            )
            duration = int(time.monotonic() * 1000) - start_ms

            action_type, content, think_content = self._parse_agent_response(response)

            if think_content and action_type != "think":
                observations.append(f"[THINK turn {turn_num}] {think_content[:500]}")

            if action_type == "answer":
                turns.append(AgentTurn(
                    turn_number=turn_num,
                    turn_type=TurnType.ANSWER,
                    input_prompt=turn_prompt[:500],
                    raw_response=response[:2000],
                    parsed_action=content[:2000],
                    duration_ms=duration,
                ))
                return self._compile_answer(content, turns, observations)

            if action_type == "tool":
                exec_result = await self._execute_tool_action(content)
                observations.append(
                    f"[TOOL turn {turn_num}] {content[:200]}\nResult: {exec_result[:2000]}"
                )
                turns.append(AgentTurn(
                    turn_number=turn_num,
                    turn_type=TurnType.TOOL_CALL,
                    input_prompt=turn_prompt[:500],
                    raw_response=response[:2000],
                    parsed_action=content[:500],
                    execution_result=exec_result[:2000],
                    duration_ms=duration,
                ))

            elif action_type == "execute":
                exec_result = await self._execute_code_action(content)
                error = exec_result if exec_result.startswith("Error") else None
                observations.append(
                    f"[CODE turn {turn_num}] {content[:200]}\nOutput: {exec_result[:2000]}"
                )
                turns.append(AgentTurn(
                    turn_number=turn_num,
                    turn_type=TurnType.CODE_EXECUTION,
                    input_prompt=turn_prompt[:500],
                    raw_response=response[:2000],
                    parsed_action=content[:500],
                    execution_result=exec_result[:2000],
                    error=error,
                    duration_ms=duration,
                ))

            else:  # think
                observations.append(f"[THINK turn {turn_num}] {content[:1000]}")
                turns.append(AgentTurn(
                    turn_number=turn_num,
                    turn_type=TurnType.THINK,
                    input_prompt=turn_prompt[:500],
                    raw_response=response[:2000],
                    parsed_action=content[:1000],
                    duration_ms=duration,
                ))

            # If token budget is exhausted and agent didn't answer, force stop
            if budget_exhausted:
                self.audit.log(
                    "agent_token_budget_stop",
                    agent_id=self.agent_id,
                    tokens_used=self._llm_tokens,
                    budget=token_budget,
                    turn=turn_num,
                )
                return self._compile_from_observations(turns, observations, task)

        # Turn budget exhausted — compile from observations
        return self._compile_from_observations(turns, observations, task)

    def _compile_answer(
        self,
        answer_content: str,
        turns: list[AgentTurn],
        observations: list[str],
    ) -> dict[str, Any]:
        """Parse ``<answer>`` content into the standard investigation result dict.

        Handles truncated JSON from max-token cutoff by attempting to repair
        the JSON before parsing.
        """
        try:
            parsed = self.llm.parse_json(answer_content)
        except Exception:
            # Attempt to repair truncated JSON
            parsed = self._repair_truncated_json(answer_content)
            if parsed is None:
                return {
                    "nodes": [],
                    "edges": [],
                    "summary": answer_content[:500],
                    "reasoning_trace": self._summarize_turns(turns),
                    "turns": turns,
                }

        nodes = self._parse_nodes_from_answer(parsed.get("entities", []))
        edges = self._parse_edges_from_answer(parsed.get("relationships", []), nodes)

        return {
            "nodes": nodes,
            "edges": edges,
            "summary": parsed.get("summary", "Multi-turn investigation complete."),
            "reasoning_trace": parsed.get(
                "reasoning_trace", self._summarize_turns(turns)
            ),
            "recommended_next": parsed.get("recommended_next"),
            "turns": turns,
        }

    @staticmethod
    def _repair_truncated_json(text: str) -> dict[str, Any] | None:
        """Attempt to repair truncated JSON from max-token cutoff.

        Strategy: find the first ``{``, then try progressively shorter
        substrings, closing any open braces/brackets.
        """
        # Find the start of JSON
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                start = i
                break
        if start == -1:
            return None

        json_text = text[start:]

        # Try parsing as-is first
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

        # Count unclosed braces/brackets and try to close them
        opens = 0
        open_brackets = 0
        in_string = False
        escape = False
        for ch in json_text:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                opens += 1
            elif ch == "}":
                opens -= 1
            elif ch == "[":
                open_brackets += 1
            elif ch == "]":
                open_brackets -= 1

        # Try closing with appropriate brackets/braces
        suffix = "]" * max(0, open_brackets) + "}" * max(0, opens)
        if suffix:
            try:
                return json.loads(json_text + suffix)
            except json.JSONDecodeError:
                pass

        # Last resort: try to find the last complete entity/relationship array
        # and build a minimal valid JSON
        try:
            # Find entities array
            entities_match = re.search(r'"entities"\s*:\s*\[(.*?)\]', json_text, re.DOTALL)
            relationships_match = re.search(r'"relationships"\s*:\s*\[(.*?)\]', json_text, re.DOTALL)
            summary_match = re.search(r'"summary"\s*:\s*"([^"]*)"', json_text)

            result: dict[str, Any] = {}
            if entities_match:
                try:
                    result["entities"] = json.loads("[" + entities_match.group(1) + "]")
                except json.JSONDecodeError:
                    result["entities"] = []
            if relationships_match:
                try:
                    result["relationships"] = json.loads("[" + relationships_match.group(1) + "]")
                except json.JSONDecodeError:
                    result["relationships"] = []
            if summary_match:
                result["summary"] = summary_match.group(1)

            if result:
                return result
        except Exception:
            pass

        return None

    def _compile_from_observations(
        self,
        turns: list[AgentTurn],
        observations: list[str],
        task: AgentTask,
    ) -> dict[str, Any]:
        """Fallback compilation when the turn budget is exhausted."""
        return {
            "nodes": [],
            "edges": [],
            "summary": (
                f"Investigation reached turn limit ({len(turns)} turns). "
                f"Observations collected: {len(observations)}"
            ),
            "reasoning_trace": self._summarize_turns(turns),
            "turns": turns,
        }

    def _parse_nodes_from_answer(self, entities: list[dict[str, Any]]) -> list[KGNode]:
        """Convert entity dicts from an LLM answer to ``KGNode`` instances."""
        nodes: list[KGNode] = []
        for entity in entities:
            try:
                node_type = NodeType(entity.get("type", "GENE"))
            except (ValueError, KeyError):
                node_type = NodeType.GENE

            ev_source = EvidenceSourceType.AGENT_REASONING
            try:
                if entity.get("evidence_source"):
                    ev_source = EvidenceSourceType(entity["evidence_source"])
            except (ValueError, KeyError):
                pass

            confidence = float(entity.get("confidence", 0.6))

            node = KGNode(
                type=node_type,
                name=entity.get("name", ""),
                description=entity.get("description", ""),
                external_ids=entity.get("external_ids", {}),
                properties=entity.get("properties", {}),
                confidence=confidence,
                sources=[
                    EvidenceSource(
                        source_type=ev_source,
                        source_id=entity.get("evidence_id"),
                        claim=entity.get("description", ""),
                        quality_score=confidence,
                        confidence=confidence,
                        agent_id=self.agent_id,
                    )
                ],
            )
            nodes.append(node)
        return nodes

    def _parse_edges_from_answer(
        self, relationships: list[dict[str, Any]], nodes: list[KGNode]
    ) -> list[KGEdge]:
        """Convert relationship dicts from an LLM answer to ``KGEdge`` instances."""
        name_to_id: dict[str, str] = {n.name: n.id for n in nodes}
        edges: list[KGEdge] = []

        for rel in relationships:
            source_name = rel.get("source", "")
            target_name = rel.get("target", "")

            source_id = name_to_id.get(source_name)
            target_id = name_to_id.get(target_name)

            # Resolve from KG if not in answer entities
            if not source_id:
                kg_node = self.kg.get_node_by_name(source_name)
                if kg_node:
                    source_id = kg_node.id
            if not target_id:
                kg_node = self.kg.get_node_by_name(target_name)
                if kg_node:
                    target_id = kg_node.id

            if not source_id or not target_id:
                continue

            try:
                relation = EdgeRelationType(rel.get("relation", "ASSOCIATED_WITH"))
            except (ValueError, KeyError):
                relation = EdgeRelationType.ASSOCIATED_WITH

            ev_source = EvidenceSourceType.AGENT_REASONING
            try:
                if rel.get("evidence_source"):
                    ev_source = EvidenceSourceType(rel["evidence_source"])
            except (ValueError, KeyError):
                pass

            confidence = float(rel.get("confidence", 0.5))

            edge = KGEdge(
                source_id=source_id,
                target_id=target_id,
                relation=relation,
                confidence=EdgeConfidence(
                    overall=confidence,
                    evidence_quality=confidence,
                    evidence_count=1,
                ),
                evidence=[
                    EvidenceSource(
                        source_type=ev_source,
                        source_id=rel.get("evidence_id"),
                        claim=rel.get("claim", ""),
                        confidence=confidence,
                        agent_id=self.agent_id,
                    )
                ],
            )
            edges.append(edge)
        return edges

    @staticmethod
    def _summarize_turns(turns: list[AgentTurn]) -> str:
        """Build a reasoning trace string from turn history."""
        parts: list[str] = []
        for t in turns:
            action = t.parsed_action[:200].replace("\n", " ")
            parts.append(f"Turn {t.turn_number} ({t.turn_type}): {action}")
        return "\n".join(parts)

    # Sub-agent spawning
    # ------------------------------------------------------------------

    async def spawn_sub_agent(
        self,
        task_description: str,
        *,
        agent_type: AgentType | None = None,
        spec: AgentSpec | None = None,
        tool_names: list[str] | None = None,
        hypothesis_branch: str | None = None,
    ) -> AgentResult:
        """Spawn a child agent to handle a sub-task.

        The sub-agent inherits the parent's KG (writes to the same graph),
        hypothesis branch, and LLM client. KG mutations are attributed with
        ``parent_agent_id`` pointing to this agent.

        Args:
            task_description: What the sub-agent should investigate.
            agent_type: Optional agent type override. Defaults to parent's type.
            spec: Optional AgentSpec for dynamic orchestration. When provided,
                the sub-agent uses the spec instead of a static template.
            tool_names: Optional specific tool list. Defaults to parent's tools.
            hypothesis_branch: Optional override. Defaults to current task branch.

        Returns:
            AgentResult from the sub-agent.

        Raises:
            AgentError: If depth limit or per-parent spawn limit is exceeded.
        """
        child_depth = self.depth + 1
        if child_depth > MAX_SUB_AGENT_DEPTH:
            raise AgentError(
                f"Sub-agent depth limit exceeded: depth {child_depth} > max {MAX_SUB_AGENT_DEPTH}",
                error_code="SUB_AGENT_DEPTH_EXCEEDED",
                details={"depth": child_depth, "max_depth": MAX_SUB_AGENT_DEPTH},
            )

        if self._sub_agents_spawned >= MAX_SUB_AGENTS_PER_PARENT:
            raise AgentError(
                f"Sub-agent spawn limit exceeded: {self._sub_agents_spawned} >= max {MAX_SUB_AGENTS_PER_PARENT}",
                error_code="SUB_AGENT_LIMIT_EXCEEDED",
                details={
                    "spawned": self._sub_agents_spawned,
                    "max": MAX_SUB_AGENTS_PER_PARENT,
                },
            )

        # Resolve tools: use specified subset, or inherit parent's tools
        if tool_names is not None:
            child_tools = {n: t for n, t in self.tools.items() if n in tool_names}
        else:
            child_tools = dict(self.tools)

        if spec is not None:
            # Dynamic: create base agent from spec
            spec.parent_agent_id = self.agent_id
            child_agent = BaseAgentImpl(
                spec=spec,
                llm=self.llm,
                kg=self.kg,
                yami=self.yami,
                tools=child_tools,
                audit_logger=self.audit,
                parent_agent_id=self.agent_id,
                depth=child_depth,
            )
            resolved_type = spec.agent_type_hint or self.agent_type
        else:
            # Static: resolve agent type and template
            resolved_type = agent_type or self.agent_type
            from agents.templates import get_template
            child_template = get_template(resolved_type)

            # Create the sub-agent (same concrete class if same type, else base)
            child_agent = self.__class__(
                template=child_template,
                llm=self.llm,
                kg=self.kg,
                yami=self.yami,
                tools=child_tools,
                audit_logger=self.audit,
                parent_agent_id=self.agent_id,
                depth=child_depth,
            ) if resolved_type == self.agent_type else BaseAgentImpl(
                template=child_template,
                llm=self.llm,
                kg=self.kg,
                yami=self.yami,
                tools=child_tools,
                audit_logger=self.audit,
                parent_agent_id=self.agent_id,
                depth=child_depth,
            )

        # Build a task for the sub-agent
        child_task = AgentTask(
            research_id="",
            agent_type=resolved_type,
            agent_id=child_agent.agent_id,
            hypothesis_branch=hypothesis_branch,
            instruction=task_description,
        )

        self.audit.log(
            "sub_agent_spawned",
            parent_agent_id=self.agent_id,
            child_agent_id=child_agent.agent_id,
            child_type=str(resolved_type),
            depth=child_depth,
        )

        self._sub_agents_spawned += 1

        # Execute the sub-agent
        result = await child_agent.execute(child_task)

        self._sub_agent_results.append(result)

        # Merge sub-agent's KG contributions into parent tracking
        self._nodes_added.extend(result.nodes_added)
        self._edges_added.extend(result.edges_added)
        self._nodes_updated.extend(result.nodes_updated)
        self._edges_updated.extend(result.edges_updated)

        self.audit.log(
            "sub_agent_completed",
            parent_agent_id=self.agent_id,
            child_agent_id=child_agent.agent_id,
            success=result.success,
            nodes_added=len(result.nodes_added),
            edges_added=len(result.edges_added),
        )

        return result

    async def spawn_sub_agents(
        self,
        tasks: list[dict[str, Any]],
        *,
        hypothesis_branch: str | None = None,
    ) -> list[AgentResult]:
        """Spawn multiple sub-agents concurrently.

        Each item in *tasks* is a dict with keys:
        - ``task_description`` (str, required)
        - ``agent_type`` (AgentType, optional)
        - ``tool_names`` (list[str], optional)

        Returns list of AgentResult in the same order as *tasks*.
        """
        coros = [
            self.spawn_sub_agent(
                t["task_description"],
                agent_type=t.get("agent_type"),
                tool_names=t.get("tool_names"),
                hypothesis_branch=hypothesis_branch,
            )
            for t in tasks
        ]
        return list(await asyncio.gather(*coros, return_exceptions=False))
