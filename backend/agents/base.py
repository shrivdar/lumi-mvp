"""BaseAgent — concrete implementation of the agent execute loop.

Provides the template-method pattern: subclasses override ``_investigate()``
while this class handles KG context, LLM calls, KG writes, falsification,
uncertainty, and audit logging.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import structlog

from core.audit import AuditLogger, Timer, set_request_context
from core.exceptions import AgentError
from core.interfaces import BaseTool, KnowledgeGraph, YamiInterface
from core.models import (
    AgentResult,
    AgentTask,
    AgentTemplate,
    AgentType,
    EvidenceSource,
    EvidenceSourceType,
    FalsificationResult,
    KGEdge,
    KGNode,
    UncertaintyVector,
)
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
        template: AgentTemplate,
        llm: Any,  # LLMClient
        kg: KnowledgeGraph,
        yami: YamiInterface | None = None,
        tools: dict[str, BaseTool] | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.agent_id = agent_id or str(uuid.uuid4())
        self.template = template
        self.agent_type = template.agent_type
        self.llm = llm
        self.kg = kg
        self.yami = yami
        self.tools = tools or {}
        self.audit = audit_logger or AuditLogger("agents")
        self._know_how_retriever = KnowHowRetriever()
        self._current_know_how: str = ""

        # Tracking state during execution
        self._nodes_added: list[KGNode] = []
        self._edges_added: list[KGEdge] = []
        self._nodes_updated: list[str] = []
        self._edges_updated: list[str] = []
        self._llm_calls: int = 0
        self._llm_tokens: int = 0
        self._errors: list[str] = []

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
            if self._edges_added and self.template.falsification_protocol:
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

            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                hypothesis_id=task.hypothesis_branch or "",
                nodes_added=self._nodes_added,
                edges_added=self._edges_added,
                nodes_updated=self._nodes_updated,
                edges_updated=self._edges_updated,
                falsification_results=falsification_results,
                uncertainty=uncertainty,
                summary=investigation.get("summary", ""),
                reasoning_trace=investigation.get("reasoning_trace", ""),
                recommended_next=investigation.get("recommended_next"),
                token_usage=self.llm.token_summary if self.llm else {},
                duration_ms=duration_ms,
                llm_calls=self._llm_calls,
                llm_tokens_used=self._llm_tokens,
                success=True,
                errors=self._errors,
            )

        except Exception as exc:
            duration_ms = int(time.monotonic() * 1000) - start_ms
            self.audit.error(
                "agent_execute_failed",
                agent_id=self.agent_id,
                task_id=task.task_id,
                error=str(exc),
            )
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                hypothesis_id=task.hypothesis_branch or "",
                nodes_added=self._nodes_added,
                edges_added=self._edges_added,
                uncertainty=self.get_uncertainty(),
                summary=f"Agent failed: {exc}",
                duration_ms=duration_ms,
                llm_calls=self._llm_calls,
                llm_tokens_used=self._llm_tokens,
                success=False,
                errors=[*self._errors, str(exc)],
            )

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
        """
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
    ) -> str:
        """Wraps LLM call with system prompt, KG injection, know-how, audit, and token tracking."""
        sys_prompt = system_prompt or self.template.system_prompt
        # Inject domain know-how if retrieved for this execution
        if self._current_know_how:
            sys_prompt = sys_prompt + "\n\n" + self._current_know_how

        self.audit.log(
            "agent_llm_call",
            agent_id=self.agent_id,
            prompt_len=len(prompt),
        )

        with Timer() as t:
            response = await self.llm.query(
                prompt,
                system_prompt=sys_prompt,
                kg_context=kg_context,
                agent_id=self.agent_id,
            )

        self._llm_calls += 1
        if hasattr(self.llm, "token_summary"):
            summary = self.llm.token_summary
            self._llm_tokens = summary.get("total_tokens", 0)

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
            counter_evidence: list[EvidenceSource] = []
            for tool_name in ["pubmed", "semantic_scholar"]:
                tool = self.tools.get(tool_name)
                if tool is None:
                    continue
                try:
                    search_results = await tool.execute(action="search", query=search_query, max_results=3)
                    for paper in search_results.get("results", []):
                        counter_evidence.append(
                            EvidenceSource(
                                source_type=(
                                    EvidenceSourceType.PUBMED
                                    if tool_name == "pubmed"
                                    else EvidenceSourceType.SEMANTIC_SCHOLAR
                                ),
                                source_id=paper.get("pmid") or paper.get("paper_id", ""),
                                title=paper.get("title", ""),
                                claim=f"Potential counter-evidence for {source_name} {edge.relation} {target_name}",
                                doi=paper.get("doi"),
                                quality_score=0.5,
                                confidence=0.4,
                                agent_id=self.agent_id,
                            )
                        )
                except Exception:
                    continue

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
