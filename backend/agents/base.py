"""BaseAgent — concrete implementation of the agent execute loop.

Provides the template-method pattern: subclasses override ``_investigate()``
while this class handles KG context, LLM calls, KG writes, falsification,
uncertainty, and audit logging.
"""

from __future__ import annotations

import json
import re
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
                turns=investigation.get("turns", []),
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
        """Wraps LLM call with system prompt, KG injection, audit, and token tracking."""
        sys_prompt = system_prompt or self.template.system_prompt

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

    # ------------------------------------------------------------------
    # Multi-turn investigation loop
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str | None:
        """Extract content from an XML tag in an LLM response."""
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    def _parse_agent_response(self, response: str) -> tuple[str, str, str | None]:
        """Parse LLM response for action tags.

        Returns ``(action_type, content, think_content)``.
        Priority: answer > tool > execute > think.
        """
        think = self._extract_tag(response, "think")
        answer = self._extract_tag(response, "answer")
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

    async def _multi_turn_investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
        *,
        max_turns: int = 20,
        investigation_focus: str = "",
    ) -> dict[str, Any]:
        """Multi-turn investigation loop: Plan → Generate → Execute → Observe → Repeat.

        Agents call this from their ``_investigate()`` override, passing
        an ``investigation_focus`` string that steers the LLM.
        """
        turns: list[AgentTurn] = []
        observations: list[str] = []

        tool_descriptions = self._build_tool_descriptions()

        allowed_nodes = ", ".join(str(t) for t in self.template.kg_write_permissions)
        allowed_edges = ", ".join(str(t) for t in self.template.kg_edge_permissions)

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
            obs_text = "\n\n".join(observations[-10:])

            turn_prompt = (
                f"Research task: {task.instruction}\n\n"
                f"Turn {turn_num}/{max_turns}\n\n"
                f"Previous observations:\n{obs_text}\n\n"
                f"Available tools:\n{tool_descriptions}\n\n"
                "Execute the next step. Use ONE of these formats:\n\n"
                "To reason: <think>your reasoning</think>\n"
                "To call a tool: <tool>tool_name:{\"arg\": \"value\"}</tool>\n"
                "To execute code: <execute>python_code</execute>\n"
                "To provide final answer:\n"
                "<answer>{\"entities\": [{\"name\": \"...\", "
                "\"type\": \"GENE|PROTEIN|DISEASE|...\", "
                "\"description\": \"...\", \"confidence\": 0.8, "
                "\"external_ids\": {}, \"properties\": {}, "
                "\"evidence_source\": \"PUBMED|...\", "
                "\"evidence_id\": \"...\"}], "
                "\"relationships\": [{\"source\": \"name\", "
                "\"target\": \"name\", "
                "\"relation\": \"ASSOCIATED_WITH|...\", "
                "\"confidence\": 0.8, \"claim\": \"...\", "
                "\"evidence_source\": \"PUBMED|...\", "
                "\"evidence_id\": \"...\"}], "
                "\"summary\": \"...\", "
                "\"reasoning_trace\": \"...\"}</answer>\n\n"
                "If you have gathered enough information, provide your <answer>."
            )

            start_ms = int(time.monotonic() * 1000)
            response = await self.query_llm(turn_prompt, kg_context=kg_context)
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

        # Budget exhausted — compile from observations
        return self._compile_from_observations(turns, observations, task)

    def _compile_answer(
        self,
        answer_content: str,
        turns: list[AgentTurn],
        observations: list[str],
    ) -> dict[str, Any]:
        """Parse ``<answer>`` content into the standard investigation result dict."""
        try:
            parsed = self.llm.parse_json(answer_content)
        except Exception:
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
