"""Swarm Composer — LLM-based agent selection and tool assignment.

Given a research query and hypothesis, the composer:
1. Asks the LLM which agent types are most relevant
2. Always includes scientific_critic (non-negotiable)
3. Selects tools from the ToolRegistry for each agent (static or dynamic via ToolRetriever)
4. Instantiates agents with the correct KG, LLM, tools, and Yami access
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from core.audit import AuditLogger
from core.constants import MANDATORY_AGENTS
from core.config import settings
from core.llm import LLMClient
from core.models import (
    AgentConstraints,
    AgentSpec,
    AgentTask,
    AgentType,
    HypothesisNode,
    ResearchConfig,
    ResearchEvent,
    ToolRegistryEntry,
)

if TYPE_CHECKING:
    from agents.tool_retriever import ToolRetriever

logger = structlog.get_logger(__name__)


class SwarmComposer:
    """Composes agent swarms for hypothesis exploration.

    Uses the LLM to decide which agent types are relevant for a given
    research query and hypothesis, then instantiates them with
    appropriate tools from the ToolRegistry.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        tool_registry_entries: list[ToolRegistryEntry] | None = None,
        session_id: str = "",
        tool_retriever: ToolRetriever | None = None,
    ) -> None:
        self.llm = llm
        self._tool_entries = tool_registry_entries or []
        self.session_id = session_id
        self.audit = AuditLogger("swarm_composer")
        self._events: list[ResearchEvent] = []
        self._tool_retriever = tool_retriever

    # ------------------------------------------------------------------
    # Main entry point — dynamic AgentSpec generation
    # ------------------------------------------------------------------

    async def compose_swarm_specs(
        self,
        query: str,
        hypothesis: HypothesisNode,
        config: ResearchConfig,
        agent_constraints: list[AgentConstraints] | None = None,
    ) -> list[AgentSpec]:
        """Dynamically generate AgentSpec objects for a hypothesis swarm.

        Instead of just picking agent types from a template library, this method
        asks the LLM to define the full role, instructions, tools, and constraints
        for each agent in the swarm. The orchestrator controls what each agent does.

        Always includes a scientific_critic spec (non-negotiable).

        Args:
            query: The research query.
            hypothesis: The hypothesis to explore.
            config: Research configuration.
            agent_constraints: Pre-allocated per-agent constraints from TokenBudgetManager.
                If provided, overrides the LLM-generated constraints.

        Returns:
            List of AgentSpec objects for the swarm.
        """
        available_types = config.agent_types or list(AgentType)

        # Build available tools summary
        tool_catalog = []
        for entry in self._tool_entries:
            if entry.enabled:
                tool_catalog.append(f"{entry.name} [{entry.category}]")

        prompt = self._build_spec_composition_prompt(
            query, hypothesis, available_types, tool_catalog, config,
        )

        specs: list[AgentSpec] = []

        try:
            response = await self.llm.query(
                prompt,
                system_prompt=(
                    "You are a research orchestrator dynamically composing an agent swarm. "
                    "For each agent, define its role, specific instructions, recommended tools, "
                    "and an optional agent_type_hint if it maps to a known specialist. "
                    "Respond with ONLY a JSON array of agent spec objects."
                ),
                research_id=self.session_id,
                model=settings.llm_fast_model,
            )
            parsed = LLMClient.parse_json(response)
            if not isinstance(parsed, list):
                parsed = parsed.get("agents", []) if isinstance(parsed, dict) else []

            for i, raw_spec in enumerate(parsed):
                if not isinstance(raw_spec, dict):
                    continue
                spec = self._parse_agent_spec(
                    raw_spec, hypothesis,
                    constraint_override=(
                        agent_constraints[i] if agent_constraints and i < len(agent_constraints)
                        else None
                    ),
                )
                if spec:
                    specs.append(spec)

        except Exception as exc:
            logger.warning("spec_composition_llm_failed", error=str(exc))
            # Fallback: generate specs from heuristic agent selection
            agent_types = self._fallback_selection(query, hypothesis, available_types)
            # Enforce mandatory agents
            for mandatory in MANDATORY_AGENTS:
                at = AgentType(mandatory)
                if at not in agent_types:
                    agent_types.append(at)
            for i, at in enumerate(agent_types):
                constraint = (
                    agent_constraints[i] if agent_constraints and i < len(agent_constraints)
                    else AgentConstraints(token_budget=config.agent_token_budget)
                )
                template_guidance = self._get_template_guidance(at)
                specs.append(AgentSpec(
                    role=f"{at.value} for hypothesis investigation",
                    instructions=self._default_instruction(query, hypothesis, at),
                    tools=self._fallback_tool_selection(at, []),
                    constraints=constraint,
                    hypothesis_branch=hypothesis.id,
                    agent_type_hint=at,
                    system_prompt=template_guidance.get("system_prompt", ""),
                    kg_write_permissions=template_guidance.get("kg_write_permissions", []),
                    kg_edge_permissions=template_guidance.get("kg_edge_permissions", []),
                    falsification_protocol=template_guidance.get("falsification_protocol", ""),
                ))

        # Enforce mandatory agent: scientific_critic
        has_critic = any(
            s.agent_type_hint == AgentType.SCIENTIFIC_CRITIC for s in specs
        )
        if not has_critic:
            critic_constraint = (
                agent_constraints[len(specs)]
                if agent_constraints and len(specs) < len(agent_constraints)
                else AgentConstraints(token_budget=config.agent_token_budget)
            )
            specs.append(AgentSpec(
                role="Scientific critic and falsifier",
                instructions=(
                    f"Critically evaluate evidence for the hypothesis: '{hypothesis.hypothesis}'. "
                    f"Search for counter-evidence and contradictions. Attempt to falsify KG edges "
                    f"added by other agents. Report any weaknesses in the evidence chain."
                ),
                tools=self._fallback_tool_selection(AgentType.SCIENTIFIC_CRITIC, []),
                constraints=critic_constraint,
                hypothesis_branch=hypothesis.id,
                agent_type_hint=AgentType.SCIENTIFIC_CRITIC,
                falsification_protocol="Search for contradicting evidence in PubMed and Semantic Scholar.",
            ))

        # Cap at max agents per swarm
        specs = specs[:config.max_agents_per_swarm]

        self.audit.log(
            "swarm_specs_composed",
            session_id=self.session_id,
            hypothesis_id=hypothesis.id,
            specs=[{"role": s.role, "hint": str(s.agent_type_hint)} for s in specs],
            count=len(specs),
        )

        self._emit(
            "swarm_specs_composed",
            hypothesis_id=hypothesis.id,
            hypothesis=hypothesis.hypothesis,
            agent_count=len(specs),
            roles=[s.role for s in specs],
        )

        return specs

    def _build_spec_composition_prompt(
        self,
        query: str,
        hypothesis: HypothesisNode,
        available_types: list[AgentType],
        tool_catalog: list[str],
        config: ResearchConfig,
    ) -> str:
        from agents.templates import AGENT_TEMPLATES

        lines = [
            f"Research query: {query}",
            f"Hypothesis to explore: {hypothesis.hypothesis}",
            f"Rationale: {hypothesis.rationale}",
            "",
            "Known specialist types with their expertise (use as agent_type_hint if applicable):",
        ]
        for at in available_types:
            template = AGENT_TEMPLATES.get(at)
            if template:
                # Include template system prompt as guidance for spec composition
                prompt_summary = template.system_prompt.split("\n")[0] if template.system_prompt else ""
                tools_str = ", ".join(template.tools) if template.tools else "none"
                lines.append(
                    f"  - {at.value}: {template.description}\n"
                    f"    Expertise: {prompt_summary}\n"
                    f"    Default tools: [{tools_str}]\n"
                    f"    Falsification: {template.falsification_protocol or 'N/A'}"
                )
            else:
                lines.append(f"  - {at.value}: Specialist agent")

        lines.extend([
            "",
            f"Available tools: {', '.join(tool_catalog[:30])}",
            "",
            f"Create {config.max_agents_per_swarm - 1} agent specs (critic is auto-added).",
            "For each agent, provide a JSON object with:",
            '  - "role": concise role description',
            '  - "instructions": 2-4 sentences of specific investigation instructions',
            '  - "tools": array of tool names from the catalog above',
            '  - "agent_type_hint": one of the specialist types if applicable, or null',
            '  - "system_prompt": optional custom system prompt (if omitted, template default is used)',
            '  - "falsification_protocol": how this agent should self-falsify its findings',
            "",
            "Use the specialist expertise descriptions above to craft precise instructions.",
            "scientific_critic is always included automatically — do not include it.",
            "Return a JSON array of agent spec objects.",
        ])
        return "\n".join(lines)

    def _parse_agent_spec(
        self,
        raw: dict[str, Any],
        hypothesis: HypothesisNode,
        constraint_override: AgentConstraints | None = None,
    ) -> AgentSpec | None:
        """Parse a raw dict from LLM response into an AgentSpec.

        Injects template guidance (system_prompt, KG permissions, falsification
        protocol) from the template library when the LLM response doesn't
        provide them. This ensures spec-composed agents retain the domain
        expertise encoded in templates.
        """
        role = raw.get("role", "")
        instructions = raw.get("instructions", "")
        if not role or not instructions:
            return None

        # Parse agent_type_hint
        agent_type_hint = None
        hint_str = raw.get("agent_type_hint", "")
        if hint_str:
            for at in AgentType:
                if at.value == str(hint_str).strip().lower():
                    agent_type_hint = at
                    break

        # Parse tools — validate against catalog
        raw_tools = raw.get("tools", [])
        valid_tool_names = {e.name for e in self._tool_entries if e.enabled}
        tools = [str(t) for t in raw_tools if str(t) in valid_tool_names]

        constraints = constraint_override or AgentConstraints()

        # Inject template guidance if agent_type_hint maps to a known template
        template_guidance = self._get_template_guidance(agent_type_hint)
        system_prompt = raw.get("system_prompt", "") or template_guidance.get("system_prompt", "")
        falsification_protocol = (
            raw.get("falsification_protocol", "") or template_guidance.get("falsification_protocol", "")
        )
        kg_write_permissions = template_guidance.get("kg_write_permissions", [])
        kg_edge_permissions = template_guidance.get("kg_edge_permissions", [])

        return AgentSpec(
            role=role,
            instructions=instructions,
            tools=tools,
            constraints=constraints,
            hypothesis_branch=hypothesis.id,
            agent_type_hint=agent_type_hint,
            system_prompt=system_prompt,
            kg_write_permissions=kg_write_permissions,
            kg_edge_permissions=kg_edge_permissions,
            falsification_protocol=falsification_protocol,
        )

    # ------------------------------------------------------------------
    # Task generation
    # ------------------------------------------------------------------

    async def generate_tasks(
        self,
        query: str,
        hypothesis: HypothesisNode,
        agent_types: list[AgentType],
        research_id: str,
        kg_context_ids: list[str] | None = None,
    ) -> list[AgentTask]:
        """Generate AgentTasks for each agent in the swarm.

        Uses the LLM to create specific instructions per agent type,
        tailored to the hypothesis being explored.
        """
        prompt = self._build_task_prompt(query, hypothesis, agent_types)

        try:
            response = await self.llm.query(
                prompt,
                system_prompt=(
                    "You are generating specific investigation instructions for "
                    "biomedical research agents. Each agent has different tools and "
                    "expertise. Respond with a JSON object mapping agent_type to "
                    "instruction string. Be specific about what each agent should "
                    "search for, analyze, or verify."
                ),
                research_id=self.session_id,
                model=settings.llm_fast_model,
            )
            parsed = LLMClient.parse_json(response)
            if not isinstance(parsed, dict):
                parsed = {}
        except Exception as exc:
            logger.warning("task_generation_llm_failed", error=str(exc))
            parsed = {}

        tasks: list[AgentTask] = []
        for agent_type in agent_types:
            type_str = str(agent_type)
            instruction = parsed.get(type_str) or parsed.get(agent_type.value, "")
            if not instruction:
                instruction = self._default_instruction(query, hypothesis, agent_type)

            task = AgentTask(
                research_id=research_id,
                agent_type=agent_type,
                hypothesis_branch=hypothesis.id,
                instruction=instruction,
                context={
                    "query": query,
                    "hypothesis": hypothesis.hypothesis,
                    "rationale": hypothesis.rationale,
                },
                kg_context=kg_context_ids or [],
            )
            tasks.append(task)

            self._emit(
                "agent_task_created",
                task_id=task.task_id,
                agent_type=type_str,
                hypothesis_id=hypothesis.id,
                instruction_preview=instruction[:200],
            )

        return tasks

    # ------------------------------------------------------------------
    # Tool selection (LLM-driven, dynamic per task)
    # ------------------------------------------------------------------

    async def select_tools_for_task(
        self,
        agent_type: AgentType,
        task: AgentTask,
        *,
        max_tools: int = 8,
    ) -> list[str]:
        """Dynamically select tools for a specific agent task via LLM.

        Uses the full tool catalog to pick the most relevant tools for
        the agent's assignment, rather than hardcoded tool lists.

        Returns:
            List of tool names from the catalog.
        """
        from agents.templates import AGENT_TEMPLATES

        template = AGENT_TEMPLATES.get(agent_type)
        default_tools = template.tools if template else []

        # Build compact catalog for LLM
        catalog_lines = []
        for entry in self._tool_entries:
            if not entry.enabled:
                continue
            caps = ", ".join(entry.capabilities[:4]) if entry.capabilities else ""
            catalog_lines.append(
                f"  - {entry.name} [{entry.category}]: {entry.description[:100]}  capabilities: [{caps}]"
            )

        prompt = (
            f"Agent type: {agent_type.value}\n"
            f"Task instruction: {task.instruction}\n"
            f"Hypothesis: {task.context.get('hypothesis', '')}\n"
            f"Research query: {task.context.get('query', '')}\n\n"
            f"Default tools for this agent type: {default_tools}\n\n"
            f"Available tools ({len(self._tool_entries)} total):\n"
            + "\n".join(catalog_lines[:80])  # cap prompt size
            + "\n\n"
            f"Select the {max_tools} most relevant tools for this specific task. "
            f"Always include the agent's default tools unless clearly irrelevant. "
            f"Add additional tools from the catalog that would help investigate the hypothesis. "
            f"Return a JSON array of tool name strings."
        )

        try:
            response = await self.llm.query(
                prompt,
                system_prompt=(
                    "You are a tool selector for a biomedical research platform. "
                    "Select the most relevant tools from the catalog for the given "
                    "agent task. Prefer tools that directly address the hypothesis. "
                    "Return ONLY a JSON array of tool name strings, nothing else."
                ),
                research_id=self.session_id,
                model=settings.llm_cheap_model,
            )
            parsed = LLMClient.parse_json(response)
            if isinstance(parsed, list):
                # Validate names against catalog
                valid_names = {e.name for e in self._tool_entries if e.enabled}
                selected = [str(n) for n in parsed if str(n) in valid_names]
                if selected:
                    self.audit.log(
                        "tools_selected",
                        agent_type=str(agent_type),
                        task_id=task.task_id,
                        tools=selected,
                        method="llm",
                    )
                    return selected[:max_tools]
        except Exception as exc:
            logger.warning("tool_selection_llm_failed", error=str(exc), agent_type=str(agent_type))

        # Fallback: use template defaults + category-matched tools
        return self._fallback_tool_selection(agent_type, default_tools)

    def _fallback_tool_selection(
        self,
        agent_type: AgentType,
        default_tools: list[str],
    ) -> list[str]:
        """Heuristic fallback: template defaults + top tools from matching categories."""
        result = list(default_tools)

        # Map agent types to relevant categories
        agent_category_map: dict[AgentType, list[str]] = {
            AgentType.LITERATURE_ANALYST: ["literature_search", "web_search"],
            AgentType.PROTEIN_ENGINEER: ["protein_analysis", "structural_biology"],
            AgentType.GENOMICS_MAPPER: ["genomics", "variant_analysis", "gene_expression"],
            AgentType.PATHWAY_ANALYST: ["pathway_analysis", "ontology_annotation", "network_analysis"],
            AgentType.DRUG_HUNTER: ["drug_discovery", "chemistry", "safety_toxicology"],
            AgentType.CLINICAL_ANALYST: ["clinical_data", "regulatory_data"],
            AgentType.SCIENTIFIC_CRITIC: ["literature_search", "web_search"],
            AgentType.EXPERIMENT_DESIGNER: [],
        }

        categories = agent_category_map.get(agent_type, [])
        for entry in self._tool_entries:
            if entry.name in result:
                continue
            if not entry.enabled:
                continue
            if entry.category in categories and entry.source_type.value == "NATIVE":
                result.append(entry.name)
            if len(result) >= 6:
                break

        return result

    def select_tools_for_agent(
        self,
        agent_type: AgentType,
        tool_names: list[str],
    ) -> list[ToolRegistryEntry]:
        """Select ToolRegistryEntry objects matching the agent's declared tool names (static)."""
        entries_by_name = {e.name: e for e in self._tool_entries}
        selected: list[ToolRegistryEntry] = []
        for name in tool_names:
            entry = entries_by_name.get(name)
            if entry and entry.enabled:
                selected.append(entry)
        return selected

    async def select_tools_dynamic(
        self,
        task_instruction: str,
        hypothesis: str = "",
        agent_type: AgentType | None = None,
        top_k: int = 4,
    ) -> list[str]:
        """Dynamically select tools using the ToolRetriever (LLM-based).

        Falls back to static template-based selection if no ToolRetriever is configured.

        Returns:
            List of tool name strings.
        """
        if self._tool_retriever is not None:
            return await self._tool_retriever.select_tools(
                task=task_instruction,
                hypothesis=hypothesis,
                top_k=top_k,
                agent_type=str(agent_type) if agent_type else "",
            )

        # Fallback: use static template tools
        if agent_type is not None:
            from agents.templates import get_template
            template = get_template(agent_type)
            return list(template.tools)

        return []

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_task_prompt(
        self,
        query: str,
        hypothesis: HypothesisNode,
        agent_types: list[AgentType],
    ) -> str:
        lines = [
            f"Research query: {query}",
            f"Hypothesis: {hypothesis.hypothesis}",
            f"Rationale: {hypothesis.rationale}",
            "",
            "Generate a specific investigation instruction for each agent:",
        ]
        for at in agent_types:
            lines.append(f"  - {at.value}")

        lines.extend([
            "",
            'Return JSON: {"agent_type": "instruction", ...}',
            "Each instruction should be 2-4 sentences, specific to what the agent should search/analyze.",
        ])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Template guidance
    # ------------------------------------------------------------------

    @staticmethod
    def _get_template_guidance(agent_type: AgentType | None) -> dict[str, Any]:
        """Extract guidance from the template library for a given agent type.

        Returns system_prompt, KG permissions, and falsification protocol
        from the canonical template. Used to enrich LLM-composed specs with
        domain knowledge that was previously hardcoded in templates.
        """
        if agent_type is None:
            return {}
        from agents.templates import AGENT_TEMPLATES
        template = AGENT_TEMPLATES.get(agent_type)
        if template is None:
            return {}
        return {
            "system_prompt": template.system_prompt,
            "kg_write_permissions": list(template.kg_write_permissions),
            "kg_edge_permissions": list(template.kg_edge_permissions),
            "falsification_protocol": template.falsification_protocol,
        }

    # ------------------------------------------------------------------
    # Fallback / parsing helpers
    # ------------------------------------------------------------------

    def _parse_agent_types(
        self,
        raw: list[Any],
        available: list[AgentType],
    ) -> list[AgentType]:
        """Parse LLM response into valid AgentType list."""
        available_set = set(available)
        result: list[AgentType] = []
        for item in raw:
            val = str(item).strip().lower()
            for at in AgentType:
                if at.value == val and at in available_set and at not in result:
                    result.append(at)
                    break
        return result

    def _fallback_selection(
        self,
        query: str,
        hypothesis: HypothesisNode,
        available: list[AgentType],
    ) -> list[AgentType]:
        """Heuristic fallback when LLM composition fails."""
        # Always start with literature
        selected: list[AgentType] = []

        query_lower = (query + " " + hypothesis.hypothesis).lower()

        priority_map = [
            (AgentType.LITERATURE_ANALYST, ["research", "study", "evidence", "literature"]),
            (AgentType.PROTEIN_ENGINEER, ["protein", "structure", "binding", "domain", "esm"]),
            (AgentType.GENOMICS_MAPPER, ["gene", "genomic", "expression", "mutation", "variant"]),
            (AgentType.PATHWAY_ANALYST, ["pathway", "signaling", "cascade", "regulation"]),
            (AgentType.DRUG_HUNTER, ["drug", "compound", "therapeutic", "treatment", "inhibitor"]),
            (AgentType.CLINICAL_ANALYST, ["clinical", "trial", "patient", "efficacy"]),
            (AgentType.EXPERIMENT_DESIGNER, ["experiment", "validate", "test"]),
        ]

        # Literature analyst is almost always useful
        if AgentType.LITERATURE_ANALYST in available:
            selected.append(AgentType.LITERATURE_ANALYST)

        for agent_type, keywords in priority_map:
            if agent_type in selected:
                continue
            if agent_type not in available:
                continue
            if any(kw in query_lower for kw in keywords):
                selected.append(agent_type)

        # Ensure at least 2 agents (besides critic)
        if len(selected) < 2:
            for at in available:
                if at not in selected and at != AgentType.SCIENTIFIC_CRITIC:
                    selected.append(at)
                if len(selected) >= 3:
                    break

        return selected

    def _default_instruction(
        self,
        query: str,
        hypothesis: HypothesisNode,
        agent_type: AgentType,
    ) -> str:
        """Generate a default instruction when LLM task generation fails."""
        return (
            f"Investigate the hypothesis: '{hypothesis.hypothesis}' "
            f"in the context of the research query: '{query}'. "
            f"Use your available tools to find evidence supporting or contradicting this hypothesis. "
            f"Report all findings as structured knowledge graph nodes and edges."
        )

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, **data: Any) -> None:
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
