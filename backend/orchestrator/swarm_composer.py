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
from core.llm import LLMClient
from core.models import (
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
    # Main entry point
    # ------------------------------------------------------------------

    async def compose_swarm(
        self,
        query: str,
        hypothesis: HypothesisNode,
        config: ResearchConfig,
    ) -> list[AgentType]:
        """Select agent types for exploring a hypothesis.

        Always includes SCIENTIFIC_CRITIC. Uses the LLM to determine
        which other agent types are most relevant.

        Returns:
            List of AgentType values for the swarm.
        """
        available_types = config.agent_types or list(AgentType)
        max_agents = config.max_agents_per_swarm

        # Ask LLM which agents to include
        prompt = self._build_composition_prompt(query, hypothesis, available_types)

        try:
            response = await self.llm.query(
                prompt,
                system_prompt=(
                    "You are a research orchestrator selecting which specialist agents "
                    "to deploy for a hypothesis investigation. Respond with ONLY a JSON "
                    "array of agent type strings. Be selective — choose only the agents "
                    "whose tools are directly relevant to the hypothesis."
                ),
                research_id=self.session_id,
            )
            parsed = LLMClient.parse_json(response)
            if not isinstance(parsed, list):
                parsed = parsed.get("agents", []) if isinstance(parsed, dict) else []
            selected = self._parse_agent_types(parsed, available_types)
        except Exception as exc:
            logger.warning("swarm_composition_llm_failed", error=str(exc))
            selected = self._fallback_selection(query, hypothesis, available_types)

        # Enforce mandatory agents
        for mandatory in MANDATORY_AGENTS:
            agent_type = AgentType(mandatory)
            if agent_type not in selected:
                selected.append(agent_type)

        # Cap at max
        selected = selected[:max_agents]

        self.audit.log(
            "swarm_composed",
            session_id=self.session_id,
            hypothesis_id=hypothesis.id,
            agents=[str(a) for a in selected],
            count=len(selected),
        )

        self._emit(
            "swarm_composed",
            hypothesis_id=hypothesis.id,
            hypothesis=hypothesis.hypothesis,
            agents=[str(a) for a in selected],
            count=len(selected),
        )

        return selected

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
        from agents.templates import AGENT_TEMPLATES

        result = list(default_tools)
        template = AGENT_TEMPLATES.get(agent_type)

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

    def _build_composition_prompt(
        self,
        query: str,
        hypothesis: HypothesisNode,
        available_types: list[AgentType],
    ) -> str:
        type_descriptions = {
            AgentType.LITERATURE_ANALYST: "Searches PubMed/Semantic Scholar for publications",
            AgentType.PROTEIN_ENGINEER: "Fetches protein data from UniProt, predicts structure via ESM",
            AgentType.GENOMICS_MAPPER: "Maps genes to pathways via MyGene and KEGG",
            AgentType.PATHWAY_ANALYST: "Deep pathway analysis via KEGG and Reactome",
            AgentType.DRUG_HUNTER: "Searches ChEMBL for drugs, ClinicalTrials.gov for trials",
            AgentType.CLINICAL_ANALYST: "Analyzes clinical trial outcomes and designs",
            AgentType.SCIENTIFIC_CRITIC: "Falsifies KG edges, searches for counter-evidence (ALWAYS INCLUDED)",
            AgentType.EXPERIMENT_DESIGNER: "Proposes experiments to resolve uncertainties",
        }

        lines = [
            f"Research query: {query}",
            f"Hypothesis to explore: {hypothesis.hypothesis}",
            f"Rationale: {hypothesis.rationale}",
            "",
            "Available agent types:",
        ]
        for at in available_types:
            lines.append(f"  - {at.value}: {type_descriptions.get(at, 'Specialist agent')}")

        lines.extend([
            "",
            "Select the 3-5 most relevant agent types for investigating this hypothesis.",
            "scientific_critic is always included automatically — do not include it.",
            "Return a JSON array of agent type strings.",
        ])
        return "\n".join(lines)

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
