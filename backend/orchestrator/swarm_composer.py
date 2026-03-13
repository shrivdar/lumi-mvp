"""Swarm Composer — LLM-based agent selection and tool assignment.

Given a research query and hypothesis, the composer:
1. Asks the LLM which agent types are most relevant
2. Always includes scientific_critic (non-negotiable)
3. Selects tools from the ToolRegistry for each agent
4. Instantiates agents with the correct KG, LLM, tools, and Yami access
"""

from __future__ import annotations

from typing import Any

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
    ) -> None:
        self.llm = llm
        self._tool_entries = tool_registry_entries or []
        self.session_id = session_id
        self.audit = AuditLogger("swarm_composer")
        self._events: list[ResearchEvent] = []

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
    # Tool selection
    # ------------------------------------------------------------------

    def select_tools_for_agent(
        self,
        agent_type: AgentType,
        tool_names: list[str],
    ) -> list[ToolRegistryEntry]:
        """Select ToolRegistryEntry objects matching the agent's declared tool names."""
        entries_by_name = {e.name: e for e in self._tool_entries}
        selected: list[ToolRegistryEntry] = []
        for name in tool_names:
            entry = entries_by_name.get(name)
            if entry and entry.enabled:
                selected.append(entry)
        return selected

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
