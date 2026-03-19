"""Agent Factory — creates agent instances with proper templates, tools, and dependencies.

This is the single entry point for agent instantiation. The orchestrator
calls ``create_agent()`` with an agent type, and gets back a fully-configured
BaseAgentImpl subclass with the correct template, LLM, KG, Yami, and tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from agents.base import BaseAgentImpl
from agents.clinical_analyst import ClinicalAnalystAgent
from agents.drug_hunter import DrugHunterAgent
from agents.experiment_designer import ExperimentDesignerAgent
from agents.genomics_mapper import GenomicsMapperAgent
from agents.literature_analyst import LiteratureAnalystAgent
from agents.pathway_analyst import PathwayAnalystAgent
from agents.protein_engineer import ProteinEngineerAgent
from agents.scientific_critic import ScientificCriticAgent
from agents.templates import AGENT_TEMPLATES
from agents.tool_creator import ToolCreatorAgent
from core.exceptions import AgentError
from core.models import AgentSpec, AgentType

if TYPE_CHECKING:
    from core.interfaces import BaseTool, KnowledgeGraph, YamiInterface
    from core.llm import LLMClient

logger = structlog.get_logger(__name__)

# Map AgentType → concrete class
_AGENT_CLASS_MAP: dict[AgentType, type[BaseAgentImpl]] = {
    AgentType.LITERATURE_ANALYST: LiteratureAnalystAgent,
    AgentType.PROTEIN_ENGINEER: ProteinEngineerAgent,
    AgentType.GENOMICS_MAPPER: GenomicsMapperAgent,
    AgentType.PATHWAY_ANALYST: PathwayAnalystAgent,
    AgentType.DRUG_HUNTER: DrugHunterAgent,
    AgentType.CLINICAL_ANALYST: ClinicalAnalystAgent,
    AgentType.SCIENTIFIC_CRITIC: ScientificCriticAgent,
    AgentType.EXPERIMENT_DESIGNER: ExperimentDesignerAgent,
    AgentType.TOOL_CREATOR: ToolCreatorAgent,
}


def create_agent(
    agent_type: AgentType,
    llm: LLMClient,
    kg: KnowledgeGraph,
    yami: YamiInterface | None = None,
    tools: dict[str, BaseTool] | None = None,
) -> BaseAgentImpl:
    """Create a fully-configured agent instance from a static template.

    Args:
        agent_type: Which specialist agent to create.
        llm: Shared LLM client for this session.
        kg: Shared knowledge graph for this session.
        yami: Optional Yami/ESM interface (required for protein_engineer).
        tools: Dict of tool_name → BaseTool instances. Dynamically selected
               by SwarmComposer per task.

    Returns:
        A concrete BaseAgentImpl subclass, ready to ``execute(task)``.
    """
    cls = _AGENT_CLASS_MAP.get(agent_type)
    if cls is None:
        raise AgentError(
            f"Unknown agent type: {agent_type}",
            error_code="UNKNOWN_AGENT_TYPE",
        )

    template = AGENT_TEMPLATES.get(agent_type)
    if template is None:
        raise AgentError(
            f"No template for agent type: {agent_type}",
            error_code="MISSING_TEMPLATE",
        )

    agent = cls(
        template=template,
        llm=llm,
        kg=kg,
        yami=yami,
        tools=tools or {},
    )

    logger.debug(
        "agent_created",
        agent_type=str(agent_type),
        agent_id=agent.agent_id,
        tool_count=len(tools) if tools else 0,
        tool_names=list((tools or {}).keys()),
    )

    return agent


def create_agent_from_spec(
    spec: AgentSpec,
    llm: LLMClient,
    kg: KnowledgeGraph,
    yami: YamiInterface | None = None,
    tools: dict[str, BaseTool] | None = None,
) -> BaseAgentImpl:
    """Create an agent from a dynamic AgentSpec (orchestrator-generated).

    If the spec has an ``agent_type_hint`` that maps to a known subclass,
    that subclass is used (preserving specialised ``_investigate()`` logic).
    Otherwise a generic ``BaseAgentImpl`` is created that runs the multi-turn
    loop driven entirely by the spec's role, instructions, and constraints.

    Args:
        spec: Dynamic agent specification from the orchestrator.
        llm: Shared LLM client for this session.
        kg: Shared knowledge graph for this session.
        yami: Optional Yami/ESM interface.
        tools: Dict of tool_name → BaseTool instances.

    Returns:
        A BaseAgentImpl (or subclass), ready to ``execute(task)``.
    """
    resolved_tools = tools or {}

    # All agents are created as generic BaseAgentImpl driven entirely by their spec.
    # The spec's role, instructions, system_prompt, and permissions define behavior.
    # No template lookup, no subclass dispatch — the orchestrator controls everything.
    agent = BaseAgentImpl(
        spec=spec,
        llm=llm,
        kg=kg,
        yami=yami,
        tools=resolved_tools,
    )

    logger.debug(
        "agent_created_from_spec",
        role=spec.role,
        agent_type_hint=str(spec.agent_type_hint) if spec.agent_type_hint else None,
        agent_id=agent.agent_id,
        tool_count=len(resolved_tools),
        tool_names=list(resolved_tools.keys()),
        constraints=spec.constraints.model_dump(),
    )

    return agent
