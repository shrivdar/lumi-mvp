"""Agent Factory — creates agent instances with proper templates, tools, and dependencies.

This is the single entry point for agent instantiation. The orchestrator
calls ``create_agent()`` with an agent type, and gets back a fully-configured
BaseAgentImpl subclass with the correct template, LLM, KG, Yami, and tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
from core.models import AgentType

if TYPE_CHECKING:
    from core.interfaces import BaseTool, KnowledgeGraph, YamiInterface
    from core.llm import LLMClient
    from core.models import AgentTemplate

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
    """Create a fully-configured agent instance.

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
