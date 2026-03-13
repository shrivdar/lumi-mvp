"""YOHAS agents — BaseAgent + 8 specialized research agents."""

from agents.base import BaseAgentImpl
from agents.clinical_analyst import ClinicalAnalystAgent
from agents.drug_hunter import DrugHunterAgent
from agents.experiment_designer import ExperimentDesignerAgent
from agents.genomics_mapper import GenomicsMapperAgent
from agents.literature_analyst import LiteratureAnalystAgent
from agents.pathway_analyst import PathwayAnalystAgent
from agents.protein_engineer import ProteinEngineerAgent
from agents.scientific_critic import ScientificCriticAgent
from agents.templates import AGENT_TEMPLATES, get_template

__all__ = [
    "BaseAgentImpl",
    "ClinicalAnalystAgent",
    "DrugHunterAgent",
    "ExperimentDesignerAgent",
    "GenomicsMapperAgent",
    "LiteratureAnalystAgent",
    "PathwayAnalystAgent",
    "ProteinEngineerAgent",
    "ScientificCriticAgent",
    "AGENT_TEMPLATES",
    "get_template",
]
