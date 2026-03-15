"""Tests for ExperimentDesignerAgent — verify EXPERIMENT nodes with required fields."""

from __future__ import annotations

import json

import pytest

from agents.experiment_designer import ExperimentDesignerAgent
from agents.templates import get_template
from core.models import AgentTask, AgentType, NodeType
from tests.test_agents.conftest import MockLLMClient


class TestExperimentDesigner:
    """Test the experiment designer agent."""

    @pytest.mark.asyncio
    async def test_creates_experiment_node(self, seeded_kg):
        """Designer should create an EXPERIMENT node with required fields."""
        llm = MockLLMClient(responses=[
            # Plan
            "<think>1. Get weak edges\n2. Get orphans\n3. Design experiment</think>",
            # Get weakest edges
            '<tool>kg_get_weakest_edges:{"n": 10}</tool>',
            # Get orphan nodes
            '<tool>kg_get_orphan_nodes:{}</tool>',
            # Answer with experiment
            '<answer>' + json.dumps({
                "entities": [
                    {
                        "name": "BRCA1 knockdown assay in MCF-7 cells",
                        "type": "EXPERIMENT",
                        "description": "BRCA1 silencing increases sensitivity to DNA-damaging agents",
                        "confidence": 0.7,
                        "properties": {
                            "experiment_type": "in_vitro",
                            "hypothesis": "BRCA1 silencing increases sensitivity to DNA-damaging agents",
                            "rationale": "Weak BRCA1-breast cancer edge needs functional validation",
                            "expected_outcome_positive": (
                                "Increased apoptosis in BRCA1-knockdown "
                                "cells after cisplatin treatment"
                            ),
                            "expected_outcome_negative": "No change in cell viability",
                            "methods": ["siRNA knockdown", "MTT viability assay", "Western blot"],
                            "materials": ["MCF-7 cells", "BRCA1 siRNA", "Cisplatin", "MTT reagent"],
                            "timeline_weeks": 4,
                            "success_criteria": ">=50% reduction in viability in knockdown vs control",
                            "information_gain_estimate": 0.8,
                            "feasibility_score": 0.9,
                        },
                    }
                ],
                "relationships": [
                    {"source": "BRCA1 knockdown assay in MCF-7 cells", "target": "BRCA1",
                     "relation": "ASSOCIATED_WITH", "confidence": 0.6,
                     "claim": "Experiment targets BRCA1 uncertainty"},
                    {"source": "BRCA1 knockdown assay in MCF-7 cells", "target": "Breast Cancer",
                     "relation": "ASSOCIATED_WITH", "confidence": 0.6,
                     "claim": "Experiment targets breast cancer uncertainty"},
                ],
                "summary": "Designed experiment: BRCA1 knockdown assay (in_vitro). Info gain: 0.8.",
            }) + '</answer>',
        ])

        template = get_template(AgentType.EXPERIMENT_DESIGNER)
        task = AgentTask(
            task_id="task-exp-001",
            research_id="research-001",
            agent_type=AgentType.EXPERIMENT_DESIGNER,
            hypothesis_branch="h-main",
            instruction="Design an experiment to resolve the biggest uncertainty in the KG.",
        )

        agent = ExperimentDesignerAgent(
            template=template, llm=llm, kg=seeded_kg,
        )

        result = await agent.execute(task)

        assert result.success is True
        assert len(result.nodes_added) >= 1

        # Find the EXPERIMENT node
        exp_nodes = [n for n in result.nodes_added if n.type == NodeType.EXPERIMENT]
        assert len(exp_nodes) == 1

        exp = exp_nodes[0]
        assert exp.name == "BRCA1 knockdown assay in MCF-7 cells"
        assert "hypothesis" in exp.properties
        assert "methods" in exp.properties
        assert "materials" in exp.properties
        assert "timeline_weeks" in exp.properties
        assert "success_criteria" in exp.properties
        assert exp.properties["experiment_type"] == "in_vitro"
        assert exp.properties["information_gain_estimate"] == 0.8

        # Verify provenance
        assert exp.created_by == agent.agent_id
        assert exp.hypothesis_branch == task.hypothesis_branch

    @pytest.mark.asyncio
    async def test_designer_links_to_uncertain_nodes(self, seeded_kg):
        """Designer should link experiment to nodes involved in weak edges."""
        llm = MockLLMClient(responses=[
            # Plan
            "<think>Analyze uncertainties and design experiment</think>",
            # Get weakest edges
            '<tool>kg_get_weakest_edges:{"n": 10}</tool>',
            # Answer
            '<answer>' + json.dumps({
                "entities": [
                    {
                        "name": "Computational binding analysis",
                        "type": "EXPERIMENT",
                        "description": "test hypothesis",
                        "confidence": 0.7,
                        "properties": {
                            "experiment_type": "in_silico",
                            "hypothesis": "test hypothesis",
                            "methods": ["molecular dynamics"],
                            "materials": ["compute cluster"],
                            "timeline_weeks": 2,
                            "success_criteria": "binding affinity < 10nM",
                            "information_gain_estimate": 0.6,
                            "feasibility_score": 0.95,
                        },
                    }
                ],
                "relationships": [
                    {"source": "Computational binding analysis", "target": "BRCA1",
                     "relation": "ASSOCIATED_WITH", "confidence": 0.6,
                     "claim": "Targets BRCA1 uncertainty"},
                ],
                "summary": "Designed in_silico experiment.",
            }) + '</answer>',
        ])

        template = get_template(AgentType.EXPERIMENT_DESIGNER)
        task = AgentTask(
            task_id="task-exp-002",
            research_id="research-001",
            agent_type=AgentType.EXPERIMENT_DESIGNER,
            hypothesis_branch="h-main",
            instruction="Design experiment.",
        )

        agent = ExperimentDesignerAgent(
            template=template, llm=llm, kg=seeded_kg,
        )

        result = await agent.execute(task)

        assert result.success is True
        # Should have edges linking experiment to KG nodes
        assert len(result.edges_added) > 0

    @pytest.mark.asyncio
    async def test_designer_no_falsification(self, seeded_kg):
        """Experiment designer should skip falsification."""
        llm = MockLLMClient(responses=[
            # Plan
            "<think>Design experiment</think>",
            # Get weakest edges
            '<tool>kg_get_weakest_edges:{"n": 10}</tool>',
            # Answer
            '<answer>' + json.dumps({
                "entities": [
                    {
                        "name": "Test experiment",
                        "type": "EXPERIMENT",
                        "description": "test",
                        "confidence": 0.7,
                        "properties": {
                            "experiment_type": "in_vitro",
                            "hypothesis": "test",
                            "methods": ["method"],
                            "materials": ["material"],
                            "timeline_weeks": 1,
                            "success_criteria": "criterion",
                            "information_gain_estimate": 0.5,
                            "feasibility_score": 0.5,
                        },
                    }
                ],
                "relationships": [
                    {"source": "Test experiment", "target": "BRCA1",
                     "relation": "ASSOCIATED_WITH", "confidence": 0.6,
                     "claim": "test"},
                ],
                "summary": "Designed test experiment.",
            }) + '</answer>',
        ])

        template = get_template(AgentType.EXPERIMENT_DESIGNER)
        task = AgentTask(
            task_id="task-exp-003",
            research_id="research-001",
            agent_type=AgentType.EXPERIMENT_DESIGNER,
            hypothesis_branch="h-main",
            instruction="Design experiment.",
        )

        agent = ExperimentDesignerAgent(
            template=template, llm=llm, kg=seeded_kg,
        )

        result = await agent.execute(task)

        assert result.success is True
        assert len(result.falsification_results) == 0
