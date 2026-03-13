"""Experiment Designer agent — reasoning-only agent that designs experiments."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgentImpl
from core.models import (
    AgentTask,
    AgentType,
    EdgeConfidence,
    EdgeRelationType,
    EvidenceSource,
    EvidenceSourceType,
    KGEdge,
    KGNode,
    NodeType,
)


class ExperimentDesignerAgent(BaseAgentImpl):
    """Reasoning-only agent — designs experiments to resolve KG uncertainties.

    No external tools. Uses LLM to analyze the KG state and propose
    the highest-value experiment to resolve the biggest uncertainty.
    """

    agent_type = AgentType.EXPERIMENT_DESIGNER

    async def _investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        # Gather uncertainty signals from KG
        weak_edges = self.kg.get_weakest_edges(n=10)
        orphan_nodes = self.kg.get_orphan_nodes()

        # Build a summary of uncertainties
        uncertainty_summary = []
        for edge in weak_edges:
            source = self.kg.get_node(edge.source_id)
            target = self.kg.get_node(edge.target_id)
            s_name = source.name if source else edge.source_id
            t_name = target.name if target else edge.target_id
            uncertainty_summary.append(
                f"  - {s_name} --[{edge.relation}]--> {t_name} (confidence: {edge.confidence.overall:.2f})"
            )

        orphan_summary = [f"  - {n.name} ({n.type})" for n in orphan_nodes[:5]]

        # Ask LLM to design the experiment
        design_prompt = (
            f"Research question: {task.instruction}\n\n"
            f"Current KG state:\n"
            f"  Nodes: {self.kg.node_count()}, Edges: {self.kg.edge_count()}\n"
            f"  Average confidence: {self.kg.avg_confidence():.2f}\n\n"
            f"Weakest edges (highest uncertainty):\n"
            + ("\n".join(uncertainty_summary[:10]) or "  None") + "\n\n"
            "Orphan nodes (not connected):\n"
            + ("\n".join(orphan_summary) or "  None") + "\n\n"
            "Design the single most informative experiment to resolve the biggest knowledge gap.\n\n"
            "Respond as JSON:\n"
            "{\n"
            "  \"experiment_type\": \"in_vitro|in_vivo|in_silico|clinical|observational\",\n"
            "  \"title\": \"concise experiment title\",\n"
            "  \"hypothesis\": \"specific testable hypothesis\",\n"
            "  \"rationale\": \"why this experiment resolves the biggest uncertainty\",\n"
            "  \"expected_outcome_positive\": \"what we observe if hypothesis is true\",\n"
            "  \"expected_outcome_negative\": \"what we observe if hypothesis is false\",\n"
            "  \"methods\": [\"technique1\", \"technique2\"],\n"
            "  \"materials\": [\"material1\", \"material2\"],\n"
            "  \"timeline_weeks\": 1-52,\n"
            "  \"success_criteria\": \"measurable endpoint\",\n"
            "  \"edges_to_resolve\": [\"edge description 1\", ...],\n"
            "  \"information_gain_estimate\": 0.0-1.0,\n"
            "  \"feasibility_score\": 0.0-1.0\n"
            "}"
        )

        try:
            design_response = await self.query_llm(design_prompt, kg_context=kg_context)
            design = self.llm.parse_json(design_response)
        except Exception:
            return {
                "nodes": [],
                "edges": [],
                "summary": "Failed to design experiment — LLM response could not be parsed.",
                "reasoning_trace": "LLM design prompt failed.",
            }

        # Create EXPERIMENT node
        experiment_node = KGNode(
            type=NodeType.EXPERIMENT,
            name=design.get("title", "Proposed experiment"),
            description=design.get("hypothesis", ""),
            properties={
                "experiment_type": design.get("experiment_type", "unknown"),
                "hypothesis": design.get("hypothesis", ""),
                "rationale": design.get("rationale", ""),
                "expected_outcome_positive": design.get("expected_outcome_positive", ""),
                "expected_outcome_negative": design.get("expected_outcome_negative", ""),
                "methods": design.get("methods", []),
                "materials": design.get("materials", []),
                "timeline_weeks": design.get("timeline_weeks", 0),
                "success_criteria": design.get("success_criteria", ""),
                "information_gain_estimate": design.get("information_gain_estimate", 0.5),
                "feasibility_score": design.get("feasibility_score", 0.5),
            },
            confidence=0.7,
            sources=[
                EvidenceSource(
                    source_type=EvidenceSourceType.AGENT_REASONING,
                    claim=f"Experiment designed to resolve: {design.get('rationale', 'uncertainty')}",
                    quality_score=0.7,
                    confidence=0.7,
                    agent_id=self.agent_id,
                )
            ],
        )

        nodes = [experiment_node]
        edges: list[KGEdge] = []

        # Link experiment to the weakest edges' nodes
        linked_nodes: set[str] = set()
        for edge in weak_edges[:3]:
            for node_id in [edge.source_id, edge.target_id]:
                if node_id not in linked_nodes:
                    linked_nodes.add(node_id)
                    edges.append(
                        KGEdge(
                            source_id=experiment_node.id,
                            target_id=node_id,
                            relation=EdgeRelationType.ASSOCIATED_WITH,
                            confidence=EdgeConfidence(
                                overall=0.6,
                                evidence_quality=0.6,
                                evidence_count=1,
                            ),
                            evidence=[
                                EvidenceSource(
                                    source_type=EvidenceSourceType.AGENT_REASONING,
                                    claim="Experiment targets uncertainty in this node",
                                    confidence=0.6,
                                    agent_id=self.agent_id,
                                )
                            ],
                        )
                    )

        return {
            "nodes": nodes,
            "edges": edges,
            "summary": f"Designed experiment: {design.get('title', 'N/A')} ({design.get('experiment_type', 'N/A')}). "
                       f"Information gain estimate: {design.get('information_gain_estimate', 'N/A')}.",
            "reasoning_trace": (
                f"Weakest edges: {len(weak_edges)}, Orphan nodes: {len(orphan_nodes)}\n"
                f"Experiment: {design.get('title', 'N/A')}\n"
                f"Hypothesis: {design.get('hypothesis', 'N/A')}\n"
                f"Methods: {design.get('methods', [])}"
            ),
            "recommended_next": design.get("rationale", ""),
        }

    async def falsify(self, edges: list[KGEdge]) -> list:
        """Experiment designer does not perform falsification — it proposes, doesn't assert."""
        return []
