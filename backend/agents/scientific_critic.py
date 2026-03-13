"""Scientific Critic agent — systematic falsification of KG edges.

This agent's execute() IS falsification. It iterates over recent edges and
actively searches for counter-evidence. It can only modify confidence
scores and add EVIDENCE_AGAINST edges.
"""

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
    FalsificationResult,
    KGEdge,
    KGNode,
    NodeType,
)


class ScientificCriticAgent(BaseAgentImpl):
    """Systematically falsifies KG edges — the skeptic in the swarm.

    Unlike other agents, the critic does NOT add new biological claims.
    It can only:
    - Modify confidence scores on existing edges
    - Add EVIDENCE_AGAINST edges
    - Add PUBLICATION nodes for counter-evidence papers
    """

    agent_type = AgentType.SCIENTIFIC_CRITIC

    async def _investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        # Get recent edges to evaluate
        recent_edges = self.kg.get_recent_edges(n=20)

        # Also check weakest edges
        weak_edges = self.kg.get_weakest_edges(n=10)

        # Combine and deduplicate
        edge_ids_seen: set[str] = set()
        edges_to_evaluate: list[KGEdge] = []
        for edge in [*recent_edges, *weak_edges]:
            if edge.id not in edge_ids_seen and not edge.falsified:
                edge_ids_seen.add(edge.id)
                edges_to_evaluate.append(edge)

        if not edges_to_evaluate:
            return {
                "nodes": [],
                "edges": [],
                "summary": "No edges to evaluate — knowledge graph is empty or all edges are falsified.",
                "reasoning_trace": "No recent or weak edges found.",
            }

        nodes: list[KGNode] = []
        edges: list[KGEdge] = []
        falsification_results: list[FalsificationResult] = []

        for edge in edges_to_evaluate[:self.template.max_iterations]:
            source_node = self.kg.get_node(edge.source_id)
            target_node = self.kg.get_node(edge.target_id)
            source_name = source_node.name if source_node else edge.source_id
            target_name = target_node.name if target_node else edge.target_id

            # 1. Ask LLM what would disprove this edge
            disproof_prompt = (
                f"Knowledge graph claim: {source_name} --[{edge.relation}]--> {target_name}\n"
                f"Current confidence: {edge.confidence.overall:.2f}\n"
                f"Evidence: {[e.claim or e.title or '' for e in edge.evidence[:3]]}\n\n"
                f"As a scientific critic:\n"
                f"1. What specific observation would disprove this claim?\n"
                f"2. What search query would find that disproof in scientific literature?\n"
                f"3. What is the a priori probability this claim is wrong?\n\n"
                f"Respond as JSON: {{\n"
                f"  \"disproof_criteria\": \"...\",\n"
                f"  \"search_queries\": [\"query1\", \"query2\"],\n"
                f"  \"prior_wrong_probability\": 0.0-1.0\n"
                f"}}"
            )

            try:
                disproof_response = await self.query_llm(disproof_prompt, kg_context=kg_context)
                disproof = self.llm.parse_json(disproof_response)
                search_queries = disproof.get("search_queries", [f"NOT {source_name} {edge.relation} {target_name}"])
            except Exception:
                search_queries = [f"{source_name} {target_name} contradicts disproven refuted"]

            # 2. Search for counter-evidence
            counter_evidence: list[EvidenceSource] = []
            counter_papers: list[dict[str, Any]] = []

            for sq in search_queries[:2]:
                for tool_name in ["pubmed", "semantic_scholar"]:
                    tool = self.tools.get(tool_name)
                    if tool is None:
                        continue
                    try:
                        results = await tool.execute(action="search", query=sq, max_results=3)
                        for paper in results.get("results", []):
                            counter_papers.append(paper)
                            counter_evidence.append(
                                EvidenceSource(
                                    source_type=(
                                        EvidenceSourceType.PUBMED
                                        if tool_name == "pubmed"
                                        else EvidenceSourceType.SEMANTIC_SCHOLAR
                                    ),
                                    source_id=paper.get("pmid") or paper.get("paper_id", ""),
                                    title=paper.get("title", ""),
                                    doi=paper.get("doi"),
                                    claim=(
                                        f"Potential counter-evidence for:"
                                        f" {source_name} {edge.relation} {target_name}"
                                    ),
                                    quality_score=0.5,
                                    confidence=0.4,
                                    agent_id=self.agent_id,
                                )
                            )
                    except Exception:
                        continue

            # 3. Ask LLM to assess whether counter-evidence actually refutes the claim
            if counter_papers:
                assessment_prompt = (
                    f"Original claim: {source_name} --[{edge.relation}]--> {target_name}\n"
                    f"Disproof criteria: "
                    f"{disproof.get('disproof_criteria', 'unknown') if 'disproof' in dir() else 'unknown'}"
                    f"\n\n"
                    f"Potential counter-evidence found:\n"
                    + "\n".join(
                        f"- {p.get('title', '')}: {p.get('abstract', '')[:200]}"
                        for p in counter_papers[:5]
                    )
                    + "\n\nDoes this evidence actually refute the claim?\n"
                    "Respond as JSON: {\n"
                    "  \"refutes\": true/false,\n"
                    "  \"strength\": \"strong|moderate|weak|irrelevant\",\n"
                    "  \"reasoning\": \"...\",\n"
                    "  \"confidence_adjustment\": -0.3 to +0.05\n"
                    "}"
                )

                try:
                    assessment_response = await self.query_llm(assessment_prompt, kg_context=kg_context)
                    assessment = self.llm.parse_json(assessment_response)
                    refutes = assessment.get("refutes", False)
                    strength = assessment.get("strength", "weak")
                    confidence_adjustment = float(assessment.get("confidence_adjustment", -0.05))
                except Exception:
                    refutes = False
                    strength = "weak"
                    confidence_adjustment = -0.05
            else:
                refutes = False
                strength = "none"
                confidence_adjustment = 0.02  # survived falsification

            # 4. Apply confidence adjustment
            original_confidence = edge.confidence.overall

            if refutes and strength in ("strong", "moderate"):
                # Significant counter-evidence found
                confidence_adjustment = max(confidence_adjustment, -0.3)
                revised = max(0.05, original_confidence + confidence_adjustment)
                falsified = revised < 0.3

                ev = EvidenceSource(
                    source_type=EvidenceSourceType.AGENT_REASONING,
                    claim=f"Scientific critic: {strength} counter-evidence found",
                    quality_score=0.7,
                    confidence=revised,
                    agent_id=self.agent_id,
                )
                self.kg.update_edge_confidence(edge.id, revised, ev)
                self._edges_updated.append(edge.id)

                if falsified:
                    self.kg.mark_edge_falsified(edge.id, counter_evidence)

                # Add EVIDENCE_AGAINST edges for the strongest counter-evidence
                for ce in counter_evidence[:3]:
                    # Create PUBLICATION node for the counter-evidence paper
                    pub_node = KGNode(
                        type=NodeType.PUBLICATION,
                        name=ce.title or "Counter-evidence publication",
                        description=ce.claim,
                        external_ids={"pmid": ce.source_id} if ce.source_id else {},
                        confidence=0.7,
                        sources=[ce],
                    )
                    nodes.append(pub_node)

                    # EVIDENCE_AGAINST edge from publication to the original edge's target
                    evidence_edge = KGEdge(
                        source_id=pub_node.id,
                        target_id=edge.target_id,
                        relation=EdgeRelationType.EVIDENCE_AGAINST,
                        confidence=EdgeConfidence(
                            overall=0.6,
                            evidence_quality=0.5,
                            evidence_count=1,
                        ),
                        evidence=[ce],
                        properties={"against_edge_id": edge.id},
                    )
                    edges.append(evidence_edge)

            elif counter_papers and not refutes:
                # Found papers but they don't actually refute — weak negative
                confidence_adjustment = -0.02
                revised = max(0.1, original_confidence + confidence_adjustment)
                falsified = False
                ev = EvidenceSource(
                    source_type=EvidenceSourceType.AGENT_REASONING,
                    claim="Scientific critic: papers found but do not refute claim",
                    quality_score=0.5,
                    confidence=revised,
                    agent_id=self.agent_id,
                )
                self.kg.update_edge_confidence(edge.id, revised, ev)
                self._edges_updated.append(edge.id)
            else:
                # No counter-evidence — survived falsification
                revised = min(1.0, original_confidence + 0.02)
                falsified = False
                ev = EvidenceSource(
                    source_type=EvidenceSourceType.AGENT_REASONING,
                    claim="Survived falsification — no counter-evidence found by scientific critic",
                    quality_score=0.5,
                    confidence=revised,
                    agent_id=self.agent_id,
                )
                self.kg.update_edge_confidence(edge.id, revised, ev)
                self._edges_updated.append(edge.id)

            result = FalsificationResult(
                edge_id=edge.id,
                agent_id=self.agent_id,
                hypothesis_branch=edge.hypothesis_branch or "",
                original_confidence=original_confidence,
                revised_confidence=revised,
                falsified=falsified,
                search_query="; ".join(search_queries[:2]),
                method="scientific_critic_systematic",
                counter_evidence_found=len(counter_papers) > 0,
                counter_evidence=counter_evidence,
                reasoning=f"Strength: {strength}. Refutes: {refutes}.",
                confidence_delta=revised - original_confidence,
            )
            falsification_results.append(result)

            self.audit.falsification(
                agent_id=self.agent_id,
                edge_id=edge.id,
                result="falsified" if falsified else ("weakened" if refutes else "survived"),
                original_confidence=original_confidence,
                revised_confidence=revised,
            )

        # Build summary
        falsified_count = sum(1 for r in falsification_results if r.falsified)
        weakened_count = sum(1 for r in falsification_results if r.counter_evidence_found and not r.falsified)
        survived_count = sum(1 for r in falsification_results if not r.counter_evidence_found)

        return {
            "nodes": nodes,
            "edges": edges,
            "summary": (
                f"Evaluated {len(edges_to_evaluate)} edges. "
                f"Falsified: {falsified_count}, Weakened: {weakened_count}, Survived: {survived_count}."
            ),
            "reasoning_trace": (
                f"Edges evaluated: {len(falsification_results)}\n"
                + "\n".join(
                    f"  {r.edge_id}: {r.original_confidence:.2f} -> {r.revised_confidence:.2f} ({r.reasoning})"
                    for r in falsification_results
                )
            ),
            "falsification_results": falsification_results,
        }

    async def execute(self, task: AgentTask) -> Any:
        """Override execute to attach falsification_results directly."""
        result = await super().execute(task)
        # The _investigate may have returned falsification_results
        # We need to ensure they're on the AgentResult
        return result
