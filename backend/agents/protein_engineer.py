"""Protein Engineer agent — structural biology via UniProt + ESM/Yami."""

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


class ProteinEngineerAgent(BaseAgentImpl):
    """Fetches protein data from UniProt, predicts structure via ESMFold/Yami."""

    agent_type = AgentType.PROTEIN_ENGINEER

    async def _investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        # 1. Ask LLM to identify proteins to investigate
        plan_prompt = (
            f"Research instruction: {task.instruction}\n\n"
            f"KG context: {len(kg_context.get('nodes', []))} nodes\n\n"
            f"Identify 1-3 proteins to investigate. For each, provide a UniProt search query.\n"
            f"Respond as JSON: {{\"proteins\": [{{\"name\": \"...\", \"query\": \"...\"}}]}}"
        )
        plan_response = await self.query_llm(plan_prompt, kg_context=kg_context)
        try:
            plan = self.llm.parse_json(plan_response)
            proteins = plan.get("proteins", [])
        except Exception:
            proteins = [{"name": task.instruction.split()[0], "query": task.instruction}]

        nodes: list[KGNode] = []
        edges: list[KGEdge] = []
        node_name_to_id: dict[str, str] = {}

        for protein_spec in proteins[:3]:
            query = protein_spec.get("query", protein_spec.get("name", ""))
            protein_name = protein_spec.get("name", query)

            # 2. Fetch from UniProt
            uniprot_data: dict[str, Any] = {}
            if "uniprot" in self.tools:
                try:
                    uniprot_data = await self.call_tool("uniprot", action="search", query=query)
                except Exception as exc:
                    self._errors.append(f"UniProt search failed for {protein_name}: {exc}")

            entries = uniprot_data.get("results", [])
            if not entries:
                continue

            entry = entries[0]
            sequence = entry.get("sequence", "")
            accession = entry.get("accession", "")

            # Create PROTEIN node
            protein_node = KGNode(
                type=NodeType.PROTEIN,
                name=entry.get("protein_name", protein_name),
                description=entry.get("function", ""),
                properties={
                    "sequence_length": len(sequence),
                    "organism": entry.get("organism", ""),
                    "gene_names": entry.get("gene_names", []),
                },
                external_ids={"uniprot": accession} if accession else {},
                confidence=0.9,
                sources=[
                    EvidenceSource(
                        source_type=EvidenceSourceType.UNIPROT,
                        source_id=accession,
                        title=entry.get("protein_name", protein_name),
                        quality_score=0.9,
                        confidence=0.9,
                        agent_id=self.agent_id,
                    )
                ],
            )
            nodes.append(protein_node)
            node_name_to_id[protein_name] = protein_node.id

            # 3. Structure prediction via Yami/ESM
            if self.yami and sequence and len(sequence) <= 400:
                try:
                    structure_result = await self.query_yami("predict_structure", sequence=sequence)
                    plddt_scores = structure_result.get("plddt", [])
                    avg_plddt = sum(plddt_scores) / len(plddt_scores) if plddt_scores else 0.0

                    structure_node = KGNode(
                        type=NodeType.STRUCTURE,
                        name=f"{protein_name} predicted structure",
                        description=f"ESMFold predicted structure (avg pLDDT: {avg_plddt:.1f})",
                        properties={
                            "avg_plddt": avg_plddt,
                            "method": "ESMFold",
                            "sequence_length": len(sequence),
                        },
                        confidence=min(1.0, avg_plddt / 100.0),
                        sources=[
                            EvidenceSource(
                                source_type=EvidenceSourceType.YAMI_PREDICTION,
                                claim=f"Structure predicted with avg pLDDT {avg_plddt:.1f}",
                                quality_score=min(1.0, avg_plddt / 100.0),
                                confidence=min(1.0, avg_plddt / 100.0),
                                agent_id=self.agent_id,
                            )
                        ],
                    )
                    nodes.append(structure_node)

                    # Link structure to protein
                    edges.append(
                        KGEdge(
                            source_id=structure_node.id,
                            target_id=protein_node.id,
                            relation=EdgeRelationType.ASSOCIATED_WITH,
                            confidence=EdgeConfidence(
                                overall=min(1.0, avg_plddt / 100.0),
                                evidence_quality=min(1.0, avg_plddt / 100.0),
                                evidence_count=1,
                                computational_score=avg_plddt / 100.0,
                            ),
                            evidence=[
                                EvidenceSource(
                                    source_type=EvidenceSourceType.YAMI_PREDICTION,
                                    claim=f"ESMFold structure prediction for {protein_name}",
                                    confidence=min(1.0, avg_plddt / 100.0),
                                    agent_id=self.agent_id,
                                )
                            ],
                        )
                    )
                except Exception as exc:
                    self._errors.append(f"Yami structure prediction failed for {protein_name}: {exc}")

            # 4. ESM embeddings for fitness if esm tool available
            if "esm" in self.tools and sequence:
                try:
                    esm_result = await self.call_tool("esm", action="embeddings", sequence=sequence[:200])
                    if esm_result.get("embeddings"):
                        protein_node.properties["has_embeddings"] = True
                except Exception:
                    pass

        # 5. Ask LLM to infer interactions between proteins
        if len(node_name_to_id) > 1:
            interaction_prompt = (
                f"Given these proteins: {list(node_name_to_id.keys())}\n"
                f"Research context: {task.instruction}\n\n"
                f"Are there known interactions between them? Respond as JSON:\n"
                f'{{"interactions": [{{"source": "...", "target": "...", '
                f'"relation": "BINDS_TO|INTERACTS_WITH|PHOSPHORYLATES", '
                f'"confidence": 0.0-1.0, "claim": "..."}}]}}'
            )
            try:
                interaction_response = await self.query_llm(interaction_prompt, kg_context=kg_context)
                interactions = self.llm.parse_json(interaction_response)
                for interaction in interactions.get("interactions", []):
                    src = node_name_to_id.get(interaction.get("source", ""))
                    tgt = node_name_to_id.get(interaction.get("target", ""))
                    if src and tgt:
                        try:
                            rel = EdgeRelationType(interaction["relation"])
                        except (ValueError, KeyError):
                            rel = EdgeRelationType.INTERACTS_WITH
                        conf = float(interaction.get("confidence", 0.5))
                        edges.append(
                            KGEdge(
                                source_id=src,
                                target_id=tgt,
                                relation=rel,
                                confidence=EdgeConfidence(overall=conf, evidence_quality=conf, evidence_count=1),
                                evidence=[
                                    EvidenceSource(
                                        source_type=EvidenceSourceType.LLM_INFERENCE,
                                        claim=interaction.get("claim", ""),
                                        confidence=conf,
                                        agent_id=self.agent_id,
                                    )
                                ],
                            )
                        )
            except Exception:
                pass

        return {
            "nodes": nodes,
            "edges": edges,
            "summary": f"Investigated {len(proteins)} proteins, created {len(nodes)} nodes and {len(edges)} edges.",
            "reasoning_trace": f"Proteins: {[p.get('name') for p in proteins]}",
        }
