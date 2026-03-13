"""Drug Hunter agent — finds drugs/compounds targeting KG entities."""

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


class DrugHunterAgent(BaseAgentImpl):
    """Finds drugs and compounds targeting proteins/genes, writes DRUG nodes."""

    agent_type = AgentType.DRUG_HUNTER

    async def _investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        # 1. Identify targets from KG context or instruction
        plan_prompt = (
            f"Research instruction: {task.instruction}\n\n"
            f"KG context: {len(kg_context.get('nodes', []))} nodes\n\n"
            f"Identify 2-3 drug targets (proteins/genes/pathways) to search compounds for.\n"
            f"Respond as JSON: {{\"targets\": [{{\"name\": \"...\", \"type\": \"protein|gene|pathway\"}}]}}"
        )
        plan_response = await self.query_llm(plan_prompt, kg_context=kg_context)
        try:
            plan = self.llm.parse_json(plan_response)
            targets = plan.get("targets", [])
        except Exception:
            targets = [{"name": task.instruction.split()[0], "type": "protein"}]

        nodes: list[KGNode] = []
        edges: list[KGEdge] = []
        node_name_to_id: dict[str, str] = {}

        for target in targets[:3]:
            target_name = target.get("name", "")

            # 2. Search ChEMBL for compounds
            if "chembl" in self.tools:
                try:
                    chembl_result = await self.call_tool("chembl", action="search", query=target_name)
                    for compound in chembl_result.get("results", [])[:5]:
                        drug_name = compound.get("pref_name") or compound.get("molecule_chembl_id", "")
                        if not drug_name or drug_name in node_name_to_id:
                            continue

                        drug_node = KGNode(
                            type=NodeType.DRUG if compound.get("max_phase", 0) >= 1 else NodeType.COMPOUND,
                            name=drug_name,
                            description=compound.get("indication_class", ""),
                            properties={
                                "max_phase": compound.get("max_phase", 0),
                                "molecule_type": compound.get("molecule_type", ""),
                                "first_approval": compound.get("first_approval"),
                            },
                            external_ids={"chembl": compound.get("molecule_chembl_id", "")},
                            confidence=0.85,
                            sources=[
                                EvidenceSource(
                                    source_type=EvidenceSourceType.CHEMBL,
                                    source_id=compound.get("molecule_chembl_id", ""),
                                    title=drug_name,
                                    quality_score=0.85,
                                    confidence=0.85,
                                    agent_id=self.agent_id,
                                )
                            ],
                        )
                        nodes.append(drug_node)
                        node_name_to_id[drug_name] = drug_node.id

                        # Create TARGETS edge
                        # Try to find existing target node in KG
                        target_node = self.kg.get_node_by_name(target_name)
                        target_id = target_node.id if target_node else None

                        if not target_id and target_name not in node_name_to_id:
                            # Create a placeholder target node
                            t_node = KGNode(
                                type=NodeType.PROTEIN,
                                name=target_name,
                                confidence=0.6,
                                sources=[
                                    EvidenceSource(
                                        source_type=EvidenceSourceType.CHEMBL,
                                        claim="Drug target identified from ChEMBL",
                                        agent_id=self.agent_id,
                                    )
                                ],
                            )
                            nodes.append(t_node)
                            node_name_to_id[target_name] = t_node.id
                            target_id = t_node.id
                        elif target_name in node_name_to_id:
                            target_id = node_name_to_id[target_name]

                        if target_id:
                            edges.append(
                                KGEdge(
                                    source_id=drug_node.id,
                                    target_id=target_id,
                                    relation=EdgeRelationType.TARGETS,
                                    confidence=EdgeConfidence(
                                        overall=0.8,
                                        evidence_quality=0.85,
                                        evidence_count=1,
                                    ),
                                    evidence=[
                                        EvidenceSource(
                                            source_type=EvidenceSourceType.CHEMBL,
                                            source_id=compound.get("molecule_chembl_id", ""),
                                            claim=f"{drug_name} targets {target_name}",
                                            quality_score=0.85,
                                            confidence=0.8,
                                            agent_id=self.agent_id,
                                        )
                                    ],
                                )
                            )
                except Exception as exc:
                    self._errors.append(f"ChEMBL search failed for {target_name}: {exc}")

            # 3. Search clinical trials
            if "clinicaltrials" in self.tools:
                try:
                    ct_result = await self.call_tool(
                        "clinicaltrials", action="search",
                        query=target_name, max_results=3,
                    )
                    for trial in ct_result.get("results", []):
                        nct_id = trial.get("nct_id", "")
                        trial_title = trial.get("title", nct_id)
                        if nct_id in node_name_to_id:
                            continue

                        trial_node = KGNode(
                            type=NodeType.CLINICAL_TRIAL,
                            name=trial_title[:100],
                            description=trial.get("brief_summary", ""),
                            properties={
                                "phase": trial.get("phase", ""),
                                "status": trial.get("status", ""),
                                "enrollment": trial.get("enrollment"),
                            },
                            external_ids={"nct": nct_id} if nct_id else {},
                            confidence=0.9,
                            sources=[
                                EvidenceSource(
                                    source_type=EvidenceSourceType.CLINICALTRIALS,
                                    source_id=nct_id,
                                    title=trial_title[:100],
                                    quality_score=0.9,
                                    confidence=0.9,
                                    agent_id=self.agent_id,
                                )
                            ],
                        )
                        nodes.append(trial_node)
                        node_name_to_id[nct_id] = trial_node.id
                except Exception as exc:
                    self._errors.append(f"ClinicalTrials search failed for {target_name}: {exc}")

        # 4. Ask LLM for drug-disease relationships
        drug_names = [n.name for n in nodes if n.type in (NodeType.DRUG, NodeType.COMPOUND)]
        if drug_names:
            relationship_prompt = (
                f"Drugs/compounds found: {drug_names[:10]}\n"
                f"Research context: {task.instruction}\n\n"
                f"What diseases do these drugs treat? What are known side effects?\n"
                f"Respond as JSON: {{\"relationships\": [{{"
                f"\"drug\": \"...\", \"disease_or_effect\": \"...\", "
                f"\"relation\": \"TREATS|SIDE_EFFECT_OF|INHIBITS\", "
                f"\"confidence\": 0.0-1.0, \"claim\": \"...\"}}]}}"
            )
            try:
                rel_response = await self.query_llm(relationship_prompt, kg_context=kg_context)
                rels = self.llm.parse_json(rel_response)
                for rel in rels.get("relationships", []):
                    src = node_name_to_id.get(rel.get("drug", ""))
                    if not src:
                        continue
                    # Check if disease/effect exists in KG
                    effect_name = rel.get("disease_or_effect", "")
                    effect_node = self.kg.get_node_by_name(effect_name)
                    if effect_node:
                        try:
                            relation = EdgeRelationType(rel["relation"])
                        except (ValueError, KeyError):
                            relation = EdgeRelationType.ASSOCIATED_WITH
                        conf = float(rel.get("confidence", 0.5))
                        edges.append(
                            KGEdge(
                                source_id=src,
                                target_id=effect_node.id,
                                relation=relation,
                                confidence=EdgeConfidence(overall=conf, evidence_quality=conf, evidence_count=1),
                                evidence=[
                                    EvidenceSource(
                                        source_type=EvidenceSourceType.LLM_INFERENCE,
                                        claim=rel.get("claim", ""),
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
            "summary": (
                f"Found {len([n for n in nodes if n.type in (NodeType.DRUG, NodeType.COMPOUND)])}"
                f" drugs/compounds for {len(targets)} targets."
            ),
            "reasoning_trace": f"Targets: {[t.get('name') for t in targets]}",
        }
