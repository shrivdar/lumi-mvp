"""Pathway Analyst agent — deep pathway analysis and signaling cascade mapping."""

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


class PathwayAnalystAgent(BaseAgentImpl):
    """Deep pathway analysis — KEGG + Reactome for signaling cascades and cross-talk."""

    agent_type = AgentType.PATHWAY_ANALYST

    async def _investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        # 1. Ask LLM to identify pathways to investigate
        plan_prompt = (
            f"Research instruction: {task.instruction}\n\n"
            f"KG context: {len(kg_context.get('nodes', []))} nodes\n\n"
            f"Identify 2-3 biological pathways most relevant to this research.\n"
            f"For each, provide a KEGG or Reactome search query.\n"
            f"Respond as JSON: {{\"pathways\": [{{\"name\": \"...\", "
            f"\"query\": \"...\", \"database\": \"kegg|reactome\"}}]}}"
        )
        plan_response = await self.query_llm(plan_prompt, kg_context=kg_context)
        try:
            plan = self.llm.parse_json(plan_response)
            pathways = plan.get("pathways", [])
        except Exception:
            pathways = [{"name": task.instruction, "query": task.instruction, "database": "kegg"}]

        nodes: list[KGNode] = []
        edges: list[KGEdge] = []
        node_name_to_id: dict[str, str] = {}

        for pw_spec in pathways[:3]:
            query = pw_spec.get("query", pw_spec.get("name", ""))
            database = pw_spec.get("database", "kegg")
            pw_name = pw_spec.get("name", query)

            # 2. Search pathway database
            pw_data: dict[str, Any] = {}
            if database == "kegg" and "kegg" in self.tools:
                try:
                    pw_data = await self.call_tool("kegg", action="search", database="pathway", query=query)
                except Exception as exc:
                    self._errors.append(f"KEGG search failed for {pw_name}: {exc}")
            elif database == "reactome" and "reactome" in self.tools:
                try:
                    pw_data = await self.call_tool("reactome", action="search", query=query)
                except Exception as exc:
                    self._errors.append(f"Reactome search failed for {pw_name}: {exc}")

            results = pw_data.get("results", [])
            if not results:
                continue

            entry = results[0]
            source_type = EvidenceSourceType.KEGG if database == "kegg" else EvidenceSourceType.REACTOME
            pw_id = entry.get("id", "")

            pathway_node = KGNode(
                type=NodeType.PATHWAY,
                name=entry.get("name", pw_name),
                description=entry.get("description", ""),
                external_ids={database: pw_id} if pw_id else {},
                confidence=0.9,
                sources=[
                    EvidenceSource(
                        source_type=source_type,
                        source_id=pw_id,
                        title=entry.get("name", pw_name),
                        quality_score=0.9,
                        confidence=0.9,
                        agent_id=self.agent_id,
                    )
                ],
            )
            nodes.append(pathway_node)
            node_name_to_id[pw_name] = pathway_node.id

            # 3. Fetch pathway components (genes)
            if database == "kegg" and "kegg" in self.tools and pw_id:
                try:
                    genes_data = await self.call_tool("kegg", action="pathway_genes", pathway_id=pw_id)
                    for gene in genes_data.get("results", [])[:8]:
                        gene_name = gene.get("name", gene.get("symbol", ""))
                        if not gene_name or gene_name in node_name_to_id:
                            continue

                        gene_node = KGNode(
                            type=NodeType.GENE,
                            name=gene_name,
                            description=gene.get("description", ""),
                            external_ids={"kegg": gene.get("id", "")},
                            confidence=0.85,
                            sources=[
                                EvidenceSource(
                                    source_type=EvidenceSourceType.KEGG,
                                    source_id=gene.get("id", ""),
                                    claim=f"{gene_name} is a member of {pw_name}",
                                    quality_score=0.85,
                                    confidence=0.85,
                                    agent_id=self.agent_id,
                                )
                            ],
                        )
                        nodes.append(gene_node)
                        node_name_to_id[gene_name] = gene_node.id

                        edges.append(
                            KGEdge(
                                source_id=gene_node.id,
                                target_id=pathway_node.id,
                                relation=EdgeRelationType.MEMBER_OF,
                                confidence=EdgeConfidence(overall=0.85, evidence_quality=0.85, evidence_count=1),
                                evidence=[
                                    EvidenceSource(
                                        source_type=EvidenceSourceType.KEGG,
                                        source_id=pw_id,
                                        claim=f"{gene_name} is a member of {pw_name}",
                                        quality_score=0.85,
                                        confidence=0.85,
                                        agent_id=self.agent_id,
                                    )
                                ],
                            )
                        )
                except Exception as exc:
                    self._errors.append(f"KEGG pathway_genes failed for {pw_id}: {exc}")

        # 4. Ask LLM to infer pathway cross-talk
        pathway_names = [p.get("name", "") for p in pathways if p.get("name", "") in node_name_to_id]
        if len(pathway_names) > 1:
            crosstalk_prompt = (
                f"Pathways: {pathway_names}\n"
                f"Research context: {task.instruction}\n\n"
                f"Identify cross-talk or regulatory relationships between these pathways.\n"
                f"Respond as JSON: {{\"crosstalk\": [{{"
                f"\"source\": \"pathway_name\", \"target\": \"pathway_name\", "
                f"\"relation\": \"UPSTREAM_OF|DOWNSTREAM_OF|REGULATES|ACTIVATES|INHIBITS\", "
                f"\"mechanism\": \"...\", \"confidence\": 0.0-1.0}}]}}"
            )
            try:
                ct_response = await self.query_llm(crosstalk_prompt, kg_context=kg_context)
                crosstalk = self.llm.parse_json(ct_response)
                for ct in crosstalk.get("crosstalk", []):
                    src = node_name_to_id.get(ct.get("source", ""))
                    tgt = node_name_to_id.get(ct.get("target", ""))
                    if src and tgt:
                        try:
                            rel = EdgeRelationType(ct["relation"])
                        except (ValueError, KeyError):
                            rel = EdgeRelationType.REGULATES
                        conf = float(ct.get("confidence", 0.5))
                        edges.append(
                            KGEdge(
                                source_id=src,
                                target_id=tgt,
                                relation=rel,
                                confidence=EdgeConfidence(overall=conf, evidence_quality=conf, evidence_count=1),
                                evidence=[
                                    EvidenceSource(
                                        source_type=EvidenceSourceType.LLM_INFERENCE,
                                        claim=ct.get("mechanism", ""),
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
            "summary": f"Analyzed {len(pathways)} pathways, created {len(nodes)} nodes and {len(edges)} edges.",
            "reasoning_trace": f"Pathways: {[p.get('name') for p in pathways]}",
        }
