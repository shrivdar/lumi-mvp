"""Genomics Mapper agent — maps genes to pathways and expression patterns."""

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


class GenomicsMapperAgent(BaseAgentImpl):
    """Maps genes to pathways and expression patterns via MyGene and KEGG."""

    agent_type = AgentType.GENOMICS_MAPPER

    async def _investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        # 1. Ask LLM for genes to investigate
        plan_prompt = (
            f"Research instruction: {task.instruction}\n\n"
            f"KG context: {len(kg_context.get('nodes', []))} nodes\n\n"
            f"Identify 2-4 genes relevant to this research question.\n"
            f"Respond as JSON: {{\"genes\": [\"BRCA1\", \"TP53\", ...]}}"
        )
        plan_response = await self.query_llm(plan_prompt, kg_context=kg_context)
        try:
            plan = self.llm.parse_json(plan_response)
            genes = plan.get("genes", [])
        except Exception:
            genes = [task.instruction.split()[0]]

        nodes: list[KGNode] = []
        edges: list[KGEdge] = []
        node_name_to_id: dict[str, str] = {}

        for gene_symbol in genes[:4]:
            # 2. Fetch gene info from MyGene
            gene_info: dict[str, Any] = {}
            if "mygene" in self.tools:
                try:
                    gene_info = await self.call_tool("mygene", action="search", query=gene_symbol)
                except Exception as exc:
                    self._errors.append(f"MyGene search failed for {gene_symbol}: {exc}")

            results = gene_info.get("results", [])
            entry = results[0] if results else {}

            gene_node = KGNode(
                type=NodeType.GENE,
                name=gene_symbol,
                description=entry.get("summary", entry.get("name", "")),
                properties={
                    "entrez_id": entry.get("entrezgene", ""),
                    "ensembl": (
                        entry.get("ensembl", {}).get("gene", "")
                        if isinstance(entry.get("ensembl"), dict)
                        else ""
                    ),
                    "type_of_gene": entry.get("type_of_gene", ""),
                },
                external_ids={
                    k: str(v)
                    for k, v in {
                        "ncbi_gene": entry.get("entrezgene"),
                        "symbol": entry.get("symbol"),
                    }.items()
                    if v
                },
                confidence=0.9 if entry else 0.5,
                sources=[
                    EvidenceSource(
                        source_type=EvidenceSourceType.MYGENE if entry else EvidenceSourceType.LLM_INFERENCE,
                        source_id=str(entry.get("entrezgene", "")),
                        title=gene_symbol,
                        quality_score=0.9 if entry else 0.4,
                        confidence=0.9 if entry else 0.5,
                        agent_id=self.agent_id,
                    )
                ],
            )
            nodes.append(gene_node)
            node_name_to_id[gene_symbol] = gene_node.id

            # 3. Fetch KEGG pathways for this gene
            if "kegg" in self.tools:
                try:
                    kegg_result = await self.call_tool(
                        "kegg", action="search",
                        database="genes", query=f"hsa:{gene_symbol}",
                    )
                    pathway_entries = kegg_result.get("results", [])

                    for pw in pathway_entries[:5]:
                        pw_name = pw.get("name", pw.get("id", ""))
                        pw_id = pw.get("id", "")

                        if pw_name not in node_name_to_id:
                            pathway_node = KGNode(
                                type=NodeType.PATHWAY,
                                name=pw_name,
                                description=pw.get("description", ""),
                                external_ids={"kegg": pw_id} if pw_id else {},
                                confidence=0.85,
                                sources=[
                                    EvidenceSource(
                                        source_type=EvidenceSourceType.KEGG,
                                        source_id=pw_id,
                                        title=pw_name,
                                        quality_score=0.85,
                                        confidence=0.85,
                                        agent_id=self.agent_id,
                                    )
                                ],
                            )
                            nodes.append(pathway_node)
                            node_name_to_id[pw_name] = pathway_node.id

                        # Gene participates in pathway
                        edges.append(
                            KGEdge(
                                source_id=gene_node.id,
                                target_id=node_name_to_id[pw_name],
                                relation=EdgeRelationType.PARTICIPATES_IN,
                                confidence=EdgeConfidence(overall=0.85, evidence_quality=0.85, evidence_count=1),
                                evidence=[
                                    EvidenceSource(
                                        source_type=EvidenceSourceType.KEGG,
                                        source_id=pw_id,
                                        claim=f"{gene_symbol} participates in {pw_name}",
                                        quality_score=0.85,
                                        confidence=0.85,
                                        agent_id=self.agent_id,
                                    )
                                ],
                            )
                        )
                except Exception as exc:
                    self._errors.append(f"KEGG search failed for {gene_symbol}: {exc}")

        # 4. Ask LLM for regulatory relationships between genes
        if len(node_name_to_id) > 1:
            regulation_prompt = (
                f"Given these genes: {list(g for g in genes if g in node_name_to_id)}\n"
                f"Research context: {task.instruction}\n\n"
                f"Identify known regulatory relationships. Respond as JSON:\n"
                f'{{"regulations": [{{"source": "GENE", "target": "GENE", '
                f'"relation": "UPREGULATES|DOWNREGULATES|ACTIVATES|INHIBITS", '
                f'"confidence": 0.0-1.0, "claim": "..."}}]}}'
            )
            try:
                reg_response = await self.query_llm(regulation_prompt, kg_context=kg_context)
                regulations = self.llm.parse_json(reg_response)
                for reg in regulations.get("regulations", []):
                    src = node_name_to_id.get(reg.get("source", ""))
                    tgt = node_name_to_id.get(reg.get("target", ""))
                    if src and tgt:
                        try:
                            rel = EdgeRelationType(reg["relation"])
                        except (ValueError, KeyError):
                            rel = EdgeRelationType.ASSOCIATED_WITH
                        conf = float(reg.get("confidence", 0.5))
                        edges.append(
                            KGEdge(
                                source_id=src,
                                target_id=tgt,
                                relation=rel,
                                confidence=EdgeConfidence(overall=conf, evidence_quality=conf, evidence_count=1),
                                evidence=[
                                    EvidenceSource(
                                        source_type=EvidenceSourceType.LLM_INFERENCE,
                                        claim=reg.get("claim", ""),
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
            "summary": f"Mapped {len(genes)} genes to pathways, created {len(nodes)} nodes and {len(edges)} edges.",
            "reasoning_trace": f"Genes: {genes}",
        }
