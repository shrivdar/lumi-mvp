"""Literature Analyst agent — extracts biological claims from publications."""

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


class LiteratureAnalystAgent(BaseAgentImpl):
    """Searches PubMed and Semantic Scholar, extracts biological relationships as KG edges."""

    agent_type = AgentType.LITERATURE_ANALYST

    async def _investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        # 1. Ask LLM for search queries
        plan_prompt = (
            f"Research instruction: {task.instruction}\n\n"
            f"Current KG context (summary): {len(kg_context.get('nodes', []))} nodes, "
            f"{len(kg_context.get('edges', []))} edges\n\n"
            f"Generate 2-3 precise PubMed/Semantic Scholar search queries to investigate this.\n"
            f"Respond as JSON: {{\"queries\": [\"query1\", \"query2\"]}}"
        )
        plan_response = await self.query_llm(plan_prompt, kg_context=kg_context)
        try:
            plan = self.llm.parse_json(plan_response)
            queries = plan.get("queries", [task.instruction])
        except Exception:
            queries = [task.instruction]

        # 2. Execute searches
        all_papers: list[dict[str, Any]] = []
        for query in queries[:3]:
            for tool_name in ["pubmed", "semantic_scholar"]:
                if tool_name not in self.tools:
                    continue
                try:
                    result = await self.call_tool(tool_name, action="search", query=query, max_results=5)
                    all_papers.extend(result.get("results", []))
                except Exception as exc:
                    self._errors.append(f"{tool_name} search failed: {exc}")

        if not all_papers:
            return {
                "nodes": [],
                "edges": [],
                "summary": "No papers found for the given queries.",
                "reasoning_trace": f"Searched with queries: {queries}",
            }

        # 3. Ask LLM to extract entities and relationships
        papers_text = "\n".join(
            f"- [{p.get('pmid') or p.get('paper_id', 'unknown')}] {p.get('title', '')}: {p.get('abstract', '')[:300]}"
            for p in all_papers[:10]
        )

        extraction_prompt = (
            f"Research question: {task.instruction}\n\n"
            f"Papers found:\n{papers_text}\n\n"
            f"Extract biological entities and their relationships from these papers.\n"
            f"Respond as JSON:\n"
            f'{{"entities": [{{"name": "...", "type": "GENE|PROTEIN|DISEASE|PATHWAY|DRUG|BIOMARKER", '
            f'"description": "..."}}, ...],\n'
            f'"relationships": [{{"source": "entity_name", "target": "entity_name", '
            f'"relation": "ASSOCIATED_WITH|INHIBITS|ACTIVATES|TREATS|...", '
            f'"evidence_pmid": "...", "claim": "...", "confidence": 0.0-1.0}}, ...],\n'
            f'"summary": "brief summary of findings"}}'
        )
        extraction_response = await self.query_llm(extraction_prompt, kg_context=kg_context)

        try:
            extraction = self.llm.parse_json(extraction_response)
        except Exception:
            return {
                "nodes": [],
                "edges": [],
                "summary": "Failed to parse LLM extraction response.",
                "reasoning_trace": f"Queries: {queries}, Papers: {len(all_papers)}",
            }

        # 4. Build KGNodes and KGEdges
        nodes: list[KGNode] = []
        node_name_to_id: dict[str, str] = {}

        for entity in extraction.get("entities", []):
            try:
                node_type = NodeType(entity["type"])
            except (ValueError, KeyError):
                node_type = NodeType.GENE  # default fallback

            node = KGNode(
                type=node_type,
                name=entity["name"],
                description=entity.get("description", ""),
                confidence=0.6,
                sources=[
                    EvidenceSource(
                        source_type=EvidenceSourceType.LLM_INFERENCE,
                        claim=f"Extracted from literature search: {task.instruction}",
                        agent_id=self.agent_id,
                    )
                ],
            )
            nodes.append(node)
            node_name_to_id[entity["name"]] = node.id

        edges: list[KGEdge] = []
        for rel in extraction.get("relationships", []):
            source_id = node_name_to_id.get(rel.get("source", ""))
            target_id = node_name_to_id.get(rel.get("target", ""))
            if not source_id or not target_id:
                continue

            try:
                relation = EdgeRelationType(rel["relation"])
            except (ValueError, KeyError):
                relation = EdgeRelationType.ASSOCIATED_WITH

            confidence_val = float(rel.get("confidence", 0.5))
            pmid = rel.get("evidence_pmid", "")

            edge = KGEdge(
                source_id=source_id,
                target_id=target_id,
                relation=relation,
                confidence=EdgeConfidence(
                    overall=confidence_val,
                    evidence_quality=confidence_val,
                    evidence_count=1,
                ),
                evidence=[
                    EvidenceSource(
                        source_type=EvidenceSourceType.PUBMED if pmid else EvidenceSourceType.LLM_INFERENCE,
                        source_id=pmid or None,
                        claim=rel.get("claim", ""),
                        confidence=confidence_val,
                        agent_id=self.agent_id,
                    )
                ],
            )
            edges.append(edge)

        return {
            "nodes": nodes,
            "edges": edges,
            "summary": extraction.get(
                "summary",
                f"Analyzed {len(all_papers)} papers, extracted"
                f" {len(nodes)} entities and {len(edges)} relationships.",
            ),
            "reasoning_trace": (
                f"Queries: {queries}\nPapers found: {len(all_papers)}"
                f"\nEntities: {len(nodes)}\nRelationships: {len(edges)}"
            ),
        }
