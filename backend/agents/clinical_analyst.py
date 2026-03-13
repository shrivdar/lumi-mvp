"""Clinical Analyst agent — clinical trial search, outcome analysis, failure analysis."""

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


class ClinicalAnalystAgent(BaseAgentImpl):
    """Searches clinical trials, reports outcomes and failure analyses."""

    agent_type = AgentType.CLINICAL_ANALYST

    async def _investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        # 1. Identify clinical questions
        plan_prompt = (
            f"Research instruction: {task.instruction}\n\n"
            f"KG context: {len(kg_context.get('nodes', []))} nodes\n\n"
            f"Identify 2-3 clinical trial search queries.\n"
            f"Focus on: interventions, conditions, outcomes.\n"
            f"Respond as JSON: {{\"queries\": [{{\"query\": \"...\", \"focus\": \"intervention|condition|outcome\"}}]}}"
        )
        plan_response = await self.query_llm(plan_prompt, kg_context=kg_context)
        try:
            plan = self.llm.parse_json(plan_response)
            queries = plan.get("queries", [])
        except Exception:
            queries = [{"query": task.instruction, "focus": "condition"}]

        nodes: list[KGNode] = []
        edges: list[KGEdge] = []
        node_name_to_id: dict[str, str] = {}
        all_trials: list[dict[str, Any]] = []

        for q in queries[:3]:
            query = q.get("query", task.instruction)

            # 2. Search ClinicalTrials.gov
            if "clinicaltrials" in self.tools:
                try:
                    ct_result = await self.call_tool("clinicaltrials", action="search", query=query, max_results=5)
                    all_trials.extend(ct_result.get("results", []))
                except Exception as exc:
                    self._errors.append(f"ClinicalTrials search failed: {exc}")

            # 3. Search PubMed for trial publications
            if "pubmed" in self.tools:
                try:
                    pub_result = await self.call_tool(
                        "pubmed", action="search",
                        query=f"{query} clinical trial",
                        max_results=3,
                    )
                    for paper in pub_result.get("results", []):
                        pmid = paper.get("pmid", "")
                        if pmid and pmid not in node_name_to_id:
                            # Check if it reports trial results
                            mesh = paper.get("mesh_terms", [])
                            is_trial = any("clinical trial" in str(m).lower() for m in mesh)
                    if is_trial or "trial" in paper.get("title", "").lower():
                                all_trials.append({
                                    "nct_id": "",
                                    "title": paper.get("title", ""),
                                    "brief_summary": paper.get("abstract", "")[:300],
                                    "phase": "",
                                    "status": "Published",
                                    "pmid": pmid,
                                })
                except Exception as exc:
                    self._errors.append(f"PubMed search failed: {exc}")

        # 4. Create nodes for unique trials
        for trial in all_trials:
            nct_id = trial.get("nct_id", "")
            trial_title = trial.get("title", nct_id)[:100]
            key = nct_id or trial_title
            if key in node_name_to_id:
                continue

            trial_node = KGNode(
                type=NodeType.CLINICAL_TRIAL,
                name=trial_title,
                description=trial.get("brief_summary", ""),
                properties={
                    "phase": trial.get("phase", ""),
                    "status": trial.get("status", ""),
                    "enrollment": trial.get("enrollment"),
                    "start_date": trial.get("start_date"),
                    "completion_date": trial.get("completion_date"),
                    "interventions": trial.get("interventions", []),
                    "conditions": trial.get("conditions", []),
                },
                external_ids={
                    k: v
                    for k, v in {
                        "nct": nct_id,
                        "pmid": trial.get("pmid", ""),
                    }.items()
                    if v
                },
                confidence=0.9,
                sources=[
                    EvidenceSource(
                        source_type=EvidenceSourceType.CLINICALTRIALS if nct_id else EvidenceSourceType.PUBMED,
                        source_id=nct_id or trial.get("pmid", ""),
                        title=trial_title,
                        quality_score=0.9,
                        confidence=0.9,
                        agent_id=self.agent_id,
                    )
                ],
            )
            nodes.append(trial_node)
            node_name_to_id[key] = trial_node.id

        # 5. Ask LLM to analyze trial outcomes and create edges
        if all_trials:
            trials_text = "\n".join(
                f"- [{t.get('nct_id', 'N/A')}] {t.get('title', '')[:80]}"
                f" | Phase: {t.get('phase', 'N/A')} | Status: {t.get('status', 'N/A')}"
                for t in all_trials[:10]
            )
            analysis_prompt = (
                f"Clinical trials found:\n{trials_text}\n\n"
                f"Research question: {task.instruction}\n\n"
                f"Analyze these trials. For each, identify:\n"
                f"1. The drug/intervention and condition being treated\n"
                f"2. Whether it provides evidence FOR or AGAINST the research hypothesis\n"
                f"3. Any failure analysis for terminated/withdrawn trials\n\n"
                f"Respond as JSON: {{"
                f"\"analyses\": [{{"
                f"\"trial_key\": \"NCT_ID or title\", "
                f"\"drug\": \"...\", \"condition\": \"...\", "
                f"\"evidence_type\": \"EVIDENCE_FOR|EVIDENCE_AGAINST|TREATS\", "
                f"\"confidence\": 0.0-1.0, \"claim\": \"...\", "
                f"\"failure_reason\": \"null or reason\"}}],\n"
                f"\"summary\": \"overall analysis\"}}"
            )
            try:
                analysis_response = await self.query_llm(analysis_prompt, kg_context=kg_context)
                analysis = self.llm.parse_json(analysis_response)

                for a in analysis.get("analyses", []):
                    trial_key = a.get("trial_key", "")
                    trial_node_id = node_name_to_id.get(trial_key)
                    if not trial_node_id:
                        # Try to match by prefix
                        for k, v in node_name_to_id.items():
                            if trial_key in k or k in trial_key:
                                trial_node_id = v
                                break
                    if not trial_node_id:
                        continue

                    # Link to existing disease node if found
                    condition = a.get("condition", "")
                    if condition:
                        disease_node = self.kg.get_node_by_name(condition)
                        if disease_node:
                            try:
                                rel = EdgeRelationType(a.get("evidence_type", "ASSOCIATED_WITH"))
                            except (ValueError, KeyError):
                                rel = EdgeRelationType.ASSOCIATED_WITH
                            conf = float(a.get("confidence", 0.6))
                            edges.append(
                                KGEdge(
                                    source_id=trial_node_id,
                                    target_id=disease_node.id,
                                    relation=rel,
                                    confidence=EdgeConfidence(overall=conf, evidence_quality=conf, evidence_count=1),
                                    evidence=[
                                        EvidenceSource(
                                            source_type=EvidenceSourceType.CLINICALTRIALS,
                                            source_id=trial_key,
                                            claim=a.get("claim", ""),
                                            quality_score=0.8,
                                            confidence=conf,
                                            agent_id=self.agent_id,
                                        )
                                    ],
                                )
                            )

                summary = analysis.get("summary", "")
            except Exception:
                summary = ""
        else:
            summary = "No clinical trials found."

        return {
            "nodes": nodes,
            "edges": edges,
            "summary": summary or f"Analyzed {len(all_trials)} clinical trials.",
            "reasoning_trace": f"Queries: {[q.get('query') for q in queries]}\nTrials found: {len(all_trials)}",
        }
