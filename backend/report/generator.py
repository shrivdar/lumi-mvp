"""Report generator — produces structured research reports from KG + hypothesis tree."""

from __future__ import annotations

from core.interfaces import KnowledgeGraph
from core.llm import LLMClient
from core.models import ResearchResult, ResearchSession


async def generate_report(
    session: ResearchSession,
    result: ResearchResult,
    kg: KnowledgeGraph,
    llm: LLMClient | None = None,
) -> str:
    """Generate a structured markdown research report.

    Sections:
    1. Executive Summary
    2. Evidence Map
    3. Competing Hypotheses
    4. Key Uncertainties
    5. Recommended Experiments
    6. Full Audit Trail

    Each claim references KG edge IDs for traceability.
    """
    sections: list[str] = []

    # --- 1. Executive Summary ---
    best = result.best_hypothesis
    sections.append("# Research Report\n")
    sections.append(f"**Query:** {session.query}\n")
    sections.append(f"**Status:** {session.status}\n")
    sections.append(f"**Duration:** {result.total_duration_ms / 1000:.1f}s\n")
    sections.append(f"**LLM Calls:** {result.total_llm_calls} ({result.total_tokens:,} tokens)\n")

    sections.append("\n## 1. Executive Summary\n")
    if best and best.hypothesis:
        sections.append(f"**Best Hypothesis:** {best.hypothesis}\n")
        sections.append(f"- Confidence: {best.confidence:.2f}\n")
        sections.append(f"- Supporting evidence: {len(best.supporting_edges)} edges\n")
        sections.append(f"- Contradicting evidence: {len(best.contradicting_edges)} edges\n")
        sections.append(f"- Visit count: {best.visit_count}\n")

    # LLM-assisted executive summary
    if llm and result.key_findings:
        summary_prompt = (
            f"Research query: {session.query}\n"
            f"Best hypothesis: {best.hypothesis if best else 'None'}\n"
            f"Key findings ({len(result.key_findings)} edges):\n"
        )
        for edge in result.key_findings[:10]:
            summary_prompt += (
                f"- {edge.source_id} --[{edge.relation}]--> {edge.target_id} "
                f"(confidence: {edge.confidence.overall:.2f})\n"
            )
        summary_prompt += "\nWrite a 3-4 sentence executive summary of these findings."

        try:
            narrative = await llm.query(
                summary_prompt,
                system_prompt="You are a scientific report writer. Be concise and precise.",
                research_id=session.id,
            )
            sections.append(f"\n{narrative}\n")
        except Exception:
            pass

    # --- 2. Evidence Map ---
    sections.append("\n## 2. Evidence Map\n")
    sections.append(f"Knowledge graph: **{result.kg_stats.get('nodes', 0)}** nodes, "
                    f"**{result.kg_stats.get('edges', 0)}** edges\n")

    if result.key_findings:
        sections.append("\n### Top Findings\n")
        sections.append("| # | Relation | Confidence | Edge ID |\n")
        sections.append("|---|----------|------------|----------|\n")
        for i, edge in enumerate(result.key_findings[:15], 1):
            sections.append(
                f"| {i} | {edge.source_id} → {edge.relation} → {edge.target_id} "
                f"| {edge.confidence.overall:.2f} | `{edge.id}` |\n"
            )

    # --- 3. Competing Hypotheses ---
    sections.append("\n## 3. Competing Hypotheses\n")
    if result.hypothesis_ranking:
        for i, h in enumerate(result.hypothesis_ranking[:10], 1):
            status_emoji = {
                "CONFIRMED": "[CONFIRMED]",
                "REFUTED": "[REFUTED]",
                "EXPLORED": "[EXPLORED]",
            }.get(str(h.status), f"[{h.status}]")
            sections.append(
                f"{i}. {status_emoji} **{h.hypothesis}** "
                f"(confidence: {h.confidence:.2f}, visits: {h.visit_count}, "
                f"info gain: {h.avg_info_gain:.3f})\n"
            )

    # --- 4. Key Uncertainties ---
    sections.append("\n## 4. Key Uncertainties\n")
    if result.uncertainties:
        for u in result.uncertainties:
            if u.is_critical:
                sections.append(f"- **CRITICAL** composite={u.composite:.2f}: "
                                f"data_quality={u.data_quality:.2f}, "
                                f"conflict={u.conflict_uncertainty:.2f}\n")
    else:
        sections.append("No critical uncertainties flagged.\n")

    # --- 5. Contradictions ---
    sections.append("\n## 5. Contradictions\n")
    if result.contradictions:
        for edge_a, edge_b in result.contradictions[:10]:
            sections.append(
                f"- `{edge_a.id}` ({edge_a.relation}, conf={edge_a.confidence.overall:.2f}) "
                f"vs `{edge_b.id}` ({edge_b.relation}, conf={edge_b.confidence.overall:.2f})\n"
            )
    else:
        sections.append("No contradictions detected.\n")

    # --- 6. Recommended Experiments ---
    sections.append("\n## 6. Recommended Experiments\n")
    if result.recommended_experiments:
        for i, exp in enumerate(result.recommended_experiments, 1):
            sections.append(f"{i}. {exp}\n")
    else:
        sections.append("No experiments recommended.\n")

    # --- 7. Audit Trail ---
    sections.append("\n## 7. Audit Trail\n")
    sections.append(f"- Total MCTS iterations: {session.current_iteration}\n")
    sections.append(f"- Hypotheses explored: {session.total_hypotheses}\n")
    sections.append(f"- KG nodes: {session.total_nodes}\n")
    sections.append(f"- KG edges: {session.total_edges}\n")
    sections.append(f"- Total tokens: {result.total_tokens:,}\n")

    return "".join(sections)
