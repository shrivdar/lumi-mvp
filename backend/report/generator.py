"""Report generator — produces structured research reports from KG + hypothesis tree.

V1: Basic markdown with sections for executive summary, evidence map, hypotheses, etc.
V2: Adds evidence chain visualization, methodology section, competing hypotheses
    comparison, confidence intervals, KG subgraph per claim.
"""

from __future__ import annotations

from typing import Any

import structlog

from core.interfaces import KnowledgeGraph
from core.llm import LLMClient
from core.models import KGEdge, ResearchResult, ResearchSession

logger = structlog.get_logger(__name__)


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


# ═══════════════════════════════════════════════════════════════════════════════
# V2 Report Generator
# ═══════════════════════════════════════════════════════════════════════════════


def _build_evidence_chain(edge: KGEdge, kg: KnowledgeGraph) -> list[dict[str, Any]]:
    """Trace an edge back through the KG to its source papers/databases.

    Returns a list of evidence chain steps, each with source info.
    """
    chain: list[dict[str, Any]] = []
    for ev in edge.evidence:
        step = {
            "source_type": str(ev.source_type),
            "source_id": ev.source_id or "",
            "doi": ev.doi or "",
            "title": ev.title or "",
            "claim": ev.claim,
            "quality_score": ev.quality_score,
            "confidence": ev.confidence,
            "retrieval_method": ev.retrieval_method,
        }
        if ev.publication_year:
            step["publication_year"] = ev.publication_year
        if ev.citation_count:
            step["citation_count"] = ev.citation_count
        chain.append(step)

    # Also trace through KG for upstream supporting edges
    upstream_edges = kg.get_edges_to(edge.source_id)
    for up_edge in upstream_edges[:3]:  # Limit to prevent unbounded recursion
        if up_edge.id != edge.id and not up_edge.falsified:
            chain.append({
                "source_type": "KG_EDGE",
                "edge_id": up_edge.id,
                "relation": str(up_edge.relation),
                "confidence": up_edge.confidence.overall,
                "claim": f"{up_edge.source_id} --[{up_edge.relation}]--> {up_edge.target_id}",
            })

    return chain


def _format_evidence_chain_markdown(chain: list[dict[str, Any]]) -> str:
    """Format an evidence chain as indented markdown."""
    if not chain:
        return "  - _No evidence chain available_\n"

    lines: list[str] = []
    for i, step in enumerate(chain):
        src_type = step.get("source_type", "unknown")
        if src_type == "KG_EDGE":
            lines.append(
                f"  {i + 1}. **KG Edge** `{step.get('edge_id', '')}`: "
                f"{step.get('claim', '')} (conf: {step.get('confidence', 0):.2f})\n"
            )
        else:
            doi_str = f" DOI: {step['doi']}" if step.get("doi") else ""
            year_str = f" ({step['publication_year']})" if step.get("publication_year") else ""
            cite_str = f" [{step['citation_count']} citations]" if step.get("citation_count") else ""
            lines.append(
                f"  {i + 1}. **{src_type}** {step.get('source_id', '')}{year_str}{doi_str}{cite_str}\n"
                f"     Claim: {step.get('claim', 'N/A')} "
                f"(quality: {step.get('quality_score', 0):.2f}, conf: {step.get('confidence', 0):.2f})\n"
            )
    return "".join(lines)


def _confidence_interval_str(edge: KGEdge) -> str:
    """Compute and format a confidence interval for an edge.

    Uses evidence count and falsification attempts to estimate bounds.
    """
    c = edge.confidence
    n = max(c.evidence_count, 1)
    # Wilson score interval approximation
    z = 1.96  # 95% CI
    p = c.overall
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominator
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return f"{c.overall:.2f} [{low:.2f}, {high:.2f}]"


async def generate_report_v2(
    session: ResearchSession,
    result: ResearchResult | None,
    kg: KnowledgeGraph,
    llm: LLMClient | None = None,
) -> str:
    """Generate a V2 structured research report with enhanced sections.

    V2 additions over V1:
    - Evidence chain visualization per claim
    - Methodology section (MCTS parameters, agent composition, tool usage)
    - Competing hypotheses comparison table
    - Confidence intervals on key findings
    - KG subgraph summary per claim
    """
    if result is None:
        return "# Research Report\n\n**Status:** No results available.\n"

    sections: list[str] = []
    best = result.best_hypothesis

    # ── Header ──
    sections.append("# Research Report V2\n")
    sections.append(f"**Query:** {session.query}\n")
    sections.append(f"**Status:** {session.status}\n")
    sections.append(f"**Duration:** {result.total_duration_ms / 1000:.1f}s\n")
    sections.append(f"**LLM Calls:** {result.total_llm_calls} ({result.total_tokens:,} tokens)\n")
    sections.append(f"**Generated:** {result.created_at}\n")

    # ── 1. Executive Summary ──
    sections.append("\n## 1. Executive Summary\n")
    if best and best.hypothesis:
        sections.append(f"**Best Hypothesis:** {best.hypothesis}\n")
        sections.append(f"- Confidence: {best.confidence:.2f}\n")
        sections.append(f"- Supporting evidence: {len(best.supporting_edges)} edges\n")
        sections.append(f"- Contradicting evidence: {len(best.contradicting_edges)} edges\n")
        sections.append(f"- MCTS visits: {best.visit_count}\n")
        sections.append(f"- Average information gain: {best.avg_info_gain:.3f}\n")

    # LLM-assisted executive summary with V2 depth
    if llm and result.key_findings:
        finding_lines = []
        for edge in result.key_findings[:10]:
            finding_lines.append(
                f"- {edge.source_id} --[{edge.relation}]--> {edge.target_id} "
                f"(confidence: {edge.confidence.overall:.2f}, "
                f"evidence count: {edge.confidence.evidence_count}, "
                f"falsification attempts: {edge.confidence.falsification_attempts})"
            )

        summary_prompt = (
            f"Research query: {session.query}\n"
            f"Best hypothesis: {best.hypothesis if best else 'None'}\n"
            f"Confidence: {best.confidence:.2f if best else 0}\n"
            f"Key findings ({len(result.key_findings)} edges):\n"
            + "\n".join(finding_lines) + "\n\n"
            f"Contradictions found: {len(result.contradictions)}\n"
            f"Competing hypotheses: {len(result.hypothesis_ranking)}\n\n"
            "Write a 4-5 sentence executive summary. Include the main conclusion, "
            "strength of evidence, key caveats, and most important next step."
        )

        try:
            narrative = await llm.query(
                summary_prompt,
                system_prompt=(
                    "You are a senior scientific report writer. Be concise, precise, "
                    "and highlight both supporting and contradicting evidence. "
                    "Use hedging language appropriate to the confidence level."
                ),
                research_id=session.id,
            )
            sections.append(f"\n{narrative}\n")
        except Exception:
            pass

    # ── 2. Methodology ──
    sections.append("\n## 2. Methodology\n")
    sections.append("### MCTS Configuration\n")
    sections.append(f"- Max hypothesis depth: {session.config.max_hypothesis_depth}\n")
    sections.append(f"- Max MCTS iterations: {session.config.max_mcts_iterations}\n")
    sections.append(f"- Confidence threshold: {session.config.confidence_threshold}\n")
    sections.append(f"- Falsification enabled: {session.config.enable_falsification}\n")
    sections.append(f"- HITL enabled: {session.config.enable_hitl}\n")
    sections.append(f"- HITL uncertainty threshold: {session.config.hitl_uncertainty_threshold}\n")

    sections.append("\n### Agent Composition\n")
    sections.append(f"- Max agents per swarm: {session.config.max_agents_per_swarm}\n")
    sections.append(f"- Max concurrent agents: {session.config.max_concurrent_agents}\n")
    sections.append(f"- Agent token budget: {session.config.agent_token_budget:,}\n")
    sections.append(f"- Session token budget: {session.config.session_token_budget:,}\n")

    sections.append("\n### Execution Summary\n")
    sections.append(f"- Iterations completed: {session.current_iteration}\n")
    sections.append(f"- Total hypotheses explored: {session.total_hypotheses}\n")
    sections.append(f"- KG nodes created: {session.total_nodes}\n")
    sections.append(f"- KG edges created: {session.total_edges}\n")
    sections.append(f"- Total tokens used: {result.total_tokens:,}\n")

    # ── 3. Evidence Map with Chains ──
    sections.append("\n## 3. Evidence Map\n")
    sections.append(
        f"Knowledge graph: **{result.kg_stats.get('nodes', 0)}** nodes, "
        f"**{result.kg_stats.get('edges', 0)}** edges\n"
    )

    if result.key_findings:
        sections.append("\n### Top Findings with Evidence Chains\n")
        sections.append(
            "| # | Relation | Confidence [95% CI] | Evidence Count | "
            "Falsification | Edge ID |\n"
        )
        sections.append(
            "|---|----------|---------------------|----------------|"
            "---------------|----------|\n"
        )
        for i, edge in enumerate(result.key_findings[:15], 1):
            ci_str = _confidence_interval_str(edge)
            fals_str = (
                f"{edge.confidence.falsification_attempts} attempts, "
                f"{edge.confidence.falsification_failures} survived"
            )
            sections.append(
                f"| {i} | {edge.source_id} -> {edge.relation} -> {edge.target_id} "
                f"| {ci_str} | {edge.confidence.evidence_count} | {fals_str} "
                f"| `{edge.id}` |\n"
            )

        # Evidence chains for top 5
        sections.append("\n### Evidence Chains (Top 5 Claims)\n")
        for i, edge in enumerate(result.key_findings[:5], 1):
            sections.append(
                f"\n**Claim {i}:** {edge.source_id} --[{edge.relation}]--> "
                f"{edge.target_id} (conf: {_confidence_interval_str(edge)})\n"
            )
            chain = _build_evidence_chain(edge, kg)
            sections.append(_format_evidence_chain_markdown(chain))

            # KG subgraph summary for this claim
            try:
                subgraph = kg.get_subgraph(edge.source_id, hops=1)
                sub_nodes = subgraph.get("nodes", [])
                sub_edges = subgraph.get("edges", [])
                if sub_nodes or sub_edges:
                    sections.append(
                        f"  **Local KG context:** {len(sub_nodes)} nodes, "
                        f"{len(sub_edges)} edges in 1-hop neighborhood\n"
                    )
            except Exception:
                pass

    # ── 4. Competing Hypotheses Comparison ──
    sections.append("\n## 4. Competing Hypotheses Comparison\n")
    if result.hypothesis_ranking and len(result.hypothesis_ranking) > 1:
        sections.append(
            "| Rank | Hypothesis | Status | Confidence | Visits | "
            "Info Gain | Supporting | Contradicting |\n"
        )
        sections.append(
            "|------|-----------|--------|------------|--------|"
            "-----------|------------|---------------|\n"
        )
        for i, h in enumerate(result.hypothesis_ranking[:10], 1):
            status_tag = str(h.status)
            hyp_text = h.hypothesis[:80] + "..." if len(h.hypothesis) > 80 else h.hypothesis
            sections.append(
                f"| {i} | {hyp_text} | {status_tag} | {h.confidence:.2f} "
                f"| {h.visit_count} | {h.avg_info_gain:.3f} "
                f"| {len(h.supporting_edges)} | {len(h.contradicting_edges)} |\n"
            )

        # Pairwise comparison of top 2 hypotheses
        if len(result.hypothesis_ranking) >= 2:
            h1 = result.hypothesis_ranking[0]
            h2 = result.hypothesis_ranking[1]
            sections.append("\n### Head-to-Head: Top Two Hypotheses\n")
            sections.append(f"**H1:** {h1.hypothesis}\n")
            sections.append(f"**H2:** {h2.hypothesis}\n\n")

            conf_diff = h1.confidence - h2.confidence
            if abs(conf_diff) < 0.1:
                sections.append(
                    "These hypotheses have **similar confidence levels** "
                    f"({h1.confidence:.2f} vs {h2.confidence:.2f}). "
                    "Further investigation may be warranted.\n"
                )
            elif conf_diff > 0:
                sections.append(
                    f"H1 leads by **{conf_diff:.2f}** confidence points "
                    f"with {len(h1.supporting_edges)} supporting edges vs "
                    f"{len(h2.supporting_edges)} for H2.\n"
                )
            else:
                sections.append(
                    f"H2 leads by **{abs(conf_diff):.2f}** confidence points "
                    f"despite being ranked lower by information gain.\n"
                )
    elif result.hypothesis_ranking:
        sections.append(f"Only one hypothesis explored: **{result.hypothesis_ranking[0].hypothesis}**\n")
    else:
        sections.append("No hypotheses were explored.\n")

    # ── 5. Key Uncertainties ──
    sections.append("\n## 5. Key Uncertainties\n")
    if result.uncertainties:
        critical = [u for u in result.uncertainties if u.is_critical]
        non_critical = [u for u in result.uncertainties if not u.is_critical]

        if critical:
            sections.append(f"\n### Critical Uncertainties ({len(critical)})\n")
            for u in critical:
                sections.append(
                    f"- **Composite: {u.composite:.2f}** | "
                    f"Ambiguity: {u.input_ambiguity:.2f}, "
                    f"Data Quality: {u.data_quality:.2f}, "
                    f"Divergence: {u.reasoning_divergence:.2f}, "
                    f"Conflict: {u.conflict_uncertainty:.2f}, "
                    f"Novelty: {u.novelty_uncertainty:.2f}\n"
                )

        if non_critical:
            sections.append(f"\n### Non-Critical Uncertainties ({len(non_critical)})\n")
            avg_composite = sum(u.composite for u in non_critical) / len(non_critical)
            sections.append(f"Average composite uncertainty: {avg_composite:.2f}\n")
    else:
        sections.append("No uncertainties recorded.\n")

    # ── 6. Contradictions ──
    sections.append("\n## 6. Contradictions\n")
    if result.contradictions:
        sections.append(f"Found **{len(result.contradictions)}** contradiction pair(s):\n\n")
        for idx, (edge_a, edge_b) in enumerate(result.contradictions[:10], 1):
            sections.append(f"### Contradiction {idx}\n")
            sections.append(
                f"- **A:** `{edge_a.id}` {edge_a.source_id} --[{edge_a.relation}]--> "
                f"{edge_a.target_id} (conf: {edge_a.confidence.overall:.2f})\n"
            )
            sections.append(
                f"- **B:** `{edge_b.id}` {edge_b.source_id} --[{edge_b.relation}]--> "
                f"{edge_b.target_id} (conf: {edge_b.confidence.overall:.2f})\n"
            )
            # Note which has stronger evidence
            if edge_a.confidence.evidence_count > edge_b.confidence.evidence_count:
                sections.append(
                    f"  Edge A has stronger evidence support "
                    f"({edge_a.confidence.evidence_count} vs {edge_b.confidence.evidence_count} sources).\n"
                )
            elif edge_b.confidence.evidence_count > edge_a.confidence.evidence_count:
                sections.append(
                    f"  Edge B has stronger evidence support "
                    f"({edge_b.confidence.evidence_count} vs {edge_a.confidence.evidence_count} sources).\n"
                )
    else:
        sections.append("No contradictions detected.\n")

    # ── 7. Recommended Experiments ──
    sections.append("\n## 7. Recommended Experiments\n")
    if result.recommended_experiments:
        for i, exp in enumerate(result.recommended_experiments, 1):
            sections.append(f"{i}. {exp}\n")
    else:
        sections.append("No experiments recommended.\n")

    # ── 8. Audit Trail ──
    sections.append("\n## 8. Audit Trail\n")
    sections.append(f"- Total MCTS iterations: {session.current_iteration}\n")
    sections.append(f"- Hypotheses explored: {session.total_hypotheses}\n")
    sections.append(f"- KG nodes: {session.total_nodes}\n")
    sections.append(f"- KG edges: {session.total_edges}\n")
    sections.append(f"- Total LLM calls: {result.total_llm_calls}\n")
    sections.append(f"- Total tokens: {result.total_tokens:,}\n")
    sections.append(f"- Duration: {result.total_duration_ms / 1000:.1f}s\n")

    if result.screening:
        sections.append("\n### Biosecurity Screening\n")
        sections.append(f"- Tier: **{result.screening.tier}**\n")
        if result.screening.flagged_categories:
            sections.append(
                f"- Flagged: {', '.join(str(c) for c in result.screening.flagged_categories)}\n"
            )
        if result.screening.disclaimer:
            sections.append(f"- Disclaimer: {result.screening.disclaimer}\n")

    return "".join(sections)
