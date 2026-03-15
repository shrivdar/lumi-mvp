"""Tests for Report Generator V2."""

from __future__ import annotations

import pytest

from core.models import (
    EdgeConfidence,
    EdgeRelationType,
    EvidenceSource,
    EvidenceSourceType,
    HypothesisNode,
    HypothesisStatus,
    KGEdge,
    KGNode,
    NodeType,
    ResearchConfig,
    ResearchResult,
    ResearchSession,
    SessionStatus,
    UncertaintyVector,
)
from report.generator import (
    _build_evidence_chain,
    _confidence_interval_str,
    _format_evidence_chain_markdown,
    generate_report,
    generate_report_v2,
)
from world_model.knowledge_graph import InMemoryKnowledgeGraph


@pytest.fixture()
def report_session() -> ResearchSession:
    return ResearchSession(
        id="test-report-session",
        query="Role of B7-H3 in NSCLC",
        status=SessionStatus.COMPLETED,
        config=ResearchConfig(
            max_hypothesis_depth=2,
            max_mcts_iterations=10,
            enable_falsification=True,
            enable_hitl=True,
        ),
        current_iteration=5,
        total_nodes=10,
        total_edges=8,
        total_hypotheses=4,
    )


@pytest.fixture()
def report_kg() -> InMemoryKnowledgeGraph:
    kg = InMemoryKnowledgeGraph(graph_id="test-report-session")
    kg.add_node(KGNode(id="n-b7h3", type=NodeType.PROTEIN, name="B7-H3",
                       created_by="agent-lit", hypothesis_branch="h1"))
    kg.add_node(KGNode(id="n-nsclc", type=NodeType.DISEASE, name="NSCLC",
                       created_by="agent-lit", hypothesis_branch="h1"))
    kg.add_edge(KGEdge(
        id="e-1", source_id="n-b7h3", target_id="n-nsclc",
        relation=EdgeRelationType.OVEREXPRESSED_IN,
        confidence=EdgeConfidence(overall=0.88, evidence_quality=0.85, evidence_count=5,
                                  falsification_attempts=2, falsification_failures=1),
        created_by="agent-lit", hypothesis_branch="h1",
        evidence=[
            EvidenceSource(source_type=EvidenceSourceType.PUBMED, source_id="PMID:12345",
                          doi="10.1234/test", quality_score=0.9, confidence=0.88,
                          claim="B7-H3 is overexpressed in NSCLC tumors", publication_year=2023,
                          citation_count=45),
        ],
    ))
    return kg


@pytest.fixture()
def report_result(report_kg: InMemoryKnowledgeGraph) -> ResearchResult:
    best = HypothesisNode(
        id="h-best",
        hypothesis="B7-H3 overexpression drives immune evasion in NSCLC",
        confidence=0.85,
        visit_count=5,
        avg_info_gain=1.2,
        supporting_edges=["e-1"],
        contradicting_edges=[],
        status=HypothesisStatus.CONFIRMED,
    )
    h2 = HypothesisNode(
        id="h-alt",
        hypothesis="B7-H3 promotes tumor angiogenesis in NSCLC",
        confidence=0.65,
        visit_count=3,
        avg_info_gain=0.8,
        supporting_edges=[],
        status=HypothesisStatus.EXPLORED,
    )

    edge = report_kg.get_edge("e-1")
    findings = [edge] if edge else []

    return ResearchResult(
        research_id="test-report-session",
        best_hypothesis=best,
        hypothesis_ranking=[best, h2],
        key_findings=findings,
        contradictions=[],
        uncertainties=[
            UncertaintyVector(
                input_ambiguity=0.3,
                data_quality=0.4,
                reasoning_divergence=0.2,
                conflict_uncertainty=0.1,
                novelty_uncertainty=0.5,
                composite=0.35,
                is_critical=False,
            ),
            UncertaintyVector(
                input_ambiguity=0.5,
                data_quality=0.7,
                reasoning_divergence=0.6,
                conflict_uncertainty=0.8,
                novelty_uncertainty=0.3,
                composite=0.65,
                is_critical=True,
            ),
        ],
        recommended_experiments=["Validate B7-H3 expression via IHC in NSCLC tissue samples"],
        total_duration_ms=30000,
        total_llm_calls=25,
        total_tokens=75000,
    )


class TestReportV1:
    @pytest.mark.asyncio
    async def test_generates_markdown(self, report_session, report_result, report_kg):
        report = await generate_report(report_session, report_result, report_kg)
        assert "# Research Report" in report
        assert "B7-H3" in report
        assert "Executive Summary" in report
        assert "Evidence Map" in report
        assert "Competing Hypotheses" in report

    @pytest.mark.asyncio
    async def test_includes_audit_trail(self, report_session, report_result, report_kg):
        report = await generate_report(report_session, report_result, report_kg)
        assert "Audit Trail" in report
        assert "75,000" in report  # token count


class TestReportV2:
    @pytest.mark.asyncio
    async def test_generates_v2_markdown(self, report_session, report_result, report_kg):
        report = await generate_report_v2(report_session, report_result, report_kg)
        assert "# Research Report V2" in report
        assert "Methodology" in report
        assert "Evidence Chains" in report

    @pytest.mark.asyncio
    async def test_includes_methodology(self, report_session, report_result, report_kg):
        report = await generate_report_v2(report_session, report_result, report_kg)
        assert "MCTS Configuration" in report
        assert "Agent Composition" in report
        assert "Execution Summary" in report

    @pytest.mark.asyncio
    async def test_includes_confidence_intervals(self, report_session, report_result, report_kg):
        report = await generate_report_v2(report_session, report_result, report_kg)
        assert "95% CI" in report
        assert "[" in report  # CI brackets

    @pytest.mark.asyncio
    async def test_includes_competing_hypotheses_comparison(self, report_session, report_result, report_kg):
        report = await generate_report_v2(report_session, report_result, report_kg)
        assert "Competing Hypotheses Comparison" in report
        assert "Head-to-Head" in report
        assert "B7-H3 promotes tumor angiogenesis" in report

    @pytest.mark.asyncio
    async def test_includes_evidence_chains(self, report_session, report_result, report_kg):
        report = await generate_report_v2(report_session, report_result, report_kg)
        assert "Evidence Chains" in report
        assert "PUBMED" in report
        assert "PMID:12345" in report

    @pytest.mark.asyncio
    async def test_includes_uncertainty_breakdown(self, report_session, report_result, report_kg):
        report = await generate_report_v2(report_session, report_result, report_kg)
        assert "Critical Uncertainties" in report
        assert "Non-Critical Uncertainties" in report

    @pytest.mark.asyncio
    async def test_handles_no_result(self, report_session, report_kg):
        report = await generate_report_v2(report_session, None, report_kg)
        assert "No results available" in report


class TestEvidenceChain:
    def test_build_evidence_chain(self, report_kg):
        edge = report_kg.get_edge("e-1")
        assert edge is not None
        chain = _build_evidence_chain(edge, report_kg)
        assert len(chain) >= 1
        assert chain[0]["source_type"] == "PUBMED"
        assert chain[0]["source_id"] == "PMID:12345"
        assert chain[0]["doi"] == "10.1234/test"

    def test_format_evidence_chain(self):
        chain = [
            {
                "source_type": "PUBMED",
                "source_id": "PMID:12345",
                "doi": "10.1234/test",
                "title": "Test paper",
                "claim": "B7-H3 is overexpressed",
                "quality_score": 0.9,
                "confidence": 0.88,
                "publication_year": 2023,
                "citation_count": 45,
            }
        ]
        md = _format_evidence_chain_markdown(chain)
        assert "PUBMED" in md
        assert "PMID:12345" in md
        assert "2023" in md
        assert "45 citations" in md

    def test_format_empty_chain(self):
        md = _format_evidence_chain_markdown([])
        assert "No evidence chain" in md


class TestConfidenceInterval:
    def test_confidence_interval_str(self):
        edge = KGEdge(
            source_id="a", target_id="b",
            relation=EdgeRelationType.ASSOCIATED_WITH,
            confidence=EdgeConfidence(overall=0.8, evidence_count=10),
        )
        ci = _confidence_interval_str(edge)
        assert "0.80" in ci
        assert "[" in ci
        assert "]" in ci

    def test_confidence_interval_low_evidence(self):
        edge = KGEdge(
            source_id="a", target_id="b",
            relation=EdgeRelationType.ASSOCIATED_WITH,
            confidence=EdgeConfidence(overall=0.8, evidence_count=1),
        )
        ci = _confidence_interval_str(edge)
        # With low evidence, interval should be wider
        assert "[" in ci
