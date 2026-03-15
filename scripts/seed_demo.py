#!/usr/bin/env python3
"""Seed demo script — populates a KG with realistic B7-H3/NSCLC data for frontend development.

Creates:
- 55+ nodes (genes, proteins, drugs, pathways, diseases, variants, trials, etc.)
- 85+ edges with varied relation types
- 5 hypotheses in different states (exploring, promising, refuted, confirmed)
- Realistic confidence scores (0.3-0.95)
- Contradictions for testing contradiction detection UI
- Falsified edges
- Exports to JSON for frontend consumption

Uses the actual InMemoryKnowledgeGraph class to validate the API.

Usage:
    python scripts/seed_demo.py
    python scripts/seed_demo.py --output frontend/public/demo-kg.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure backend is on sys.path
_backend = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_backend))

from core.models import (  # noqa: E402
    EdgeConfidence,
    EdgeRelationType,
    EvidenceSource,
    EvidenceSourceType,
    HypothesisNode,
    HypothesisStatus,
    KGEdge,
    KGNode,
    NodeType,
)
from world_model.knowledge_graph import InMemoryKnowledgeGraph  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Hypothesis definitions
# ─────────────────────────────────────────────────────────────────────

HYPOTHESES = [
    HypothesisNode(
        id="h-root",
        hypothesis="B7-H3 (CD276) plays a critical role in NSCLC immune evasion",
        rationale="Root hypothesis",
        depth=0,
        status=HypothesisStatus.EXPLORED,
        visit_count=12,
        total_info_gain=8.5,
        avg_info_gain=0.71,
        confidence=0.82,
    ),
    HypothesisNode(
        id="h-checkpoint",
        parent_id="h-root",
        hypothesis="B7-H3 directly suppresses CD8+ T-cell cytotoxicity via unknown inhibitory receptor",
        rationale="B7-H3 is a B7 family checkpoint molecule; direct T-cell suppression is likely",
        depth=1,
        status=HypothesisStatus.CONFIRMED,
        visit_count=6,
        total_info_gain=5.2,
        avg_info_gain=0.87,
        confidence=0.88,
        supporting_edges=["e-b7h3-cd8-inhib", "e-b7h3-nsclc-overexp", "e-b7h3-tlt2-bind"],
    ),
    HypothesisNode(
        id="h-pi3k",
        parent_id="h-root",
        hypothesis="B7-H3 promotes NSCLC metastasis through PI3K/AKT/mTOR signaling",
        rationale="B7-H3 has non-immunological signaling roles in cancer cell migration",
        depth=1,
        status=HypothesisStatus.EXPLORING,
        visit_count=4,
        total_info_gain=2.1,
        avg_info_gain=0.53,
        confidence=0.62,
        supporting_edges=["e-b7h3-pi3k-act"],
    ),
    HypothesisNode(
        id="h-adc",
        parent_id="h-root",
        hypothesis="Anti-B7-H3 ADCs can overcome B7-H3-mediated immune evasion in NSCLC",
        rationale="ADC approach targets B7-H3+ tumor cells while avoiding systemic immune effects",
        depth=1,
        status=HypothesisStatus.EXPLORED,
        visit_count=5,
        total_info_gain=3.8,
        avg_info_gain=0.76,
        confidence=0.79,
        supporting_edges=["e-ds7300-b7h3-target", "e-ds7300-nsclc-treat"],
    ),
    HypothesisNode(
        id="h-costim",
        parent_id="h-checkpoint",
        hypothesis="B7-H3 has a dual costimulatory/coinhibitory role depending on tumor microenvironment",
        rationale="Some studies suggest B7-H3 can activate T cells in certain contexts",
        depth=2,
        status=HypothesisStatus.REFUTED,
        visit_count=3,
        total_info_gain=1.2,
        avg_info_gain=0.40,
        confidence=0.25,
        contradicting_edges=["e-b7h3-cd8-costim"],
    ),
]


# ─────────────────────────────────────────────────────────────────────
# Node definitions
# ─────────────────────────────────────────────────────────────────────

def _ev(source_type: EvidenceSourceType, pmid: str, quality: float, agent: str, title: str = "") -> EvidenceSource:
    """Convenience helper for creating evidence sources."""
    return EvidenceSource(
        source_type=source_type,
        source_id=pmid,
        title=title,
        quality_score=quality,
        agent_id=agent,
        is_peer_reviewed=True,
    )


NODES: list[dict] = [
    # === Proteins / Checkpoint molecules ===
    dict(id="n-b7h3", type=NodeType.PROTEIN, name="B7-H3", aliases=["CD276", "B7RP-2"],
         description="B7 family member 3, immune checkpoint ligand overexpressed in NSCLC",
         confidence=0.95, created_by="agent-lit-001", hypothesis_branch="h-checkpoint",
         external_ids={"uniprot": "Q5ZPR3", "ncbi_gene": "80381"},
         sources=[_ev(EvidenceSourceType.UNIPROT, "Q5ZPR3", 0.95, "agent-lit-001", "UniProt B7-H3")]),
    dict(id="n-pdl1", type=NodeType.PROTEIN, name="PD-L1", aliases=["CD274", "B7-H1"],
         description="Programmed death-ligand 1, established immune checkpoint in NSCLC",
         confidence=0.98, created_by="agent-lit-001", hypothesis_branch="h-checkpoint",
         external_ids={"uniprot": "Q9NZQ7"}),
    dict(id="n-pd1", type=NodeType.PROTEIN, name="PD-1", aliases=["CD279", "PDCD1"],
         description="Programmed cell death protein 1, inhibitory receptor on T cells",
         confidence=0.98, created_by="agent-lit-001", hypothesis_branch="h-checkpoint",
         external_ids={"uniprot": "Q15116"}),
    dict(id="n-ctla4", type=NodeType.PROTEIN, name="CTLA-4", aliases=["CD152"],
         description="Cytotoxic T-lymphocyte associated protein 4",
         confidence=0.97, created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="n-tlt2", type=NodeType.PROTEIN, name="TLT-2", aliases=["TREML2"],
         description="TREM-like transcript 2, putative B7-H3 binding partner",
         confidence=0.55, created_by="agent-prot-002", hypothesis_branch="h-checkpoint"),
    dict(id="n-b7h3-igv", type=NodeType.STRUCTURE, name="B7-H3 IgV Domain",
         description="Immunoglobulin variable domain responsible for receptor interaction",
         confidence=0.85, created_by="agent-prot-002", hypothesis_branch="h-checkpoint"),
    dict(id="n-b7h3-igc", type=NodeType.STRUCTURE, name="B7-H3 IgC Domain",
         description="Immunoglobulin constant domain, structural scaffold",
         confidence=0.82, created_by="agent-prot-002", hypothesis_branch="h-checkpoint"),
    dict(id="n-nkg2d", type=NodeType.PROTEIN, name="NKG2D",
         description="NK cell activating receptor, potential B7-H3 interactor",
         confidence=0.50, created_by="agent-prot-002", hypothesis_branch="h-checkpoint"),

    # === Genes ===
    dict(id="n-cd276", type=NodeType.GENE, name="CD276",
         description="Gene encoding B7-H3 protein, chr15q24.1",
         confidence=0.99, created_by="agent-gen-005", hypothesis_branch="h-checkpoint",
         external_ids={"ensembl": "ENSG00000103855", "ncbi_gene": "80381"}),
    dict(id="n-egfr", type=NodeType.GENE, name="EGFR",
         description="Epidermal growth factor receptor, frequently mutated in NSCLC",
         confidence=0.99, created_by="agent-gen-005", hypothesis_branch="h-pi3k",
         external_ids={"ensembl": "ENSG00000146648"}),
    dict(id="n-kras", type=NodeType.GENE, name="KRAS",
         description="KRAS proto-oncogene, key driver of NSCLC",
         confidence=0.99, created_by="agent-gen-005", hypothesis_branch="h-pi3k",
         external_ids={"ensembl": "ENSG00000133703"}),
    dict(id="n-alk", type=NodeType.GENE, name="ALK",
         description="Anaplastic lymphoma kinase, rearranged in ~5% NSCLC",
         confidence=0.97, created_by="agent-gen-005", hypothesis_branch="h-pi3k"),
    dict(id="n-tp53", type=NodeType.GENE, name="TP53",
         description="Tumor protein p53, most commonly mutated gene in NSCLC",
         confidence=0.99, created_by="agent-gen-005", hypothesis_branch="h-checkpoint"),
    dict(id="n-stk11", type=NodeType.GENE, name="STK11", aliases=["LKB1"],
         description="Serine/threonine kinase 11, loss associated with immune cold tumors",
         confidence=0.93, created_by="agent-gen-005", hypothesis_branch="h-checkpoint"),

    # === Diseases ===
    dict(id="n-nsclc", type=NodeType.DISEASE, name="Non-Small Cell Lung Cancer",
         aliases=["NSCLC"],
         description="Most common type of lung cancer (~85%), includes adenocarcinoma and squamous cell",
         confidence=0.99, created_by="agent-lit-001", hypothesis_branch="h-root"),
    dict(id="n-nsclc-adeno", type=NodeType.DISEASE, name="Lung Adenocarcinoma",
         description="Most common subtype of NSCLC",
         confidence=0.99, created_by="agent-lit-001", hypothesis_branch="h-root"),
    dict(id="n-nsclc-sq", type=NodeType.DISEASE, name="Lung Squamous Cell Carcinoma",
         description="Second most common NSCLC subtype",
         confidence=0.97, created_by="agent-lit-001", hypothesis_branch="h-root"),
    dict(id="n-melanoma", type=NodeType.DISEASE, name="Melanoma",
         description="Malignant melanoma, high B7-H3 expression reported",
         confidence=0.90, created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="n-tnbc", type=NodeType.DISEASE, name="Triple-Negative Breast Cancer",
         aliases=["TNBC"],
         description="Aggressive breast cancer subtype with B7-H3 overexpression",
         confidence=0.85, created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),

    # === Cell Types ===
    dict(id="n-cd8", type=NodeType.CELL_TYPE, name="CD8+ T cells",
         aliases=["Cytotoxic T lymphocytes", "CTLs"],
         description="Primary adaptive immune effectors against tumors",
         confidence=0.99, created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="n-cd4", type=NodeType.CELL_TYPE, name="CD4+ T helper cells",
         description="Helper T cells coordinating immune response",
         confidence=0.98, created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="n-nk", type=NodeType.CELL_TYPE, name="NK cells",
         aliases=["Natural killer cells"],
         description="Innate immune cells with anti-tumor activity",
         confidence=0.96, created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="n-treg", type=NodeType.CELL_TYPE, name="Regulatory T cells",
         aliases=["Tregs", "CD4+CD25+ T cells"],
         description="Immunosuppressive T cells enriched in tumor microenvironment",
         confidence=0.94, created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="n-mdsc", type=NodeType.CELL_TYPE, name="MDSCs",
         aliases=["Myeloid-derived suppressor cells"],
         description="Immunosuppressive myeloid cells recruited to tumors",
         confidence=0.88, created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="n-tam", type=NodeType.CELL_TYPE, name="Tumor-associated macrophages",
         aliases=["TAMs"],
         description="M2-polarized macrophages promoting tumor immune evasion",
         confidence=0.86, created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),

    # === Pathways ===
    dict(id="n-pi3k", type=NodeType.PATHWAY, name="PI3K/AKT/mTOR Signaling",
         description="Key survival and proliferation pathway; B7-H3 may activate it intracellularly",
         confidence=0.90, created_by="agent-path-006", hypothesis_branch="h-pi3k"),
    dict(id="n-jak-stat", type=NodeType.PATHWAY, name="JAK-STAT Signaling",
         description="Cytokine signaling pathway regulating immune gene expression",
         confidence=0.92, created_by="agent-path-006", hypothesis_branch="h-checkpoint"),
    dict(id="n-nfkb", type=NodeType.PATHWAY, name="NF-κB Signaling",
         description="Key pro-inflammatory and survival signaling pathway",
         confidence=0.91, created_by="agent-path-006", hypothesis_branch="h-pi3k"),
    dict(id="n-wnt", type=NodeType.PATHWAY, name="Wnt/β-catenin Pathway",
         description="Developmental pathway reactivated in cancer, linked to immune exclusion",
         confidence=0.78, created_by="agent-path-006", hypothesis_branch="h-pi3k"),
    dict(id="n-mapk", type=NodeType.PATHWAY, name="RAS-MAPK Signaling",
         description="Proliferative signaling cascade downstream of KRAS",
         confidence=0.95, created_by="agent-path-006", hypothesis_branch="h-pi3k"),
    dict(id="n-ifn-gamma", type=NodeType.PATHWAY, name="IFN-γ Signaling",
         description="Interferon-gamma pathway driving B7-H3 upregulation",
         confidence=0.83, created_by="agent-path-006", hypothesis_branch="h-checkpoint"),

    # === Drugs ===
    dict(id="n-ds7300", type=NodeType.DRUG, name="DS-7300",
         aliases=["Ifinatamab deruxtecan"],
         description="B7-H3-targeting ADC by Daiichi Sankyo, DXd payload",
         confidence=0.82, created_by="agent-drug-003", hypothesis_branch="h-adc"),
    dict(id="n-omburtamab", type=NodeType.DRUG, name="Omburtamab",
         aliases=["8H9"],
         description="Anti-B7-H3 monoclonal antibody (Y-mAbs Therapeutics)",
         confidence=0.88, created_by="agent-drug-003", hypothesis_branch="h-adc"),
    dict(id="n-mgc018", type=NodeType.DRUG, name="MGC018",
         aliases=["Vobramitamab duocarmazine"],
         description="B7-H3-targeting ADC by MacroGenics, duocarmycin payload",
         confidence=0.75, created_by="agent-drug-003", hypothesis_branch="h-adc"),
    dict(id="n-enoblituzumab", type=NodeType.DRUG, name="Enoblituzumab",
         aliases=["MGA271"],
         description="Fc-optimized anti-B7-H3 antibody enhancing ADCC",
         confidence=0.80, created_by="agent-drug-003", hypothesis_branch="h-adc"),
    dict(id="n-pembrolizumab", type=NodeType.DRUG, name="Pembrolizumab",
         aliases=["Keytruda"],
         description="Anti-PD-1 checkpoint inhibitor, standard of care in NSCLC",
         confidence=0.99, created_by="agent-drug-003", hypothesis_branch="h-checkpoint"),
    dict(id="n-nivolumab", type=NodeType.DRUG, name="Nivolumab",
         aliases=["Opdivo"],
         description="Anti-PD-1 checkpoint inhibitor approved for NSCLC",
         confidence=0.98, created_by="agent-drug-003", hypothesis_branch="h-checkpoint"),
    dict(id="n-osimertinib", type=NodeType.DRUG, name="Osimertinib",
         aliases=["Tagrisso"],
         description="Third-generation EGFR TKI for EGFR-mutant NSCLC",
         confidence=0.97, created_by="agent-drug-003", hypothesis_branch="h-pi3k"),

    # === Clinical Trials ===
    dict(id="n-nct05280470", type=NodeType.CLINICAL_TRIAL, name="NCT05280470",
         description="Phase II trial of DS-7300 in advanced NSCLC",
         confidence=0.75, created_by="agent-clin-007", hypothesis_branch="h-adc"),
    dict(id="n-nct04145622", type=NodeType.CLINICAL_TRIAL, name="NCT04145622",
         description="Phase I/II trial of MGC018 in solid tumors including NSCLC",
         confidence=0.68, created_by="agent-clin-007", hypothesis_branch="h-adc"),
    dict(id="n-nct02982941", type=NodeType.CLINICAL_TRIAL, name="NCT02982941",
         description="Phase I trial of Enoblituzumab + Pembrolizumab in solid tumors",
         confidence=0.72, created_by="agent-clin-007", hypothesis_branch="h-adc"),

    # === Biomarkers ===
    dict(id="n-tmb", type=NodeType.BIOMARKER, name="Tumor Mutational Burden",
         aliases=["TMB"],
         description="Number of somatic mutations per megabase; predicts immunotherapy response",
         confidence=0.88, created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="n-pdl1-tps", type=NodeType.BIOMARKER, name="PD-L1 TPS",
         description="PD-L1 tumor proportion score, guides pembrolizumab eligibility",
         confidence=0.92, created_by="agent-clin-007", hypothesis_branch="h-checkpoint"),
    dict(id="n-b7h3-ihc", type=NodeType.BIOMARKER, name="B7-H3 IHC Score",
         description="Immunohistochemistry score for B7-H3 expression in tumor",
         confidence=0.70, created_by="agent-clin-007", hypothesis_branch="h-adc"),

    # === Tissue ===
    dict(id="n-tme", type=NodeType.TISSUE, name="Tumor Microenvironment",
         aliases=["TME"],
         description="Complex ecosystem of tumor cells, immune cells, stroma, and vasculature",
         confidence=0.95, created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="n-lung-tissue", type=NodeType.TISSUE, name="Lung Epithelium",
         description="Normal lung tissue; low B7-H3 expression baseline",
         confidence=0.93, created_by="agent-lit-001", hypothesis_branch="h-root"),

    # === Mechanisms ===
    dict(id="n-immune-evasion", type=NodeType.MECHANISM, name="Immune Evasion",
         description="Tumor escape from immune surveillance via checkpoint upregulation",
         confidence=0.94, created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="n-emt", type=NodeType.MECHANISM, name="Epithelial-Mesenchymal Transition",
         aliases=["EMT"],
         description="Process by which tumor cells gain migratory and invasive properties",
         confidence=0.80, created_by="agent-lit-001", hypothesis_branch="h-pi3k"),
    dict(id="n-adcc", type=NodeType.MECHANISM, name="ADCC",
         aliases=["Antibody-dependent cellular cytotoxicity"],
         description="Fc-mediated killing of antibody-coated target cells by NK cells",
         confidence=0.92, created_by="agent-drug-003", hypothesis_branch="h-adc"),

    # === Publications (counter-evidence) ===
    dict(id="n-pub-costim", type=NodeType.PUBLICATION, name="Hashiguchi et al. 2008",
         description="Reports B7-H3 costimulatory function enhancing T-cell proliferation",
         confidence=0.60, created_by="agent-critic-004", hypothesis_branch="h-costim"),
    dict(id="n-pub-inhib", type=NodeType.PUBLICATION, name="Lee et al. 2017",
         description="Confirms B7-H3 coinhibitory role suppressing T-cell and NK-cell function",
         confidence=0.85, created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),

    # === Variants ===
    dict(id="n-egfr-l858r", type=NodeType.BIOMARKER, name="EGFR L858R",
         description="Activating EGFR mutation, sensitive to TKI therapy",
         confidence=0.97, created_by="agent-gen-005", hypothesis_branch="h-pi3k"),
    dict(id="n-kras-g12c", type=NodeType.BIOMARKER, name="KRAS G12C",
         description="KRAS driver mutation targetable by sotorasib/adagrasib",
         confidence=0.96, created_by="agent-gen-005", hypothesis_branch="h-pi3k"),
]


# ─────────────────────────────────────────────────────────────────────
# Edge definitions
# ─────────────────────────────────────────────────────────────────────

def _ec(overall: float, quality: float, count: int, fals_attempts: int = 0, fals_failures: int = 0) -> EdgeConfidence:
    return EdgeConfidence(
        overall=overall, evidence_quality=quality, evidence_count=count,
        falsification_attempts=fals_attempts, falsification_failures=fals_failures,
    )


EDGES: list[dict] = [
    # === B7-H3 core biology ===
    dict(id="e-b7h3-nsclc-overexp", source_id="n-b7h3", target_id="n-nsclc",
         relation=EdgeRelationType.OVEREXPRESSED_IN, confidence=_ec(0.92, 0.88, 12, 2, 2),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-cd276-b7h3", source_id="n-cd276", target_id="n-b7h3",
         relation=EdgeRelationType.ENCODES, confidence=_ec(0.99, 0.99, 20),
         created_by="agent-gen-005", hypothesis_branch="h-checkpoint"),
    dict(id="e-b7h3-cd8-inhib", source_id="n-b7h3", target_id="n-cd8",
         relation=EdgeRelationType.INHIBITS, confidence=_ec(0.82, 0.78, 8, 3, 2),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-b7h3-nk-inhib", source_id="n-b7h3", target_id="n-nk",
         relation=EdgeRelationType.INHIBITS, confidence=_ec(0.75, 0.70, 5, 1, 1),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-b7h3-tlt2-bind", source_id="n-b7h3", target_id="n-tlt2",
         relation=EdgeRelationType.BINDS_TO, confidence=_ec(0.48, 0.40, 3, 2, 0),
         created_by="agent-prot-002", hypothesis_branch="h-checkpoint"),
    dict(id="e-b7h3-nkg2d-interact", source_id="n-b7h3", target_id="n-nkg2d",
         relation=EdgeRelationType.INTERACTS_WITH, confidence=_ec(0.38, 0.35, 2),
         created_by="agent-prot-002", hypothesis_branch="h-checkpoint"),
    dict(id="e-b7h3-igv-domain", source_id="n-b7h3", target_id="n-b7h3-igv",
         relation=EdgeRelationType.DOMAIN_OF, confidence=_ec(0.95, 0.95, 10),
         created_by="agent-prot-002", hypothesis_branch="h-checkpoint"),
    dict(id="e-b7h3-igc-domain", source_id="n-b7h3", target_id="n-b7h3-igc",
         relation=EdgeRelationType.DOMAIN_OF, confidence=_ec(0.95, 0.95, 10),
         created_by="agent-prot-002", hypothesis_branch="h-checkpoint"),

    # === CONTRADICTION: B7-H3 costimulatory (contradicts INHIBITS) ===
    dict(id="e-b7h3-cd8-costim", source_id="n-b7h3", target_id="n-cd8",
         relation=EdgeRelationType.ACTIVATES, confidence=_ec(0.35, 0.30, 2, 1, 0),
         created_by="agent-critic-004", hypothesis_branch="h-costim"),

    # === Immune evasion mechanism ===
    dict(id="e-b7h3-immune-evasion", source_id="n-b7h3", target_id="n-immune-evasion",
         relation=EdgeRelationType.PARTICIPATES_IN, confidence=_ec(0.85, 0.80, 9),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-pdl1-cd8-inhib", source_id="n-pdl1", target_id="n-cd8",
         relation=EdgeRelationType.INHIBITS, confidence=_ec(0.95, 0.93, 25),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-pd1-pdl1-bind", source_id="n-pd1", target_id="n-pdl1",
         relation=EdgeRelationType.BINDS_TO, confidence=_ec(0.99, 0.99, 30),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-ctla4-cd8-inhib", source_id="n-ctla4", target_id="n-cd4",
         relation=EdgeRelationType.INHIBITS, confidence=_ec(0.94, 0.92, 20),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-b7h3-treg-upreg", source_id="n-b7h3", target_id="n-treg",
         relation=EdgeRelationType.UPREGULATES, confidence=_ec(0.58, 0.50, 3),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-b7h3-mdsc-recruit", source_id="n-b7h3", target_id="n-mdsc",
         relation=EdgeRelationType.UPREGULATES, confidence=_ec(0.52, 0.45, 2),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),

    # === TME context ===
    dict(id="e-cd8-tme-member", source_id="n-cd8", target_id="n-tme",
         relation=EdgeRelationType.MEMBER_OF, confidence=_ec(0.95, 0.95, 15),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-treg-tme-member", source_id="n-treg", target_id="n-tme",
         relation=EdgeRelationType.MEMBER_OF, confidence=_ec(0.93, 0.90, 12),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-tam-tme-member", source_id="n-tam", target_id="n-tme",
         relation=EdgeRelationType.MEMBER_OF, confidence=_ec(0.90, 0.88, 10),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-mdsc-tme-member", source_id="n-mdsc", target_id="n-tme",
         relation=EdgeRelationType.MEMBER_OF, confidence=_ec(0.88, 0.85, 8),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),

    # === PI3K/AKT signaling (h-pi3k) ===
    dict(id="e-b7h3-pi3k-act", source_id="n-b7h3", target_id="n-pi3k",
         relation=EdgeRelationType.ACTIVATES, confidence=_ec(0.62, 0.55, 4),
         created_by="agent-path-006", hypothesis_branch="h-pi3k"),
    dict(id="e-b7h3-emt-promotes", source_id="n-b7h3", target_id="n-emt",
         relation=EdgeRelationType.ACTIVATES, confidence=_ec(0.55, 0.48, 3),
         created_by="agent-path-006", hypothesis_branch="h-pi3k"),
    dict(id="e-pi3k-emt-upreg", source_id="n-pi3k", target_id="n-emt",
         relation=EdgeRelationType.UPSTREAM_OF, confidence=_ec(0.78, 0.75, 8),
         created_by="agent-path-006", hypothesis_branch="h-pi3k"),
    dict(id="e-egfr-pi3k-act", source_id="n-egfr", target_id="n-pi3k",
         relation=EdgeRelationType.ACTIVATES, confidence=_ec(0.95, 0.93, 25),
         created_by="agent-path-006", hypothesis_branch="h-pi3k"),
    dict(id="e-kras-mapk-act", source_id="n-kras", target_id="n-mapk",
         relation=EdgeRelationType.ACTIVATES, confidence=_ec(0.97, 0.95, 30),
         created_by="agent-path-006", hypothesis_branch="h-pi3k"),
    dict(id="e-kras-pi3k-act", source_id="n-kras", target_id="n-pi3k",
         relation=EdgeRelationType.ACTIVATES, confidence=_ec(0.82, 0.78, 10),
         created_by="agent-path-006", hypothesis_branch="h-pi3k"),

    # === IFN-γ regulation of B7-H3 ===
    dict(id="e-ifng-b7h3-upreg", source_id="n-ifn-gamma", target_id="n-b7h3",
         relation=EdgeRelationType.UPREGULATES, confidence=_ec(0.72, 0.68, 5),
         created_by="agent-path-006", hypothesis_branch="h-checkpoint"),
    dict(id="e-ifng-pdl1-upreg", source_id="n-ifn-gamma", target_id="n-pdl1",
         relation=EdgeRelationType.UPREGULATES, confidence=_ec(0.90, 0.88, 15),
         created_by="agent-path-006", hypothesis_branch="h-checkpoint"),
    dict(id="e-ifng-jak-upreg", source_id="n-ifn-gamma", target_id="n-jak-stat",
         relation=EdgeRelationType.UPSTREAM_OF, confidence=_ec(0.93, 0.90, 18),
         created_by="agent-path-006", hypothesis_branch="h-checkpoint"),
    dict(id="e-nfkb-b7h3-upreg", source_id="n-nfkb", target_id="n-b7h3",
         relation=EdgeRelationType.UPREGULATES, confidence=_ec(0.60, 0.55, 3),
         created_by="agent-path-006", hypothesis_branch="h-pi3k"),

    # === Genetic context ===
    dict(id="e-egfr-nsclc-assoc", source_id="n-egfr", target_id="n-nsclc",
         relation=EdgeRelationType.ASSOCIATED_WITH, confidence=_ec(0.97, 0.95, 30),
         created_by="agent-gen-005", hypothesis_branch="h-pi3k"),
    dict(id="e-kras-nsclc-assoc", source_id="n-kras", target_id="n-nsclc",
         relation=EdgeRelationType.ASSOCIATED_WITH, confidence=_ec(0.97, 0.95, 30),
         created_by="agent-gen-005", hypothesis_branch="h-pi3k"),
    dict(id="e-alk-nsclc-assoc", source_id="n-alk", target_id="n-nsclc",
         relation=EdgeRelationType.ASSOCIATED_WITH, confidence=_ec(0.93, 0.90, 15),
         created_by="agent-gen-005", hypothesis_branch="h-pi3k"),
    dict(id="e-tp53-nsclc-risk", source_id="n-tp53", target_id="n-nsclc",
         relation=EdgeRelationType.RISK_OF, confidence=_ec(0.95, 0.92, 25),
         created_by="agent-gen-005", hypothesis_branch="h-checkpoint"),
    dict(id="e-stk11-immune-evasion", source_id="n-stk11", target_id="n-immune-evasion",
         relation=EdgeRelationType.ASSOCIATED_WITH, confidence=_ec(0.78, 0.72, 6),
         created_by="agent-gen-005", hypothesis_branch="h-checkpoint"),
    dict(id="e-egfr-l858r-variant", source_id="n-egfr-l858r", target_id="n-egfr",
         relation=EdgeRelationType.BIOMARKER_FOR, confidence=_ec(0.97, 0.95, 20),
         created_by="agent-gen-005", hypothesis_branch="h-pi3k"),
    dict(id="e-kras-g12c-variant", source_id="n-kras-g12c", target_id="n-kras",
         relation=EdgeRelationType.BIOMARKER_FOR, confidence=_ec(0.96, 0.94, 18),
         created_by="agent-gen-005", hypothesis_branch="h-pi3k"),

    # === Disease subtypes ===
    dict(id="e-nsclc-adeno", source_id="n-nsclc-adeno", target_id="n-nsclc",
         relation=EdgeRelationType.ASSOCIATED_WITH, confidence=_ec(0.99, 0.99, 50),
         created_by="agent-lit-001", hypothesis_branch="h-root"),
    dict(id="e-nsclc-sq", source_id="n-nsclc-sq", target_id="n-nsclc",
         relation=EdgeRelationType.ASSOCIATED_WITH, confidence=_ec(0.99, 0.99, 50),
         created_by="agent-lit-001", hypothesis_branch="h-root"),
    dict(id="e-b7h3-melanoma-overexp", source_id="n-b7h3", target_id="n-melanoma",
         relation=EdgeRelationType.OVEREXPRESSED_IN, confidence=_ec(0.80, 0.75, 6),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-b7h3-tnbc-overexp", source_id="n-b7h3", target_id="n-tnbc",
         relation=EdgeRelationType.OVEREXPRESSED_IN, confidence=_ec(0.77, 0.72, 5),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),

    # === Drug targeting ===
    dict(id="e-ds7300-b7h3-target", source_id="n-ds7300", target_id="n-b7h3",
         relation=EdgeRelationType.TARGETS, confidence=_ec(0.88, 0.85, 5),
         created_by="agent-drug-003", hypothesis_branch="h-adc"),
    dict(id="e-omburtamab-b7h3-target", source_id="n-omburtamab", target_id="n-b7h3",
         relation=EdgeRelationType.TARGETS, confidence=_ec(0.90, 0.88, 7),
         created_by="agent-drug-003", hypothesis_branch="h-adc"),
    dict(id="e-mgc018-b7h3-target", source_id="n-mgc018", target_id="n-b7h3",
         relation=EdgeRelationType.TARGETS, confidence=_ec(0.78, 0.72, 4),
         created_by="agent-drug-003", hypothesis_branch="h-adc"),
    dict(id="e-enoblituzumab-b7h3-target", source_id="n-enoblituzumab", target_id="n-b7h3",
         relation=EdgeRelationType.TARGETS, confidence=_ec(0.85, 0.82, 6),
         created_by="agent-drug-003", hypothesis_branch="h-adc"),
    dict(id="e-ds7300-nsclc-treat", source_id="n-ds7300", target_id="n-nsclc",
         relation=EdgeRelationType.TREATS, confidence=_ec(0.65, 0.58, 2),
         created_by="agent-drug-003", hypothesis_branch="h-adc"),
    dict(id="e-pembrolizumab-pd1-target", source_id="n-pembrolizumab", target_id="n-pd1",
         relation=EdgeRelationType.TARGETS, confidence=_ec(0.99, 0.99, 30),
         created_by="agent-drug-003", hypothesis_branch="h-checkpoint"),
    dict(id="e-nivolumab-pd1-target", source_id="n-nivolumab", target_id="n-pd1",
         relation=EdgeRelationType.TARGETS, confidence=_ec(0.99, 0.99, 28),
         created_by="agent-drug-003", hypothesis_branch="h-checkpoint"),
    dict(id="e-pembrolizumab-nsclc-treat", source_id="n-pembrolizumab", target_id="n-nsclc",
         relation=EdgeRelationType.TREATS, confidence=_ec(0.93, 0.90, 20),
         created_by="agent-drug-003", hypothesis_branch="h-checkpoint"),
    dict(id="e-osimertinib-egfr-target", source_id="n-osimertinib", target_id="n-egfr",
         relation=EdgeRelationType.TARGETS, confidence=_ec(0.97, 0.95, 15),
         created_by="agent-drug-003", hypothesis_branch="h-pi3k"),
    dict(id="e-osimertinib-nsclc-treat", source_id="n-osimertinib", target_id="n-nsclc",
         relation=EdgeRelationType.TREATS, confidence=_ec(0.90, 0.88, 12),
         created_by="agent-drug-003", hypothesis_branch="h-pi3k"),
    dict(id="e-enoblituzumab-adcc", source_id="n-enoblituzumab", target_id="n-adcc",
         relation=EdgeRelationType.ACTIVATES, confidence=_ec(0.82, 0.78, 5),
         created_by="agent-drug-003", hypothesis_branch="h-adc"),

    # === Clinical trials ===
    dict(id="e-ds7300-nct05280470", source_id="n-ds7300", target_id="n-nct05280470",
         relation=EdgeRelationType.EVIDENCE_FOR, confidence=_ec(0.72, 0.68, 2),
         created_by="agent-clin-007", hypothesis_branch="h-adc"),
    dict(id="e-mgc018-nct04145622", source_id="n-mgc018", target_id="n-nct04145622",
         relation=EdgeRelationType.EVIDENCE_FOR, confidence=_ec(0.65, 0.60, 1),
         created_by="agent-clin-007", hypothesis_branch="h-adc"),
    dict(id="e-enoblituzumab-nct02982941", source_id="n-enoblituzumab", target_id="n-nct02982941",
         relation=EdgeRelationType.EVIDENCE_FOR, confidence=_ec(0.70, 0.65, 2),
         created_by="agent-clin-007", hypothesis_branch="h-adc"),

    # === Biomarkers ===
    dict(id="e-tmb-nsclc-biomarker", source_id="n-tmb", target_id="n-nsclc",
         relation=EdgeRelationType.BIOMARKER_FOR, confidence=_ec(0.85, 0.82, 10),
         created_by="agent-clin-007", hypothesis_branch="h-checkpoint"),
    dict(id="e-pdl1-tps-nsclc", source_id="n-pdl1-tps", target_id="n-nsclc",
         relation=EdgeRelationType.BIOMARKER_FOR, confidence=_ec(0.90, 0.88, 15),
         created_by="agent-clin-007", hypothesis_branch="h-checkpoint"),
    dict(id="e-b7h3-ihc-nsclc", source_id="n-b7h3-ihc", target_id="n-nsclc",
         relation=EdgeRelationType.BIOMARKER_FOR, confidence=_ec(0.65, 0.58, 3),
         created_by="agent-clin-007", hypothesis_branch="h-adc"),

    # === B7-H3 co-expression with other checkpoints ===
    dict(id="e-b7h3-pdl1-corr", source_id="n-b7h3", target_id="n-pdl1",
         relation=EdgeRelationType.CORRELATES_WITH, confidence=_ec(0.68, 0.62, 4),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),

    # === Publications as evidence ===
    dict(id="e-pub-costim-b7h3", source_id="n-pub-costim", target_id="n-b7h3",
         relation=EdgeRelationType.EVIDENCE_AGAINST, confidence=_ec(0.55, 0.50, 1),
         created_by="agent-critic-004", hypothesis_branch="h-costim"),
    dict(id="e-pub-inhib-b7h3", source_id="n-pub-inhib", target_id="n-b7h3",
         relation=EdgeRelationType.EVIDENCE_FOR, confidence=_ec(0.85, 0.82, 1),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),

    # === Tissue expression ===
    dict(id="e-b7h3-tme-expr", source_id="n-b7h3", target_id="n-tme",
         relation=EdgeRelationType.EXPRESSED_IN, confidence=_ec(0.88, 0.85, 8),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-b7h3-lung-low", source_id="n-b7h3", target_id="n-lung-tissue",
         relation=EdgeRelationType.UNDEREXPRESSED_IN, confidence=_ec(0.80, 0.75, 5),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),

    # === Wnt pathway crosstalk ===
    dict(id="e-wnt-emt-act", source_id="n-wnt", target_id="n-emt",
         relation=EdgeRelationType.ACTIVATES, confidence=_ec(0.82, 0.78, 8),
         created_by="agent-path-006", hypothesis_branch="h-pi3k"),
    dict(id="e-wnt-immune-evasion", source_id="n-wnt", target_id="n-immune-evasion",
         relation=EdgeRelationType.ASSOCIATED_WITH, confidence=_ec(0.65, 0.60, 4),
         created_by="agent-path-006", hypothesis_branch="h-pi3k"),

    # === STK11/LKB1 and immune context ===
    dict(id="e-stk11-cd8-downreg", source_id="n-stk11", target_id="n-cd8",
         relation=EdgeRelationType.DOWNREGULATES, confidence=_ec(0.72, 0.68, 5),
         created_by="agent-gen-005", hypothesis_branch="h-checkpoint"),
    dict(id="e-stk11-pdl1-corr", source_id="n-stk11", target_id="n-pdl1",
         relation=EdgeRelationType.CORRELATES_WITH, confidence=_ec(0.55, 0.50, 3),
         created_by="agent-gen-005", hypothesis_branch="h-checkpoint"),

    # === Additional edges for density (80+ target) ===
    dict(id="e-b7h3-cd4-inhib", source_id="n-b7h3", target_id="n-cd4",
         relation=EdgeRelationType.INHIBITS, confidence=_ec(0.60, 0.55, 3),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-pdl1-nsclc-overexp", source_id="n-pdl1", target_id="n-nsclc",
         relation=EdgeRelationType.OVEREXPRESSED_IN, confidence=_ec(0.88, 0.85, 15),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-pdl1-immune-evasion", source_id="n-pdl1", target_id="n-immune-evasion",
         relation=EdgeRelationType.PARTICIPATES_IN, confidence=_ec(0.93, 0.90, 20),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-tam-immune-evasion", source_id="n-tam", target_id="n-immune-evasion",
         relation=EdgeRelationType.PARTICIPATES_IN, confidence=_ec(0.80, 0.75, 7),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-treg-immune-evasion", source_id="n-treg", target_id="n-immune-evasion",
         relation=EdgeRelationType.PARTICIPATES_IN, confidence=_ec(0.85, 0.80, 10),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-mdsc-immune-evasion", source_id="n-mdsc", target_id="n-immune-evasion",
         relation=EdgeRelationType.PARTICIPATES_IN, confidence=_ec(0.78, 0.72, 6),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-nk-tme-member", source_id="n-nk", target_id="n-tme",
         relation=EdgeRelationType.MEMBER_OF, confidence=_ec(0.92, 0.90, 12),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-b7h3-adeno-overexp", source_id="n-b7h3", target_id="n-nsclc-adeno",
         relation=EdgeRelationType.OVEREXPRESSED_IN, confidence=_ec(0.85, 0.80, 7),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-b7h3-sq-overexp", source_id="n-b7h3", target_id="n-nsclc-sq",
         relation=EdgeRelationType.OVEREXPRESSED_IN, confidence=_ec(0.78, 0.72, 5),
         created_by="agent-lit-001", hypothesis_branch="h-checkpoint"),
    dict(id="e-egfr-b7h3-corr", source_id="n-egfr", target_id="n-b7h3",
         relation=EdgeRelationType.CORRELATES_WITH, confidence=_ec(0.55, 0.48, 3),
         created_by="agent-gen-005", hypothesis_branch="h-pi3k"),
    dict(id="e-pembrolizumab-nsclc-adeno", source_id="n-pembrolizumab", target_id="n-nsclc-adeno",
         relation=EdgeRelationType.TREATS, confidence=_ec(0.90, 0.88, 18),
         created_by="agent-drug-003", hypothesis_branch="h-checkpoint"),
    dict(id="e-nivolumab-nsclc-treat", source_id="n-nivolumab", target_id="n-nsclc",
         relation=EdgeRelationType.TREATS, confidence=_ec(0.91, 0.88, 16),
         created_by="agent-drug-003", hypothesis_branch="h-checkpoint"),
    dict(id="e-omburtamab-nsclc-treat", source_id="n-omburtamab", target_id="n-nsclc",
         relation=EdgeRelationType.TREATS, confidence=_ec(0.55, 0.48, 2),
         created_by="agent-drug-003", hypothesis_branch="h-adc"),
    dict(id="e-mgc018-nsclc-treat", source_id="n-mgc018", target_id="n-nsclc",
         relation=EdgeRelationType.TREATS, confidence=_ec(0.50, 0.42, 1),
         created_by="agent-drug-003", hypothesis_branch="h-adc"),
    dict(id="e-mapk-nsclc-assoc", source_id="n-mapk", target_id="n-nsclc",
         relation=EdgeRelationType.ASSOCIATED_WITH, confidence=_ec(0.88, 0.85, 12),
         created_by="agent-path-006", hypothesis_branch="h-pi3k"),
    dict(id="e-pi3k-nsclc-assoc", source_id="n-pi3k", target_id="n-nsclc",
         relation=EdgeRelationType.ASSOCIATED_WITH, confidence=_ec(0.85, 0.80, 10),
         created_by="agent-path-006", hypothesis_branch="h-pi3k"),
    dict(id="e-nfkb-nsclc-assoc", source_id="n-nfkb", target_id="n-nsclc",
         relation=EdgeRelationType.ASSOCIATED_WITH, confidence=_ec(0.75, 0.70, 6),
         created_by="agent-path-006", hypothesis_branch="h-pi3k"),
    dict(id="e-jak-stat-immune-evasion", source_id="n-jak-stat", target_id="n-immune-evasion",
         relation=EdgeRelationType.REGULATES, confidence=_ec(0.78, 0.72, 7),
         created_by="agent-path-006", hypothesis_branch="h-checkpoint"),
    dict(id="e-b7h3-ihc-adc", source_id="n-b7h3-ihc", target_id="n-ds7300",
         relation=EdgeRelationType.BIOMARKER_FOR, confidence=_ec(0.62, 0.55, 2),
         created_by="agent-clin-007", hypothesis_branch="h-adc"),
    dict(id="e-tmb-pembrolizumab", source_id="n-tmb", target_id="n-pembrolizumab",
         relation=EdgeRelationType.BIOMARKER_FOR, confidence=_ec(0.80, 0.75, 8),
         created_by="agent-clin-007", hypothesis_branch="h-checkpoint"),
    dict(id="e-osimertinib-nsclc-adeno", source_id="n-osimertinib", target_id="n-nsclc-adeno",
         relation=EdgeRelationType.TREATS, confidence=_ec(0.88, 0.85, 10),
         created_by="agent-drug-003", hypothesis_branch="h-pi3k"),
]


# Edges to mark as falsified
FALSIFIED_EDGES = {"e-b7h3-nkg2d-interact", "e-b7h3-cd8-costim"}


# ─────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────

def build_seed_kg() -> InMemoryKnowledgeGraph:
    """Build the seed KG using the actual InMemoryKnowledgeGraph class."""
    kg = InMemoryKnowledgeGraph(graph_id="seed-demo-kg")

    # Add nodes
    for node_data in NODES:
        node = KGNode(**node_data)
        kg.add_node(node)

    # Add edges (with evidence)
    for edge_data in EDGES:
        ev_source = EvidenceSource(
            source_type=EvidenceSourceType.PUBMED,
            source_id=f"PMID:{abs(hash(edge_data['id'])) % 90000000 + 10000000}",
            quality_score=edge_data["confidence"].evidence_quality,
            agent_id=edge_data["created_by"],
        )
        edge = KGEdge(
            evidence=[ev_source],
            **edge_data,
        )
        kg.add_edge(edge)

    # Falsify specific edges
    for edge_id in FALSIFIED_EDGES:
        edge = kg.get_edge(edge_id)
        if edge:
            kg.mark_edge_falsified(
                edge_id,
                evidence=[
                    EvidenceSource(
                        source_type=EvidenceSourceType.PUBMED,
                        source_id="PMID:99999999",
                        title="Counter-evidence from recent meta-analysis",
                        quality_score=0.85,
                        agent_id="agent-critic-004",
                    )
                ],
            )

    return kg


def build_output(kg: InMemoryKnowledgeGraph) -> dict:
    """Build the complete demo output including KG, hypotheses, and metadata."""
    kg_json = kg.to_json()
    cytoscape = kg.to_cytoscape()

    return {
        "graph": kg_json,
        "cytoscape": cytoscape,
        "hypotheses": [h.model_dump(mode="json") for h in HYPOTHESES],
        "metadata": {
            "query": "Role of B7-H3 (CD276) in non-small cell lung cancer immune evasion",
            "session_id": "seed-demo-session",
            "node_count": kg.node_count(),
            "edge_count": kg.edge_count(),
            "hypothesis_count": len(HYPOTHESES),
            "contradiction_count": sum(1 for e in kg._edges.values() if e.is_contradiction),
            "falsified_count": sum(1 for e in kg._edges.values() if e.falsified),
            "avg_confidence": round(kg.avg_confidence(), 3),
            "node_types": {
                ntype.value: len(nids)
                for ntype, nids in sorted(kg._type_index.items(), key=lambda x: -len(x[1]))
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed YOHAS demo KG for frontend development")
    parser.add_argument(
        "--output", "-o",
        default="frontend/public/demo-kg.json",
        help="Output JSON file path (default: frontend/public/demo-kg.json)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON (default: true)",
    )
    args = parser.parse_args()

    print("🌱 YOHAS 3.0 — Seed Demo KG")
    print("   Building knowledge graph...")

    kg = build_seed_kg()
    output = build_output(kg)

    # Print summary
    meta = output["metadata"]
    print(f"\n   Nodes:          {meta['node_count']}")
    print(f"   Edges:          {meta['edge_count']}")
    print(f"   Hypotheses:     {meta['hypothesis_count']}")
    print(f"   Contradictions: {meta['contradiction_count']}")
    print(f"   Falsified:      {meta['falsified_count']}")
    print(f"   Avg confidence: {meta['avg_confidence']:.3f}")
    print("\n   Node types:")
    for ntype, count in meta["node_types"].items():
        print(f"     {ntype:<20} {count}")

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    indent = 2 if args.pretty else None
    with open(out_path, "w") as f:
        json.dump(output, f, indent=indent, default=str)

    size_kb = out_path.stat().st_size / 1024
    print(f"\n   Written to: {out_path} ({size_kb:.1f} KB)")

    # Also write markdown summary
    md_path = out_path.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write(kg.to_markdown_summary())
    print(f"   Summary:    {md_path}")

    print("\n   Done! ✅\n")


if __name__ == "__main__":
    main()
