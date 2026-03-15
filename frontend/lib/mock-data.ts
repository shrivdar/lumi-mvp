// Mock data for standalone UI demo — B7-H3 targeting in NSCLC research session
import type {
  ResearchSession,
  KGNode,
  KGEdge,
  HypothesisNode,
  AgentInfo,
  ResearchEvent,
  ResearchResult,
  ToolRegistryEntry,
  BenchmarkRun,
} from "./types";

const SESSION_ID = "demo-session-001";
const NOW = new Date().toISOString();
const MINUTES_AGO = (m: number) =>
  new Date(Date.now() - m * 60_000).toISOString();

// ─── Helper ──────────────────────────────────────────────────────────────────

function node(
  id: string,
  type: KGNode["type"],
  name: string,
  description: string,
  confidence: number,
  createdBy: string,
  branch: string | null = "hyp-001",
  extra: Partial<KGNode> = {},
): KGNode {
  return {
    id,
    type,
    name,
    aliases: [],
    description,
    properties: {},
    external_ids: {},
    confidence,
    sources: [],
    created_by: createdBy,
    hypothesis_branch: branch,
    created_at: MINUTES_AGO(15),
    updated_at: MINUTES_AGO(5),
    viz: {
      position_x: null,
      position_y: null,
      cluster_id: type,
      visual_weight: confidence,
      color_override: null,
      animation_state: null,
      pinned: false,
      layer: 0,
    },
    ...extra,
  };
}

function edge(
  id: string,
  src: string,
  tgt: string,
  relation: KGEdge["relation"],
  overall: number,
  createdBy: string,
  branch: string | null = "hyp-001",
  falsified = false,
): KGEdge {
  return {
    id,
    source_id: src,
    target_id: tgt,
    relation,
    confidence: {
      overall,
      evidence_quality: overall * 0.9,
      evidence_count: Math.ceil(overall * 8),
      replication_count: Math.ceil(overall * 3),
      falsification_attempts: falsified ? 2 : 1,
      falsification_failures: falsified ? 1 : 0,
      computational_score: overall * 0.85,
      llm_assessment: overall * 0.95,
      last_evaluated: MINUTES_AGO(3),
    },
    properties: {},
    evidence: [],
    created_by: createdBy,
    hypothesis_branch: branch,
    is_contradiction: falsified,
    contradicted_by: [],
    falsified,
    falsification_evidence: null,
    created_at: MINUTES_AGO(12),
    updated_at: MINUTES_AGO(3),
    viz: {
      position_x: null,
      position_y: null,
      cluster_id: null,
      visual_weight: overall,
      color_override: null,
      animation_state: falsified ? "falsified" : overall > 0.7 ? "stable" : "pulsing",
      pinned: false,
      layer: 0,
    },
  };
}

// ─── Knowledge Graph Nodes ───────────────────────────────────────────────────

export const MOCK_NODES: KGNode[] = [
  // Core disease & target
  node("n-b7h3", "PROTEIN", "B7-H3 (CD276)", "Immune checkpoint protein overexpressed in multiple solid tumors including NSCLC. Member of B7 superfamily.", 0.95, "literature_analyst"),
  node("n-nsclc", "DISEASE", "Non-Small Cell Lung Cancer", "Most common form of lung cancer, accounting for ~85% of lung cancer cases. High unmet medical need.", 0.98, "literature_analyst"),
  node("n-egfr", "GENE", "EGFR", "Epidermal growth factor receptor. Frequently mutated in NSCLC (10-35% of cases). Key oncogenic driver.", 0.92, "genomics_mapper"),
  node("n-kras", "GENE", "KRAS", "GTPase involved in cell signaling. KRAS G12C mutation found in ~13% of NSCLC patients.", 0.90, "genomics_mapper"),
  node("n-tp53", "GENE", "TP53", "Tumor suppressor gene. Mutated in ~50% of NSCLC cases. Loss of function promotes tumor growth.", 0.93, "genomics_mapper"),
  node("n-pdl1", "PROTEIN", "PD-L1", "Programmed death-ligand 1. Primary immune checkpoint target in NSCLC immunotherapy.", 0.96, "literature_analyst"),
  node("n-pd1", "PROTEIN", "PD-1", "Programmed cell death protein 1. Receptor for PD-L1 on T cells.", 0.94, "literature_analyst"),

  // Pathways
  node("n-pi3k", "PATHWAY", "PI3K/AKT/mTOR Signaling", "Key survival pathway frequently activated in NSCLC. B7-H3 signals through this pathway.", 0.85, "pathway_analyst"),
  node("n-nfkb", "PATHWAY", "NF-κB Pathway", "Pro-inflammatory transcription factor pathway. Activated by B7-H3 in tumor microenvironment.", 0.78, "pathway_analyst"),
  node("n-jak-stat", "PATHWAY", "JAK-STAT Signaling", "Cytokine signaling pathway. Mediates interferon response in tumor immunity.", 0.82, "pathway_analyst"),
  node("n-mapk", "PATHWAY", "MAPK/ERK Cascade", "Mitogen-activated protein kinase cascade. Downstream of EGFR signaling.", 0.88, "pathway_analyst"),

  // Drugs
  node("n-enob", "DRUG", "Enoblituzumab", "Anti-B7-H3 monoclonal antibody (MacroGenics). In Phase 2 clinical trials for NSCLC.", 0.82, "drug_hunter"),
  node("n-mgd009", "DRUG", "MGD009 (Orlotamab)", "Bispecific DART molecule targeting B7-H3 and CD3. Redirects T cells to B7-H3+ tumor cells.", 0.70, "drug_hunter"),
  node("n-ds7300", "DRUG", "DS-7300 (Ifinatamab)", "B7-H3-targeted ADC (antibody-drug conjugate) by Daiichi Sankyo. Topoisomerase I inhibitor payload.", 0.88, "drug_hunter"),
  node("n-pembro", "DRUG", "Pembrolizumab", "Anti-PD-1 checkpoint inhibitor. Standard of care for PD-L1+ NSCLC.", 0.97, "drug_hunter"),
  node("n-osim", "DRUG", "Osimertinib", "Third-generation EGFR TKI. Standard of care for EGFR-mutant NSCLC.", 0.95, "drug_hunter"),

  // Clinical trials
  node("n-trial1", "CLINICAL_TRIAL", "NCT02923180", "Phase 1/2 trial of enoblituzumab in B7-H3+ solid tumors including NSCLC.", 0.80, "clinical_analyst"),
  node("n-trial2", "CLINICAL_TRIAL", "NCT05180578", "Phase 2 trial of DS-7300 (ifinatamab) in advanced NSCLC after prior therapy.", 0.85, "clinical_analyst"),
  node("n-trial3", "CLINICAL_TRIAL", "NCT05280470", "Combination trial: B7-H3 CAR-T + pembrolizumab in PD-L1 refractory NSCLC.", 0.65, "clinical_analyst"),

  // Mechanisms & biomarkers
  node("n-tme", "MECHANISM", "Tumor Microenvironment Remodeling", "B7-H3 promotes immunosuppressive TME by inhibiting CD8+ T cell infiltration and function.", 0.84, "literature_analyst"),
  node("n-cd8", "PROTEIN", "CD8+ T cells", "Cytotoxic T lymphocytes. B7-H3 impairs their anti-tumor activity.", 0.90, "literature_analyst"),
  node("n-nk", "PROTEIN", "NK Cells", "Natural killer cells. B7-H3 inhibits NK cell-mediated cytotoxicity in vitro.", 0.72, "literature_analyst"),
  node("n-biom-b7h3", "BIOMARKER", "B7-H3 Expression (IHC)", "Immunohistochemistry-based B7-H3 expression. H-score >150 associated with response to anti-B7-H3 therapy.", 0.78, "clinical_analyst"),
  node("n-biom-pdl1", "BIOMARKER", "PD-L1 TPS", "PD-L1 tumor proportion score. TPS ≥50% predicts response to pembrolizumab monotherapy.", 0.92, "clinical_analyst"),

  // Compounds / variants
  node("n-capt1", "COMPOUND", "CAR-T (B7-H3)", "Chimeric antigen receptor T cells targeting B7-H3. Preclinical efficacy in NSCLC xenografts.", 0.68, "drug_hunter"),
  node("n-egfr-mut", "PROTEIN", "EGFR L858R", "EGFR exon 21 L858R point mutation. Sensitizing mutation for EGFR TKIs.", 0.91, "protein_engineer", "hyp-002"),
  node("n-kras-g12c", "PROTEIN", "KRAS G12C", "KRAS glycine-to-cysteine substitution at codon 12. Targetable by covalent inhibitors.", 0.89, "protein_engineer", "hyp-002"),

  // Publications
  node("n-pub1", "PUBLICATION", "Zhang et al. 2023", "B7-H3 promotes immune evasion in NSCLC via PI3K/AKT pathway activation. Nature Immunology.", 0.94, "scientific_critic"),
  node("n-pub2", "PUBLICATION", "Chen et al. 2024", "Ifinatamab deruxtecan shows durable responses in B7-H3+ NSCLC: Phase 2 results. J Clin Oncol.", 0.91, "scientific_critic"),

  // Experiments
  node("n-exp1", "EXPERIMENT", "B7-H3 + PD-L1 Dual Blockade", "Proposed: Combinatorial blockade of B7-H3 and PD-L1 to overcome resistance in checkpoint-refractory NSCLC.", 0.55, "experiment_designer"),
  node("n-exp2", "EXPERIMENT", "B7-H3 CAR-T + TME Modulator", "Proposed: B7-H3 CAR-T therapy with JAK inhibitor to reshape immunosuppressive TME.", 0.50, "experiment_designer"),
];

// ─── Knowledge Graph Edges ───────────────────────────────────────────────────

export const MOCK_EDGES: KGEdge[] = [
  // B7-H3 core relationships
  edge("e-01", "n-b7h3", "n-nsclc", "OVEREXPRESSED_IN", 0.93, "literature_analyst"),
  edge("e-02", "n-b7h3", "n-pi3k", "ACTIVATES", 0.82, "pathway_analyst"),
  edge("e-03", "n-b7h3", "n-nfkb", "ACTIVATES", 0.75, "pathway_analyst"),
  edge("e-04", "n-b7h3", "n-tme", "CAUSES", 0.86, "literature_analyst"),
  edge("e-05", "n-b7h3", "n-cd8", "INHIBITS", 0.88, "literature_analyst"),
  edge("e-06", "n-b7h3", "n-nk", "INHIBITS", 0.70, "literature_analyst"),
  edge("e-07", "n-b7h3", "n-pdl1", "CORRELATES_WITH", 0.64, "literature_analyst"),

  // Drug → target interactions
  edge("e-08", "n-enob", "n-b7h3", "TARGETS", 0.90, "drug_hunter"),
  edge("e-09", "n-mgd009", "n-b7h3", "TARGETS", 0.78, "drug_hunter"),
  edge("e-10", "n-ds7300", "n-b7h3", "TARGETS", 0.92, "drug_hunter"),
  edge("e-11", "n-pembro", "n-pd1", "TARGETS", 0.98, "drug_hunter"),
  edge("e-12", "n-osim", "n-egfr", "TARGETS", 0.96, "drug_hunter"),
  edge("e-13", "n-capt1", "n-b7h3", "TARGETS", 0.72, "drug_hunter"),

  // Drug → disease treatment
  edge("e-14", "n-ds7300", "n-nsclc", "TREATS", 0.85, "drug_hunter"),
  edge("e-15", "n-pembro", "n-nsclc", "TREATS", 0.94, "drug_hunter"),
  edge("e-16", "n-osim", "n-nsclc", "TREATS", 0.93, "drug_hunter"),

  // Clinical trials
  edge("e-17", "n-enob", "n-trial1", "ASSOCIATED_WITH", 0.82, "clinical_analyst"),
  edge("e-18", "n-ds7300", "n-trial2", "ASSOCIATED_WITH", 0.88, "clinical_analyst"),
  edge("e-19", "n-capt1", "n-trial3", "ASSOCIATED_WITH", 0.65, "clinical_analyst"),

  // Genetic context
  edge("e-20", "n-egfr", "n-nsclc", "ASSOCIATED_WITH", 0.91, "genomics_mapper"),
  edge("e-21", "n-kras", "n-nsclc", "ASSOCIATED_WITH", 0.88, "genomics_mapper"),
  edge("e-22", "n-tp53", "n-nsclc", "ASSOCIATED_WITH", 0.90, "genomics_mapper"),
  edge("e-23", "n-egfr", "n-mapk", "ACTIVATES", 0.89, "genomics_mapper"),
  edge("e-24", "n-egfr", "n-pi3k", "ACTIVATES", 0.86, "pathway_analyst"),

  // Pathway interactions
  edge("e-25", "n-jak-stat", "n-tme", "REGULATES", 0.76, "pathway_analyst"),
  edge("e-26", "n-nfkb", "n-tme", "ACTIVATES", 0.72, "pathway_analyst"),
  edge("e-27", "n-pi3k", "n-mapk", "INTERACTS_WITH", 0.68, "pathway_analyst"),

  // Biomarkers
  edge("e-28", "n-biom-b7h3", "n-b7h3", "BIOMARKER_FOR", 0.80, "clinical_analyst"),
  edge("e-29", "n-biom-pdl1", "n-pdl1", "BIOMARKER_FOR", 0.91, "clinical_analyst"),

  // Evidence / publications
  edge("e-30", "n-pub1", "n-b7h3", "EVIDENCE_FOR", 0.92, "scientific_critic"),
  edge("e-31", "n-pub2", "n-ds7300", "EVIDENCE_FOR", 0.88, "scientific_critic"),

  // Mutation variants
  edge("e-32", "n-egfr-mut", "n-egfr", "VARIANT_OF", 0.95, "protein_engineer", "hyp-002"),
  edge("e-33", "n-kras-g12c", "n-kras", "VARIANT_OF", 0.93, "protein_engineer", "hyp-002"),

  // Falsified edge — contradicted finding
  edge("e-34", "n-b7h3", "n-pd1", "BINDS_TO", 0.18, "scientific_critic", "hyp-001", true),

  // Experiments
  edge("e-35", "n-exp1", "n-b7h3", "TARGETS", 0.55, "experiment_designer"),
  edge("e-36", "n-exp1", "n-pdl1", "TARGETS", 0.55, "experiment_designer"),
  edge("e-37", "n-exp2", "n-capt1", "DERIVED_FROM", 0.50, "experiment_designer"),
  edge("e-38", "n-exp2", "n-jak-stat", "TARGETS", 0.50, "experiment_designer"),
];

// ─── Hypothesis Tree ─────────────────────────────────────────────────────────

export const MOCK_HYPOTHESES: HypothesisNode[] = [
  {
    id: "hyp-root",
    parent_id: null,
    hypothesis: "B7-H3 is a viable therapeutic target for NSCLC treatment",
    rationale: "Root research question — investigate B7-H3 as an immune checkpoint target in NSCLC",
    depth: 0,
    visit_count: 15,
    total_info_gain: 8.4,
    avg_info_gain: 0.56,
    ucb_score: 0.0,
    children: ["hyp-001", "hyp-002", "hyp-003"],
    status: "EXPLORED",
    supporting_edges: ["e-01", "e-04", "e-05", "e-08", "e-10", "e-14"],
    contradicting_edges: ["e-34"],
    agents_assigned: ["literature_analyst", "protein_engineer"],
    confidence: 0.82,
    created_at: MINUTES_AGO(30),
    updated_at: MINUTES_AGO(2),
  },
  {
    id: "hyp-001",
    parent_id: "hyp-root",
    hypothesis: "Anti-B7-H3 antibody-drug conjugates (ADCs) show superior efficacy over naked antibodies in NSCLC by combining immune checkpoint blockade with cytotoxic payload delivery",
    rationale: "DS-7300 Phase 2 data shows higher ORR than enoblituzumab. ADC mechanism bypasses need for direct immune activation.",
    depth: 1,
    visit_count: 8,
    total_info_gain: 5.6,
    avg_info_gain: 0.70,
    ucb_score: 1.12,
    children: ["hyp-001a", "hyp-001b"],
    status: "CONFIRMED",
    supporting_edges: ["e-10", "e-14", "e-18", "e-31"],
    contradicting_edges: [],
    agents_assigned: ["drug_hunter", "clinical_analyst", "scientific_critic"],
    confidence: 0.88,
    created_at: MINUTES_AGO(25),
    updated_at: MINUTES_AGO(3),
  },
  {
    id: "hyp-001a",
    parent_id: "hyp-001",
    hypothesis: "Topoisomerase I inhibitor payloads (DXd) are optimal for B7-H3 ADCs due to high bystander effect in heterogeneous tumors",
    rationale: "DS-7300 uses DXd payload (same as T-DXd). Membrane-permeable payload enables bystander killing of B7-H3-negative adjacent cells.",
    depth: 2,
    visit_count: 3,
    total_info_gain: 1.8,
    avg_info_gain: 0.60,
    ucb_score: 0.95,
    children: [],
    status: "EXPLORING",
    supporting_edges: ["e-31"],
    contradicting_edges: [],
    agents_assigned: ["drug_hunter", "scientific_critic"],
    confidence: 0.72,
    created_at: MINUTES_AGO(15),
    updated_at: MINUTES_AGO(2),
  },
  {
    id: "hyp-001b",
    parent_id: "hyp-001",
    hypothesis: "B7-H3 ADC efficacy is independent of PD-L1 expression status",
    rationale: "If ADC mechanism is primarily cytotoxic, PD-L1 status should not predict response. This would expand patient population.",
    depth: 2,
    visit_count: 2,
    total_info_gain: 0.8,
    avg_info_gain: 0.40,
    ucb_score: 0.88,
    children: [],
    status: "EXPLORING",
    supporting_edges: [],
    contradicting_edges: [],
    agents_assigned: ["clinical_analyst"],
    confidence: 0.55,
    created_at: MINUTES_AGO(12),
    updated_at: MINUTES_AGO(5),
  },
  {
    id: "hyp-002",
    parent_id: "hyp-root",
    hypothesis: "B7-H3 overexpression is driven by oncogenic signaling through EGFR and KRAS mutations, suggesting a biomarker-guided combination strategy",
    rationale: "EGFR/KRAS activate PI3K/AKT which upregulates B7-H3. Combining TKI + anti-B7-H3 may be synergistic.",
    depth: 1,
    visit_count: 5,
    total_info_gain: 2.5,
    avg_info_gain: 0.50,
    ucb_score: 0.98,
    children: [],
    status: "EXPLORED",
    supporting_edges: ["e-02", "e-23", "e-24", "e-32", "e-33"],
    contradicting_edges: [],
    agents_assigned: ["genomics_mapper", "pathway_analyst", "protein_engineer"],
    confidence: 0.68,
    created_at: MINUTES_AGO(22),
    updated_at: MINUTES_AGO(4),
  },
  {
    id: "hyp-003",
    parent_id: "hyp-root",
    hypothesis: "B7-H3 directly binds PD-1 to create a secondary immune checkpoint axis",
    rationale: "Some studies suggest B7-H3 may interact with PD-1 directly. If true, dual blockade could be synergistic.",
    depth: 1,
    visit_count: 4,
    total_info_gain: 0.8,
    avg_info_gain: 0.20,
    ucb_score: 0.45,
    children: [],
    status: "REFUTED",
    supporting_edges: [],
    contradicting_edges: ["e-34"],
    agents_assigned: ["scientific_critic", "protein_engineer"],
    confidence: 0.12,
    created_at: MINUTES_AGO(20),
    updated_at: MINUTES_AGO(6),
  },
];

// ─── Agent Info ──────────────────────────────────────────────────────────────

export const MOCK_AGENTS: AgentInfo[] = [
  { agent_id: "ag-lit", agent_type: "literature_analyst", status: "COMPLETED", hypothesis_branch: "hyp-001", task_count: 3, nodes_added: 8, edges_added: 10 },
  { agent_id: "ag-prot", agent_type: "protein_engineer", status: "COMPLETED", hypothesis_branch: "hyp-002", task_count: 2, nodes_added: 4, edges_added: 5 },
  { agent_id: "ag-gen", agent_type: "genomics_mapper", status: "COMPLETED", hypothesis_branch: "hyp-002", task_count: 2, nodes_added: 3, edges_added: 4 },
  { agent_id: "ag-path", agent_type: "pathway_analyst", status: "COMPLETED", hypothesis_branch: "hyp-001", task_count: 2, nodes_added: 4, edges_added: 6 },
  { agent_id: "ag-drug", agent_type: "drug_hunter", status: "COMPLETED", hypothesis_branch: "hyp-001", task_count: 3, nodes_added: 6, edges_added: 9 },
  { agent_id: "ag-clin", agent_type: "clinical_analyst", status: "COMPLETED", hypothesis_branch: "hyp-001", task_count: 2, nodes_added: 5, edges_added: 6 },
  { agent_id: "ag-crit", agent_type: "scientific_critic", status: "COMPLETED", hypothesis_branch: null, task_count: 4, nodes_added: 2, edges_added: 3 },
  { agent_id: "ag-exp", agent_type: "experiment_designer", status: "COMPLETED", hypothesis_branch: null, task_count: 1, nodes_added: 2, edges_added: 4 },
];

// ─── Research Events (live feed) ─────────────────────────────────────────────

export const MOCK_EVENTS: ResearchEvent[] = [
  { session_id: SESSION_ID, event_type: "session_created", data: { query: "Investigate B7-H3 as a therapeutic target in NSCLC" }, timestamp: MINUTES_AGO(30) },
  { session_id: SESSION_ID, event_type: "initialization_started", data: {}, timestamp: MINUTES_AGO(29) },
  { session_id: SESSION_ID, event_type: "hypothesis_generated", data: { hypothesis: "B7-H3 is a viable therapeutic target for NSCLC treatment" }, timestamp: MINUTES_AGO(28) },
  { session_id: SESSION_ID, event_type: "mcts_iteration_start", data: { iteration: 1, total_visits: 0, node_count: 1 }, timestamp: MINUTES_AGO(27) },
  { session_id: SESSION_ID, event_type: "hypothesis_selected", data: { hypothesis: "Anti-B7-H3 ADCs show superior efficacy", node_id: "hyp-001" }, timestamp: MINUTES_AGO(26) },
  { session_id: SESSION_ID, event_type: "agents_composed", data: { agents: ["literature_analyst", "drug_hunter", "clinical_analyst", "scientific_critic"] }, timestamp: MINUTES_AGO(25) },
  { session_id: SESSION_ID, event_type: "agent_started", data: { agent_type: "literature_analyst", agent_id: "ag-lit" }, timestamp: MINUTES_AGO(24) },
  { session_id: SESSION_ID, event_type: "agent_started", data: { agent_type: "drug_hunter", agent_id: "ag-drug" }, timestamp: MINUTES_AGO(24) },
  { session_id: SESSION_ID, event_type: "node_created", data: { node_name: "B7-H3 (CD276)", node_type: "PROTEIN" }, timestamp: MINUTES_AGO(23) },
  { session_id: SESSION_ID, event_type: "node_created", data: { node_name: "Non-Small Cell Lung Cancer", node_type: "DISEASE" }, timestamp: MINUTES_AGO(23) },
  { session_id: SESSION_ID, event_type: "edge_created", data: { source_name: "B7-H3", relation: "OVEREXPRESSED_IN", target_name: "NSCLC", confidence: 0.93 }, timestamp: MINUTES_AGO(22) },
  { session_id: SESSION_ID, event_type: "node_created", data: { node_name: "DS-7300 (Ifinatamab)", node_type: "DRUG" }, timestamp: MINUTES_AGO(21) },
  { session_id: SESSION_ID, event_type: "edge_created", data: { source_name: "DS-7300", relation: "TARGETS", target_name: "B7-H3", confidence: 0.92 }, timestamp: MINUTES_AGO(20) },
  { session_id: SESSION_ID, event_type: "node_created", data: { node_name: "Pembrolizumab", node_type: "DRUG" }, timestamp: MINUTES_AGO(19) },
  { session_id: SESSION_ID, event_type: "agent_started", data: { agent_type: "scientific_critic", agent_id: "ag-crit" }, timestamp: MINUTES_AGO(18) },
  { session_id: SESSION_ID, event_type: "confidence_updated", data: { edge_id: "e-01", confidence: 0.93, source_name: "B7-H3", target_name: "NSCLC" }, timestamp: MINUTES_AGO(17) },
  { session_id: SESSION_ID, event_type: "edge_falsified", data: { edge_id: "e-34", source_name: "B7-H3", relation: "BINDS_TO", target_name: "PD-1", original_confidence: 0.45, revised_confidence: 0.18 }, timestamp: MINUTES_AGO(15) },
  { session_id: SESSION_ID, event_type: "hypothesis_generated", data: { hypothesis: "B7-H3 ADC efficacy independent of PD-L1 status", parent: "hyp-001" }, timestamp: MINUTES_AGO(14) },
  { session_id: SESSION_ID, event_type: "mcts_iteration_start", data: { iteration: 2, total_visits: 5, node_count: 4 }, timestamp: MINUTES_AGO(13) },
  { session_id: SESSION_ID, event_type: "hypothesis_selected", data: { hypothesis: "B7-H3 overexpression driven by EGFR/KRAS mutations", node_id: "hyp-002" }, timestamp: MINUTES_AGO(12) },
  { session_id: SESSION_ID, event_type: "agents_composed", data: { agents: ["genomics_mapper", "pathway_analyst", "protein_engineer", "scientific_critic"] }, timestamp: MINUTES_AGO(11) },
  { session_id: SESSION_ID, event_type: "agent_started", data: { agent_type: "genomics_mapper", agent_id: "ag-gen" }, timestamp: MINUTES_AGO(10) },
  { session_id: SESSION_ID, event_type: "agent_started", data: { agent_type: "pathway_analyst", agent_id: "ag-path" }, timestamp: MINUTES_AGO(10) },
  { session_id: SESSION_ID, event_type: "node_created", data: { node_name: "PI3K/AKT/mTOR Signaling", node_type: "PATHWAY" }, timestamp: MINUTES_AGO(9) },
  { session_id: SESSION_ID, event_type: "node_created", data: { node_name: "EGFR L858R", node_type: "PROTEIN" }, timestamp: MINUTES_AGO(8) },
  { session_id: SESSION_ID, event_type: "edge_created", data: { source_name: "B7-H3", relation: "ACTIVATES", target_name: "PI3K/AKT/mTOR", confidence: 0.82 }, timestamp: MINUTES_AGO(7) },
  { session_id: SESSION_ID, event_type: "uncertainty_aggregated", data: { composite: 0.35, agents: 8, is_critical: false }, timestamp: MINUTES_AGO(6) },
  { session_id: SESSION_ID, event_type: "agent_started", data: { agent_type: "experiment_designer", agent_id: "ag-exp" }, timestamp: MINUTES_AGO(5) },
  { session_id: SESSION_ID, event_type: "node_created", data: { node_name: "B7-H3 + PD-L1 Dual Blockade", node_type: "EXPERIMENT" }, timestamp: MINUTES_AGO(4) },
  { session_id: SESSION_ID, event_type: "mcts_backpropagation", data: { node_id: "hyp-001", info_gain: 0.70 }, timestamp: MINUTES_AGO(3) },
  { session_id: SESSION_ID, event_type: "research_completed", data: { best_hypothesis: "hyp-001", total_nodes: 28, total_edges: 38, total_iterations: 15 }, timestamp: MINUTES_AGO(1) },
];

// ─── Research Session ────────────────────────────────────────────────────────

export const MOCK_SESSION: ResearchSession = {
  id: SESSION_ID,
  query: "Investigate B7-H3 as a therapeutic target in NSCLC — evaluate ADCs, combination strategies, and biomarker-guided patient selection",
  status: "COMPLETED",
  config: {
    max_hypothesis_depth: 2,
    max_mcts_iterations: 15,
    max_agents: 8,
    max_agents_per_swarm: 4,
    confidence_threshold: 0.7,
    hitl_uncertainty_threshold: 0.6,
    hitl_timeout_seconds: 600,
    max_llm_calls_per_agent: 20,
    agent_types: null,
    enable_falsification: true,
    enable_hitl: true,
    slack_channel_id: null,
  },
  swarm_composition: [
    "literature_analyst", "protein_engineer", "genomics_mapper",
    "pathway_analyst", "drug_hunter", "clinical_analyst",
    "scientific_critic", "experiment_designer",
  ],
  hypothesis_tree_id: "tree-001",
  knowledge_graph_id: "kg-001",
  current_iteration: 15,
  total_nodes: MOCK_NODES.length,
  total_edges: MOCK_EDGES.length,
  total_hypotheses: MOCK_HYPOTHESES.length,
  total_tokens_used: 284_391,
  created_at: MINUTES_AGO(30),
  updated_at: MINUTES_AGO(1),
  completed_at: MINUTES_AGO(1),
  result: {
    research_id: SESSION_ID,
    best_hypothesis: MOCK_HYPOTHESES[1],
    hypothesis_ranking: [MOCK_HYPOTHESES[1], MOCK_HYPOTHESES[4], MOCK_HYPOTHESES[2]],
    key_findings: MOCK_EDGES.filter((e) => e.confidence.overall > 0.8),
    contradictions: [[MOCK_EDGES[33], MOCK_EDGES[6]]],
    uncertainties: [
      {
        input_ambiguity: 0.15,
        data_quality: 0.22,
        reasoning_divergence: 0.18,
        model_disagreement: 0.12,
        conflict_uncertainty: 0.28,
        novelty_uncertainty: 0.35,
        composite: 0.22,
        is_critical: false,
      },
    ],
    recommended_experiments: [
      "Phase 1b combination trial: DS-7300 + pembrolizumab in B7-H3+/PD-L1+ NSCLC patients",
      "Correlative biomarker study: B7-H3 IHC H-score vs DS-7300 ORR in retrospective cohort",
      "In vitro assay: B7-H3 CAR-T + ruxolitinib (JAK1/2 inhibitor) combination in NSCLC organoids",
    ],
    report_markdown: `# B7-H3 as a Therapeutic Target in NSCLC\n\n## Executive Summary\n\nThis investigation evaluated B7-H3 (CD276) as a therapeutic target in non-small cell lung cancer using an MCTS-driven multi-agent research approach. **B7-H3 antibody-drug conjugates (ADCs), particularly DS-7300 (ifinatamab deruxtecan), emerged as the most promising therapeutic modality**, showing superior efficacy signals over naked antibodies and bispecific approaches.\n\n## Key Findings\n\n### 1. B7-H3 is broadly overexpressed in NSCLC (Confidence: 93%)\nB7-H3 expression is elevated in the majority of NSCLC tumors, making it an attractive therapeutic target. Expression correlates with poor prognosis and immune evasion.\n\n### 2. ADCs outperform naked anti-B7-H3 antibodies (Confidence: 88%)\nDS-7300 (ifinatamab deruxtecan) with a topoisomerase I inhibitor payload shows higher objective response rates than enoblituzumab in early clinical trials, likely due to bystander killing effect.\n\n### 3. B7-H3 signals through PI3K/AKT/mTOR (Confidence: 82%)\nB7-H3 activates the PI3K/AKT/mTOR survival pathway, promoting tumor cell survival and resistance to apoptosis. This may explain resistance to standard chemotherapy.\n\n### 4. B7-H3 does NOT directly bind PD-1 (REFUTED, Confidence: 12%)\nThe hypothesis that B7-H3 creates a secondary checkpoint axis by directly binding PD-1 was **falsified** through systematic counter-evidence search. The interaction is indirect, mediated through shared downstream signaling.\n\n### 5. EGFR/KRAS mutations may drive B7-H3 upregulation (Confidence: 68%)\nOncogenic EGFR and KRAS signaling through PI3K/AKT may upregulate B7-H3 expression, suggesting mutation-guided patient stratification.\n\n## Recommended Experiments\n\n1. **Phase 1b: DS-7300 + pembrolizumab** in B7-H3+/PD-L1+ NSCLC\n2. **Biomarker study**: B7-H3 IHC H-score correlation with ADC response\n3. **Organoid assay**: B7-H3 CAR-T + JAK inhibitor combination\n\n## Methodology\n\n- **MCTS iterations**: 15\n- **Agents deployed**: 8 (literature, protein, genomics, pathway, drug, clinical, critic, experiment)\n- **Knowledge graph**: 28 nodes, 38 edges\n- **Hypotheses explored**: 6 (1 confirmed, 1 refuted, 4 explored)\n- **LLM calls**: 284,391 tokens consumed\n- **Self-falsification**: 1 hypothesis refuted, 1 edge falsified\n`,
    graph_snapshot: {},
    kg_stats: { node_count: 28, edge_count: 38, avg_confidence: 0.79 },
    total_duration_ms: 1_740_000,
    total_llm_calls: 127,
    total_tokens: 284_391,
    created_at: MINUTES_AGO(1),
  },
};

// A running session for the active demo
export const MOCK_RUNNING_SESSION: ResearchSession = {
  ...MOCK_SESSION,
  id: "demo-session-002",
  query: "Evaluate dual B7-H3/PD-L1 blockade strategies for checkpoint-refractory NSCLC",
  status: "RUNNING",
  current_iteration: 7,
  total_nodes: 14,
  total_edges: 18,
  total_hypotheses: 3,
  total_tokens_used: 142_000,
  completed_at: null,
  result: null,
};

// ─── Tool Registry ───────────────────────────────────────────────────────────

export const MOCK_TOOLS: ToolRegistryEntry[] = [
  { name: "pubmed", description: "Search PubMed for biomedical literature", source_type: "NATIVE", category: "Literature", mcp_server: null, enabled: true, registered_at: MINUTES_AGO(60) },
  { name: "semantic_scholar", description: "Semantic Scholar academic paper search and citation analysis", source_type: "NATIVE", category: "Literature", mcp_server: null, enabled: true, registered_at: MINUTES_AGO(60) },
  { name: "uniprot", description: "UniProt protein database queries", source_type: "NATIVE", category: "Protein", mcp_server: null, enabled: true, registered_at: MINUTES_AGO(60) },
  { name: "esm", description: "ESM-2/ESMFold protein language model and structure prediction", source_type: "NATIVE", category: "Protein", mcp_server: null, enabled: true, registered_at: MINUTES_AGO(60) },
  { name: "kegg", description: "KEGG pathway database", source_type: "NATIVE", category: "Pathway", mcp_server: null, enabled: true, registered_at: MINUTES_AGO(60) },
  { name: "reactome", description: "Reactome pathway analysis and visualization", source_type: "NATIVE", category: "Pathway", mcp_server: null, enabled: true, registered_at: MINUTES_AGO(60) },
  { name: "mygene", description: "MyGene.info gene annotation service", source_type: "NATIVE", category: "Genomics", mcp_server: null, enabled: true, registered_at: MINUTES_AGO(60) },
  { name: "chembl", description: "ChEMBL compound and bioactivity database", source_type: "NATIVE", category: "Drug Discovery", mcp_server: null, enabled: true, registered_at: MINUTES_AGO(60) },
  { name: "clinicaltrials", description: "ClinicalTrials.gov clinical trial registry", source_type: "NATIVE", category: "Clinical", mcp_server: null, enabled: true, registered_at: MINUTES_AGO(60) },
  { name: "slack", description: "Slack integration for human-in-the-loop requests", source_type: "NATIVE", category: "Communication", mcp_server: null, enabled: true, registered_at: MINUTES_AGO(60) },
  { name: "mcp-alphafold", description: "AlphaFold2 structure prediction via MCP server", source_type: "MCP", category: "Protein", mcp_server: "mcp-alphafold", enabled: true, registered_at: MINUTES_AGO(45) },
  { name: "mcp-opentargets", description: "Open Targets genetics & drug target evidence", source_type: "MCP", category: "Drug Discovery", mcp_server: "mcp-opentargets", enabled: true, registered_at: MINUTES_AGO(45) },
  { name: "mcp-string-db", description: "STRING protein-protein interaction network", source_type: "MCP", category: "Protein", mcp_server: "mcp-string-db", enabled: false, registered_at: MINUTES_AGO(45) },
  { name: "container-blast", description: "NCBI BLAST sequence alignment (containerized)", source_type: "CONTAINER", category: "Genomics", mcp_server: null, enabled: true, registered_at: MINUTES_AGO(30) },
  { name: "container-hmmer", description: "HMMER protein homology search (containerized)", source_type: "CONTAINER", category: "Protein", mcp_server: null, enabled: false, registered_at: MINUTES_AGO(30) },
];

// ─── Benchmark Runs ──────────────────────────────────────────────────────────

export const MOCK_BENCHMARKS: BenchmarkRun[] = [
  {
    id: "bench-001",
    benchmark_name: "LAB-Bench",
    version: "v1.0",
    metrics: [
      { name: "Accuracy", value: 0.847, unit: "%", metadata: {} },
      { name: "Recall", value: 0.812, unit: "%", metadata: {} },
      { name: "F1 Score", value: 0.829, unit: "", metadata: {} },
      { name: "Avg Confidence", value: 0.79, unit: "", metadata: {} },
      { name: "Falsification Rate", value: 0.15, unit: "", metadata: {} },
      { name: "Avg Duration", value: 29.4, unit: "min", metadata: {} },
    ],
    baseline_comparison: {
      "BioMNI": 0.72,
      "YOHAS (ours)": 0.847,
      "GPT-4 (single agent)": 0.68,
      "Claude (single agent)": 0.71,
      "PaperQA2": 0.61,
    },
    total_tasks: 150,
    correct_tasks: 127,
    accuracy: 0.847,
    started_at: MINUTES_AGO(180),
    completed_at: MINUTES_AGO(120),
  },
  {
    id: "bench-002",
    benchmark_name: "BioASQ",
    version: "11b",
    metrics: [
      { name: "Accuracy", value: 0.891, unit: "%", metadata: {} },
      { name: "MAP", value: 0.823, unit: "", metadata: {} },
      { name: "MRR", value: 0.876, unit: "", metadata: {} },
    ],
    baseline_comparison: {
      "BioMNI": 0.81,
      "YOHAS (ours)": 0.891,
      "GPT-4 (single agent)": 0.79,
      "PubMedBERT": 0.74,
    },
    total_tasks: 500,
    correct_tasks: 446,
    accuracy: 0.891,
    started_at: MINUTES_AGO(240),
    completed_at: MINUTES_AGO(200),
  },
];
