// TypeScript types mirroring backend/core/models.py

export type NodeType =
  | "PROTEIN" | "GENE" | "DISEASE" | "PATHWAY" | "DRUG"
  | "CELL_TYPE" | "TISSUE" | "CLINICAL_TRIAL" | "MECHANISM"
  | "MODALITY" | "SIDE_EFFECT" | "BIOMARKER" | "ORGANISM"
  | "COMPOUND" | "EXPERIMENT" | "PUBLICATION" | "STRUCTURE";

export type EdgeRelationType =
  | "OVEREXPRESSED_IN" | "UNDEREXPRESSED_IN" | "EXPRESSED_IN"
  | "INHIBITS" | "ACTIVATES" | "BINDS_TO" | "PHOSPHORYLATES"
  | "INTERACTS_WITH" | "UPREGULATES" | "DOWNREGULATES"
  | "MEMBER_OF" | "REGULATES" | "CATALYZES" | "PARTICIPATES_IN"
  | "UPSTREAM_OF" | "DOWNSTREAM_OF"
  | "TREATS" | "TARGETS" | "SIDE_EFFECT_OF" | "CAUSES" | "RISK_OF"
  | "BIOMARKER_FOR" | "CONTRAINDICATED_WITH" | "SYNERGIZES_WITH"
  | "ANTAGONIZES" | "METABOLIZED_BY"
  | "ASSOCIATED_WITH" | "CORRELATES_WITH"
  | "ENCODES" | "TRANSCRIBES" | "TRANSLATES_TO"
  | "MUTANT_OF" | "VARIANT_OF" | "HOMOLOGOUS_TO" | "DOMAIN_OF"
  | "EVIDENCE_FOR" | "EVIDENCE_AGAINST" | "SUPPORTED_BY"
  | "CONTRADICTS" | "DERIVED_FROM";

export type AgentType =
  | "literature_analyst" | "protein_engineer" | "genomics_mapper"
  | "pathway_analyst" | "drug_hunter" | "clinical_analyst"
  | "scientific_critic" | "experiment_designer";

export type EvidenceSourceType =
  | "PUBMED" | "SEMANTIC_SCHOLAR" | "UNIPROT" | "KEGG" | "REACTOME"
  | "CHEMBL" | "CLINICALTRIALS" | "MYGENE" | "ESM" | "DATABASE"
  | "TOOL_OUTPUT" | "YAMI_PREDICTION" | "LLM_INFERENCE"
  | "AGENT_REASONING" | "EXPERT_INPUT" | "HUMAN_INPUT" | "COMPUTATIONAL";

export type HypothesisStatus =
  | "UNEXPLORED" | "EXPLORING" | "EXPLORED"
  | "PRUNED" | "CONFIRMED" | "REFUTED";

export type TaskStatus = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED" | "WAITING_HITL";
export type SessionStatus = "INITIALIZING" | "RUNNING" | "WAITING_HITL" | "COMPLETED" | "FAILED" | "CANCELLED";
export type ToolSourceType = "NATIVE" | "MCP" | "CONTAINER";

export interface EvidenceSource {
  source_type: EvidenceSourceType;
  source_id: string | null;
  doi: string | null;
  title: string | null;
  claim: string;
  url: string | null;
  retrieval_method: string;
  snippet: string;
  quality_score: number;
  confidence: number;
  publication_year: number | null;
  citation_count: number | null;
  journal_impact_factor: number | null;
  is_peer_reviewed: boolean;
  agent_id: string;
  timestamp: string;
}

export interface VisualizationHints {
  position_x: number | null;
  position_y: number | null;
  cluster_id: string | null;
  visual_weight: number;
  color_override: string | null;
  animation_state: string | null;
  pinned: boolean;
  layer: number;
}

export interface KGNode {
  id: string;
  type: NodeType;
  name: string;
  aliases: string[];
  description: string;
  properties: Record<string, unknown>;
  external_ids: Record<string, string>;
  confidence: number;
  sources: EvidenceSource[];
  created_by: string;
  hypothesis_branch: string | null;
  created_at: string;
  updated_at: string;
  viz: VisualizationHints;
}

export interface EdgeConfidence {
  overall: number;
  evidence_quality: number;
  evidence_count: number;
  replication_count: number;
  falsification_attempts: number;
  falsification_failures: number;
  computational_score: number | null;
  llm_assessment: number | null;
  last_evaluated: string;
}

export interface KGEdge {
  id: string;
  source_id: string;
  target_id: string;
  relation: EdgeRelationType;
  confidence: EdgeConfidence;
  properties: Record<string, unknown>;
  evidence: EvidenceSource[];
  created_by: string;
  hypothesis_branch: string | null;
  is_contradiction: boolean;
  contradicted_by: string[];
  falsified: boolean;
  falsification_evidence: EvidenceSource[] | null;
  created_at: string;
  updated_at: string;
  viz: VisualizationHints;
}

export interface HypothesisNode {
  id: string;
  parent_id: string | null;
  hypothesis: string;
  rationale: string;
  depth: number;
  visit_count: number;
  total_info_gain: number;
  avg_info_gain: number;
  ucb_score: number;
  children: string[];
  status: HypothesisStatus;
  supporting_edges: string[];
  contradicting_edges: string[];
  agents_assigned: string[];
  confidence: number;
  created_at: string;
  updated_at: string;
}

export interface UncertaintyVector {
  input_ambiguity: number;
  data_quality: number;
  reasoning_divergence: number;
  model_disagreement: number;
  conflict_uncertainty: number;
  novelty_uncertainty: number;
  composite: number;
  is_critical: boolean;
}

export interface FalsificationResult {
  edge_id: string;
  agent_id: string;
  hypothesis_branch: string;
  original_confidence: number;
  revised_confidence: number;
  falsified: boolean;
  search_query: string;
  method: string;
  counter_evidence_found: boolean;
  counter_evidence: EvidenceSource[];
  reasoning: string;
  confidence_delta: number;
  timestamp: string;
}

export interface ResearchConfig {
  max_hypothesis_depth?: number;
  max_mcts_iterations?: number;
  max_agents?: number;
  max_agents_per_swarm?: number;
  confidence_threshold?: number;
  hitl_uncertainty_threshold?: number;
  hitl_timeout_seconds?: number;
  max_llm_calls_per_agent?: number;
  agent_types?: AgentType[] | null;
  enable_falsification?: boolean;
  enable_hitl?: boolean;
  slack_channel_id?: string | null;
}

export interface ResearchResult {
  research_id: string;
  best_hypothesis: HypothesisNode;
  hypothesis_ranking: HypothesisNode[];
  key_findings: KGEdge[];
  contradictions: [KGEdge, KGEdge][];
  uncertainties: UncertaintyVector[];
  recommended_experiments: string[];
  report_markdown: string;
  graph_snapshot: Record<string, unknown>;
  kg_stats: Record<string, number>;
  total_duration_ms: number;
  total_llm_calls: number;
  total_tokens: number;
  created_at: string;
}

export interface ResearchSession {
  id: string;
  query: string;
  status: SessionStatus;
  config: ResearchConfig;
  swarm_composition: string[];
  hypothesis_tree_id: string;
  knowledge_graph_id: string;
  current_iteration: number;
  total_nodes: number;
  total_edges: number;
  total_hypotheses: number;
  total_tokens_used: number;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  result: ResearchResult | null;
}

export interface ResearchEvent {
  session_id: string;
  event_type: string;
  data: Record<string, unknown>;
  timestamp: string;
}

export interface ToolRegistryEntry {
  name: string;
  description: string;
  source_type: ToolSourceType;
  category: string;
  mcp_server: string | null;
  enabled: boolean;
  registered_at: string;
}

export interface BenchmarkMetric {
  name: string;
  value: number;
  unit: string;
  metadata: Record<string, unknown>;
}

export interface BenchmarkRun {
  id: string;
  benchmark_name: string;
  version: string;
  metrics: BenchmarkMetric[];
  baseline_comparison: Record<string, number>;
  total_tasks: number;
  correct_tasks: number;
  accuracy: number;
  started_at: string;
  completed_at: string | null;
}

export interface AgentInfo {
  agent_id: string;
  agent_type: AgentType;
  status: TaskStatus;
  hypothesis_branch: string | null;
  task_count: number;
  nodes_added: number;
  edges_added: number;
}

// Node type to color mapping (matches tailwind.config.ts)
export const NODE_COLORS: Record<NodeType, string> = {
  PROTEIN: "#4A90D9",
  GENE: "#6B5CE7",
  DISEASE: "#E74C3C",
  PATHWAY: "#2ECC71",
  DRUG: "#F39C12",
  CELL_TYPE: "#8E44AD",
  TISSUE: "#E91E63",
  CLINICAL_TRIAL: "#1ABC9C",
  MECHANISM: "#95A5A6",
  MODALITY: "#607D8B",
  SIDE_EFFECT: "#FF5722",
  BIOMARKER: "#009688",
  ORGANISM: "#795548",
  COMPOUND: "#FF9800",
  EXPERIMENT: "#E67E22",
  PUBLICATION: "#3F51B5",
  STRUCTURE: "#00BCD4",
};

export const AGENT_COLORS: Record<AgentType, string> = {
  literature_analyst: "#4A90D9",
  protein_engineer: "#6B5CE7",
  genomics_mapper: "#E74C3C",
  pathway_analyst: "#2ECC71",
  drug_hunter: "#F39C12",
  clinical_analyst: "#1ABC9C",
  scientific_critic: "#E67E22",
  experiment_designer: "#8E44AD",
};

export const AGENT_LABELS: Record<AgentType, string> = {
  literature_analyst: "Literature Analyst",
  protein_engineer: "Protein Engineer",
  genomics_mapper: "Genomics Mapper",
  pathway_analyst: "Pathway Analyst",
  drug_hunter: "Drug Hunter",
  clinical_analyst: "Clinical Analyst",
  scientific_critic: "Scientific Critic",
  experiment_designer: "Experiment Designer",
};

export const STATUS_COLORS: Record<SessionStatus, string> = {
  INITIALIZING: "#F39C12",
  RUNNING: "#3498DB",
  WAITING_HITL: "#E67E22",
  COMPLETED: "#2ECC71",
  FAILED: "#E74C3C",
  CANCELLED: "#95A5A6",
};
