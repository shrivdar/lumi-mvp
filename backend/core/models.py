"""Pydantic data models — single source of truth for all YOHAS data structures.

Every module (agents, orchestrator, API, world_model) imports from here.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(UTC)


def _uuid() -> str:
    return str(uuid.uuid4())


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════


class NodeType(enum.StrEnum):
    PROTEIN = "PROTEIN"
    GENE = "GENE"
    DISEASE = "DISEASE"
    PATHWAY = "PATHWAY"
    DRUG = "DRUG"
    CELL_TYPE = "CELL_TYPE"
    TISSUE = "TISSUE"
    CLINICAL_TRIAL = "CLINICAL_TRIAL"
    MECHANISM = "MECHANISM"
    MODALITY = "MODALITY"
    SIDE_EFFECT = "SIDE_EFFECT"
    BIOMARKER = "BIOMARKER"
    ORGANISM = "ORGANISM"
    COMPOUND = "COMPOUND"
    EXPERIMENT = "EXPERIMENT"
    PUBLICATION = "PUBLICATION"
    STRUCTURE = "STRUCTURE"


class EdgeRelationType(enum.StrEnum):
    # Protein / Gene relations
    OVEREXPRESSED_IN = "OVEREXPRESSED_IN"
    UNDEREXPRESSED_IN = "UNDEREXPRESSED_IN"
    EXPRESSED_IN = "EXPRESSED_IN"
    INHIBITS = "INHIBITS"
    ACTIVATES = "ACTIVATES"
    BINDS_TO = "BINDS_TO"
    PHOSPHORYLATES = "PHOSPHORYLATES"
    INTERACTS_WITH = "INTERACTS_WITH"
    UPREGULATES = "UPREGULATES"
    DOWNREGULATES = "DOWNREGULATES"

    # Pathway relations
    MEMBER_OF = "MEMBER_OF"
    REGULATES = "REGULATES"
    CATALYZES = "CATALYZES"
    PARTICIPATES_IN = "PARTICIPATES_IN"
    UPSTREAM_OF = "UPSTREAM_OF"
    DOWNSTREAM_OF = "DOWNSTREAM_OF"

    # Disease / Drug relations
    TREATS = "TREATS"
    TARGETS = "TARGETS"
    SIDE_EFFECT_OF = "SIDE_EFFECT_OF"
    CAUSES = "CAUSES"
    RISK_OF = "RISK_OF"
    BIOMARKER_FOR = "BIOMARKER_FOR"
    CONTRAINDICATED_WITH = "CONTRAINDICATED_WITH"
    SYNERGIZES_WITH = "SYNERGIZES_WITH"
    ANTAGONIZES = "ANTAGONIZES"
    METABOLIZED_BY = "METABOLIZED_BY"

    # General relations
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    CORRELATES_WITH = "CORRELATES_WITH"

    # Genomic relations
    ENCODES = "ENCODES"
    TRANSCRIBES = "TRANSCRIBES"
    TRANSLATES_TO = "TRANSLATES_TO"
    MUTANT_OF = "MUTANT_OF"
    VARIANT_OF = "VARIANT_OF"
    HOMOLOGOUS_TO = "HOMOLOGOUS_TO"
    DOMAIN_OF = "DOMAIN_OF"

    # Evidence relations
    EVIDENCE_FOR = "EVIDENCE_FOR"
    EVIDENCE_AGAINST = "EVIDENCE_AGAINST"
    SUPPORTED_BY = "SUPPORTED_BY"
    CONTRADICTS = "CONTRADICTS"
    DERIVED_FROM = "DERIVED_FROM"


class AgentType(enum.StrEnum):
    """Agent specializations."""

    LITERATURE_ANALYST = "literature_analyst"
    PROTEIN_ENGINEER = "protein_engineer"
    GENOMICS_MAPPER = "genomics_mapper"
    PATHWAY_ANALYST = "pathway_analyst"
    DRUG_HUNTER = "drug_hunter"
    CLINICAL_ANALYST = "clinical_analyst"
    SCIENTIFIC_CRITIC = "scientific_critic"
    EXPERIMENT_DESIGNER = "experiment_designer"
    TOOL_CREATOR = "tool_creator"


class EvidenceSourceType(enum.StrEnum):
    PUBMED = "PUBMED"
    SEMANTIC_SCHOLAR = "SEMANTIC_SCHOLAR"
    UNIPROT = "UNIPROT"
    KEGG = "KEGG"
    REACTOME = "REACTOME"
    CHEMBL = "CHEMBL"
    CLINICALTRIALS = "CLINICALTRIALS"
    MYGENE = "MYGENE"
    ESM = "ESM"
    DATABASE = "DATABASE"
    TOOL_OUTPUT = "TOOL_OUTPUT"
    YAMI_PREDICTION = "YAMI_PREDICTION"
    LLM_INFERENCE = "LLM_INFERENCE"
    AGENT_REASONING = "AGENT_REASONING"
    EXPERT_INPUT = "EXPERT_INPUT"
    HUMAN_INPUT = "HUMAN_INPUT"
    COMPUTATIONAL = "COMPUTATIONAL"


class HypothesisStatus(enum.StrEnum):
    UNEXPLORED = "UNEXPLORED"
    EXPLORING = "EXPLORING"
    EXPLORED = "EXPLORED"
    PRUNED = "PRUNED"
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"


class TaskStatus(enum.StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    WAITING_HITL = "WAITING_HITL"


class SessionStatus(enum.StrEnum):
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    WAITING_HITL = "WAITING_HITL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class BiosecurityCategory(enum.StrEnum):
    """Categories of dual-use/biosecurity concern."""

    PATHOGEN_ENHANCEMENT = "pathogen_enhancement"
    TOXIN_SYNTHESIS = "toxin_synthesis"
    WEAPONS_POTENTIAL = "weapons_potential"
    GAIN_OF_FUNCTION = "gain_of_function"
    DUAL_USE_CONCERN = "dual_use_concern"


class ScreeningTier(enum.StrEnum):
    """Biosecurity screening decision tiers."""

    CLEAR = "CLEAR"        # Proceed normally
    WARNING = "WARNING"    # Add disclaimer to output
    BLOCKED = "BLOCKED"    # Refuse to present results


class ToolSourceType(enum.StrEnum):
    """How a tool is provided to the system."""

    NATIVE = "NATIVE"
    MCP = "MCP"
    CONTAINER = "CONTAINER"
    DYNAMIC = "DYNAMIC"


class MCPTransportType(enum.StrEnum):
    """MCP server transport types."""

    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"


# ═══════════════════════════════════════════════════════════════════════════════
# Evidence & Provenance
# ═══════════════════════════════════════════════════════════════════════════════


class EvidenceSource(BaseModel):
    """A single piece of evidence supporting a KG edge or claim.

    Granular quality scoring for benchmark dominance on LAB-Bench.
    """

    source_type: EvidenceSourceType
    source_id: str | None = None  # e.g. PMID, UniProt accession
    doi: str | None = None
    title: str | None = None
    claim: str = ""
    url: str | None = None
    retrieval_method: str = ""  # e.g. "keyword_search", "citation_chase", "mcp_tool"
    snippet: str = ""  # relevant excerpt
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    publication_year: int | None = None
    citation_count: int | None = None
    journal_impact_factor: float | None = None
    is_peer_reviewed: bool = True
    agent_id: str = ""
    timestamp: datetime = Field(default_factory=_utcnow)


# ═══════════════════════════════════════════════════════════════════════════════
# Knowledge Graph Models
# ═══════════════════════════════════════════════════════════════════════════════


class VisualizationHints(BaseModel):
    """Frontend visualization metadata for KG nodes and edges."""

    position_x: float | None = None
    position_y: float | None = None
    cluster_id: str | None = None
    visual_weight: float = 1.0  # affects node size / edge thickness
    color_override: str | None = None  # hex color
    animation_state: str | None = None  # e.g. "pulsing", "highlighted", "dimmed"
    pinned: bool = False
    layer: int = 0


class KGNode(BaseModel):
    """A node in the knowledge graph."""

    id: str = Field(default_factory=_uuid)
    type: NodeType
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)
    external_ids: dict[str, str] = Field(default_factory=dict)  # e.g. {"uniprot": "P12345"}
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    sources: list[EvidenceSource] = Field(default_factory=list)
    created_by: str = ""  # agent_id — no anonymous mutations
    hypothesis_branch: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    viz: VisualizationHints = Field(default_factory=VisualizationHints)


class EdgeConfidence(BaseModel):
    """Granular confidence scoring for a KG edge.

    Goes beyond a single float — tracks evidence quality, replication,
    and falsification attempts for benchmark-quality provenance.
    """

    overall: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_count: int = 0
    replication_count: int = 0  # independent sources confirming
    falsification_attempts: int = 0
    falsification_failures: int = 0  # failed refutations strengthen confidence
    computational_score: float | None = None  # from ESM/structural analysis
    llm_assessment: float | None = None
    last_evaluated: datetime = Field(default_factory=_utcnow)


class KGEdge(BaseModel):
    """An edge (relationship) in the knowledge graph."""

    id: str = Field(default_factory=_uuid)
    source_id: str
    target_id: str
    relation: EdgeRelationType
    confidence: EdgeConfidence = Field(default_factory=EdgeConfidence)
    properties: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceSource] = Field(default_factory=list)
    created_by: str = ""  # agent_id — no anonymous mutations
    hypothesis_branch: str | None = None
    is_contradiction: bool = False
    contradicted_by: list[str] = Field(default_factory=list)  # edge IDs
    falsified: bool = False
    falsification_evidence: list[EvidenceSource] | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    viz: VisualizationHints = Field(default_factory=VisualizationHints)


# ═══════════════════════════════════════════════════════════════════════════════
# Hypothesis Tree / MCTS
# ═══════════════════════════════════════════════════════════════════════════════


class HypothesisNode(BaseModel):
    """A node in the MCTS hypothesis tree."""

    id: str = Field(default_factory=_uuid)
    parent_id: str | None = None
    hypothesis: str
    rationale: str = ""
    depth: int = 0
    visit_count: int = 0
    total_info_gain: float = 0.0
    avg_info_gain: float = 0.0
    ucb_score: float = 0.0
    children: list[str] = Field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.UNEXPLORED
    supporting_edges: list[str] = Field(default_factory=list)
    contradicting_edges: list[str] = Field(default_factory=list)
    agents_assigned: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Models
# ═══════════════════════════════════════════════════════════════════════════════


class AgentTemplate(BaseModel):
    """Template defining an agent's capabilities and configuration."""

    agent_type: AgentType
    display_name: str
    description: str = ""
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    kg_write_permissions: list[NodeType] = Field(default_factory=list)
    kg_edge_permissions: list[EdgeRelationType] = Field(default_factory=list)
    requires_yami: bool = False
    falsification_protocol: str = ""
    max_iterations: int = 10
    timeout_seconds: int = 300


class AgentConstraints(BaseModel):
    """Resource constraints for a dynamically-spawned agent."""

    max_turns: int = 200
    token_budget: int = 200_000
    timeout_seconds: int = 300
    max_llm_calls: int = 20


class AgentSpec(BaseModel):
    """Dynamic agent specification — generated by the orchestrator LLM per-task.

    Replaces the static AgentTemplate for dynamic orchestration: the orchestrator
    decides what role the agent plays, what tools it gets, and what constraints
    apply.  Agent types become optional hints, not hard requirements.
    """

    role: str
    instructions: str
    tools: list[str] = Field(default_factory=list)
    constraints: AgentConstraints = Field(default_factory=AgentConstraints)
    parent_agent_id: str | None = None
    hypothesis_branch: str = ""
    agent_type_hint: AgentType | None = None

    # Optional overrides — when present, these take precedence over template defaults
    system_prompt: str = ""
    kg_write_permissions: list[NodeType] = Field(default_factory=list)
    kg_edge_permissions: list[EdgeRelationType] = Field(default_factory=list)
    falsification_protocol: str = ""


class AgentTask(BaseModel):
    """A task assigned to an agent by the orchestrator."""

    task_id: str = Field(default_factory=_uuid)
    research_id: str
    agent_type: AgentType
    agent_id: str = ""
    hypothesis_branch: str | None = None
    instruction: str
    context: dict[str, Any] = Field(default_factory=dict)
    kg_context: list[str] = Field(default_factory=list)  # relevant node/edge IDs
    priority: int = 0
    depends_on: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.QUEUED
    created_at: datetime = Field(default_factory=_utcnow)


class FalsificationResult(BaseModel):
    """Result of an agent's self-falsification attempt on an edge."""

    edge_id: str
    agent_id: str = ""
    hypothesis_branch: str = ""
    original_confidence: float = 0.5
    revised_confidence: float = 0.5
    falsified: bool = False
    search_query: str = ""
    method: str = ""
    counter_evidence_found: bool = False
    counter_evidence: list[EvidenceSource] = Field(default_factory=list)
    reasoning: str = ""
    confidence_delta: float = 0.0
    timestamp: datetime = Field(default_factory=_utcnow)


class TurnType(enum.StrEnum):
    """Type of action in a multi-turn agent investigation loop."""

    THINK = "think"
    TOOL_CALL = "tool_call"
    CODE_EXECUTION = "code_execution"
    ANSWER = "answer"


class AgentTurn(BaseModel):
    """A single turn in a multi-turn agent investigation loop."""

    turn_number: int
    turn_type: TurnType
    input_prompt: str = ""
    raw_response: str = ""
    parsed_action: str = ""
    execution_result: str = ""
    error: str | None = None
    tokens_used: int = 0
    duration_ms: int = 0
    timestamp: datetime = Field(default_factory=_utcnow)


class UncertaintyVector(BaseModel):
    """Multi-dimensional uncertainty assessment from an agent."""

    input_ambiguity: float = Field(default=0.0, ge=0.0, le=1.0)
    data_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning_divergence: float = Field(default=0.0, ge=0.0, le=1.0)
    model_disagreement: float = Field(default=0.0, ge=0.0, le=1.0)
    conflict_uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty_uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    composite: float = Field(default=0.0, ge=0.0, le=1.0)
    is_critical: bool = False

    def compute_composite(
        self,
        w_ambiguity: float = 0.15,
        w_quality: float = 0.25,
        w_divergence: float = 0.20,
        w_model: float = 0.15,
        w_conflict: float = 0.15,
        w_novelty: float = 0.10,
    ) -> float:
        self.composite = (
            w_ambiguity * self.input_ambiguity
            + w_quality * self.data_quality
            + w_divergence * self.reasoning_divergence
            + w_model * self.model_disagreement
            + w_conflict * self.conflict_uncertainty
            + w_novelty * self.novelty_uncertainty
        )
        return self.composite


class AgentResult(BaseModel):
    """Result produced by an agent after executing a task."""

    task_id: str
    agent_id: str
    agent_type: AgentType
    hypothesis_id: str = ""
    parent_agent_id: str | None = None
    depth: int = 0
    nodes_added: list[KGNode] = Field(default_factory=list)
    edges_added: list[KGEdge] = Field(default_factory=list)
    nodes_updated: list[str] = Field(default_factory=list)
    edges_updated: list[str] = Field(default_factory=list)
    falsification_results: list[FalsificationResult] = Field(default_factory=list)
    uncertainty: UncertaintyVector = Field(default_factory=UncertaintyVector)
    summary: str = ""
    reasoning_trace: str = ""
    recommended_next: str | None = None
    sub_agent_results: list[AgentResult] = Field(default_factory=list)
    token_usage: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = 0
    llm_calls: int = 0
    llm_tokens_used: int = 0
    success: bool = True
    turns: list[AgentTurn] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


# Rebuild for self-referential sub_agent_results
AgentResult.model_rebuild()


# ═══════════════════════════════════════════════════════════════════════════════
# Research Session Models
# ═══════════════════════════════════════════════════════════════════════════════


class ResearchConfig(BaseModel):
    """Configuration for a specific research session."""

    max_hypothesis_depth: int = 5
    max_mcts_iterations: int = 30
    max_agents: int = 8
    max_agents_per_swarm: int = 15
    confidence_threshold: float = 0.7
    hitl_uncertainty_threshold: float = 0.6
    hitl_timeout_seconds: int = 600
    max_llm_calls_per_agent: int = 20
    agent_types: list[AgentType] | None = None  # None = auto-select
    enable_falsification: bool = True
    enable_hitl: bool = True
    slack_channel_id: str | None = None

    # --- Scaled orchestration (per-hypothesis swarms) ---
    max_concurrent_agents: int = 100  # Global concurrency limit (semaphore)
    max_total_agents: int = 10_000  # Hard cap across entire session
    max_hypothesis_breadth: int = 50  # Max competing hypotheses per tree level
    agent_token_budget: int = 200_000  # Per-agent token limit
    session_token_budget: int = 10_000_000  # Total session token limit
    session_timeout_seconds: int = 1800  # Wall-clock timeout (default 30 min)


class ResearchResult(BaseModel):
    """Final compiled result of a research session."""

    research_id: str
    best_hypothesis: HypothesisNode
    hypothesis_ranking: list[HypothesisNode] = Field(default_factory=list)
    key_findings: list[KGEdge] = Field(default_factory=list)
    contradictions: list[tuple[KGEdge, KGEdge]] = Field(default_factory=list)
    uncertainties: list[UncertaintyVector] = Field(default_factory=list)
    recommended_experiments: list[str] = Field(default_factory=list)
    report_markdown: str = ""
    graph_snapshot: dict[str, Any] = Field(default_factory=dict)
    kg_stats: dict[str, int] = Field(default_factory=dict)
    total_duration_ms: int = 0
    total_llm_calls: int = 0
    total_tokens: int = 0
    screening: ScreeningResult | None = None
    living_document: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class ScreeningResult(BaseModel):
    """Result of biosecurity screening on a research output."""

    research_id: str
    tier: ScreeningTier
    flagged_categories: list[BiosecurityCategory] = Field(default_factory=list)
    reasoning: str = ""
    disclaimer: str = ""
    screened_at: datetime = Field(default_factory=_utcnow)


class ResearchSession(BaseModel):
    """A research session — the top-level unit of work."""

    id: str = Field(default_factory=_uuid)
    query: str
    status: SessionStatus = SessionStatus.INITIALIZING
    config: ResearchConfig = Field(default_factory=ResearchConfig)
    swarm_composition: list[str] = Field(default_factory=list)
    hypothesis_tree_id: str = ""
    knowledge_graph_id: str = ""
    current_iteration: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    total_hypotheses: int = 0
    total_tokens_used: int = 0
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None
    result: ResearchResult | None = None


# Rebuild to resolve forward reference
ResearchSession.model_rebuild()


# ═══════════════════════════════════════════════════════════════════════════════
# MCP & Tool Models
# ═══════════════════════════════════════════════════════════════════════════════


class MCPServerConfig(BaseModel):
    """Configuration for connecting to an MCP tool server."""

    name: str
    transport: MCPTransportType = MCPTransportType.STDIO
    command: str | None = None  # for stdio transport
    args: list[str] = Field(default_factory=list)
    url: str | None = None  # for SSE/HTTP transport
    env: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = 10
    max_reconnect_attempts: int = 5
    enabled: bool = True


class MCPToolParameter(BaseModel):
    """A parameter in an MCP tool's input schema."""

    name: str
    type: str  # JSON Schema type
    description: str = ""
    required: bool = False
    default: Any = None


class MCPToolManifest(BaseModel):
    """Manifest describing an MCP-provided tool."""

    name: str
    description: str = ""
    server_name: str = ""
    parameters: list[MCPToolParameter] = Field(default_factory=list)
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ContainerToolConfig(BaseModel):
    """Configuration for a containerized (sandboxed) tool."""

    name: str
    image: str
    command: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    memory_limit: str = "512m"
    cpu_limit: str = "1.0"
    timeout_seconds: int = 120
    network_mode: str = "none"  # sandboxed by default
    volumes: dict[str, str] = Field(default_factory=dict)


class DynamicToolStatus(enum.StrEnum):
    """Lifecycle status of a dynamically created tool."""

    DRAFT = "DRAFT"
    TESTING = "TESTING"
    VALIDATED = "VALIDATED"
    FAILED = "FAILED"
    REGISTERED = "REGISTERED"


class DynamicToolSpec(BaseModel):
    """Specification for a dynamically generated tool wrapper."""

    name: str
    description: str = ""
    api_base_url: str = ""
    api_documentation: str = ""
    wrapper_code: str = ""
    test_code: str = ""
    test_results: list[dict[str, Any]] = Field(default_factory=list)
    status: DynamicToolStatus = DynamicToolStatus.DRAFT
    category: str = "dynamic"
    capabilities: list[str] = Field(default_factory=list)
    example_tasks: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    created_by: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    validated_at: datetime | None = None


class ToolRegistryEntry(BaseModel):
    """An entry in the unified tool registry."""

    name: str
    description: str = ""
    source_type: ToolSourceType
    category: str = ""  # e.g. "literature", "protein", "pathway"
    capabilities: list[str] = Field(default_factory=list)
    mcp_server: str | None = None
    mcp_manifest: MCPToolManifest | None = None
    container_config: ContainerToolConfig | None = None
    cache_ttl: int = 86_400
    rate_limit_rps: float = 10.0
    enabled: bool = True
    registered_at: datetime = Field(default_factory=_utcnow)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Models
# ═══════════════════════════════════════════════════════════════════════════════


class BenchmarkMetric(BaseModel):
    """A single metric in a benchmark run."""

    name: str
    value: float
    unit: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class LABBenchResult(BaseModel):
    """Result of a LAB-Bench benchmark evaluation."""

    task_id: str
    task_type: str  # e.g. "literature_qa", "protocol_generation"
    correct: bool = False
    predicted: str = ""
    expected: str = ""
    score: float = 0.0
    reasoning_trace: str = ""
    evidence_sources_used: int = 0
    falsification_applied: bool = False


class BenchmarkRun(BaseModel):
    """A complete benchmark run comparing YOHAS against baselines."""

    id: str = Field(default_factory=_uuid)
    benchmark_name: str  # e.g. "LAB-Bench", "BioASQ"
    version: str = ""
    metrics: list[BenchmarkMetric] = Field(default_factory=list)
    lab_bench_results: list[LABBenchResult] = Field(default_factory=list)
    baseline_comparison: dict[str, float] = Field(default_factory=dict)
    total_tasks: int = 0
    correct_tasks: int = 0
    accuracy: float = 0.0
    run_config: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket / Live Event Models
# ═══════════════════════════════════════════════════════════════════════════════


class ResearchEvent(BaseModel):
    """An event emitted during research, streamed via WebSocket."""

    session_id: str
    event_type: str  # e.g. "agent_started", "node_created", "hypothesis_expanded"
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utcnow)
