"""Constants used across the YOHAS platform."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Agent type identifiers
# ---------------------------------------------------------------------------
AGENT_LITERATURE = "literature_analyst"
AGENT_PROTEIN = "protein_engineer"
AGENT_GENOMICS = "genomics_mapper"
AGENT_PATHWAY = "pathway_analyst"
AGENT_DRUG = "drug_hunter"
AGENT_CLINICAL = "clinical_analyst"
AGENT_CRITIC = "scientific_critic"
AGENT_EXPERIMENT = "experiment_designer"

ALL_AGENT_TYPES: list[str] = [
    AGENT_LITERATURE,
    AGENT_PROTEIN,
    AGENT_GENOMICS,
    AGENT_PATHWAY,
    AGENT_DRUG,
    AGENT_CLINICAL,
    AGENT_CRITIC,
    AGENT_EXPERIMENT,
]

# scientific_critic is always included in every swarm composition
MANDATORY_AGENTS: list[str] = [AGENT_CRITIC]

# ---------------------------------------------------------------------------
# Cache TTLs (seconds)
# ---------------------------------------------------------------------------
CACHE_TTL_PUBMED = 86_400         # 24 h
CACHE_TTL_SEMANTIC_SCHOLAR = 86_400
CACHE_TTL_UNIPROT = 86_400        # 24 h
CACHE_TTL_KEGG = 604_800          # 7 d
CACHE_TTL_REACTOME = 604_800
CACHE_TTL_CHEMBL = 604_800
CACHE_TTL_CLINICALTRIALS = 3_600  # 1 h — trials update frequently
CACHE_TTL_MYGENE = 86_400
CACHE_TTL_ESM = 604_800           # embeddings/structures don't change
CACHE_TTL_LLM = 3_600             # 1 h (optional prompt-hash cache)
CACHE_TTL_OPENTARGETS = 86_400    # 24 h
CACHE_TTL_CLINVAR = 86_400        # 24 h
CACHE_TTL_GTEX = 604_800          # 7 d — expression data stable
CACHE_TTL_GNOMAD = 604_800        # 7 d — allele freqs stable
CACHE_TTL_HPO = 604_800           # 7 d — ontology stable
CACHE_TTL_OMIM = 86_400           # 24 h
CACHE_TTL_BIOGRID = 86_400        # 24 h
CACHE_TTL_DEPMAP = 604_800        # 7 d — release-based
CACHE_TTL_CELLXGENE = 604_800     # 7 d — census snapshots
CACHE_TTL_STRING_DB = 604_800     # 7 d — interactions stable
CACHE_TTL_DEFAULT = 86_400

# ---------------------------------------------------------------------------
# Rate limits (requests per second)
# ---------------------------------------------------------------------------
RATE_LIMIT_PUBMED = 3.0           # NCBI E-utilities default; 10/s with key
RATE_LIMIT_SEMANTIC_SCHOLAR = 10.0
RATE_LIMIT_UNIPROT = 5.0
RATE_LIMIT_KEGG = 5.0
RATE_LIMIT_REACTOME = 5.0
RATE_LIMIT_CHEMBL = 5.0
RATE_LIMIT_CLINICALTRIALS = 5.0
RATE_LIMIT_MYGENE = 10.0
RATE_LIMIT_ESM = 1.0
RATE_LIMIT_OPENTARGETS = 10.0     # Open Targets Platform — generous
RATE_LIMIT_CLINVAR = 3.0          # NCBI E-utils default
RATE_LIMIT_GTEX = 5.0
RATE_LIMIT_GNOMAD = 5.0           # GraphQL API
RATE_LIMIT_HPO = 10.0             # JAX HPO API — generous
RATE_LIMIT_OMIM = 3.0             # API key required, conservative
RATE_LIMIT_BIOGRID = 5.0
RATE_LIMIT_DEPMAP = 5.0
RATE_LIMIT_CELLXGENE = 5.0
RATE_LIMIT_STRING_DB = 5.0
RATE_LIMIT_DEFAULT = 10.0

# ---------------------------------------------------------------------------
# Retry / timeout defaults
# ---------------------------------------------------------------------------
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = [1.0, 2.0, 4.0]
DEFAULT_TIMEOUT_SECONDS = 30

# ---------------------------------------------------------------------------
# MCTS defaults
# ---------------------------------------------------------------------------
DEFAULT_UCB_EXPLORATION_CONSTANT = 1.414  # sqrt(2)

# ---------------------------------------------------------------------------
# Knowledge graph defaults
# ---------------------------------------------------------------------------
KG_MAX_NODES = 10_000
KG_MAX_EDGES = 50_000
KG_SNAPSHOT_INTERVAL_SECONDS = 300  # auto-persist every 5 min

# ---------------------------------------------------------------------------
# Research defaults (supplement config.py settings)
# ---------------------------------------------------------------------------
DEFAULT_MAX_AGENTS_PER_SWARM = 8
DEFAULT_MAX_LLM_CALLS_PER_AGENT = 20
DEFAULT_SESSION_TIMEOUT_S = 7_200  # 2 h
DEFAULT_MIN_EVIDENCE_FOR_CONFIDENCE = 3

# ---------------------------------------------------------------------------
# Sub-agent spawning limits
# ---------------------------------------------------------------------------
MAX_SUB_AGENT_DEPTH = 5
MAX_SUB_AGENTS_PER_PARENT = 10

# ---------------------------------------------------------------------------
# Evidence quality thresholds
# ---------------------------------------------------------------------------
EVIDENCE_QUALITY_HIGH = 0.8
EVIDENCE_QUALITY_MEDIUM = 0.5
EVIDENCE_QUALITY_LOW = 0.3

# ---------------------------------------------------------------------------
# API versioning
# ---------------------------------------------------------------------------
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"
