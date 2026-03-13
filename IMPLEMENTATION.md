# YOHAS 3.0 — Implementation Plan

## Overview

YOHAS (Your Own Hypothesis-driven Agentic Scientist) is an autonomous biomedical research platform. A user submits a research question (e.g., "What are the therapeutic approaches for B7-H3 in NSCLC?"), and YOHAS spawns a swarm of specialized AI agents that collaboratively build a knowledge graph, explore hypotheses via Monte Carlo Tree Search, self-falsify findings, request human input when uncertain (via Slack), and produce a structured research report.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Frontend (Next.js 14)                      │
│  Dashboard │ Research Detail │ KG Visualization │ Report View       │
│  Port 3000 │ WebSocket-driven live feed │ Cytoscape.js graph       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ REST + WebSocket
┌──────────────────────────────▼──────────────────────────────────────┐
│                        API Layer (FastAPI)                           │
│  Port 8000 │ Routes: research, graph, agents, hypothesis, health    │
│  Middleware: API key auth, CORS, request ID propagation             │
│  WebSocket: /api/v1/research/{id}/ws (real-time event streaming)    │
└───────┬───────────────────────┬──────────────────────┬──────────────┘
        │                       │                      │
┌───────▼───────┐   ┌──────────▼──────────┐   ┌──────▼───────────┐
│  PostgreSQL   │   │       Redis          │   │  Celery Workers  │
│  Port 5432    │   │  Port 6379           │   │  Agent execution │
│  Research     │   │  Job queue (Celery)  │   │  Concurrency: 4  │
│  sessions,    │   │  API response cache  │   │                  │
│  KG state,    │   │  Rate limit tokens   │   │                  │
│  audit logs   │   │                      │   │                  │
└───────────────┘   └──────────────────────┘   └────────┬─────────┘
                                                        │
        ┌───────────────────────────────────────────────▼─────────┐
        │                    Agent Swarm                           │
        │  literature_analyst │ protein_engineer │ genomics_mapper │
        │  pathway_analyst │ drug_hunter │ clinical_analyst        │
        │  scientific_critic │ experiment_designer                 │
        │                                                         │
        │  Each agent: LLM (Claude) + Tools + KG access + Yami   │
        └──────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────▼──────────────────────────────────────────┐
        │                  External Integrations                   │
        │  PubMed │ Semantic Scholar │ UniProt │ KEGG │ Reactome  │
        │  MyGene.info │ ChEMBL │ ClinicalTrials.gov │ ESM-2     │
        │  Slack (HITL) │ Anthropic API (LLM)                     │
        └─────────────────────────────────────────────────────────┘
```

### Key Modules

| Module | Path | Responsibility |
|--------|------|---------------|
| **Core** | `backend/core/` | Data models, interfaces/protocols, config, LLM wrapper, exceptions, audit logging |
| **World Model** | `backend/world_model/` | Knowledge graph (in-memory + persistence), Yami/ESM interface |
| **Agents** | `backend/agents/` | Base agent class + 8 specialized agent implementations |
| **Orchestrator** | `backend/orchestrator/` | MCTS hypothesis tree, swarm composer, uncertainty/HITL logic, main research loop |
| **Integrations** | `backend/integrations/` | External API clients (PubMed, UniProt, KEGG, etc.) with caching + rate limiting |
| **API** | `backend/api/` | FastAPI app, routes, WebSocket, middleware, dependency injection |
| **Workers** | `backend/workers/` | Celery app config + task definitions for agent execution |
| **Report** | `backend/report/` | KG + hypothesis tree → structured markdown report |
| **Database** | `backend/db/` | SQLAlchemy async models + session management |
| **Frontend** | `frontend/` | Next.js 14 app with Cytoscape.js, D3, WebSocket-driven live feeds |

### Data Flow

1. User submits research query via frontend → `POST /api/v1/research`
2. API creates `ResearchSession` in Postgres, dispatches to Celery
3. Orchestrator seeds KG with initial entities, generates hypotheses, composes agent swarm
4. MCTS loop: select hypothesis → dispatch agents → agents query tools + write to KG → critique → evaluate info gain → backpropagate → check HITL → check termination
5. Events stream to frontend via WebSocket throughout
6. On completion: compile results, generate report, return `ResearchResult`

---

## Task Breakdown

### Phase 1: Foundation (Core + Infrastructure)

#### 1.1 Project Scaffolding
- [ ] **1.1.1** Create root `docker-compose.yml` with all 5 services (api, worker, redis, postgres, frontend)
- [ ] **1.1.2** Create `docker-compose.prod.yml` override for production
- [ ] **1.1.3** Create `.env.example` with all environment variables documented
- [ ] **1.1.4** Create root `Makefile` with targets: `dev`, `test`, `seed`, `lint`, `migrate`, `build`
- [ ] **1.1.5** Initialize `backend/pyproject.toml` (Python 3.11+, uv/poetry) with all dependencies
- [ ] **1.1.6** Initialize `frontend/package.json` (Next.js 14, TypeScript) with all dependencies
- [ ] **1.1.7** Create Dockerfiles for backend (Python) and frontend (Node)

#### 1.2 Core Data Models (`backend/core/models.py`)
- [ ] **1.2.1** Define `NodeType` and `EdgeRelationType` enums with all values
- [ ] **1.2.2** Define `EvidenceSource` Pydantic model
- [ ] **1.2.3** Define `KGNode` Pydantic model (with type-specific properties dict)
- [ ] **1.2.4** Define `KGEdge` Pydantic model (with falsification fields)
- [ ] **1.2.5** Define `HypothesisNode` Pydantic model (with MCTS fields: visit_count, total_info_gain, etc.)
- [ ] **1.2.6** Define `AgentTemplate` Pydantic model
- [ ] **1.2.7** Define `AgentTask` Pydantic model (with status enum, depends_on)
- [ ] **1.2.8** Define `AgentResult` Pydantic model (with mutations, falsification, uncertainty)
- [ ] **1.2.9** Define `FalsificationResult` Pydantic model
- [ ] **1.2.10** Define `UncertaintyVector` Pydantic model (with composite computation)
- [ ] **1.2.11** Define `ResearchSession`, `ResearchConfig`, `ResearchResult` models

#### 1.3 Core Infrastructure (`backend/core/`)
- [ ] **1.3.1** `config.py` — Pydantic Settings class loading from environment variables
- [ ] **1.3.2** `constants.py` — Relation types, node types, agent type strings, cache TTLs
- [ ] **1.3.3** `exceptions.py` — Exception hierarchy: `YOHASError` → `ToolError`, `AgentError`, `OrchestrationError`, `GraphError`
- [ ] **1.3.4** `interfaces.py` — `BaseAgent` ABC, `KnowledgeGraph` Protocol, `YamiInterface` Protocol, `BaseTool` ABC
- [ ] **1.3.5** `llm.py` — Anthropic SDK wrapper: `query()` with KG context injection, token tracking, audit logging, structured output parsing
- [ ] **1.3.6** `audit.py` — Structured audit logger using `structlog` (JSON format, request_id/agent_id/research_id propagation)

#### 1.4 Database Layer (`backend/db/`)
- [ ] **1.4.1** `session.py` — SQLAlchemy async engine + session factory
- [ ] **1.4.2** `tables.py` — SQLAlchemy models: `research_sessions`, `kg_snapshots`, `audit_logs`
- [ ] **1.4.3** Set up Alembic: `alembic.ini`, `env.py`, initial migration

---

### Phase 2: Knowledge Graph + World Model

#### 2.1 Knowledge Graph (`backend/world_model/knowledge_graph.py`)
- [ ] **2.1.1** Implement in-memory KG store (nodes dict, edges dict, adjacency index)
- [ ] **2.1.2** CRUD operations: `add_node`, `add_edge`, `get_node`, `get_node_by_name`, `get_edge`
- [ ] **2.1.3** Edge queries: `get_edges_from`, `get_edges_to`, `get_edges_between`
- [ ] **2.1.4** `update_node`, `update_edge_confidence`, `mark_edge_falsified`
- [ ] **2.1.5** Subgraph extraction: `get_subgraph(center_id, hops)` via BFS
- [ ] **2.1.6** Contradiction detection: `get_contradictions(edge)` — find edges with opposing relations between same nodes
- [ ] **2.1.7** Query helpers: `get_recent_edges`, `get_edges_by_hypothesis`, `get_weakest_edges`, `get_orphan_nodes`
- [ ] **2.1.8** Graph traversal: `shortest_path`, `get_upstream`, `get_downstream`
- [ ] **2.1.9** Stats: `node_count`, `edge_count`, `avg_confidence`, `edges_added_since`
- [ ] **2.1.10** Serialization: `to_cytoscape()`, `to_json()`, `to_markdown_summary()`
- [ ] **2.1.11** Persistence: `save(session_id)` and `load(session_id)` via Postgres JSON

#### 2.2 Yami/ESM Interface (`backend/world_model/yami.py`)
- [ ] **2.2.1** Implement `YamiInterface` wrapping ESM-2 (HuggingFace API or local)
- [ ] **2.2.2** `get_logits(sequence)` — fitness scoring
- [ ] **2.2.3** `get_embeddings(sequence)` — mean pooled last layer
- [ ] **2.2.4** `predict_structure(sequence)` — ESMFold via HuggingFace API
- [ ] **2.2.5** Caching layer for ESM predictions (Redis)

#### 2.3 Knowledge Graph Tests
- [ ] **2.3.1** Test add/get/update/delete for nodes and edges
- [ ] **2.3.2** Test contradiction detection with known opposing edges
- [ ] **2.3.3** Test subgraph extraction (verify correct BFS behavior)
- [ ] **2.3.4** Test serialization formats (cytoscape, json, markdown)

---

### Phase 3: Tool Integrations

#### 3.1 Base Tool Framework (`backend/integrations/base_tool.py`)
- [ ] **3.1.1** `BaseTool` ABC with `httpx.AsyncClient`, connection pooling
- [ ] **3.1.2** Redis cache decorator (key = hash of query params, configurable TTL per tool)
- [ ] **3.1.3** Token-bucket rate limiter (per tool, backed by Redis)
- [ ] **3.1.4** Retry with exponential backoff (3 retries: 1s/2s/4s)
- [ ] **3.1.5** Timeout handling (30s default), structured error logging, `ToolError` normalization

#### 3.2 Individual Tools
- [ ] **3.2.1** `pubmed.py` — `search()`, `fetch()`, `fetch_abstract()` via NCBI E-utilities
- [ ] **3.2.2** `semantic_scholar.py` — `search()`, `get_paper()`, `get_citations()`, `get_references()`
- [ ] **3.2.3** `uniprot.py` — `search()`, `get_protein()`, `get_sequence()`, `get_features()`
- [ ] **3.2.4** `esm.py` — `get_logits()`, `get_embeddings()`, `predict_structure()` (wraps HuggingFace)
- [ ] **3.2.5** `kegg.py` — `get_pathway()`, `get_gene_pathways()`, `get_pathway_genes()`, `search()`
- [ ] **3.2.6** `reactome.py` — `get_pathways_for_protein()`, `get_pathway_detail()`, `get_reactions()`
- [ ] **3.2.7** `mygene.py` — `get_gene()`, `query()`
- [ ] **3.2.8** `drugbank.py` — ChEMBL API: `search_by_target()`, `get_compound()`, `get_target()`, `get_activities()`
- [ ] **3.2.9** `clinicaltrials.py` — `search()`, `get_trial()` via ClinicalTrials.gov v2 API
- [ ] **3.2.10** `slack.py` — `post_hitl_request()`, `wait_for_response()`, `post_update()` via slack-bolt

#### 3.3 Tool Tests
- [ ] **3.3.1** Unit tests for each tool: mock HTTP responses (use `responses` library), verify parsing
- [ ] **3.3.2** Test caching behavior: verify cache hit/miss, TTL expiry
- [ ] **3.3.3** Test rate limiter: verify backpressure when limit exceeded
- [ ] **3.3.4** Test retry logic: verify exponential backoff on transient failures

---

### Phase 4: Agents

#### 4.1 Base Agent (`backend/agents/base.py`)
- [ ] **4.1.1** Implement `BaseAgent` ABC with: `agent_id`, `template`, `llm`, `kg`, `yami`, `tools`, `audit`
- [ ] **4.1.2** `query_llm()` — wraps LLM call with KG subgraph injection, audit logging, token tracking
- [ ] **4.1.3** `query_yami()` — wraps Yami call with audit logging, error handling, caching
- [ ] **4.1.4** `write_node()` / `write_edge()` — write to KG with audit trail, auto-trigger contradiction check
- [ ] **4.1.5** Default `falsify()` — for each edge, ask LLM for counter-evidence, search tools, adjust confidence
- [ ] **4.1.6** `get_uncertainty()` — compute `UncertaintyVector` from agent state

#### 4.2 Specialized Agents (`backend/agents/`)
- [ ] **4.2.1** `literature_analyst.py` — Tools: pubmed, semantic_scholar. Finds papers, extracts claims as KG edges.
- [ ] **4.2.2** `protein_engineer.py` — Tools: uniprot, esm. Fetches protein data, predicts structure, writes PROTEIN nodes.
- [ ] **4.2.3** `genomics_mapper.py` — Tools: mygene, kegg. Maps genes to pathways, writes GENE/PATHWAY nodes.
- [ ] **4.2.4** `pathway_analyst.py` — Tools: kegg, reactome. Deep pathway analysis, writes PATHWAY/MECHANISM edges.
- [ ] **4.2.5** `drug_hunter.py` — Tools: drugbank/chembl, clinicaltrials. Finds drugs/compounds targeting entities.
- [ ] **4.2.6** `clinical_analyst.py` — Tools: clinicaltrials, pubmed. Searches trials, reports outcomes/failures.
- [ ] **4.2.7** `scientific_critic.py` — Tools: pubmed, semantic_scholar. Iterates `get_recent_edges()`, actively tries to disprove each. Only modifies confidence + adds EVIDENCE_AGAINST.
- [ ] **4.2.8** `experiment_designer.py` — No tools (reasoning-only). Proposes experiments to resolve KG uncertainties. Writes EXPERIMENT nodes.

#### 4.3 Agent Template Definitions
- [ ] **4.3.1** Define system prompts for all 8 agent types
- [ ] **4.3.2** Define tool permissions, KG write permissions, and falsification protocols per agent

#### 4.4 Agent Tests
- [ ] **4.4.1** Test each agent with mocked tools + LLM: verify correct KG mutations
- [ ] **4.4.2** Test falsification flow: agent receives edges → searches counter-evidence → adjusts confidence
- [ ] **4.4.3** Test uncertainty vector computation

---

### Phase 5: Orchestrator

#### 5.1 Hypothesis Tree / MCTS (`backend/orchestrator/hypothesis_tree.py`)
- [ ] **5.1.1** `HypothesisTree` class with root node, nodes dict
- [ ] **5.1.2** `select_next()` — UCB1 selection from root to leaf (exploration constant = √2)
- [ ] **5.1.3** `backpropagate(node_id, info_gain)` — walk to root, update visit_count + total_info_gain
- [ ] **5.1.4** `expand(parent_id, new_hypotheses)` — add child nodes
- [ ] **5.1.5** `prune(node_id, reason)` — mark falsified/low-confidence
- [ ] **5.1.6** `get_best_path()` — root to highest-confidence leaf
- [ ] **5.1.7** `get_tree_state()` — serialize for frontend visualization

#### 5.2 Swarm Composer (`backend/orchestrator/swarm_composer.py`)
- [ ] **5.2.1** `select_agents()` — given query + hypotheses + template library, call Claude to select agent types
- [ ] **5.2.2** Always include `scientific_critic` (non-negotiable)
- [ ] **5.2.3** Instantiate selected agents with their tools, KG access, Yami access

#### 5.3 Uncertainty + HITL (`backend/orchestrator/uncertainty.py`)
- [ ] **5.3.1** `aggregate_uncertainty()` — combine vectors from multiple agents
- [ ] **5.3.2** `should_trigger_hitl()` — check composite threshold + critical path + unresolved contradictions
- [ ] **5.3.3** `compose_slack_message()` — clear, actionable Slack message with context and options

#### 5.4 Main Orchestrator Loop (`backend/orchestrator/orchestrator.py`)
- [ ] **5.4.1** INITIALIZE: seed KG with initial entities (LLM call), generate hypotheses, build HypothesisTree
- [ ] **5.4.2** COMPOSE SWARM: call `swarm_composer.select_agents()`, instantiate agents
- [ ] **5.4.3** MCTS LOOP — SELECT: `hypothesis_tree.select_next()`
- [ ] **5.4.4** MCTS LOOP — DISPATCH: generate per-agent tasks via LLM, submit to Celery
- [ ] **5.4.5** MCTS LOOP — EXECUTE: await agent results
- [ ] **5.4.6** MCTS LOOP — CRITIQUE: run `scientific_critic` on new edges
- [ ] **5.4.7** MCTS LOOP — EVALUATE: compute info gain (new edges + confidence changes + contradictions)
- [ ] **5.4.8** MCTS LOOP — BACKPROPAGATE: update hypothesis tree
- [ ] **5.4.9** MCTS LOOP — CHECK HITL: aggregate uncertainty → Slack if triggered → inject response
- [ ] **5.4.10** MCTS LOOP — CHECK TERMINATION: confidence threshold, all pruned, or budget exhausted
- [ ] **5.4.11** COMPILE RESULTS: best hypothesis path, supporting KG, generate report, return `ResearchResult`

#### 5.5 Orchestrator Tests
- [ ] **5.5.1** Test UCB1 selection correctness with known visit counts / info gains
- [ ] **5.5.2** Test backpropagation (verify visit counts propagate to root)
- [ ] **5.5.3** Test pruning and best-path extraction
- [ ] **5.5.4** Test swarm composition (mock LLM, verify critic always included)
- [ ] **5.5.5** Test HITL trigger logic with various uncertainty vectors
- [ ] **5.5.6** Integration test: mock agents, verify full MCTS loop produces expected KG state

---

### Phase 6: API Layer

#### 6.1 FastAPI App (`backend/api/`)
- [ ] **6.1.1** `main.py` — FastAPI app factory with lifespan, CORS, middleware registration
- [ ] **6.1.2** `middleware.py` — API key auth (X-API-Key header), request ID injection, timing
- [ ] **6.1.3** `deps.py` — Dependency injection: DB session, Redis, KG instance, config

#### 6.2 Routes
- [ ] **6.2.1** `routes/research.py` — `POST /research`, `GET /research/{id}`, `GET /research/{id}/result`, `GET /research` (paginated), `POST /research/{id}/cancel`, `POST /research/{id}/feedback`
- [ ] **6.2.2** `routes/graph.py` — `GET /research/{id}/graph` (cytoscape/json/summary), `GET .../subgraph`, `GET .../nodes`, `GET .../edges`, `GET .../contradictions`, `GET .../stats`
- [ ] **6.2.3** `routes/agents.py` — `GET /research/{id}/agents`, `GET .../agents/{agent_id}/log`, `GET .../agents/{agent_id}/result`
- [ ] **6.2.4** `routes/hypothesis.py` — `GET /research/{id}/hypotheses`, `GET .../{node_id}`, `GET .../best`
- [ ] **6.2.5** `routes/health.py` — `GET /health`, `GET /health/ready`, `GET /templates`
- [ ] **6.2.6** `websocket.py` — WebSocket endpoint at `/research/{id}/ws`, streams all event types

#### 6.3 Celery Workers (`backend/workers/`)
- [ ] **6.3.1** `celery_app.py` — Celery config with Redis broker, result backend
- [ ] **6.3.2** `tasks.py` — `run_research` task (wraps orchestrator), `run_agent` task (wraps single agent execution)

#### 6.4 Report Generator (`backend/report/generator.py`)
- [ ] **6.4.1** KG + hypothesis tree → structured markdown: Executive Summary, Evidence Map, Competing Hypotheses, Key Uncertainties, Recommended Experiments, Audit Trail
- [ ] **6.4.2** Each claim links to supporting KG edge ID

#### 6.5 API Tests
- [ ] **6.5.1** Test all research CRUD endpoints
- [ ] **6.5.2** Test graph endpoints with seeded KG data
- [ ] **6.5.3** Test WebSocket event streaming (connect, verify event types)
- [ ] **6.5.4** Test auth middleware (missing key, invalid key, valid key)

---

### Phase 7: Frontend

#### 7.1 App Shell + Layout
- [ ] **7.1.1** `app/layout.tsx` — Root layout with navigation, theme
- [ ] **7.1.2** `lib/api.ts` — Backend API client (typed fetch wrapper)
- [ ] **7.1.3** `lib/websocket.ts` — WebSocket connection manager (auto-reconnect, typed events)

#### 7.2 Dashboard (`app/page.tsx`)
- [ ] **7.2.1** New research form (text input + optional config accordion)
- [ ] **7.2.2** Recent research sessions list with status badges
- [ ] **7.2.3** Quick stats (total runs, active runs)

#### 7.3 Research Detail (`app/research/[id]/page.tsx`)
- [ ] **7.3.1** Header: query text, status, duration, swarm composition
- [ ] **7.3.2** Live swarm activity feed (WebSocket-driven, `swarm-feed.tsx`)
- [ ] **7.3.3** Hypothesis tree mini-view (`hypothesis-tree.tsx`)
- [ ] **7.3.4** Quick KG stats panel
- [ ] **7.3.5** Action buttons: View Graph, View Report, Cancel

#### 7.4 KG Visualization (`app/research/[id]/graph/page.tsx`)
- [ ] **7.4.1** `graph-viewer.tsx` — Cytoscape.js React wrapper with Cola.js layout
- [ ] **7.4.2** Node coloring by type (Protein=#4A90D9, Gene=#6B5CE7, Disease=#E74C3C, etc.)
- [ ] **7.4.3** Edge styling: thickness = confidence, color = green (supporting) / red (contradicting)
- [ ] **7.4.4** Click interactions: node → side panel (properties, edges, evidence); edge → side panel (relation, confidence, citations)
- [ ] **7.4.5** Filters: by node type, hypothesis branch, confidence threshold, agent
- [ ] **7.4.6** Search: find node by name

#### 7.5 Report View (`app/research/[id]/report/page.tsx`)
- [ ] **7.5.1** `report-renderer.tsx` — Markdown → React with citation hover cards
- [ ] **7.5.2** Download as markdown or PDF
- [ ] **7.5.3** Claims linked to KG edges

#### 7.6 Shared Components
- [ ] **7.6.1** `status-badge.tsx` — Status indicators for research/agents
- [ ] **7.6.2** `research-form.tsx` — Query input with advanced config
- [ ] **7.6.3** Set up shadcn/ui component library

---

### Phase 8: Integration + Polish

#### 8.1 End-to-End Integration
- [ ] **8.1.1** Verify full flow: submit query → agents run → KG builds → report generated → frontend displays
- [ ] **8.1.2** Test with canonical query: "What are the therapeutic approaches for B7-H3 in NSCLC?"
- [ ] **8.1.3** Verify WebSocket events stream correctly to frontend

#### 8.2 Scripts
- [ ] **8.2.1** `scripts/seed_demo.py` — Seed a demo research session with pre-built KG for frontend development
- [ ] **8.2.2** `scripts/benchmark_vs_vbiotech.py` — Run B7-H3 query, compare metrics vs Virtual Biotech paper
- [ ] **8.2.3** `scripts/export_graph.py` — Export KG to various formats

#### 8.3 Production Hardening
- [ ] **8.3.1** Global timeout per research session (default 30 min)
- [ ] **8.3.2** Celery task retries (max 2) for transient failures
- [ ] **8.3.3** Per-tool rate limiters verified working
- [ ] **8.3.4** LLM call budget per session (orchestrator reduces MCTS iterations if near limit)
- [ ] **8.3.5** CORS configured for frontend origin only
- [ ] **8.3.6** Docker containers run as non-root
- [ ] **8.3.7** structlog + request ID propagation verified end-to-end

#### 8.4 Documentation
- [ ] **8.4.1** README.md with setup instructions, architecture overview, quickstart
- [ ] **8.4.2** API documentation (FastAPI auto-generated OpenAPI + supplemental)
- [ ] **8.4.3** Agent template documentation (how to add new agent types)

---

## Task Summary

| Phase | Area | Tasks |
|-------|------|-------|
| 1 | Foundation | 20 tasks |
| 2 | Knowledge Graph | 16 tasks |
| 3 | Tool Integrations | 14 tasks |
| 4 | Agents | 16 tasks |
| 5 | Orchestrator | 17 tasks |
| 6 | API Layer | 12 tasks |
| 7 | Frontend | 16 tasks |
| 8 | Integration + Polish | 13 tasks |
| **Total** | | **124 tasks** |
