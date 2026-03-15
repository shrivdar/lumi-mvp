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
- [x] **1.1.1** Create root `docker-compose.yml` with all 5 services (api, worker, redis, postgres, frontend)
- [x] **1.1.2** Create `docker-compose.prod.yml` override for production
- [x] **1.1.3** Create `.env.example` with all environment variables documented
- [x] **1.1.4** Create root `Makefile` with targets: `dev`, `test`, `seed`, `lint`, `migrate`, `build`
- [x] **1.1.5** Initialize `backend/pyproject.toml` (Python 3.11+, uv/poetry) with all dependencies
- [x] **1.1.6** Initialize `frontend/package.json` (Next.js 14, TypeScript) with all dependencies
- [x] **1.1.7** Create Dockerfiles for backend (Python) and frontend (Node)

#### 1.2 Core Data Models (`backend/core/models.py`)
- [x] **1.2.1** Define `NodeType` and `EdgeRelationType` enums with all values
- [x] **1.2.2** Define `EvidenceSource` Pydantic model
- [x] **1.2.3** Define `KGNode` Pydantic model (with type-specific properties dict)
- [x] **1.2.4** Define `KGEdge` Pydantic model (with falsification fields)
- [x] **1.2.5** Define `HypothesisNode` Pydantic model (with MCTS fields: visit_count, total_info_gain, etc.)
- [x] **1.2.6** Define `AgentTemplate` Pydantic model
- [x] **1.2.7** Define `AgentTask` Pydantic model (with status enum, depends_on)
- [x] **1.2.8** Define `AgentResult` Pydantic model (with mutations, falsification, uncertainty)
- [x] **1.2.9** Define `FalsificationResult` Pydantic model
- [x] **1.2.10** Define `UncertaintyVector` Pydantic model (with composite computation)
- [x] **1.2.11** Define `ResearchSession`, `ResearchConfig`, `ResearchResult` models

#### 1.3 Core Infrastructure (`backend/core/`)
- [x] **1.3.1** `config.py` — Pydantic Settings class loading from environment variables
- [x] **1.3.2** `constants.py` — Relation types, node types, agent type strings, cache TTLs
- [x] **1.3.3** `exceptions.py` — Exception hierarchy: `YOHASError` → `ToolError`, `AgentError`, `OrchestrationError`, `GraphError`
- [x] **1.3.4** `interfaces.py` — `BaseAgent` ABC, `KnowledgeGraph` Protocol, `YamiInterface` Protocol, `BaseTool` ABC
- [x] **1.3.5** `llm.py` — Anthropic SDK wrapper: `query()` with KG context injection, token tracking, audit logging, structured output parsing
- [x] **1.3.6** `audit.py` — Structured audit logger using `structlog` (JSON format, request_id/agent_id/research_id propagation)

#### 1.4 Database Layer (`backend/db/`)
- [x] **1.4.1** `session.py` — SQLAlchemy async engine + session factory
- [x] **1.4.2** `tables.py` — SQLAlchemy models: `research_sessions`, `kg_snapshots`, `audit_logs`
- [x] **1.4.3** Set up Alembic: `alembic.ini`, `env.py`, initial migration

---

### Phase 2: Knowledge Graph + World Model

#### 2.1 Knowledge Graph (`backend/world_model/knowledge_graph.py`)
- [x] **2.1.1** Implement in-memory KG store (nodes dict, edges dict, adjacency index)
- [x] **2.1.2** CRUD operations: `add_node`, `add_edge`, `get_node`, `get_node_by_name`, `get_edge`
- [x] **2.1.3** Edge queries: `get_edges_from`, `get_edges_to`, `get_edges_between`
- [x] **2.1.4** `update_node`, `update_edge_confidence`, `mark_edge_falsified`
- [x] **2.1.5** Subgraph extraction: `get_subgraph(center_id, hops)` via BFS
- [x] **2.1.6** Contradiction detection: `get_contradictions(edge)` — find edges with opposing relations between same nodes
- [x] **2.1.7** Query helpers: `get_recent_edges`, `get_edges_by_hypothesis`, `get_weakest_edges`, `get_orphan_nodes`
- [x] **2.1.8** Graph traversal: `shortest_path`, `get_upstream`, `get_downstream`
- [x] **2.1.9** Stats: `node_count`, `edge_count`, `avg_confidence`, `edges_added_since`
- [x] **2.1.10** Serialization: `to_cytoscape()`, `to_json()`, `to_markdown_summary()`
- [x] **2.1.11** Persistence: `save(session_id)` and `load(session_id)` via Postgres JSON

#### 2.2 Yami/ESM Interface (`backend/world_model/yami.py`)
- [x] **2.2.1** Implement `YamiInterface` wrapping ESM-2 (HuggingFace API or local)
- [x] **2.2.2** `get_logits(sequence)` — fitness scoring
- [x] **2.2.3** `get_embeddings(sequence)` — mean pooled last layer
- [x] **2.2.4** `predict_structure(sequence)` — ESMFold via HuggingFace API
- [x] **2.2.5** Caching layer for ESM predictions (Redis)

#### 2.3 Knowledge Graph Tests
- [x] **2.3.1** Test add/get/update/delete for nodes and edges
- [x] **2.3.2** Test contradiction detection with known opposing edges
- [x] **2.3.3** Test subgraph extraction (verify correct BFS behavior)
- [x] **2.3.4** Test serialization formats (cytoscape, json, markdown)

---

### Phase 3: Tool Integrations

#### 3.1 Base Tool Framework (`backend/integrations/base_tool.py`)
- [x] **3.1.1** `BaseTool` ABC with `httpx.AsyncClient`, connection pooling
- [x] **3.1.2** Redis cache decorator (key = hash of query params, configurable TTL per tool)
- [x] **3.1.3** Token-bucket rate limiter (per tool, backed by Redis)
- [x] **3.1.4** Retry with exponential backoff (3 retries: 1s/2s/4s)
- [x] **3.1.5** Timeout handling (30s default), structured error logging, `ToolError` normalization

#### 3.2 Individual Tools
- [x] **3.2.1** `pubmed.py` — `search()`, `fetch()`, `fetch_abstract()` via NCBI E-utilities
- [x] **3.2.2** `semantic_scholar.py` — `search()`, `get_paper()`, `get_citations()`, `get_references()`
- [x] **3.2.3** `uniprot.py` — `search()`, `get_protein()`, `get_sequence()`, `get_features()`
- [x] **3.2.4** `esm.py` — `get_logits()`, `get_embeddings()`, `predict_structure()` (wraps HuggingFace)
- [x] **3.2.5** `kegg.py` — `get_pathway()`, `get_gene_pathways()`, `get_pathway_genes()`, `search()`
- [x] **3.2.6** `reactome.py` — `get_pathways_for_protein()`, `get_pathway_detail()`, `get_reactions()`
- [x] **3.2.7** `mygene.py` — `get_gene()`, `query()`
- [x] **3.2.8** `drugbank.py` — ChEMBL API: `search_by_target()`, `get_compound()`, `get_target()`, `get_activities()`
- [x] **3.2.9** `clinicaltrials.py` — `search()`, `get_trial()` via ClinicalTrials.gov v2 API
- [x] **3.2.10** `slack.py` — `post_hitl_request()`, `wait_for_response()`, `post_update()` via slack-bolt

#### 3.3 Tool Tests
- [x] **3.3.1** Unit tests for each tool: mock HTTP responses (use `responses` library), verify parsing
- [x] **3.3.2** Test caching behavior: verify cache hit/miss, TTL expiry
- [x] **3.3.3** Test rate limiter: verify backpressure when limit exceeded
- [x] **3.3.4** Test retry logic: verify exponential backoff on transient failures

---

### Phase 4: Agents

#### 4.1 Base Agent (`backend/agents/base.py`)
- [x] **4.1.1** Implement `BaseAgent` ABC with: `agent_id`, `template`, `llm`, `kg`, `yami`, `tools`, `audit`
- [x] **4.1.2** `query_llm()` — wraps LLM call with KG subgraph injection, audit logging, token tracking
- [x] **4.1.3** `query_yami()` — wraps Yami call with audit logging, error handling, caching
- [x] **4.1.4** `write_node()` / `write_edge()` — write to KG with audit trail, auto-trigger contradiction check
- [x] **4.1.5** Default `falsify()` — for each edge, ask LLM for counter-evidence, search tools, adjust confidence
- [x] **4.1.6** `get_uncertainty()` — compute `UncertaintyVector` from agent state

#### 4.2 Specialized Agents (`backend/agents/`)
- [x] **4.2.1** `literature_analyst.py` — Tools: pubmed, semantic_scholar. Finds papers, extracts claims as KG edges.
- [x] **4.2.2** `protein_engineer.py` — Tools: uniprot, esm. Fetches protein data, predicts structure, writes PROTEIN nodes.
- [x] **4.2.3** `genomics_mapper.py` — Tools: mygene, kegg. Maps genes to pathways, writes GENE/PATHWAY nodes.
- [x] **4.2.4** `pathway_analyst.py` — Tools: kegg, reactome. Deep pathway analysis, writes PATHWAY/MECHANISM edges.
- [x] **4.2.5** `drug_hunter.py` — Tools: drugbank/chembl, clinicaltrials. Finds drugs/compounds targeting entities.
- [x] **4.2.6** `clinical_analyst.py` — Tools: clinicaltrials, pubmed. Searches trials, reports outcomes/failures.
- [x] **4.2.7** `scientific_critic.py` — Tools: pubmed, semantic_scholar. Iterates `get_recent_edges()`, actively tries to disprove each. Only modifies confidence + adds EVIDENCE_AGAINST.
- [x] **4.2.8** `experiment_designer.py` — No tools (reasoning-only). Proposes experiments to resolve KG uncertainties. Writes EXPERIMENT nodes.

#### 4.3 Agent Template Definitions
- [x] **4.3.1** Define system prompts for all 8 agent types
- [x] **4.3.2** Define tool permissions, KG write permissions, and falsification protocols per agent

#### 4.4 Agent Tests
- [x] **4.4.1** Test each agent with mocked tools + LLM: verify correct KG mutations
- [x] **4.4.2** Test falsification flow: agent receives edges → searches counter-evidence → adjusts confidence
- [x] **4.4.3** Test uncertainty vector computation

---

### Phase 5: Orchestrator

#### 5.1 Hypothesis Tree / MCTS (`backend/orchestrator/hypothesis_tree.py`)
- [x] **5.1.1** `HypothesisTree` class with root node, nodes dict
- [x] **5.1.2** `select_next()` — UCB1 selection from root to leaf (exploration constant = √2)
- [x] **5.1.3** `backpropagate(node_id, info_gain)` — walk to root, update visit_count + total_info_gain
- [x] **5.1.4** `expand(parent_id, new_hypotheses)` — add child nodes
- [x] **5.1.5** `prune(node_id, reason)` — mark falsified/low-confidence
- [x] **5.1.6** `get_best_path()` — root to highest-confidence leaf
- [x] **5.1.7** `get_tree_state()` — serialize for frontend visualization

#### 5.2 Swarm Composer (`backend/orchestrator/swarm_composer.py`)
- [x] **5.2.1** `select_agents()` — given query + hypotheses + template library, call Claude to select agent types
- [x] **5.2.2** Always include `scientific_critic` (non-negotiable)
- [x] **5.2.3** Instantiate selected agents with their tools, KG access, Yami access

#### 5.3 Uncertainty + HITL (`backend/orchestrator/uncertainty.py`)
- [x] **5.3.1** `aggregate_uncertainty()` — combine vectors from multiple agents
- [x] **5.3.2** `should_trigger_hitl()` — check composite threshold + critical path + unresolved contradictions
- [x] **5.3.3** `compose_slack_message()` — clear, actionable Slack message with context and options

#### 5.4 Main Orchestrator Loop (`backend/orchestrator/research_loop.py`)
- [x] **5.4.1** INITIALIZE: seed KG with initial entities (LLM call), generate hypotheses, build HypothesisTree
- [x] **5.4.2** COMPOSE SWARM: call `swarm_composer.select_agents()`, instantiate agents
- [x] **5.4.3** MCTS LOOP — SELECT: `hypothesis_tree.select_next()`
- [x] **5.4.4** MCTS LOOP — DISPATCH: generate per-agent tasks via LLM, submit to Celery
- [x] **5.4.5** MCTS LOOP — EXECUTE: await agent results
- [x] **5.4.6** MCTS LOOP — CRITIQUE: run `scientific_critic` on new edges
- [x] **5.4.7** MCTS LOOP — EVALUATE: compute info gain (new edges + confidence changes + contradictions)
- [x] **5.4.8** MCTS LOOP — BACKPROPAGATE: update hypothesis tree
- [x] **5.4.9** MCTS LOOP — CHECK HITL: aggregate uncertainty → Slack if triggered → inject response
- [x] **5.4.10** MCTS LOOP — CHECK TERMINATION: confidence threshold, all pruned, or budget exhausted
- [x] **5.4.11** COMPILE RESULTS: best hypothesis path, supporting KG, generate report, return `ResearchResult`

#### 5.5 Orchestrator Tests
- [x] **5.5.1** Test UCB1 selection correctness with known visit counts / info gains
- [x] **5.5.2** Test backpropagation (verify visit counts propagate to root)
- [x] **5.5.3** Test pruning and best-path extraction
- [x] **5.5.4** Test swarm composition (mock LLM, verify critic always included)
- [x] **5.5.5** Test HITL trigger logic with various uncertainty vectors
- [x] **5.5.6** Integration test: mock agents, verify full MCTS loop produces expected KG state

---

### Phase 6: API Layer

#### 6.1 FastAPI App (`backend/api/`)
- [x] **6.1.1** `main.py` — FastAPI app factory with lifespan, CORS, middleware registration
- [x] **6.1.2** `middleware.py` — API key auth (X-API-Key header), request ID injection, timing
- [x] **6.1.3** `deps.py` — Dependency injection: DB session, Redis, KG instance, config

#### 6.2 Routes
- [x] **6.2.1** `routes/research.py` — `POST /research`, `GET /research/{id}`, `GET /research/{id}/result`, `GET /research` (paginated), `POST /research/{id}/cancel`, `POST /research/{id}/feedback`
- [x] **6.2.2** `routes/graph.py` — `GET /research/{id}/graph` (cytoscape/json/summary), `GET .../subgraph`, `GET .../nodes`, `GET .../edges`, `GET .../contradictions`, `GET .../stats`
- [x] **6.2.3** `routes/agents.py` — `GET /research/{id}/agents`, `GET .../agents/{agent_id}/log`, `GET .../agents/{agent_id}/result`
- [x] **6.2.4** `routes/hypothesis.py` — `GET /research/{id}/hypotheses`, `GET .../{node_id}`, `GET .../best`
- [x] **6.2.5** `routes/health.py` — `GET /health`, `GET /health/ready`, `GET /templates`
- [x] **6.2.6** `websocket.py` — WebSocket endpoint at `/research/{id}/ws`, streams all event types

#### 6.3 Celery Workers (`backend/workers/`)
- [x] **6.3.1** `celery_app.py` — Celery config with Redis broker, result backend
- [x] **6.3.2** `tasks.py` — `run_research` task (wraps orchestrator), `run_agent` task (real impl on `feat/production-mode`)

#### 6.4 Report Generator (`backend/report/generator.py`)
- [x] **6.4.1** KG + hypothesis tree → structured markdown: Executive Summary, Evidence Map, Competing Hypotheses, Key Uncertainties, Recommended Experiments, Audit Trail
- [x] **6.4.2** Each claim links to supporting KG edge ID

#### 6.5 API Tests
- [x] **6.5.1** Test all research CRUD endpoints
- [x] **6.5.2** Test graph endpoints with seeded KG data
- [x] **6.5.3** Test WebSocket event streaming (connect, verify event types)
- [x] **6.5.4** Test auth middleware (missing key, invalid key, valid key)

---

### Phase 7: Frontend

#### 7.1 App Shell + Layout
- [x] **7.1.1** `app/layout.tsx` — Root layout with navigation, theme
- [x] **7.1.2** `lib/api.ts` — Backend API client (typed fetch wrapper)
- [x] **7.1.3** `lib/websocket.ts` — WebSocket connection manager (auto-reconnect, typed events)

#### 7.2 Dashboard (`app/page.tsx`)
- [x] **7.2.1** New research form (text input + optional config accordion)
- [x] **7.2.2** Recent research sessions list with status badges
- [x] **7.2.3** Quick stats (total runs, active runs)

#### 7.3 Research Detail (`app/research/[id]/page.tsx`)
- [x] **7.3.1** Header: query text, status, duration, swarm composition
- [x] **7.3.2** Live swarm activity feed (WebSocket-driven, `live-feed.tsx`)
- [x] **7.3.3** Hypothesis tree mini-view (`hypothesis-tree.tsx`)
- [x] **7.3.4** Quick KG stats panel
- [x] **7.3.5** Action buttons: View Graph, View Report, Cancel

#### 7.4 KG Visualization (`app/research/[id]/graph/page.tsx`)
- [x] **7.4.1** `knowledge-graph.tsx` — Cytoscape.js React wrapper with Cola.js layout
- [x] **7.4.2** Node coloring by type (Protein=#4A90D9, Gene=#6B5CE7, Disease=#E74C3C, etc.)
- [x] **7.4.3** Edge styling: thickness = confidence, color = green (supporting) / red (contradicting)
- [x] **7.4.4** Click interactions: node → side panel (`node-detail-panel.tsx`); edge → side panel
- [x] **7.4.5** Filters: by node type, hypothesis branch, confidence threshold, agent
- [x] **7.4.6** Search: find node by name

#### 7.5 Report View (`app/research/[id]/report/page.tsx`)
- [x] **7.5.1** Report renderer — Markdown → React display
- [x] **7.5.2** Download as markdown or PDF
- [x] **7.5.3** Claims linked to KG edges

#### 7.6 Shared Components
- [x] **7.6.1** `badge.tsx` — Status indicators for research/agents
- [x] **7.6.2** `research-form.tsx` — Query input with advanced config
- [x] **7.6.3** Set up shadcn/ui component library (card, badge, tabs, tooltip, spinner, progress-bar, stat-card, empty-state)

---

### Phase 8: Integration + Polish

#### 8.1 End-to-End Integration
- [x] **8.1.1** Verify full flow: submit query → agents run → KG builds → report generated → frontend displays — `scripts/dry_run.py` (843 lines) + `scripts/validate_environment.py` (643 lines)
- [x] **8.1.2** Test with canonical query: "What are the therapeutic approaches for B7-H3 in NSCLC?" — `scripts/dry_run.py` line 168, `scripts/seed_demo.py`
- [x] **8.1.3** Verify WebSocket events stream correctly to frontend — `backend/api/websocket.py` (175 lines)

#### 8.2 Scripts
- [x] **8.2.1** `scripts/seed_demo.py` — Seed a demo research session with pre-built KG for frontend development (782 lines)
- [x] **8.2.2** `scripts/benchmark_vs_vbiotech.py` — Run B7-H3 query, compare metrics vs Virtual Biotech paper
- [x] **8.2.3** `scripts/export_graph.py` — Export KG to JSON, Cytoscape, Markdown, GraphML

#### 8.3 Production Hardening
- [x] **8.3.1** Global timeout per research session (default 30 min) — `session_timeout_seconds` in `ResearchConfig`, `asyncio.wait_for` in `research_loop.py`
- [x] **8.3.2** Celery task retries (max 2) for transient failures — `workers/tasks.py` with `max_retries=2, acks_late=True`
- [x] **8.3.3** Per-tool rate limiters verified working — `base_tool.py` Redis-backed token-bucket with Lua script
- [x] **8.3.4** LLM call budget per session — `TokenBudgetManager` with hierarchical budget distribution + agent-level enforcement
- [x] **8.3.5** CORS configured for frontend origin only — `frontend_url` setting in `config.py`, used in `CORSMiddleware`
- [x] **8.3.6** Docker containers run as non-root — `yohas` user in both `backend/Dockerfile` and `frontend/Dockerfile`
- [x] **8.3.7** structlog + request ID propagation verified end-to-end — `core/audit.py` + `RequestIDMiddleware` in `middleware.py`

#### 8.4 Documentation
- [x] **8.4.1** README.md with setup instructions, architecture overview, quickstart (300 lines)
- [x] **8.4.2** API documentation — `docs/api.md` (710 lines) + FastAPI auto-generated OpenAPI at `/docs` and `/redoc`
- [x] **8.4.3** Agent template documentation — `docs/agent-templates.md` (454 lines)

---

## Task Summary

| Phase | Area | Done | Total | Status |
|-------|------|------|-------|--------|
| 1 | Foundation | 20 | 20 | ✅ Complete |
| 2 | Knowledge Graph | 16 | 16 | ✅ Complete |
| 3 | Tool Integrations | 14 | 14 | ✅ Complete |
| 4 | Agents | 16 | 16 | ✅ Complete |
| 5 | Orchestrator | 17 | 17 | ✅ Complete |
| 6 | API Layer | 12 | 12 | ✅ Complete (`run_agent` implemented on `feat/production-mode`) |
| 7 | Frontend | 16 | 16 | ✅ Complete |
| 8 | Integration + Polish | 13 | 13 | ✅ Complete |
| **Total** | | **124** | **124** | **✅ 100% complete** |
