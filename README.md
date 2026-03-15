# YOHAS 3.0

**Your Own Hypothesis-driven Agentic Scientist** — an autonomous biomedical research platform that spawns AI agent swarms to build knowledge graphs, explore hypotheses via MCTS, self-falsify findings, and produce structured research reports.

<!-- badges -->
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![TypeScript](https://img.shields.io/badge/typescript-5.x-blue)
![Tests](https://img.shields.io/badge/tests-501%20passing-brightgreen)
![License](https://img.shields.io/badge/license-proprietary-lightgrey)

---

## Overview

YOHAS 3.0 takes a biomedical research question (e.g., *"What are the therapeutic approaches for B7-H3 in NSCLC?"*) and autonomously investigates it using a swarm of specialized AI agents coordinated by Monte Carlo Tree Search.

**Core differentiators:**

| Capability | Description |
|---|---|
| **MCTS Hypothesis Tree** | UCB1-guided exploration of hypothesis space — select, expand, evaluate, backpropagate |
| **Agent Swarms** | 8+ specialized agents (literature, protein, genomics, pathway, drug, clinical, critic, experiment) dynamically composed per query |
| **Self-Falsification** | Every agent runs `falsify()` on edges it creates; a dedicated `scientific_critic` actively disproves claims |
| **Knowledge Graph** | In-memory KG with provenance tracking — every node/edge has `agent_id`, `hypothesis_branch`, and evidence sources |
| **Human-in-the-Loop** | Uncertainty-triggered Slack escalation when agents hit unresolvable contradictions |
| **RL Training Pipeline** | Trajectory collection → SFT → GRPO for continuous agent improvement |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js 14)                        │
│  Dashboard │ Research Detail │ KG Visualization │ Report View       │
│  Port 3000 │ WebSocket live feed │ Cytoscape.js + D3.js            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ REST + WebSocket
┌──────────────────────────────▼──────────────────────────────────────┐
│                        API Layer (FastAPI)                           │
│  Port 8000 │ Routes: research, graph, agents, hypothesis, health    │
│  Middleware: API key auth, CORS, request ID propagation             │
│  WebSocket: /api/v1/research/{id}/ws                                │
└───────┬───────────────────────┬──────────────────────┬──────────────┘
        │                       │                      │
┌───────▼───────┐   ┌──────────▼──────────┐   ┌──────▼───────────┐
│  PostgreSQL   │   │       Redis          │   │  Celery Workers  │
│  Port 5432    │   │  Port 6379           │   │  Agent execution │
│  Sessions,    │   │  Job queue           │   │  Concurrency: 4  │
│  KG state,    │   │  API cache           │   │                  │
│  audit logs   │   │  Rate limit tokens   │   │                  │
└───────────────┘   └──────────────────────┘   └────────┬─────────┘
                                                        │
        ┌───────────────────────────────────────────────▼─────────┐
        │                    Agent Swarm                           │
        │  literature_analyst │ protein_engineer │ genomics_mapper │
        │  pathway_analyst │ drug_hunter │ clinical_analyst        │
        │  scientific_critic │ experiment_designer                 │
        │  + tool_creator (dynamic) │ sub-agent spawning           │
        └──────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────▼──────────────────────────────────────────┐
        │                  External Integrations                   │
        │  PubMed │ Semantic Scholar │ UniProt │ KEGG │ Reactome  │
        │  MyGene.info │ ChEMBL │ ClinicalTrials.gov │ ESM-2     │
        │  Slack (HITL) │ Anthropic Claude (LLM)                  │
        └─────────────────────────────────────────────────────────┘
```

### Key Modules

| Module | Path | Responsibility |
|--------|------|----------------|
| **Core** | `backend/core/` | Pydantic models, interfaces/protocols, config, LLM wrapper, exceptions, audit |
| **World Model** | `backend/world_model/` | In-memory knowledge graph + Yami/ESM-2 protein interface |
| **Agents** | `backend/agents/` | Base agent + 8 specialized agents + tool retriever + tool creator |
| **Orchestrator** | `backend/orchestrator/` | MCTS hypothesis tree, swarm composer, token budget, uncertainty/HITL, research loop |
| **Integrations** | `backend/integrations/` | External API clients with Redis caching, rate limiting, retry |
| **API** | `backend/api/` | FastAPI routes, WebSocket, middleware, dependency injection |
| **Workers** | `backend/workers/` | Celery task definitions for agent execution |
| **Report** | `backend/report/` | KG + hypothesis tree → structured markdown report |
| **RL** | `backend/rl/` | Trajectory collection, SFT pipeline, GRPO training, model serving |
| **Frontend** | `frontend/` | Next.js 14 app — dashboard, live feed, KG viz, report view |

---

## Tech Stack

**Backend:**
- Python 3.11+ · FastAPI · Celery · SQLAlchemy (async) · Alembic · Pydantic v2
- Claude API (Anthropic) · ESM-2 / ESMFold (HuggingFace)
- PostgreSQL 16 · Redis 7

**Frontend:**
- TypeScript · Next.js 14 · Cytoscape.js · D3.js · shadcn/ui · Tailwind CSS

**Infrastructure:**
- Docker Compose (5 services: api, worker, redis, postgres, frontend)
- Ruff (Python linting) · ESLint (TS linting) · pytest · Vitest

**External APIs:**
PubMed · Semantic Scholar · UniProt · KEGG · Reactome · MyGene.info · ChEMBL · ClinicalTrials.gov · Slack

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- An [Anthropic API key](https://console.anthropic.com/)

### Setup

```bash
# Clone
git clone <repo-url> && cd lumi-mvp

# Configure environment
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY (required), optionally HF_API_TOKEN, NCBI_API_KEY, S2_API_KEY

# Build and run
make build
make dev
```

### Access

| Service | URL |
|---------|-----|
| Frontend | [http://localhost:3000](http://localhost:3000) |
| API | [http://localhost:8000](http://localhost:8000) |
| API Docs (OpenAPI) | [http://localhost:8000/docs](http://localhost:8000/docs) |
| PostgreSQL | `localhost:5432` |
| Redis | `localhost:6379` |

---

## Development

### Commands

```bash
make build       # Docker compose build
make dev         # Start all services (api:8000, frontend:3000, redis, postgres)
make down        # Stop all services
make test        # pytest (backend) + vitest (frontend)
make lint        # ruff check (backend) + eslint (frontend)
make migrate     # alembic upgrade head
make seed        # Seed demo research session
make clean       # Stop services, remove volumes, clear caches
make benchmark   # Run benchmark suite
make prod        # Production mode (detached)
```

### Project Structure

```
├── backend/
│   ├── core/            # Models, config, LLM wrapper, interfaces, exceptions
│   ├── agents/          # Base agent + 8 specialized + tool retriever/creator
│   ├── orchestrator/    # MCTS tree, swarm composer, token budget, research loop
│   ├── world_model/     # Knowledge graph, Yami/ESM interface
│   ├── integrations/    # External API clients (PubMed, UniProt, KEGG, etc.)
│   ├── api/             # FastAPI app, routes, WebSocket, middleware
│   ├── workers/         # Celery tasks
│   ├── report/          # Report generator (KG → markdown)
│   ├── rl/              # RL pipeline (trajectories, SFT, GRPO, serving)
│   ├── db/              # SQLAlchemy models, session management
│   ├── know_how/        # Protocol docs + retriever
│   ├── data/            # Bio data lake (GO, Reactome, MSigDB, DrugBank)
│   ├── benchmarks/      # Benchmark harness
│   ├── alembic/         # Database migrations
│   └── tests/           # Test suite (501+ tests)
├── frontend/
│   ├── app/             # Next.js pages (dashboard, research, graph, report)
│   ├── components/      # React components (KG viz, live feed, forms)
│   └── lib/             # API client, WebSocket manager, types
├── configs/             # Training config presets
├── scripts/             # Seed, benchmark, export, validation scripts
├── docker/              # Dockerfiles, sandbox configs
├── infra/               # MCP tool infrastructure
├── docker-compose.yml
├── docker-compose.prod.yml
├── docker-compose.benchmark.yml
└── Makefile
```

---

## How It Works

```
Query → Seed KG → Generate Hypotheses → MCTS Loop → Report
```

1. **Submit** — User posts a research question via the frontend or API
2. **Seed** — Orchestrator extracts initial entities, seeds the knowledge graph
3. **Hypothesize** — LLM generates candidate hypotheses, builds a hypothesis tree
4. **Compose Swarm** — Swarm composer dynamically selects agents + tools per hypothesis (critic always included)
5. **MCTS Loop** — For each iteration:
   - **Select** — UCB1 picks the most promising hypothesis branch
   - **Dispatch** — Agents receive tasks, query external APIs, write to KG
   - **Critique** — Scientific critic reviews new edges, searches for counter-evidence
   - **Evaluate** — Compute information gain (new edges, confidence changes, contradictions)
   - **Backpropagate** — Update hypothesis tree scores
   - **HITL Check** — If uncertainty exceeds threshold, escalate to human via Slack
   - **Terminate** — When confidence threshold met, all branches pruned, or budget exhausted
6. **Report** — Compile best hypothesis path, supporting evidence, competing hypotheses, and recommended experiments into a structured markdown report
7. **Stream** — All events stream to the frontend via WebSocket throughout the process

---

## API

All endpoints are prefixed with `/api/v1`. Full OpenAPI docs available at `/docs`.

### Core Endpoints

```
POST   /api/v1/research                  # Submit a new research query
GET    /api/v1/research                  # List research sessions (paginated)
GET    /api/v1/research/{id}             # Get session status + metadata
GET    /api/v1/research/{id}/result      # Get completed research result
POST   /api/v1/research/{id}/cancel      # Cancel a running session
POST   /api/v1/research/{id}/feedback    # Submit feedback on results
```

### Knowledge Graph

```
GET    /api/v1/research/{id}/graph       # Full KG (cytoscape/json/summary)
GET    /api/v1/research/{id}/graph/subgraph?center={node}&hops=2
GET    /api/v1/research/{id}/graph/nodes
GET    /api/v1/research/{id}/graph/edges
GET    /api/v1/research/{id}/graph/contradictions
GET    /api/v1/research/{id}/graph/stats
```

### Agents & Hypotheses

```
GET    /api/v1/research/{id}/agents              # List agents in swarm
GET    /api/v1/research/{id}/agents/{agent_id}/log
GET    /api/v1/research/{id}/hypotheses          # Hypothesis tree
GET    /api/v1/research/{id}/hypotheses/best     # Best hypothesis path
```

### WebSocket

```
WS     /api/v1/research/{id}/ws          # Real-time event stream
```

Events: `agent_started`, `agent_completed`, `edge_added`, `hypothesis_updated`, `hitl_requested`, `research_completed`

### Monitoring

```
GET    /monitoring/overview                      # System-wide stats
GET    /monitoring/research/{id}/stats           # Per-session details
GET    /monitoring/research/{id}/agents          # Agent status list
```

---

## Contributing

### Branch Naming

```
feat/module-name        # New features
fix/module-name         # Bug fixes
refactor/module-name    # Refactors
```

### PR Process

1. Branch from `main`
2. PR title format: `[MODULE] description` (e.g., `[agents] Add scientific_critic agent`)
3. All tests must pass (`make test`)
4. Linting must pass (`make lint`)
5. Run CodeRabbit review before push

### Coding Rules

- All Pydantic models live in `core/models.py`
- Every external API call goes through `BaseTool` (caching, rate limiting, retry)
- Every LLM call goes through `core/llm.py` (token tracking, audit logging)
- Every KG write includes `agent_id` and `hypothesis_branch`
- `scientific_critic` is always included in swarm composition
- Never skip self-falsification — every agent runs `falsify()` on edges it creates
- Never write KG nodes/edges without evidence sources

---

## License

Proprietary. All rights reserved.
