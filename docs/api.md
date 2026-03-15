# YOHAS 3.0 — Supplemental API Documentation

> **Auto-generated OpenAPI docs** are available at [`/docs`](http://localhost:8000/docs) (Swagger UI) and [`/redoc`](http://localhost:8000/redoc) (ReDoc) when the server is running. This document provides supplemental context, architecture notes, and usage patterns not captured by the OpenAPI spec.

---

## Architecture Overview

The YOHAS 3.0 API is a **FastAPI** application defined in `backend/api/main.py`. The application factory `create_app()` assembles:

- **Routers** — one per domain (`research`, `graph`, `agents`, `hypothesis`, `monitoring`, `health`) plus a WebSocket router, all mounted under the `/api/v1` prefix.
- **Middleware stack** (applied bottom-to-top):
  1. `RequestIDMiddleware` — injects `X-Request-ID` into every request and structlog context.
  2. `TimingMiddleware` — logs request duration and sets `X-Response-Time-Ms` header.
  3. `APIKeyAuthMiddleware` — validates `X-API-Key` header (see [Authentication](#authentication)).
  4. `CORSMiddleware` — allows the configured `FRONTEND_URL` origin.
- **Lifespan** — configures audit logging on startup.

**Base URL:** `http://localhost:8000/api/v1`

In production, the app is served via `uvicorn api.main:app` and long-running research is dispatched to **Celery** workers. In development, research runs as asyncio background tasks.

---

## Authentication

All endpoints (except public paths) require an API key passed via the `X-API-Key` HTTP header.

| Setting | Env Var | Default |
|---------|---------|---------|
| API key | `API_KEY` | `dev-api-key-change-me` |

**Public paths** (no auth required):
- `GET /api/v1/health`
- `GET /api/v1/health/ready`
- `GET /docs`, `GET /redoc`, `GET /openapi.json`
- WebSocket connections (auth is per-connection, not via middleware)

**Example:**

```bash
curl -H "X-API-Key: dev-api-key-change-me" http://localhost:8000/api/v1/health
```

**Error on missing/invalid key:**

```json
{"error": "Missing API key", "detail": "Set X-API-Key header"}
```

```json
{"error": "Invalid API key"}
```

---

## Error Format

All error responses follow the standard FastAPI/HTTP exception format:

```json
{
  "detail": "Research session not found"
}
```

Some domain errors include an `error_code` field:

```json
{
  "detail": "Unknown agent type: foo",
  "error_code": "UNKNOWN_AGENT_TYPE"
}
```

HTTP status codes used:
- `201` — resource created (POST /research)
- `400` — bad request / validation error
- `401` — missing or invalid API key
- `404` — resource not found
- `409` — conflict (e.g., research not yet completed, cannot cancel)

---

## Endpoint Reference

### Health

**Source:** `backend/api/routes/health.py`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Basic liveness check |
| `GET` | `/health/ready` | Readiness probe (checks Redis, Postgres, Celery) |
| `GET` | `/templates` | List all registered agent templates |

#### `GET /health`

```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime_seconds": 42
}
```

#### `GET /health/ready`

```json
{
  "redis": true,
  "postgres": true,
  "celery": true
}
```

#### `GET /templates`

Returns an array of agent template summaries:

```json
[
  {
    "agent_type": "literature_analyst",
    "display_name": "Literature Analyst",
    "description": "Searches biomedical literature...",
    "tools": ["pubmed", "semantic_scholar"],
    "requires_yami": false
  }
]
```

---

### Research

**Source:** `backend/api/routes/research.py`  
**Prefix:** `/research`

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/research` | Create a new research session |
| `GET` | `/research` | List research sessions (paginated, filterable) |
| `GET` | `/research/{research_id}` | Get full session state |
| `GET` | `/research/{research_id}/result` | Get final result (completed sessions only) |
| `POST` | `/research/{research_id}/cancel` | Cancel a running session |
| `POST` | `/research/{research_id}/feedback` | Submit human feedback on a KG edge |

#### `POST /research` — Create Research Session

**Status:** `201 Created`

**Request body:**

```json
{
  "query": "What are the molecular mechanisms linking BRCA1 mutations to triple-negative breast cancer?",
  "config": {
    "max_hypothesis_depth": 5,
    "max_mcts_iterations": 15,
    "max_agents": 8,
    "max_agents_per_swarm": 5,
    "confidence_threshold": 0.7,
    "hitl_uncertainty_threshold": 0.6,
    "hitl_timeout_seconds": 600,
    "max_llm_calls_per_agent": 20,
    "agent_types": null,
    "enable_falsification": true,
    "enable_hitl": true,
    "slack_channel_id": null,
    "max_concurrent_agents": 50,
    "max_total_agents": 10000,
    "max_hypothesis_breadth": 30,
    "agent_token_budget": 50000,
    "session_token_budget": 10000000,
    "session_timeout_seconds": 1800
  }
}
```

All `config` fields are optional and have sensible defaults (see `ResearchConfig` in `core/models.py`). The minimal request is:

```json
{"query": "Your research question here"}
```

**Response:**

```json
{
  "research_id": "a1b2c3d4-...",
  "status": "INITIALIZING"
}
```

In **production** mode (`ENVIRONMENT=production`), the research is dispatched to a Celery worker. In **development** mode, it runs as an asyncio background task.

#### `GET /research` — List Sessions

**Query parameters:**
- `status` (optional) — filter by session status (`INITIALIZING`, `RUNNING`, `WAITING_HITL`, `COMPLETED`, `FAILED`, `CANCELLED`)
- `offset` (default: `0`) — pagination offset
- `limit` (default: `20`) — page size

**Response:**

```json
{
  "items": [ /* ResearchSession objects */ ],
  "total": 5,
  "offset": 0,
  "limit": 20
}
```

#### `GET /research/{research_id}` — Get Session

Returns the full `ResearchSession` model as JSON.

#### `GET /research/{research_id}/result` — Get Result

Returns the `ResearchResult` model. Only available when `status == COMPLETED`.

**Error:** `409 Conflict` if research is not yet completed.

**Response includes:**
- `best_hypothesis` — the top-ranked hypothesis node
- `hypothesis_ranking` — all hypotheses ranked by info gain
- `key_findings` — high-confidence KG edges
- `contradictions` — contradicting edge pairs
- `recommended_experiments` — suggested next experiments
- `report_markdown` — full research report
- `graph_snapshot` — serialized KG state
- `screening` — biosecurity screening result (if applicable)

#### `POST /research/{research_id}/cancel`

Cancels a running session. Only works for sessions in `RUNNING`, `INITIALIZING`, or `WAITING_HITL` state.

**Error:** `409 Conflict` if session is in a non-cancellable state.

#### `POST /research/{research_id}/feedback` — Human Feedback

Submit feedback on a specific KG edge (human-in-the-loop).

**Request body:**

```json
{
  "edge_id": "uuid-of-edge",
  "feedback": "agree",
  "confidence_override": 0.9
}
```

- `feedback` — `"agree"`, `"disagree"`, or free-text
- `confidence_override` (optional) — if provided, updates the edge confidence and adds a `HUMAN_INPUT` evidence source

---

### Knowledge Graph

**Source:** `backend/api/routes/graph.py`  
**Prefix:** `/research/{research_id}/graph`

All graph endpoints are scoped to a specific research session.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/research/{research_id}/graph` | Full graph (cytoscape, json, or summary format) |
| `GET` | `/research/{research_id}/graph/subgraph` | Subgraph around a center node |
| `GET` | `/research/{research_id}/graph/nodes` | List nodes (filterable by type, paginated) |
| `GET` | `/research/{research_id}/graph/edges` | List edges (filterable by source/target/relation, paginated) |
| `GET` | `/research/{research_id}/graph/contradictions` | List contradicting edge pairs |
| `GET` | `/research/{research_id}/graph/stats` | Graph statistics |

#### `GET /research/{research_id}/graph`

**Query parameters:**
- `format` — `cytoscape` (default), `json`, or `summary`

**Response (cytoscape format):**

```json
{
  "format": "cytoscape",
  "data": {
    "elements": {
      "nodes": [{"data": {"id": "...", "label": "BRCA1", "type": "GENE"}}],
      "edges": [{"data": {"source": "...", "target": "...", "relation": "ASSOCIATED_WITH"}}]
    }
  }
}
```

#### `GET /research/{research_id}/graph/subgraph`

**Query parameters:**
- `center` (required) — center node ID
- `hops` (default: `2`, range: 1–5) — traversal depth

#### `GET /research/{research_id}/graph/nodes`

**Query parameters:**
- `type` (optional) — filter by `NodeType` (e.g., `GENE`, `PROTEIN`, `DISEASE`, `PATHWAY`, `DRUG`, etc.)
- `offset`, `limit` — pagination (default limit: 50)

#### `GET /research/{research_id}/graph/edges`

**Query parameters:**
- `source` (optional) — filter by source node ID
- `target` (optional) — filter by target node ID
- `relation` (optional) — filter by `EdgeRelationType`
- `offset`, `limit` — pagination (default limit: 50)

#### `GET /research/{research_id}/graph/stats`

```json
{
  "node_count": 42,
  "edge_count": 87,
  "avg_confidence": 0.72,
  "type_distribution": {
    "GENE": 12,
    "PROTEIN": 8,
    "DISEASE": 5,
    "PATHWAY": 10,
    "DRUG": 7
  }
}
```

---

### Agents

**Source:** `backend/api/routes/agents.py`  
**Prefix:** `/research/{research_id}/agents`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/research/{research_id}/agents` | List agents that participated in a session |
| `GET` | `/research/{research_id}/agents/{agent_id}/log` | Full audit trail for an agent |
| `GET` | `/research/{research_id}/agents/{agent_id}/result` | Full `AgentResult` for an agent |

#### `GET /research/{research_id}/agents`

```json
{
  "agents": [
    {
      "agent_id": "uuid",
      "agent_type": "literature_analyst",
      "task_id": "uuid",
      "success": true,
      "nodes_added": 5,
      "edges_added": 8,
      "duration_ms": 12340,
      "summary": "Found 5 genes associated with..."
    }
  ],
  "count": 3
}
```

#### `GET /research/{research_id}/agents/{agent_id}/log`

Returns the full audit trail:

```json
{
  "agent_id": "uuid",
  "agent_type": "literature_analyst",
  "reasoning_trace": "Step 1: Searched PubMed for...",
  "falsification_results": [ /* FalsificationResult objects */ ],
  "errors": [],
  "token_usage": {"total_tokens": 15000},
  "duration_ms": 12340
}
```

#### `GET /research/{research_id}/agents/{agent_id}/result`

Returns the full `AgentResult` model including `nodes_added`, `edges_added`, `turns`, `uncertainty`, `sub_agent_results`, etc.

---

### Hypotheses

**Source:** `backend/api/routes/hypothesis.py`  
**Prefix:** `/research/{research_id}/hypotheses`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/research/{research_id}/hypotheses` | Full hypothesis tree state |
| `GET` | `/research/{research_id}/hypotheses/best` | Best hypothesis + ranking |
| `GET` | `/research/{research_id}/hypotheses/{node_id}` | Single hypothesis node with edge details |

#### `GET /research/{research_id}/hypotheses`

```json
{
  "root_id": "uuid",
  "nodes": [
    {
      "id": "uuid",
      "parent_id": null,
      "hypothesis": "BRCA1 loss drives TNBC via homologous recombination deficiency",
      "rationale": "...",
      "depth": 0,
      "visit_count": 5,
      "total_info_gain": 2.3,
      "avg_info_gain": 0.46,
      "ucb_score": 1.2,
      "children": ["uuid-1", "uuid-2"],
      "status": "EXPLORING",
      "confidence": 0.75
    }
  ],
  "total_visits": 15,
  "node_count": 8
}
```

#### `GET /research/{research_id}/hypotheses/best`

```json
{
  "best": { /* HypothesisNode */ },
  "ranking": [ /* HypothesisNode[] sorted by info gain */ ]
}
```

#### `GET /research/{research_id}/hypotheses/{node_id}`

Returns a single hypothesis node enriched with resolved edge details:

```json
{
  "id": "uuid",
  "hypothesis": "...",
  "supporting_edge_details": [ /* KGEdge objects */ ],
  "contradicting_edge_details": [ /* KGEdge objects */ ]
}
```

---

### Monitoring

**Source:** `backend/api/routes/monitoring.py`  
**Prefix:** `/monitoring`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/monitoring/overview` | Global platform overview |
| `GET` | `/monitoring/research/{research_id}/stats` | Per-session real-time stats |
| `GET` | `/monitoring/research/{research_id}/agents` | Agent constellation data |
| `GET` | `/monitoring/research/{research_id}/uncertainty` | Uncertainty radar data |

#### `GET /monitoring/overview`

```json
{
  "sessions": {
    "total": 5,
    "active": 2,
    "completed": 2,
    "failed": 1
  },
  "knowledge_graph": {
    "total_nodes": 150,
    "total_edges": 320
  },
  "tokens": {
    "total_used": 500000
  },
  "agents": {
    "total_spawned": 25
  }
}
```

#### `GET /monitoring/research/{research_id}/stats`

Comprehensive real-time stats for a specific session:

```json
{
  "research_id": "uuid",
  "status": "RUNNING",
  "current_iteration": 3,
  "agents": {
    "total_spawned": 8,
    "max_total": 10000,
    "type_counts": {
      "literature_analyst": 3,
      "protein_engineer": 2,
      "drug_hunter": 1,
      "scientific_critic": 2
    }
  },
  "hypothesis_tree": {
    "node_count": 6,
    "total_visits": 12,
    "max_depth": 3,
    "confirmed_count": 1,
    "refuted_count": 1,
    "exploring_count": 2,
    "unexplored_count": 2,
    "best_hypothesis": {
      "id": "uuid",
      "hypothesis": "...",
      "confidence": 0.82,
      "avg_info_gain": 0.65
    }
  },
  "knowledge_graph": {
    "node_count": 42,
    "edge_count": 87,
    "avg_confidence": 0.72
  },
  "uncertainty": { /* trend data */ },
  "tokens": {
    "session_tokens_used": 150000,
    "session_token_budget": 10000000,
    "budget_utilization": 0.015
  }
}
```

#### `GET /monitoring/research/{research_id}/agents`

Returns data shaped for the frontend `agent-constellation.tsx` component:

```json
{
  "research_id": "uuid",
  "agents": [
    {
      "agent_id": "uuid",
      "agent_type": "literature_analyst",
      "status": "COMPLETED",
      "hypothesis_branch": "uuid",
      "task_count": 2,
      "nodes_added": 8,
      "edges_added": 12,
      "tokens_used": 15000
    }
  ],
  "total_spawned": 5
}
```

#### `GET /monitoring/research/{research_id}/uncertainty`

Returns data shaped for the frontend `uncertainty-radar.tsx` component:

```json
{
  "research_id": "uuid",
  "current": {
    "input_ambiguity": 0.1,
    "data_quality": 0.3,
    "reasoning_divergence": 0.2,
    "model_disagreement": 0.15,
    "conflict_uncertainty": 0.25,
    "novelty_uncertainty": 0.1,
    "composite": 0.22,
    "is_critical": false
  },
  "history": [ /* UncertaintyVector[] */ ],
  "trend": "decreasing",
  "hitl_triggered": false,
  "hitl_response_count": 0
}
```

---

## WebSocket

**Source:** `backend/api/websocket.py`  
**Endpoint:** `ws://localhost:8000/api/v1/research/{research_id}/ws`

The WebSocket provides **real-time event streaming** for a research session and accepts **HITL (human-in-the-loop) responses** from the frontend.

### Connection

```javascript
const ws = new WebSocket("ws://localhost:8000/api/v1/research/{research_id}/ws");
```

If the research session does not exist, the connection is closed with code `4004` and reason `"Research session not found"`.

### Server → Client Events

Events are JSON objects with `event_type` and `data` fields:

```json
{
  "event_type": "agent_started",
  "data": { /* event-specific payload */ }
}
```

| Event Type | Description |
|------------|-------------|
| `agent_started` | An agent has begun executing a task |
| `agent_completed` | An agent has finished (includes summary, nodes/edges added) |
| `node_added` | A new node was added to the knowledge graph |
| `edge_added` | A new edge was added to the knowledge graph |
| `edge_falsified` | An edge was falsified or had its confidence adjusted |
| `hypothesis_explored` | A hypothesis node was explored by MCTS |
| `hitl_request` | The system is requesting human input (high uncertainty) |
| `hitl_resolved` | A HITL request was resolved |
| `research_finished` | Research session has ended (`COMPLETED`, `FAILED`, or `CANCELLED`) |
| `error` | An error occurred (e.g., session removed) |
| `pong` | Response to a client `ping` |

#### `hitl_request` Event

```json
{
  "event_type": "hitl_request",
  "data": {
    "hypothesis_id": "uuid",
    "hypothesis": "BRCA1 loss drives TNBC via HR deficiency",
    "uncertainty_composite": 0.72,
    "reason": "High conflict uncertainty between contradicting edges",
    "message": "Please review the following hypothesis...",
    "timeout_seconds": 600
  }
}
```

### Client → Server Messages

Messages are JSON objects with a `type` field:

| Message Type | Description |
|--------------|-------------|
| `hitl_response` | Human feedback in response to a `hitl_request` |
| `ping` | Keepalive ping (server responds with `pong`) |

#### `hitl_response` Message

```json
{
  "type": "hitl_response",
  "response": "I agree with this hypothesis. The HR deficiency mechanism is well-supported by recent PARP inhibitor trial data."
}
```

When a `hitl_response` is received:
1. The response is recorded in the uncertainty tracker
2. If the session was in `WAITING_HITL` status, it resumes to `RUNNING`

### Polling Behavior

The WebSocket server polls the orchestrator for events every ~500ms. When the session reaches a terminal state (`COMPLETED`, `FAILED`, `CANCELLED`), a `research_finished` event is sent and the connection is closed.

---

## Data Models Reference

All models are defined in `backend/core/models.py`. Key models used in API responses:

| Model | Description |
|-------|-------------|
| `ResearchSession` | Top-level session state (id, query, status, config, iteration count, etc.) |
| `ResearchConfig` | Session configuration (MCTS depth, agent limits, token budgets, HITL settings) |
| `ResearchResult` | Final output (best hypothesis, key findings, contradictions, report) |
| `KGNode` | Knowledge graph node (type, name, aliases, properties, confidence, evidence) |
| `KGEdge` | Knowledge graph edge (source, target, relation, confidence, evidence, falsification) |
| `HypothesisNode` | MCTS tree node (hypothesis text, visit count, UCB score, status) |
| `AgentResult` | Agent execution result (nodes/edges added, falsification, uncertainty, turns) |
| `UncertaintyVector` | 6-dimensional uncertainty assessment |
| `FalsificationResult` | Result of an agent's self-falsification attempt |
| `AgentTurn` | Single turn in a multi-turn agent investigation |

### Key Enums

| Enum | Values |
|------|--------|
| `SessionStatus` | `INITIALIZING`, `RUNNING`, `WAITING_HITL`, `COMPLETED`, `FAILED`, `CANCELLED` |
| `AgentType` | `literature_analyst`, `protein_engineer`, `genomics_mapper`, `pathway_analyst`, `drug_hunter`, `clinical_analyst`, `scientific_critic`, `experiment_designer`, `tool_creator` |
| `NodeType` | `PROTEIN`, `GENE`, `DISEASE`, `PATHWAY`, `DRUG`, `CELL_TYPE`, `TISSUE`, `CLINICAL_TRIAL`, `MECHANISM`, `MODALITY`, `SIDE_EFFECT`, `BIOMARKER`, `ORGANISM`, `COMPOUND`, `EXPERIMENT`, `PUBLICATION`, `STRUCTURE` |
| `HypothesisStatus` | `UNEXPLORED`, `EXPLORING`, `EXPLORED`, `PRUNED`, `CONFIRMED`, `REFUTED` |

---

## Response Headers

Every response includes:

| Header | Description |
|--------|-------------|
| `X-Request-ID` | Unique request identifier (pass your own via request header, or auto-generated) |
| `X-Response-Time-Ms` | Server-side processing time in milliseconds |

---

## OpenAPI / Auto-Generated Docs

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI JSON:** [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

These are generated automatically by FastAPI from the route definitions and Pydantic models.
