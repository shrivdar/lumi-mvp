# CLAUDE.md

## Project
YOHAS 3.0 (Your Own Hypothesis-driven Agentic Scientist) — an autonomous biomedical research platform that spawns AI agent swarms to build knowledge graphs, explore hypotheses via MCTS, self-falsify findings, and produce structured research reports. Built for computational biologists and drug discovery researchers.

## Team
- Darsh Shrivastava — Lead, full-stack ownership (backend + frontend + infra)

## Stack
- Python 3.11+ / FastAPI + Celery + SQLAlchemy (async) + Alembic
- TypeScript / Next.js 14 + Cytoscape.js + D3.js + shadcn/ui + Tailwind
- Claude API (LLM), ESM-2/ESMFold (HuggingFace), PostgreSQL 16, Redis 7, Docker Compose
- APIs: PubMed, Semantic Scholar, UniProt, KEGG, Reactome, ChEMBL, ClinicalTrials.gov, Slack

## Architecture
```
backend/core/        — Data models (Pydantic), interfaces (ABCs/Protocols), config, LLM wrapper, exceptions
backend/world_model/ — In-memory knowledge graph + Yami/ESM interface
backend/agents/      — Base agent + 8 specialized agents (literature, protein, genomics, pathway, drug, clinical, critic, experiment)
backend/orchestrator/ — MCTS hypothesis tree, swarm composer, uncertainty/HITL, main research loop
backend/integrations/ — External API clients with caching (Redis) + rate limiting + retry
backend/api/         — FastAPI routes + WebSocket; backend/workers/ — Celery tasks
frontend/app/        — Dashboard, research detail (live feed), KG visualization, report view
```

## Common Commands
```
build:    make build            # docker compose build
test:     make test             # pytest backend + vitest frontend
lint:     make lint             # ruff check + eslint
run dev:  make dev              # docker compose up (api:8000, frontend:3000, redis:6379, pg:5432)
migrate:  make migrate          # alembic upgrade head
seed:     make seed             # python scripts/seed_demo.py
```

## Git Workflow
- Branch naming: `feat/module-name`, `fix/module-name`, `refactor/module-name`
- Worktrees per parallel agent (Superset)
- Always run `/coderabbit:review uncommitted` before push
- PR title format: `[MODULE] description` (e.g. `[agents] Add scientific_critic agent`)
- Merge requires: tests passing + CodeRabbit clear

## Coding Rules
- All Pydantic models in `core/models.py` — agents, orchestrator, API all import from there
- Every external API call goes through `BaseTool` (caching, rate limiting, retry, error normalization)
- Every LLM call goes through `core/llm.py` (token tracking, audit logging, KG context injection)
- Every KG write includes `agent_id` and `hypothesis_branch` — no anonymous mutations
- `scientific_critic` is always included in swarm composition — non-negotiable

## Anti-Patterns (Never Do)
- Never store API keys in code — env vars only, loaded via `core/config.py`
- Never let one agent crash kill the research session — isolate agent failures
- Never skip self-falsification — every agent runs `falsify()` on edges it creates
- Never call external APIs directly — always go through the tool in `integrations/`
- Never write KG nodes/edges without evidence sources — every mutation needs provenance

## Verification
- Run `pytest` (backend) and `npm test` (frontend) + `ruff check` + `eslint` before marking done
- Agent changes: verify mocked tools produce correct KG mutations
- API changes: verify status codes and response shapes; frontend: verify WebSocket renders
- Full integration: submit B7-H3 NSCLC query, verify KG builds and report generates

## Reference Docs (read on demand, not auto-loaded)
- [Sprint Roadmap]: /Users/DarshShrivastava/Downloads/YOHAS_SPRINT_ROADMAP.md
- [Implementation Plan]: ./IMPLEMENTATION.md
