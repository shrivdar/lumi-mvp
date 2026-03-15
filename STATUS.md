# YOHAS 3.0 — Current Status

> Last updated: 2026-03-15 ~18:30 UTC

## Architecture Summary

YOHAS 3.0 is a multi-agent biomedical research platform competing against Biomni, STELLA, and Edison.
Core differentiators: MCTS hypothesis tree + agent swarms + self-falsification + knowledge graph.

## What's on `main` (merged, tests passing)

**501 tests passing** on main. 17 feature branches merged across Day 1-3.

### Layer 1: Environment
| Component | File | Status | Notes |
|-----------|------|--------|-------|
| PythonREPLTool | `integrations/python_repl.py` | ✅ LOCAL | `feat/repl-local` merged — local subprocess, namespace persists |
| Bio Data Lake | `data/download_data_lake.py` | ⚠️ PARTIAL | Gene Ontology, Reactome, MSigDB hallmark, DrugBank vocabulary present. GWAS/ClinVar interrupted. |
| Know-How Retriever | `know_how/retriever.py` (189 lines) | ✅ WIRED + REAL | Real Sonnet LLM call. 8 protocol docs + 2 tool recipes. Validated live. |

### Layer 2: Agent Architecture
| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Multi-turn Loop | `agents/base.py` (~1200 lines) | ✅ FIXED | `feat/multiturn-live` merged — truncated XML, max_tokens, urgency all fixed |
| Dynamic Tool Selection | `orchestrator/research_loop.py` | ✅ WIRED | `select_tools_for_task()` called before creating each agent |
| PythonREPL injection | `orchestrator/research_loop.py` | ✅ WIRED | `python_repl` always injected into every agent's tools dict |
| ToolRetriever | `agents/tool_retriever.py` (368 lines) | ✅ VALIDATED | LLM-based per-task tool selection. Validated live. |
| Sub-Agent Spawning | `agents/base.py` | ✅ EXISTS | max_depth=3, max_per_parent=5. |
| Tool Creator Agent | `agents/tool_creator.py` (397 lines) | ✅ EXISTS | STELLA-style. Registered in factory. Not tested live. |
| Agent Factory | `agents/factory.py` (96 lines) | ✅ COMPLETE | 9 agent types |

### Layer 3: RL Training
| Component | Status | Notes |
|-----------|--------|-------|
| Trajectory Collector | ✅ WIRED | Hooked into orchestrator. 19 JSONL files in `data/trajectories/` |
| SFT Pipeline | ✅ EXISTS | Filters trajectories for SFT |
| Reward Functions | ✅ EXISTS | 5-component reward |
| Training Scripts | ✅ EXISTS | Targets DGX Spark. Dry-run mode. |

### Layer 4: Integration
| Component | Status | Notes |
|-----------|--------|-------|
| LivingDocument | ✅ WIRED | Attaches to KG, renders markdown. Validated live. |
| Slack MCP | ✅ EXISTS | 4 MCP tools. Not tested live. |
| Biosecurity Screener | ✅ WIRED | safe→CLEAR, dangerous→BLOCKED. Validated live. |

## Active Worktrees — Batch 3 Results

### `feat/dynamic-orchestrator` — 2 COMMITS, COMPLETE ✅
**WS1 from MASTER_PLAN — 1,993 lines added, 814 test lines added across 10 files.**

Commit 1: `[orchestrator] Add AgentSpec model and spec-driven agent creation`
- `AgentSpec` + `AgentConstraints` Pydantic models in `core/models.py` (42 lines)
- `create_agent_from_spec()` factory function in `agents/factory.py` (73 lines)
- `spawn_sub_agent()` updated in `base.py` to accept `spec: AgentSpec` parameter (181 lines changed)
- `test_agent_spec.py` (342 lines, 15+ tests)

Commit 2: `[orchestrator] Dynamic orchestrator: spec generation, token budget, benchmark mode`
- **`token_budget.py`** (238 lines) — `TokenBudgetManager` class with hierarchical budget distribution: session → hypothesis → swarm → agent. Tracks usage per-hypothesis and per-agent, hard enforcement at session level, soft warnings at agent level.
- **`swarm_composer.py`** updated (223 lines added) — new `compose_swarm_specs()` method that asks LLM to dynamically generate `AgentSpec` objects (role, instructions, tools, constraints) instead of just picking from template library. Always includes `scientific_critic`. Includes `_build_spec_composition_prompt()` and `_parse_agent_spec()`.
- **`research_loop.py`** updated (484 lines added) — integrated `TokenBudgetManager`, wired `spec_factory` for dynamic agent creation, `_compose_for_hypothesis()` now uses `compose_swarm_specs()` + budget allocation, benchmark mode orchestrator added.
- `test_dynamic_orchestrator.py` (354 lines) — tests for spec generation, benchmark mode, full loop
- `test_token_budget.py` (114 lines) — tests for budget allocation, tracking, exhaustion

### `feat/environment-complete` — 2 COMMITS, COMPLETE ✅
**WS2 from MASTER_PLAN — 659 lines added, 239 test lines added across 9 files.**

Commit 1: `[environment] Add data lake integration and agent prompt injection`
- `integrations/data_lake.py` (260 lines) — context provider for 11 biomedical datasets with metadata, example queries, manifest validation
- `base.py` — data lake context injected into agent system prompts
- `python_repl.py` — `DATA_LAKE_DIR` env var added
- `docker-compose.yml` — `./data:/data:ro` volume mounts
- `test_data_lake.py` (163 lines), `test_python_repl.py` (+19 lines)

Commit 2: `[environment] Max turns bump, token budget, observation compression, manifest validation, benchmark overlay`
- **`base.py`** updated (85 lines) — `max_turns` bumped 20→200, `token_budget` parameter added to `_multi_turn_investigate()` with enforcement (forces answer when budget exhausted), `_compress_observations()` method (LLM-based history compression — summarizes older turns, keeps last 10 verbatim)
- `docker-compose.benchmark.yml` enhanced (+77 lines) — increased resource limits
- `test_multi_turn.py` (+54 lines) — tests for compression, budget enforcement

### `feat/production-mode` — UNCOMMITTED, IN PROGRESS ⚠️
**WS3 from MASTER_PLAN — 1,179 lines modified + 1,230 lines in new files (uncommitted).**

What's been built (all uncommitted):
- **`workers/tasks.py`** rewritten (369 lines added) — `run_agent` Celery task now has real implementation: receives agent config, instantiates via factory, runs agent loop, returns result. Agent failure isolation (catches exceptions, returns error status). Supports `AgentSpec`-based dispatch.
- **`db/persistence.py`** created (279 lines) — `SessionPersistence` class with async SQLAlchemy: create/update sessions, save/load checkpoints, persist KG snapshots, HITL request tracking.
- **`db/tables.py`** updated (66 lines added) — new `SessionCheckpointRow` and `HITLRequestRow` ORM models for checkpoint/resume and HITL tracking.
- **`alembic/versions/002_production_mode.py`** (80 lines) — migration for new tables.
- **`api/routes/monitoring.py`** created (252 lines) — new monitoring endpoints: `GET /monitoring/overview` (sessions, KG size, tokens, agents), `GET /monitoring/research/{id}/stats` (per-session details), `GET /monitoring/research/{id}/agents` (agent list with status).
- **`api/routes/research.py`** updated (98 lines) — wired persistence calls, session checkpoint after iterations.
- **`api/websocket.py`** updated (112 lines) — HITL request/response over WebSocket, real-time monitoring events.
- **`report/generator.py`** updated (385 lines added) — Report V2: evidence chain visualization (`_build_evidence_chain()`), methodology section, competing hypotheses comparison, confidence intervals, KG subgraph per claim.
- **`api/main.py`** — monitoring router registered.
- **Frontend**: `hooks.ts` (+130 lines real-time monitoring hooks), `types.ts` (+4 lines), `agent-constellation.tsx` (+1 line).
- **Test suite**: 5 test files (619 lines total): `test_celery_tasks.py`, `test_checkpoint_callback.py`, `test_db_tables.py`, `test_monitoring.py`, `test_report_v2.py`.

### `feat/rl-pipeline` — 1 COMMIT, COMPLETE ✅
**WS4 from MASTER_PLAN — 1,430 lines added, 396 test lines added across 12 files.**

Commit: `[rl] RL pipeline execution — trajectory collection, SFT formatting, training configs`
- **`rl/collect_at_scale.py`** created (290 lines) — batch trajectory collection pipeline: loads Biomni-Eval1 instances via adapter, runs through evaluator, converts to RL trajectories, saves JSONL. CLI with `--limit`, `--live`, `--concurrency`, `--suite` flags.
- **`rl/sft_pipeline.py`** enhanced (157 lines added) — quality filtering, rejection sampling (best trajectory per task), HuggingFace Dataset export, chat-format conversations.
- **`rl/training/config.py`** enhanced (112 lines added) — concrete DGX Spark hyperparameters: SFT 8B (lr=2e-5, 3 epochs), SFT 32B QLoRA (r=64, alpha=128), GRPO (kl=0.1, clip=0.2). Presets: `for_fast_iteration()`, `for_dry_run()`. JSON config file support.
- **`rl/training/rl.py`** enhanced (61 lines) — GRPO + PPO support, reward-as-advantage.
- **`rl/training/sft.py`** enhanced (18 lines) — config file loading.
- **`rl/training/eval.py`** updated (8 lines).
- **`rl/training/serve.py`** created (202 lines) — `ServeConfig` with SGLang + vLLM support, LoRA adapter serving, `build_sglang_cmd()` / `build_vllm_cmd()`, CLI.
- **`docker-compose.serve.yml`** created (74 lines) — model server + nginx proxy, GPU reservation, health checks.
- **`configs/train_default.json`** (79 lines) + **`configs/train_sft_8b.json`** (53 lines) — training config presets.
- **Tests**: `test_collect_at_scale.py` (144 lines), `test_sft_enhancements.py` (252 lines).

### `feat/benchmark-campaign` — NO CHANGES YET

## Code Totals — Batch 3

| Branch | Lines Added | Test Lines | Commits | Status |
|--------|------------|------------|---------|--------|
| dynamic-orchestrator | 1,993 | 814 | 2 committed | ✅ Complete |
| environment-complete | 659 | 239 | 2 committed | ✅ Complete |
| production-mode | ~2,409 | 619 | 0 (uncommitted) | ⚠️ Needs commit |
| rl-pipeline | 1,430 | 396 | 1 committed | ✅ Complete |
| **Total** | **~6,491** | **~2,068** | **5 commits** | |

## Known Bugs

All previously reported bugs on main are fixed. No new regressions detected.
- ⚠️ `feat/production-mode` changes are uncommitted — need to commit before merge
- ⚠️ Tests cannot run locally (system Python 3.9, project needs 3.11+) — Docker required
- ⚠️ GRPO implementation in `rl/training/rl.py` is simplified (not per-token log-prob)
- ⚠️ Eval script uses placeholder F1 scoring, needs LLM-as-judge

## Critical Next Steps (priority order)

1. **Commit `feat/production-mode`** — all work is uncommitted
2. **Merge all 4 branches to main** — dynamic-orchestrator, environment-complete, production-mode, rl-pipeline
3. **Start `feat/benchmark-campaign` (WS5)** — full Biomni-Eval1 (433 instances), LAB-Bench, BixBench
4. **Run full-stack validation** — real orchestrator + dynamic specs + real LLM + mock tools
5. **Run 10 Biomni instances** — first real benchmark with full pipeline
6. **Phase 8 items** — end-to-end integration, seed scripts, production hardening, documentation
