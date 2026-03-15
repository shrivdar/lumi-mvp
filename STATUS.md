# YOHAS 3.0 — Current Status

> Last updated: 2026-03-15 ~04:00 UTC

## Architecture Summary

YOHAS 3.0 is a multi-agent biomedical research platform competing against Biomni, STELLA, and Edison.
Core differentiators: MCTS hypothesis tree + agent swarms + self-falsification + knowledge graph.

## What's on `main` (merged, tests passing)

**501 tests passing** on main. 13 feature branches merged across Day 1-2.

### Layer 1: Environment
| Component | File | Status | Notes |
|-----------|------|--------|-------|
| PythonREPLTool | `integrations/python_repl.py` | ⚠️ DOCKER-ONLY on main | `feat/repl-local` rewrites to local subprocess — 511 tests pass, committed |
| Bio Data Lake | `data/download_data_lake.py` | ⚠️ 0 PARQUET FILES on main | `feat/datalake-slim` has Gene Ontology (15MB). GWAS/ClinVar interrupted. |
| Know-How Retriever | `know_how/retriever.py` (189 lines) | ✅ WIRED + REAL | Real Sonnet LLM call. 8 protocol docs + 2 tool recipes. Validated live. |

### Layer 2: Agent Architecture
| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Multi-turn Loop | `agents/base.py` (~1200 lines) | ⚠️ BUGS on main | `feat/multiturn-live` fixes truncated XML, max_tokens, urgency. 501 tests pass. |
| Dynamic Tool Selection | `orchestrator/research_loop.py` | ✅ WIRED | `select_tools_for_task()` called before creating each agent |
| PythonREPL injection | `orchestrator/research_loop.py` | ✅ WIRED | `python_repl` always injected into every agent's tools dict |
| ToolRetriever | `agents/tool_retriever.py` (368 lines) | ✅ VALIDATED | LLM-based per-task tool selection. Validated live. |
| Sub-Agent Spawning | `agents/base.py` | ✅ EXISTS | max_depth=3, max_per_parent=5. |
| Tool Creator Agent | `agents/tool_creator.py` (397 lines) | ✅ EXISTS | STELLA-style. Registered in factory. Not tested live. |
| Agent Factory | `agents/factory.py` (96 lines) | ✅ COMPLETE | 9 agent types |

### Layer 3: RL Training
| Component | Status | Notes |
|-----------|--------|-------|
| Trajectory Collector | ✅ WIRED | Hooked into orchestrator |
| SFT Pipeline | ✅ EXISTS | Filters trajectories for SFT |
| Reward Functions | ✅ EXISTS | 5-component reward |
| Training Scripts | ✅ EXISTS | Targets DGX Spark. Dry-run mode. |

### Layer 4: Integration
| Component | Status | Notes |
|-----------|--------|-------|
| LivingDocument | ✅ WIRED | Attaches to KG, renders markdown. Validated live. |
| Slack MCP | ✅ EXISTS | 4 MCP tools. Not tested live. |
| Biosecurity Screener | ✅ WIRED | safe→CLEAR, dangerous→BLOCKED. Validated live. |

## Unmerged Branches (ready to merge)

### `feat/repl-local` — commit `a053ffe`
- PythonREPLTool: Docker → local subprocess (in-process exec + asyncio.to_thread)
- 511 tests passing. Namespace persists across calls.

### `feat/multiturn-live` — commit `0d9b037`
- Fixed 4 bugs: truncated XML handling, max_tokens for answers, urgency prompting, token_usage type
- Live test: 4 turns, 7 nodes, 7 edges. 501 tests passing.

### `feat/env-validate` — commit `3ff9f85`
- Fixed async blocking in llm.py, biosecurity.py, retriever.py (sync→asyncio.to_thread)
- Validation script: 7/7 components pass with real API.

### `feat/datalake-slim` — PARTIAL, uncommitted
- Gene Ontology downloaded (15MB). GWAS/ClinVar interrupted. Script exists.

## Critical Next Steps (priority order)

1. **Merge 3 ready branches** to main
2. **Fix model ID** on main: `claude-opus-4-6` (not `claude-opus-4-20250805`)
3. **Finish data lake** — at minimum GWAS Catalog for benchmarks
4. **Run full-stack validation** — real orchestrator + real LLM + mock tools
5. **Run 10 Biomni instances** — first real benchmark with full stack

## Known Bugs on main (fixed on branches)

- `config.py` default model is `claude-opus-4-20250805` → should be `claude-opus-4-6`
- `llm.py` calls sync Anthropic SDK inside async def → blocks event loop (fixed on env-validate)
- `base.py` multi-turn loop doesn't handle truncated `<answer>` tags (fixed on multiturn-live)
- `base.py` max_tokens=4096 too low for answer JSON (fixed on multiturn-live)
