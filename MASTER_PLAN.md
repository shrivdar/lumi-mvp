# YOHAS 3.0 — Master Plan v3 (Source of Truth)
**Date:** March 15, 2026
**Status:** APPROVED — Active Execution
**Previous plans:** v1 superseded (5-workstream plan), v2 superseded (8-workstream plan with Phase 1 partially executed)

---

## The Competition (Full Landscape)

| System | Lab | Architecture | Scale | Key Benchmark | Key Strength | Key Weakness |
|--------|-----|-------------|-------|---------------|--------------|--------------|
| **Biomni A1** | Stanford (Leskovec) | Single agent, code execution, RAG planning | 1 agent, 6-24 steps | LAB-Bench DbQA 74.4%, BixBench 52.2% | 314 tools/DBs, zero-shot generalization | No self-correction, single agent, no KG |
| **Biomni Lab** | Stanford (Leskovec) | Single agent + curated environment | 1 agent | BixBench verified 88.7% | Lab-grade verified answers | Still single agent, no hypothesis exploration |
| **Virtual Biotech** | Stanford (Zou) | CSO → domain scientists, MCP servers | 37,075 agents, 55,984 trials | B7-H3 case ($46, <1 day) | Massive data-parallel, 184× speedup | Each agent does same narrow task, no KG, no falsification |
| **Stanford Virtual Lab** | Stanford (Zou) | AI PI → specialists, team meetings | ~5 agents | 92 nanobodies designed, 2 validated (Nature) | Wet-lab validated, published in Nature | Small scale, serial meetings, no MCTS, no KG |
| **Kosmos** | Edison Scientific | Data + literature agents, world model | 200 rollouts/run | 79.4% statement accuracy, 7 discoveries | 12h autonomous runs, 1,500 papers/42K lines code per run | 20% error rate, "goes down rabbit holes", no falsification, $200/run |
| **SciAgents** | MIT (Buehler) | Ontologist → Scientist → Critic → Reviewer, KG reasoning | ~6 agents | Novel biocomposite materials (Adv. Materials) | KG-driven reasoning (closest to YOHAS) | Materials science only, small scale, no MCTS |
| **SciMaster** | — | Scattered exploration + stacked selection | Multi-trial | HLE Bio 27.6% | Good on open-ended reasoning | Not biomedical-specialized |
| **STELLA** | FutureHouse | Manager/Dev/Critic/ToolCreation, self-evolving | 4 agents, multi-trial (9x) | LAB-Bench DbQA 54%, LitQA 63% | Tool creation, improves with trials | Low single-trial accuracy, slow |
| **YOHAS 3.0** | Lumigenic | MCTS + dynamic swarms + KG + falsification | Config: 50K, practical: TBD | **Untested** | Causal KG, mandatory falsification, MCTS | Falsification broken, 20 tools, untested |

### What Actually Matters (from BioML-bench):
> "biomedical specialization alone does not guarantee superior performance... architecture and scaffolding may be stronger determinants"

---

## YOHAS's Unique Architectural Advantages (Already Built)

No competitor has ALL of these together:

| Innovation | Status | Lines | What It Does | Who Else Has It |
|-----------|--------|-------|-------------|-----------------|
| **MCTS Hypothesis Tree** | ✅ | 597 | UCB1 selection, parallel exploration of 30+ competing hypotheses, auto-pruning, backpropagation | Nobody |
| **Self-Falsification** | ✅ (bug fix needed) | ~120 | Every agent searches PubMed/S2 for counter-evidence to its own claims | Nobody (SVL has critic, but no evidence search) |
| **Causal Knowledge Graph** | ✅ | 826 | Thread-safe, contradiction detection (INHIBITS↔ACTIVATES, UPREGULATES↔DOWNREGULATES), provenance | SciAgents has KG but no contradiction detection |
| **Dynamic Agent Spawning** | ✅ | ~260 | Orchestrator generates AgentSpec objects via LLM — no templates required | VBiotech has CSO delegation, but agents are predefined roles |
| **Slack MCP Integration** | ✅ | 444 | Agents post findings to Slack, notify humans, request input | Nobody |
| **HITL (WebSocket + Slack)** | ✅ | ~470 | Human-in-the-loop when uncertainty exceeds threshold — both browser and Slack | Kosmos has no HITL; SVL has 1% human involvement |
| **Living Document** | ✅ | 417 | Auto-updating research report that grows as agents discover — real-time, not post-hoc | Kosmos generates final reports; this is live |
| **Biosecurity Screening** | ✅ | 162 | Screens all KG mutations against biosecurity watchlists before allowing writes | Nobody |
| **Scientific Critic (Devil's Advocate)** | ✅ | mandatory | Always included in swarm composition — non-negotiable; challenges other agents' findings | SVL and SciAgents have critics but not mandatory |
| **Multi-Turn Investigation Loop** | ✅ | ~160 | Plan → (Think/Tool/Execute/Answer) × N with observation compression, token budget, urgency signals | Kosmos has longer runs but no structured multi-turn protocol |
| **Token Budget Manager** | ✅ | 238 | Hierarchical budget: session → hypothesis → swarm → agent. Hard enforcement at session level | Nobody — all competitors use time limits |
| **Trajectory Collector for RL** | ✅ | 244 | Records every agent's turns, tool calls, KG mutations, rewards for RL fine-tuning | Nobody has RL pipeline for scientific agents |
| **Know-How Retriever** | ✅ | 189 | LLM-based retrieval of relevant protocols/recipes injected into agent prompts | Nobody |
| **Tool Creator (STELLA-style)** | ✅ | 397 | Agent that can create new tools at runtime | STELLA has tool creation; we do too |
| **Python REPL for Data Analysis** | ✅ | 336 | Agents write and execute Python code with persistent namespace, access to data lake | Biomni, Kosmos also have code execution |

---

## Current State (Honest)

| Component | Status | Critical Issue |
|-----------|--------|----------------|
| MCTS Hypothesis Tree | ✅ Real (597 lines) | Works |
| Orchestrator | ✅ Real (1,455 lines) | Works |
| Agent Loop | ✅ Real (1,539 lines) | Falsification key mismatch bug |
| KG | ✅ Real (826 lines) | Works |
| Tool Integrations | ✅ 20 tools (10 original + 10 new) | Works |
| Self-Falsification | 🔴 BROKEN | `"results"` key mismatch with PubMed `"articles"` / S2 `"papers"` |
| Static Templates | ⚠️ Still exist as fallback | compose_swarm_specs() coexists with compose_swarm() |
| Scale | ⚠️ Config ready | max_mcts_iterations=15 limits typical run to ~225 agents |
| Data Lake | ⚠️ Partial | 4/11 parquets downloaded. No ClinVar, GWAS, UniProt bulk |
| Trajectory Data | 🔴 Mock | 55 JSONL files with empty turns and hardcoded reward=1.0 |

---

## 10 Workstreams

### WS-A: Fix Falsification Bug + Harden Agent Loop
**Priority: CRITICAL — our #1 differentiator is broken**

1. Fix `base.py:544` — handle PubMed `"articles"` key and Semantic Scholar `"papers"` key
2. Add counter-evidence LLM evaluation — ask LLM "does this paper's abstract contradict the claim?"
3. Verify falsification results propagate to KG edge metadata
4. Add integration tests with correct tool response keys

**Files**: `backend/agents/base.py`, `backend/tests/test_falsification.py`

### WS-B: Kill Static Templates — Orchestrator Creates Everything
**Priority: CRITICAL — core vision from Principle #1**

1. Make `compose_swarm_specs()` the ONLY path — remove `compose_swarm()` fallback
2. Delete 8 thin agent wrapper files (literature_analyst.py, drug_hunter.py, etc.)
3. Move useful prompt content from templates.py into spec composition prompt
4. Remove `_AGENT_CLASS_MAP` — everything through `create_agent_from_spec()`
5. Keep AgentType enum as hints/tags, not dispatch keys
6. Update all tests

**Files**: `backend/agents/*.py`, `backend/agents/factory.py`, `backend/agents/templates.py`, `backend/orchestrator/swarm_composer.py`, `backend/orchestrator/research_loop.py`

### WS-C: Real Agent Scale (Target: 10K+ per session)
**Priority: HIGH — matches VBiotech's 37K claim**

1. Increase max_mcts_iterations: 15 → 100
2. Increase max_agents_per_swarm: 5 → 20
3. Add `<spawn>` tag parsing to _multi_turn_investigate() for self-delegation
4. Add data-parallel mode for bulk analysis tasks
5. Wire Celery for truly parallel execution across workers
6. Verify semaphore + budget enforcement at 10K scale

**Scale math**: 30 hyp × 20 agents × 5 sub-agents × 100 iter ÷ convergence = 10K-50K achievable

### WS-D: Agent Depth — Make Agents Actually Good
**Priority: HIGH — quality matches Kosmos's 42K lines of code per run**

1. Enhance multi-turn prompt for deeper investigation (cross-reference, cite PMIDs, evidence chains)
2. Add incremental KG writes during investigation (not just at end)
3. Add inter-agent KG awareness (query current KG state mid-investigation)
4. Improve observation compression for 50+ turn agents
5. Add investigation protocols to spec generation
6. Deep Python REPL integration during investigation

### WS-E: MCP Server Integration (Expand from 20 to 50+ Tools)
**Priority: HIGH — environment parity with Biomni's 150 tools**

1. Enhance MCP client for production use
2. Deploy MCP servers for: cBioPortal, GEO, STRING-DB, GTEx, ENCODE, ClinVar, dbSNP
3. Wire into docker-compose.mcp.yml
4. Update ToolRetriever for dynamic MCP tool discovery
5. Target: 50+ tools

### WS-F: Data Lake Completion
**Priority: MEDIUM — needed for benchmarks**

1. Fix GWAS download with FTP fallback
2. Download ClinVar, UniProt bulk, ChEMBL bulk
3. Verify all parquets load via data_lake_context()
4. Integration test: agent queries parquet via REPL

### WS-G: End-to-End Integration Test
**Priority: HIGH — we've never tested the full pipeline**

1. Create scripts/run_b7h3_live.py — real Claude API, real tools, real KG, real falsification
2. Budget cap: session_token_budget=500K (~$5-10)
3. Compare results against VBiotech's B7-H3 case study
4. Document: agent count, tool calls, KG growth, falsification outcomes, time, cost

### WS-H: Benchmark Adapter Validation
**Priority: MEDIUM**

1. Download LAB-Bench from HuggingFace
2. Run 5 DbQA + 5 BixBench questions through full pipeline
3. Measure accuracy, agent count, tool calls, time, cost
4. Identify failure modes before full campaign

### WS-I: Production Hardening
**Priority: MEDIUM**

1. Verify Celery run_agent dispatches agents end-to-end
2. Verify checkpoint/resume works
3. Verify WebSocket streaming
4. Health check verifying Redis + PostgreSQL
5. Explicit logging when Redis down (currently silently no-ops)

### WS-J: Documentation + Status Updates
**Priority: LOW**

1. Update MASTER_PLAN.md with this plan ← DONE
2. Update IMPLEMENTATION.md with accurate Phase 8 status
3. Update STATUS.md post-all-merges

---

## Execution Order

```
PARALLEL GROUP 1 (No dependencies — start immediately):
├── WS-A: Fix falsification bug (30 min)
├── WS-B: Kill static templates (2-3 hours)
├── WS-F: Data lake completion (1-2 hours)
└── WS-J: Documentation cleanup (30 min) ← DONE

PARALLEL GROUP 2 (After WS-A and WS-B merge):
├── WS-C: Real agent scale (2-3 hours)
├── WS-D: Agent depth improvements (2-3 hours)
└── WS-E: MCP server integration (2-3 hours)

PARALLEL GROUP 3 (After Groups 1+2):
├── WS-G: End-to-end live test (1-2 hours + API costs)
├── WS-H: Benchmark adapter validation (1-2 hours)
└── WS-I: Production hardening (1-2 hours)
```

---

## Competitive Targets

| Metric | Biomni A1 | Biomni Lab | VBiotech | Kosmos | Virtual Lab | SciAgents | **YOHAS Target** |
|--------|-----------|------------|----------|--------|-------------|-----------|-----------------|
| LAB-Bench DbQA | 74.4% | 88.7% | N/A | N/A | N/A | N/A | **>80%** |
| BixBench | 52.2% | 52.2% | N/A | N/A | N/A | N/A | **>55%** |
| Agent count/session | 1 | 1 | 37,075 | 200 | 5 | ~6 | **10,000+** |
| Knowledge structure | Flat | Flat | CSO synthesis | World model | Debate log | Ontological KG | **Causal KG** |
| Self-correction | None | None | Reviewer | None | Critic debate | Critic agent | **MCTS + falsification** |
| HITL | None | None | None | None | 1% human | None | **Slack + WebSocket** |
| Hypothesis exploration | None | None | None | None | Group meetings | KG path sampling | **MCTS UCB1 (50+ hyp)** |
| B7-H3 cost | Unknown | Unknown | $46 | $200/run | Unknown | Unknown | **<$100** |
| Tools available | 150 | 150+ | 100+ MCP | Code exec | ESM/AF | KG + LLM | **50+ (API + MCP + REPL)** |
| Living Document | No | No | No | Final report | No | Final doc | **Yes (real-time)** |
| Biosecurity | No | No | No | No | No | No | **Yes** |

---

## Why YOHAS Wins

1. **MCTS gives us what nobody else has** — multiple competing hypotheses explored simultaneously, with UCB1 allocating resources to the most promising branches. All competitors explore linearly or in fixed parallel tracks.

2. **Self-falsification is genuinely unique** — once the bug is fixed, every agent actively searches for counter-evidence. Kosmos has 20% error rate with no self-correction. Our falsification catches weak claims in real-time.

3. **The KG is genuine shared memory** — agents see what other agents found, enabling real collaboration. VBiotech agents work independently. Biomni is single-agent. Our KG has contradiction detection no competitor has.

4. **Scale with purpose** — VBiotech's 37K agents each did the same narrow task. Our 10K+ agents do different things: hypothesis exploration (MCTS), deep investigation (multi-turn), cross-validation (falsification), meta-reasoning (critic), data analysis (REPL). Heterogeneous swarm, not homogeneous farm.

5. **HITL + Living Document + Biosecurity** — the only system with real-time human oversight (Slack + WebSocket), continuously updating reports, and biosecurity screening. Critical for clinical/drug discovery deployment.

6. **RL pipeline for continuous improvement** — complete trajectory → SFT → GRPO pipeline. No competitor has this for scientific agents.

---

## What's Already Done

- ✅ AgentSpec model + create_agent_from_spec() factory
- ✅ compose_swarm_specs() — LLM-driven dynamic AgentSpec generation
- ✅ TokenBudgetManager — hierarchical budget distribution
- ✅ data_lake_context() — injected into agent prompts
- ✅ max_turns bumped to 200 in AgentConstraints
- ✅ Observation compression (_compress_observations)
- ✅ Session persistence (checkpoint/resume)
- ✅ Monitoring endpoints
- ✅ Report Generator V2 (evidence chains, methodology)
- ✅ Celery dispatch (real implementation)
- ✅ WebSocket HITL + Slack MCP
- ✅ Living Document tool
- ✅ Biosecurity screening
- ✅ RL pipeline (trajectory collector, SFT pipeline, training scripts, model serving)
- ✅ Benchmark adapters (Biomni-Eval1, LAB-Bench, BixBench)
- ✅ Phase 8 scripts (dry_run.py, seed_demo.py, export_graph.py, benchmark_vs_vbiotech.py)
- ✅ Documentation (README, API docs, agent template guide)
- ✅ 501+ tests passing
- ✅ CORS, non-root Docker, structlog, request ID propagation
- ✅ 10 new tool integrations (OpenTargets, ClinVar, GTEx, gnomAD, HPO, OMIM, BioGRID, DepMap, CellXGene, STRING-DB)

## What's NOT Done

- ❌ **Falsification broken** — key mismatch (WS-A)
- ❌ **Static templates still default** — compose_swarm() still used (WS-B)
- ❌ **Scale constants too low** — ~225 agents/session (WS-C)
- ❌ **Agents too shallow** — ~5 turns per investigation (WS-D)
- ❌ **Only 20 tools** — need 50+ via MCP (WS-E)
- ❌ **Data lake incomplete** — 4/11 parquets (WS-F)
- ❌ **No end-to-end validation** — never tested full pipeline (WS-G)
- ❌ **No benchmark runs** — zero instances scored (WS-H)
