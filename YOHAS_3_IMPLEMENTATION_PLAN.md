# YOHAS 3.0 — Full Implementation Plan

**Date:** March 14, 2026
**Author:** Darsh Shrivastava + Mastra Code
**Objective:** Build the world's best autonomous biomedical research platform by combining YOHAS's unique KG + MCTS + falsification architecture with code execution, data lake, dynamic agent scaling, and RL training.

---

## 0. STRATEGIC CONTEXT

### Competitive Landscape (March 2026)

| System | Architecture | LAB-Bench DbQA | HLE:Bio | BixBench | Key Strength |
|--------|-------------|-----------------|---------|----------|-------------|
| Biomni A1 | Single agent + code REPL | 74.4% | 17.3% | — | 150 tools, 11GB data lake, persistent REPL |
| Biomni R0 | RL-trained A1 (32B) | — | — | — | 0.669 avg (SFT+RL, up from 0.346) |
| STELLA | Multi-agent + tool creation | 54% | 26% | — | Self-evolving tools, doubles accuracy w/ trials |
| Edison Kosmos | Analysis agent | 70% | — | 42.4% | — |
| YOHAS (current) | KG + MCTS + swarm (zero-shot) | 40%* | — | — | Hypothesis competition, falsification |

*n=5 instances, not statistically significant

### YOHAS's Structural Advantages (Already Built)
1. **Causal Knowledge Graph** — edges have provenance, confidence, agent attribution, hypothesis branch
2. **MCTS Hypothesis Tree** — UCB1 selection explores competing hypotheses, doesn't stop at first answer
3. **Swarm Composition** — LLM-driven agent selection, 8 specialist types, mandatory ScientificCritic
4. **Self-Falsification** — every agent runs counter-evidence search, adjusts edge confidence
5. **Uncertainty Quantification** — triggers HITL when confidence drops below threshold

### What's Missing (The Gap)
- **A. Environment** — No code execution, no local data, no know-how docs
- **B. Agent Scaling** — 8 fixed types, no dynamic spawning, no sub-agents
- **C. Learning Loop** — No RL, no trajectory collection, no model improvement
- **D. Advanced Integrations** — No biosecurity screening, no Living Document, limited Yami

---

## 1. LAYER 1: ENVIRONMENT

### 1.1 Sandboxed Python REPL (`backend/integrations/python_repl.py`)

**What:** Persistent Python execution environment inside Docker container. Agents can write and execute arbitrary Python code with access to scientific libraries and the data lake.

**Architecture:**
```
Agent._investigate()
  → self.call_tool("python_repl", action="execute", code="import pandas as pd; ...")
  → PythonREPLTool.execute()
    → DockerContainer.exec("python3 -c '...'")  # OR persistent subprocess
    → Returns stdout/stderr (truncated to 10K chars)
    → Agent observes output, generates next code block
```

**Key Design Decisions:**
- **Persistent namespace** — variables/imports from step 1 available in step 5 (like Biomni)
- **Container-based** — runs inside Docker (reuse existing `ContainerTool` infrastructure)
- **Pre-installed packages** — numpy, pandas, scipy, scikit-learn, biopython, networkx, matplotlib, seaborn, rdkit
- **Data lake mount** — `/data/` directory mounted read-only with all bio data files
- **Timeout** — 120s per execution (configurable), kill on timeout
- **Output truncation** — 10K chars max, with warning if truncated
- **Security** — no network access from REPL container (data lake is local), no filesystem writes outside `/tmp/`

**Implementation:**
```python
class PythonREPLTool(BaseTool):
    """Persistent Python REPL in sandboxed Docker container."""
    
    def __init__(self):
        super().__init__(
            name="python_repl",
            rate_limit=10.0,  # 10 executions/sec
            cache_ttl=0,      # Never cache (side effects)
            timeout=120
        )
        self._container_id: str | None = None
        self._namespace_file = "/tmp/repl_namespace.pkl"
    
    async def execute(self, code: str, timeout: int = 120) -> dict:
        """Execute Python code in persistent container."""
        container = await self._ensure_container()
        wrapped_code = self._wrap_with_namespace(code)
        result = await container.exec(wrapped_code, timeout=timeout)
        return {
            "stdout": result.stdout[:10000],
            "stderr": result.stderr[:2000],
            "truncated": len(result.stdout) > 10000,
            "execution_time": result.duration
        }
```

**Files to create/modify:**
- `backend/integrations/python_repl.py` — New tool
- `backend/integrations/__init__.py` — Export new tool
- `docker/repl/Dockerfile` — REPL container image with scientific packages
- `docker/repl/requirements.txt` — Package list for REPL container
- `backend/core/models.py` — Add `PYTHON_REPL` to any tool enums if needed

### 1.2 Local Bio Data Lake (`data/`)

**What:** Pre-downloaded biomedical reference databases as parquet/CSV files, mounted into REPL container and queryable by agents.

**Datasets (Priority Order):**

| Dataset | Size | Format | Update Freq | Source |
|---------|------|--------|-------------|--------|
| Gene Ontology | ~200MB | OBO → parquet | Monthly | geneontology.org |
| MSigDB (gene sets) | ~50MB | GMT → parquet | Quarterly | broadinstitute.org/msigdb |
| ClinVar (variants) | ~1.5GB | VCF → parquet | Monthly | ncbi.nlm.nih.gov/clinvar |
| dbSNP (common variants) | ~3GB | VCF → parquet | Annual | ncbi.nlm.nih.gov/snp |
| GWAS Catalog | ~100MB | TSV → parquet | Weekly | ebi.ac.uk/gwas |
| DrugBank (open subset) | ~50MB | XML → parquet | Quarterly | drugbank.ca |
| UniProt (human reviewed) | ~500MB | FASTA + TSV → parquet | Monthly | uniprot.org |
| Reactome pathways | ~100MB | TSV → parquet | Quarterly | reactome.org |
| ChEMBL activities | ~2GB | SQL → parquet | Quarterly | ebi.ac.uk/chembl |
| OMIM (gene-disease) | ~50MB | TSV → parquet | Monthly | omim.org |
| Ensembl gene annotations | ~500MB | GTF → parquet | Quarterly | ensembl.org |

**Total: ~8GB** (comparable to Biomni's ~11GB)

**Implementation:**
```
data/
├── download_data_lake.py     # Script to download + convert all datasets
├── gene_ontology/
│   └── go_annotations.parquet
├── msigdb/
│   └── gene_sets.parquet
├── clinvar/
│   └── variant_summary.parquet
├── gwas_catalog/
│   └── associations.parquet
├── drugbank/
│   └── drugs.parquet
├── uniprot/
│   └── human_reviewed.parquet
├── reactome/
│   └── pathways.parquet
├── chembl/
│   └── activities.parquet
└── README.md                  # Data dictionary
```

**Files to create:**
- `data/download_data_lake.py` — Download + ETL script
- `data/README.md` — Data dictionary with schema descriptions
- `docker/repl/Dockerfile` — Mount `/data/` read-only

### 1.3 Know-How Document Library (`backend/know_how/`)

**What:** Markdown documents containing domain-specific protocols, best practices, troubleshooting guides, and analytical procedures. Injected into agent context via retrieval.

**Document Categories:**
```
backend/know_how/
├── protocols/
│   ├── gwas_analysis.md          # How to interpret GWAS data, LD, fine-mapping
│   ├── drug_repurposing.md       # Target identification, binding analysis, ADMET
│   ├── gene_disease_association.md # Evidence hierarchy, Mendelian vs complex
│   ├── pathway_analysis.md       # Enrichment analysis, network topology
│   ├── protein_function.md       # Domain analysis, conservation, structure
│   ├── clinical_trial_analysis.md # Phase interpretation, endpoints, statistics
│   ├── variant_interpretation.md  # ACMG guidelines, pathogenicity scoring
│   └── sequence_analysis.md      # Alignment, motif finding, primer design
├── tools/
│   ├── pandas_bio_recipes.md     # Common pandas patterns for bio data
│   ├── biopython_recipes.md      # Sequence manipulation, BLAST parsing
│   ├── networkx_recipes.md       # Graph analysis for pathways
│   └── rdkit_recipes.md          # Chemical structure analysis
└── index.json                    # Metadata for retrieval
```

**Retrieval Mechanism:**
- LLM-based retrieval (like Biomni's `ToolRetriever`): given task prompt, select relevant know-how docs
- Inject selected docs into agent system prompt before `_investigate()`
- Maximum 3 docs per agent call (context budget management)

**Files to create:**
- `backend/know_how/` directory with all markdown docs
- `backend/know_how/retriever.py` — LLM-based document retrieval
- Modify `backend/agents/base_agent.py` — inject know-how into system prompt

---

## 2. LAYER 2: AGENT ARCHITECTURE UPGRADE

### 2.1 Multi-Turn Agent Loop

**What:** Replace single-shot `_investigate()` with iterative generate→execute→observe cycle. Agents can now: generate code, execute it, read output, adjust strategy, repeat.

**Current Flow (single-shot):**
```
Agent._investigate(task, hypothesis)
  → query_llm("analyze this hypothesis")
  → call_tool("pubmed", query="...")
  → call_tool("uniprot", query="...")
  → return AgentResult
```

**New Flow (multi-turn):**
```
Agent._investigate(task, hypothesis)
  → Plan: query_llm("make a plan for this task")  # Returns numbered checklist
  → Loop (max_turns=20):
    → Generate: query_llm("execute next step", context=observations)
    → Parse: extract <execute>code</execute> or <tool>name:args</tool> or <answer>result</answer>
    → Execute: run code via REPL or call tool
    → Observe: append stdout/result to observation history
    → Check: if <answer> tag found, extract and return AgentResult
  → Compile: aggregate observations into AgentResult with KG mutations
```

**Key Design:**
- **Observation history** — each turn appends execution result, LLM sees full history
- **Checklist tracking** — agent maintains numbered plan, marks steps ✓ or ✗
- **Early termination** — if agent produces `<answer>` tag, stop immediately
- **Failure recovery** — if code execution fails, agent sees error, can retry with different approach
- **Budget limits** — max_turns=20 (configurable), max_tokens per turn, max_total_tokens per agent
- **XML tag protocol** — `<execute>` for code, `<tool>` for API calls, `<answer>` for final result, `<think>` for reasoning

**Files to modify:**
- `backend/agents/base_agent.py` — New `_multi_turn_investigate()` method
- All 8 agent subclasses — Override `_investigate()` to use multi-turn loop
- `backend/core/models.py` — Add `AgentTurn` model for observation history

### 2.2 Dynamic Tool Selection

**What:** Any agent can use any tool, selected dynamically per task. Replace static tool assignment (LiteratureAnalyst always gets PubMed + SemanticScholar) with LLM-driven selection.

**Current Flow:**
```
SwarmComposer → selects agent types → each type gets pre-assigned tools
LiteratureAnalyst: [pubmed, semantic_scholar]
ProteinEngineer: [uniprot, esm]
GenomicsMapper: [mygene, kegg]
```

**New Flow:**
```
SwarmComposer → selects agent types → ToolRetriever selects tools per agent per task
Any agent can get: [pubmed, semantic_scholar, uniprot, esm, mygene, kegg, 
                     chembl, clinical_trials, reactome, python_repl, ...]
```

**Implementation:**
```python
class ToolRetriever:
    """LLM-based tool selection given task context."""
    
    TOOL_DESCRIPTIONS = {
        "pubmed": "Search biomedical literature (PubMed/MEDLINE). Use for finding papers, reviews, clinical studies.",
        "python_repl": "Execute Python code with access to bio data lake. Use for data analysis, statistics, visualization.",
        # ... all tools
    }
    
    async def select_tools(self, task: str, hypothesis: str, 
                           available_tools: list[str], max_tools: int = 5) -> list[str]:
        """Select most relevant tools for this specific task."""
        prompt = f"Given this research task and hypothesis, select the {max_tools} most relevant tools..."
        response = await self.llm.query(prompt)
        return self._parse_tool_selection(response)
```

**Files to create/modify:**
- `backend/agents/tool_retriever.py` — New module
- `backend/agents/base_agent.py` — Use ToolRetriever before `_investigate()`
- `backend/orchestrator/swarm_composer.py` — Pass all available tools, let ToolRetriever select

### 2.3 Sub-Agent Spawning

**What:** Agents can spawn child agents for focused subtasks. A DrugHunter investigating a target can spawn a LiteratureAnalyst sub-agent to search for papers about that specific target, while continuing its own analysis.

**Implementation:**
```python
class BaseAgentImpl:
    async def spawn_sub_agent(self, task: str, agent_type: AgentType | None = None,
                               tools: list[str] | None = None) -> AgentResult:
        """Spawn a child agent for a focused subtask."""
        if agent_type is None:
            agent_type = await self._select_agent_type(task)
        
        sub_agent = self.agent_factory(
            agent_type=agent_type,
            llm=self.llm,
            kg=self.kg,
            parent_agent_id=self.agent_id,
            depth=self.depth + 1
        )
        return await sub_agent.execute(
            AgentTask(description=task, hypothesis=self.current_hypothesis),
            max_turns=10  # Sub-agents get smaller budgets
        )
```

**Constraints:**
- Max depth: 3 (agent → sub-agent → sub-sub-agent)
- Max sub-agents per parent: 5
- Sub-agents share parent's KG (write to same graph)
- Sub-agents inherit parent's hypothesis branch
- Token budget divided among sub-agents

**Files to modify:**
- `backend/agents/base_agent.py` — Add `spawn_sub_agent()` method
- `backend/core/models.py` — Add `parent_agent_id` and `depth` to AgentResult

### 2.4 Orchestrator Scaling (1000s of Agents)

**What:** Scale MCTS exploration to spawn hundreds of agents per iteration. Currently: 3-5 agents per MCTS iteration × 15 iterations = ~75 agents max. Target: dynamic scaling based on hypothesis tree breadth.

**Changes:**
- SwarmComposer generates **per-hypothesis swarms** (not one swarm per iteration)
- Each leaf node in hypothesis tree gets its own swarm of 3-8 agents
- Concurrent execution via `asyncio.gather()` with semaphore limiting
- Agent results feed back into MCTS backpropagation in real-time (not batch)
- Hypothesis tree can grow wider (more competing hypotheses) not just deeper

**Scaling Controls:**
```python
class ResearchConfig:
    # Existing
    max_agents: int = 8           # Per swarm
    max_mcts_iterations: int = 15
    
    # New
    max_concurrent_agents: int = 20    # Global concurrency limit
    max_total_agents: int = 500        # Hard cap across entire session
    max_hypothesis_breadth: int = 10   # Max competing hypotheses per level
    agent_token_budget: int = 50000    # Per agent token limit
    session_token_budget: int = 2000000  # Total session token limit
```

**Files to modify:**
- `backend/orchestrator/research_loop.py` — Parallel swarm dispatch
- `backend/orchestrator/swarm_composer.py` — Per-hypothesis swarm composition
- `backend/orchestrator/hypothesis_tree.py` — Wider expansion, real-time backprop
- `backend/core/models.py` — New config fields

---

## 3. LAYER 3: LEARNING LOOP (RL)

### 3.1 Trajectory Collector (`backend/rl/trajectory_collector.py`)

**What:** Record every agent's multi-turn trajectory (prompts, code, outputs, tool calls, KG mutations, final answer) as training data.

**Trajectory Format:**
```python
@dataclass
class Trajectory:
    task_id: str                    # Benchmark instance ID
    task_name: str                  # e.g., "gwas_causal_gene_opentargets"
    agent_type: str                 # e.g., "literature_analyst"
    turns: list[Turn]               # Multi-turn history
    final_answer: str               # Extracted answer
    reward: float                   # 0.0-1.0 from evaluator
    kg_mutations: list[KGMutation]  # Nodes/edges added
    hypothesis_branch: str          # Which hypothesis this explored
    token_usage: TokenUsage         # Input/output tokens
    wall_time: float                # Seconds
    
@dataclass
class Turn:
    role: str                       # "assistant" or "observation"
    content: str                    # LLM output or tool/REPL output
    tool_calls: list[ToolCall]      # Structured tool invocations
    code_executions: list[CodeExec] # Python REPL executions
    timestamp: float
```

**Storage:** JSONL files in `data/trajectories/`, one file per benchmark run, indexed by task_name.

**Files to create:**
- `backend/rl/trajectory_collector.py`
- `backend/rl/trajectory_format.py` — Pydantic models for trajectory data
- `backend/rl/__init__.py`

### 3.2 SFT Data Pipeline (`backend/rl/sft_pipeline.py`)

**What:** Filter trajectories to keep only successful ones (reward=1.0), convert to SFT format for fine-tuning.

**Pipeline:**
1. Run all 433 Biomni-Eval1 instances with Opus 4.6 (collect trajectories)
2. Filter: keep only trajectories where `reward == 1.0`
3. For each successful trajectory, format as conversation:
   - System prompt (with know-how docs + tool descriptions)
   - User message (task prompt)
   - Assistant turns (code generation, tool calls, reasoning)
   - Final answer
4. Export as HuggingFace Dataset or JSONL for training

**Rejection Sampling:** For each task instance, run N=5 times with temperature=0.8. Keep the successful trajectories. This gives more training data per instance.

### 3.3 RL Training Scaffold (`backend/rl/training/`)

**What:** RL training pipeline using YOHAS's richer reward signals. Target: fine-tune a 7-32B parameter model on DGX Spark.

**Reward Function (YOHAS-specific):**
```python
def compute_reward(trajectory: Trajectory, kg: KnowledgeGraph) -> float:
    """Multi-dimensional reward combining correctness + scientific rigor."""
    
    # R1: Answer correctness (from evaluator) — weight 0.5
    r_correct = trajectory.reward  # 0.0 or 1.0
    
    # R2: KG quality — weight 0.2
    r_kg = compute_kg_quality(kg, trajectory)
    #   - Edge confidence calibration (are high-confidence edges actually correct?)
    #   - Provenance coverage (do edges have real citations?)
    #   - Contradiction detection (did system find real contradictions?)
    
    # R3: Hypothesis efficiency — weight 0.15
    r_hypo = compute_hypothesis_efficiency(trajectory)
    #   - Info gain per MCTS iteration
    #   - Did system explore multiple hypotheses before converging?
    #   - Pruning effectiveness (were bad hypotheses abandoned quickly?)
    
    # R4: Falsification quality — weight 0.1
    r_falsify = compute_falsification_quality(trajectory)
    #   - Did critic find real counter-evidence?
    #   - Were falsified edges actually wrong?
    
    # R5: Format compliance — weight 0.05
    r_format = 1.0 if trajectory.final_answer else 0.0
    
    return (0.5 * r_correct + 0.2 * r_kg + 0.15 * r_hypo + 
            0.1 * r_falsify + 0.05 * r_format)
```

**Training Infrastructure (DGX Spark):**
- Framework: veRL or SkyRL (both support multi-turn RL)
- Base model: Qwen2.5-32B-Instruct or Llama-3.1-8B-Instruct (start small)
- GPU: DGX Spark (your hardware)
- Phase 1: SFT on successful trajectories (1-2 epochs)
- Phase 2: RL with multi-dimensional reward (PPO or GRPO)
- Eval: Run Biomni-Eval1 after each checkpoint, compare to base model

**Files to create:**
- `backend/rl/training/config.py` — Training hyperparameters
- `backend/rl/training/reward.py` — Multi-dimensional reward function
- `backend/rl/training/sft.py` — SFT training script
- `backend/rl/training/rl.py` — RL training script (veRL/SkyRL integration)
- `backend/rl/training/eval.py` — Checkpoint evaluation script

### 3.4 Yami Integration Space

**What:** Reserve architectural space for custom small model (couple-million parameters) that provides specialized protein/genomics predictions.

**Current Architecture:**
```
backend/world_model/yami.py → YamiInterface (Protocol)
  → HuggingFaceYami (ESM-2 via HF API)
  → LocalYami (ESM-2 via local GPU)  # Placeholder
```

**Extended Architecture:**
```
backend/world_model/yami.py → YamiInterface (Protocol)
  → HuggingFaceYami (ESM-2 via HF API)
  → LocalYami (Custom model on DGX Spark)
  → EnsembleYami (combines multiple models)
  → ImitatorYami (distilled from Opus outputs)  # Train on REPL trajectories
```

**Imitator Training Pipeline:**
1. During benchmark runs, collect (input_sequence, Opus_prediction) pairs
2. Fine-tune small model (ESM-2-based, ~650M params) to predict Opus outputs
3. Use distilled model for fast inference in REPL (local GPU, <100ms)
4. Gradually replace Opus calls with Yami predictions where confidence is high

**Files to modify:**
- `backend/world_model/yami.py` — Add `LocalYami` and `ImitatorYami` classes
- `backend/core/config.py` — Add `yami_model_path`, `yami_device` settings

---

## 4. LAYER 4: ADVANCED INTEGRATIONS

### 4.1 Biosecurity Screener (`backend/integrations/biosecurity.py`)

**What:** Screen all research outputs for dual-use/biosecurity concerns before presenting to user.

**Implementation:**
- LLM-based screening with structured prompt
- Runs on every `ResearchResult` before delivery
- Flags: pathogen enhancement, toxin synthesis, biological weapons, gain-of-function without oversight
- Three tiers: CLEAR (proceed), WARNING (add disclaimer), BLOCKED (refuse to present)
- Audit log of all screening decisions

```python
class BiosecurityScreener:
    SCREENING_PROMPT = """Analyze this research output for biosecurity concerns.
    Categories: pathogen_enhancement, toxin_synthesis, weapons_potential, 
    gain_of_function, dual_use_concern
    
    For each category, rate risk: NONE, LOW, MEDIUM, HIGH, CRITICAL
    If any category is HIGH or CRITICAL, flag for review."""
    
    async def screen(self, result: ResearchResult) -> ScreeningResult:
        response = await self.llm.query(
            self.SCREENING_PROMPT + "\n\nResearch output:\n" + result.report_markdown,
            system_prompt="You are a biosecurity expert..."
        )
        return self._parse_screening(response)
```

**Integration Point:** 
- `ResearchOrchestrator.run()` → after compiling results, before returning
- Block delivery if risk is HIGH/CRITICAL
- Log all decisions for audit

**Files to create:**
- `backend/integrations/biosecurity.py`
- Modify `backend/orchestrator/research_loop.py` — Add screening step

### 4.2 Living Document (`backend/integrations/living_document.py`)

**What:** Continuously-updated research document that evolves as the KG grows. Each agent's findings are automatically appended to a structured document.

**Architecture:**
- Markdown document with sections matching KG structure
- Auto-updated when new KG nodes/edges are added (via KG event listener)
- Sections: Executive Summary, Hypotheses Under Investigation, Evidence Map, Key Findings, Contradictions, Uncertainties, Recommended Experiments
- Version history (git-style diffs between updates)

**Files to create:**
- `backend/integrations/living_document.py` — Document generator + updater
- Modify `backend/world_model/knowledge_graph.py` — Add event listener for doc updates

### 4.3 Slack MCP Integration

**What:** Expose YOHAS capabilities as MCP tools accessible via Slack.

**Current:** Slack integration is one-way HITL (system asks human, human responds).

**Extended:** Bidirectional — humans can query YOHAS via Slack, get KG visualizations, trigger research sessions, review findings.

**MCP Tools to Expose:**
- `yohas_research(query)` — Start a research session
- `yohas_query_kg(question)` — Query the knowledge graph
- `yohas_status(session_id)` — Check research session status
- `yohas_findings(session_id)` — Get key findings
- `yohas_visualize(session_id, node_id)` — Generate subgraph visualization

**Files to create:**
- `backend/integrations/slack_mcp.py` — MCP server exposing YOHAS tools
- `backend/integrations/mcp_tools/` — Individual MCP tool definitions

---

## 5. MODEL CONFIGURATION

### 5.1 Switch Default LLM to Opus 4.6

**What:** Change default model from `claude-sonnet-4-20250514` to `claude-opus-4-20250805`.

**Files to modify:**
- `backend/core/config.py` line 14: change default to `"claude-opus-4-20250805"`
- Verify token limits are appropriate for Opus (may support larger context)

### 5.2 Multi-Model Support

**What:** Allow different models for different purposes.

```python
class Settings:
    # Primary (hypothesis generation, complex reasoning)
    llm_model: str = "claude-opus-4-20250805"
    
    # Fast (tool selection, formatting, simple classification)
    llm_fast_model: str = "claude-sonnet-4-20250514"
    
    # Cheap (biosecurity screening, document generation)
    llm_cheap_model: str = "claude-haiku-3-20250305"
    
    # Custom (Yami, RL-trained model)
    llm_custom_model: str | None = None
    llm_custom_endpoint: str | None = None
```

---

## 6. EXECUTION PLAN (2-DAY PARALLEL BUILD)

### Day 1 — Foundation (8 Parallel Worktrees)

| # | Worktree | Task | Est. Time | Dependencies |
|---|----------|------|-----------|-------------|
| 1 | `env-repl` | Build PythonREPLTool + Docker container with scientific packages | 3-4h | None |
| 2 | `env-datalake` | Download script for all 11 bio datasets, convert to parquet | 4-5h | None |
| 3 | `env-knowhow` | Write 8 protocol docs + 4 recipe docs + LLM retriever | 3-4h | None |
| 4 | `agent-multiturn` | Refactor BaseAgentImpl with multi-turn generate→execute→observe loop | 4-5h | Worktree 1 (REPL) |
| 5 | `agent-dynamic` | Build ToolRetriever + sub-agent spawning + dynamic tool selection | 3-4h | None |
| 6 | `agent-scale` | Scale orchestrator: per-hypothesis swarms, concurrent execution, wider tree | 3-4h | Worktree 5 |
| 7 | `model-upgrade` | Switch to Opus 4.6 + multi-model support | 1-2h | None |
| 8 | `bench-opus` | Run all 433 Biomni-Eval1 instances with Opus 4.6 (baseline) | 2-3h | Worktree 7 |

**Dependency Graph:**
```
[1: REPL] ──────────→ [4: Multi-turn] ──→ (merge)
[2: Data Lake] ──────────────────────────→ (merge)
[3: Know-how] ──────────────────────────→ (merge)
[5: Dynamic Tools] ──→ [6: Scale] ──────→ (merge)
[7: Model Upgrade] ──→ [8: Bench Opus] ─→ (merge)
```

### Day 2 — Advanced Features (6 Parallel Worktrees)

| # | Worktree | Task | Est. Time | Dependencies |
|---|----------|------|-----------|-------------|
| 9 | `rl-collector` | Trajectory collector + SFT data pipeline | 3-4h | Day 1 merge |
| 10 | `rl-training` | RL training scaffold (veRL/SkyRL config for DGX Spark) | 4-5h | Worktree 9 |
| 11 | `tool-creation` | Tool Creation Agent (STELLA-style autonomous tool discovery) | 4-5h | Day 1 merge |
| 12 | `biosecurity` | Biosecurity screener integration | 2-3h | None |
| 13 | `living-doc` | Living Document + Slack MCP integration | 3-4h | None |
| 14 | `bench-full` | Full benchmark suite with all upgrades (Biomni + BixBench + LAB-Bench) | 3-4h | Day 1 merge |

### Merge Strategy
- Each worktree is an independent feature branch
- Merge order: model-upgrade → env-repl → env-datalake → env-knowhow → agent-dynamic → agent-multiturn → agent-scale → bench-opus → rl-collector → rl-training → tool-creation → biosecurity → living-doc → bench-full
- Run full test suite after each merge
- If conflicts arise, resolve in merge commit

---

## 7. VERIFICATION & BENCHMARKING

### Phase 1: Baseline (Day 1, Worktree 8)
- Run all 433 Biomni-Eval1 instances with Opus 4.6 zero-shot
- Record per-task accuracy + token usage + latency
- Compare to published baselines (Biomni A1: 0.744 DbQA, R0: 0.669 avg)

### Phase 2: Full Stack (Day 2, Worktree 14)
- Run all 433 Biomni-Eval1 with full YOHAS (multi-turn + tools + data lake + know-how)
- Run BixBench 205 instances (trajectory collection for external grading)
- Run LAB-Bench selected categories (DbQA, SeqQA, LitQA2)
- Record everything for trajectory collection

### Expected Results
| Task | Zero-shot (Opus) | YOHAS Full Stack | Biomni A1 | Target |
|------|-----------------|-----------------|-----------|--------|
| lab_bench_dbqa | ~50%* | 65-75% | 74.4% | ≥75% |
| lab_bench_seqqa | ~40%* | 55-65% | 81.9% | ≥70% |
| gwas_causal_gene | ~80%* | 85-90% | — | ≥90% |
| rare_disease_diagnosis | ~30%* | 50-60% | — | ≥60% |
| screen_gene_retrieval | ~40%* | 55-65% | — | ≥65% |

*Estimates based on 5-instance smoke tests and published LLM baselines

### Phase 3: RL Training (Day 2+)
- Collect trajectories from Phase 2
- SFT on successful trajectories (DGX Spark)
- RL with multi-dimensional reward
- Re-evaluate on Biomni-Eval1

---

## 8. ARCHITECTURE DIAGRAM (POST-UPGRADE)

```
┌─────────────────────────────────────────────────────────────────┐
│                        YOHAS 3.0                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ Research      │    │ Hypothesis   │    │ Swarm            │  │
│  │ Orchestrator  │───→│ Tree (MCTS)  │───→│ Composer         │  │
│  │               │    │ UCB1 Select  │    │ (per-hypothesis) │  │
│  └──────┬───────┘    └──────────────┘    └────────┬─────────┘  │
│         │                                          │             │
│         │    ┌─────────────────────────────────────┘             │
│         │    │                                                   │
│         ▼    ▼                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Agent Swarm (100s of agents)                │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐           │   │
│  │  │ Agent  │ │ Agent  │ │ Agent  │ │ Critic │           │   │
│  │  │ (Lit)  │ │ (Drug) │ │ (Gen)  │ │        │  ...      │   │
│  │  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘           │   │
│  │      │          │          │          │                   │   │
│  │      ▼          ▼          ▼          ▼                   │   │
│  │  ┌──────────────────────────────────────────────────┐    │   │
│  │  │         Multi-Turn Loop (per agent)              │    │   │
│  │  │  Plan → Generate → Execute → Observe → Repeat   │    │   │
│  │  │         (up to 20 turns per agent)               │    │   │
│  │  └──────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Tool Layer                             │   │
│  │  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌──────────────┐   │   │
│  │  │ PubMed  │ │ UniProt  │ │ KEGG   │ │ Python REPL  │   │   │
│  │  │ S.Scholar│ │ ESM/Yami │ │Reactome│ │ (persistent) │   │   │
│  │  │ ChEMBL  │ │ ClinTrials│ │MyGene │ │              │   │   │
│  │  └─────────┘ └──────────┘ └────────┘ └──────┬───────┘   │   │
│  │                                              │            │   │
│  │                                    ┌─────────▼─────────┐  │   │
│  │                                    │   Bio Data Lake   │  │   │
│  │                                    │  GO, MSigDB,      │  │   │
│  │                                    │  ClinVar, GWAS,   │  │   │
│  │                                    │  DrugBank, OMIM   │  │   │
│  │                                    │  (~8GB parquets)  │  │   │
│  │                                    └───────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Causal Knowledge Graph                       │   │
│  │  Nodes: 17 types (Protein, Gene, Drug, Disease, ...)     │   │
│  │  Edges: 31 relation types, confidence scores,            │   │
│  │         provenance, agent attribution, hypothesis branch │   │
│  │  Features: contradiction detection, falsification,       │   │
│  │           uncertainty quantification, subgraph queries   │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Output Layer                                 │   │
│  │  ├── Biosecurity Screener (LLM-based, 3 tiers)           │   │
│  │  ├── Living Document (auto-updated, versioned)           │   │
│  │  ├── Report Generator (markdown + citations)             │   │
│  │  ├── Slack MCP (bidirectional, HITL)                     │   │
│  │  └── Trajectory Collector (for RL training)              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Learning Loop (DGX Spark)                    │   │
│  │  Trajectories → SFT → RL (multi-dimensional reward)      │   │
│  │  Reward: correctness + KG quality + hypothesis            │   │
│  │          efficiency + falsification + format              │   │
│  │  Target: Qwen-32B or Llama-8B fine-tuned for YOHAS       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Yami (Custom Model Space)                    │   │
│  │  ├── HuggingFace backend (ESM-2, ESMFold)               │   │
│  │  ├── Local backend (DGX Spark)                           │   │
│  │  ├── Imitator (distilled from Opus)                      │   │
│  │  └── Ensemble (combine multiple models)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. WHY THIS BEATS EVERYONE

**vs Biomni A1/R0:**
- Multi-agent swarm (100s) vs single agent
- Causal KG with provenance vs no persistent knowledge structure
- MCTS hypothesis competition vs linear code execution
- Self-falsification vs no counter-evidence search
- Richer RL reward signals (5 dimensions vs 1)

**vs STELLA:**
- Code execution against real data (STELLA has limited tool creation)
- MCTS exploration (STELLA repeats randomly)
- KG accumulation (STELLA has template library but no causal graph)
- RL training (STELLA has no learning loop)

**vs Edison Kosmos:**
- Multi-agent (Edison is single analysis agent)
- Code execution + data lake (Edison uses basic tool calls)
- Hypothesis competition (Edison picks one approach)

**The unique combination nobody else has:**
1. Causal KG with 17 node types + 31 edge relations + provenance
2. MCTS hypothesis tree with UCB1 selection
3. Multi-agent swarm (100s of specialist agents)
4. Multi-turn code execution in persistent REPL
5. Self-falsification loop (every claim gets counter-evidence searched)
6. Local bio data lake (8GB of reference databases)
7. RL training with multi-dimensional reward (correctness + KG quality + hypothesis efficiency + falsification quality)
8. Yami custom model for specialized predictions
9. Biosecurity screening
10. Living document with version history

---

## 10. RISKS & MITIGATIONS

| Risk | Impact | Mitigation |
|------|--------|------------|
| API costs explode (100s of agents × Opus 4.6) | High | Token budgets per agent/session, use Sonnet for simple tasks |
| REPL security (code execution) | Critical | Docker container, no network, read-only data, timeout enforcement |
| Data lake download fails | Medium | Graceful fallback to API-only mode, retry logic |
| RL training doesn't converge | Medium | Start with SFT only, add RL gradually, eval after each checkpoint |
| Merge conflicts between 14 worktrees | Medium | Clear module boundaries, merge in dependency order |
| DGX Spark insufficient for 32B model RL | Low | Start with 8B model, scale up if needed |
