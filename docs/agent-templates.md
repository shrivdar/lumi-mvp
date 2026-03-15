# YOHAS 3.0 — How to Add New Agent Types

This guide explains the agent architecture and walks through adding a new specialist agent to the platform.

---

## Architecture Overview

### The Agent Execution Model

Every agent in YOHAS 3.0 follows a **template-method pattern** implemented in `BaseAgentImpl` (`backend/agents/base.py`). The execution flow is:

```
execute(task)
  ├── 1. Set audit context
  ├── 2. Build KG context (subgraph around relevant nodes)
  ├── 3. Retrieve domain know-how (RAG injection)
  ├── 4. _investigate(task, kg_context)  ← SUBCLASS HOOK
  │     └── _multi_turn_investigate()
  │           ├── Phase 1: Planning (LLM creates 3-8 step plan)
  │           └── Phase 2: Multi-turn loop
  │                 ├── <think> — reasoning
  │                 ├── <tool> — call external tools
  │                 ├── <execute> — run Python code
  │                 └── <answer> — final structured JSON output
  ├── 5. Write nodes/edges to KG (with permission checks)
  ├── 6. Self-falsification (search for counter-evidence)
  ├── 7. Compute uncertainty vector
  └── 8. Return AgentResult
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `BaseAgentImpl` | `agents/base.py` | Concrete base class with full execute loop, LLM integration, KG writes, falsification, uncertainty |
| `AgentTemplate` | `core/models.py` | Static configuration: system prompt, tools, KG permissions, falsification protocol |
| `AgentSpec` | `core/models.py` | Dynamic specification: orchestrator-generated per-task role, instructions, constraints |
| `AgentConstraints` | `core/models.py` | Resource limits: max turns, token budget, timeout, max LLM calls |
| Agent subclasses | `agents/*.py` | Thin subclasses that override `_investigate()` with a domain-specific `investigation_focus` |
| `AGENT_TEMPLATES` | `agents/templates.py` | Registry mapping `AgentType` → `AgentTemplate` |
| `_AGENT_CLASS_MAP` | `agents/factory.py` | Registry mapping `AgentType` → concrete Python class |
| `create_agent()` | `agents/factory.py` | Factory for template-based agent creation |
| `create_agent_from_spec()` | `agents/factory.py` | Factory for dynamic spec-based agent creation |

### Two Creation Paths

1. **Template-based** (`create_agent`): The orchestrator picks an `AgentType`, the factory looks up the template and class, and creates a fully-configured agent. Used for standard agent types.

2. **Spec-based** (`create_agent_from_spec`): The orchestrator LLM generates an `AgentSpec` with a custom role, instructions, tools, and constraints. If the spec has an `agent_type_hint` matching a known type, the specialized subclass is used; otherwise a generic `BaseAgentImpl` runs the multi-turn loop driven entirely by the spec.

---

## How to Add a New Agent Type

### Step 1: Add to the `AgentType` Enum

**File:** `backend/core/models.py`

Add your new agent type to the `AgentType` enum:

```python
class AgentType(enum.StrEnum):
    """Agent specializations."""

    LITERATURE_ANALYST = "literature_analyst"
    PROTEIN_ENGINEER = "protein_engineer"
    GENOMICS_MAPPER = "genomics_mapper"
    PATHWAY_ANALYST = "pathway_analyst"
    DRUG_HUNTER = "drug_hunter"
    CLINICAL_ANALYST = "clinical_analyst"
    SCIENTIFIC_CRITIC = "scientific_critic"
    EXPERIMENT_DESIGNER = "experiment_designer"
    TOOL_CREATOR = "tool_creator"
    # ↓ Add your new type here
    METABOLOMICS_ANALYST = "metabolomics_analyst"
```

The string value (e.g., `"metabolomics_analyst"`) is used in API responses, logs, and the factory lookup.

### Step 2: Create a Template

**File:** `backend/agents/templates.py`

Define an `AgentTemplate` with the agent's system prompt, tools, KG permissions, and falsification protocol:

```python
from core.models import AgentTemplate, AgentType, EdgeRelationType, NodeType

METABOLOMICS_ANALYST_TEMPLATE = AgentTemplate(
    agent_type=AgentType.METABOLOMICS_ANALYST,
    display_name="Metabolomics Analyst",
    description=(
        "Analyzes metabolomic data, maps metabolite-pathway relationships, "
        "and identifies biomarker signatures."
    ),
    system_prompt=(
        "You are a metabolomics specialist with expertise in mass spectrometry "
        "data interpretation and metabolic pathway analysis.\n\n"
        "Given a research question and knowledge graph context, you will:\n"
        "1. Identify relevant metabolites and their biochemical pathways\n"
        "2. Map metabolite-enzyme-gene relationships\n"
        "3. Assess metabolite biomarker potential for diseases in the KG\n"
        "4. Cross-reference with KEGG metabolic pathways\n"
        "5. Flag metabolites with known drug interactions\n\n"
        "Output structured JSON with COMPOUND/PATHWAY/BIOMARKER nodes "
        "and metabolic relationship edges.\n"
        "Every claim must cite a database source (KEGG, HMDB, PubMed)."
    ),
    tools=["kegg", "pubmed"],
    kg_write_permissions=[
        NodeType.COMPOUND,
        NodeType.PATHWAY,
        NodeType.BIOMARKER,
        NodeType.GENE,
        NodeType.PROTEIN,
    ],
    kg_edge_permissions=[
        EdgeRelationType.MEMBER_OF,
        EdgeRelationType.PARTICIPATES_IN,
        EdgeRelationType.CATALYZES,
        EdgeRelationType.METABOLIZED_BY,
        EdgeRelationType.BIOMARKER_FOR,
        EdgeRelationType.ASSOCIATED_WITH,
        EdgeRelationType.CORRELATES_WITH,
    ],
    requires_yami=False,
    falsification_protocol=(
        "Cross-reference metabolite-pathway associations across KEGG and HMDB; "
        "search PubMed for contradicting metabolomics studies."
    ),
    max_iterations=10,
    timeout_seconds=300,
)
```

Then register it in the `AGENT_TEMPLATES` dict at the bottom of the file:

```python
AGENT_TEMPLATES: dict[AgentType, AgentTemplate] = {
    # ... existing entries ...
    AgentType.METABOLOMICS_ANALYST: METABOLOMICS_ANALYST_TEMPLATE,
}
```

### Step 3: Create the Agent Subclass

**File:** `backend/agents/metabolomics_analyst.py` (new file)

Most agents are thin subclasses that override `_investigate()` to call `_multi_turn_investigate()` with a domain-specific `investigation_focus`:

```python
"""Metabolomics Analyst agent — metabolite-pathway mapping and biomarker discovery."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgentImpl
from core.models import AgentTask, AgentType


class MetabolomicsAnalystAgent(BaseAgentImpl):
    """Maps metabolite-pathway relationships and identifies biomarker signatures."""

    agent_type = AgentType.METABOLOMICS_ANALYST

    async def _investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._multi_turn_investigate(
            task,
            kg_context,
            investigation_focus=(
                "Identify metabolites relevant to the research question. "
                "Map metabolite-enzyme-gene relationships using KEGG pathways. "
                "Assess biomarker potential by searching PubMed for metabolomics studies. "
                "Create COMPOUND and BIOMARKER nodes with KEGG IDs as external references."
            ),
        )
```

**What you can override:**

| Method | When to Override |
|--------|-----------------|
| `_investigate(task, kg_context)` | **Always** — this is the main subclass hook. Most agents delegate to `_multi_turn_investigate()` with a custom `investigation_focus`. |
| `falsify(edges)` | Only if you need a custom falsification strategy (e.g., structural validation instead of literature search). The default searches PubMed/Semantic Scholar for counter-evidence. |
| `get_uncertainty()` | Only if you have domain-specific uncertainty signals (e.g., pLDDT scores for structural predictions). |

**When you DON'T need a subclass:** If the template's system prompt and tools are sufficient, you can skip the subclass entirely and use `BaseAgentImpl` directly via `create_agent_from_spec()`. The base `_investigate()` method runs the multi-turn loop using the spec's role and instructions when no subclass override exists.

### Step 4: Register in the Factory

**File:** `backend/agents/factory.py`

Add the import and register in `_AGENT_CLASS_MAP`:

```python
from agents.metabolomics_analyst import MetabolomicsAnalystAgent

_AGENT_CLASS_MAP: dict[AgentType, type[BaseAgentImpl]] = {
    # ... existing entries ...
    AgentType.METABOLOMICS_ANALYST: MetabolomicsAnalystAgent,
}
```

### Step 5: Add Tests

Create `backend/tests/test_metabolomics_analyst.py`:

```python
"""Tests for the Metabolomics Analyst agent."""

import pytest

from agents.factory import create_agent
from agents.templates import AGENT_TEMPLATES
from core.models import AgentType


def test_template_registered():
    """Template exists in the registry."""
    assert AgentType.METABOLOMICS_ANALYST in AGENT_TEMPLATES
    template = AGENT_TEMPLATES[AgentType.METABOLOMICS_ANALYST]
    assert template.display_name == "Metabolomics Analyst"
    assert "kegg" in template.tools


def test_factory_creates_agent(mock_llm, mock_kg):
    """Factory creates the correct agent class."""
    agent = create_agent(
        AgentType.METABOLOMICS_ANALYST,
        llm=mock_llm,
        kg=mock_kg,
    )
    assert agent.agent_type == AgentType.METABOLOMICS_ANALYST
    assert agent.template is not None


@pytest.mark.asyncio
async def test_investigate_returns_expected_keys(mock_llm, mock_kg, sample_task):
    """Agent investigation returns the expected result structure."""
    agent = create_agent(
        AgentType.METABOLOMICS_ANALYST,
        llm=mock_llm,
        kg=mock_kg,
    )
    result = await agent.execute(sample_task)
    assert result.agent_type == AgentType.METABOLOMICS_ANALYST
    assert isinstance(result.nodes_added, list)
    assert isinstance(result.edges_added, list)
```

---

## Template Fields Reference

The `AgentTemplate` model (`core/models.py`) has the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `agent_type` | `AgentType` | Enum value identifying this agent specialization |
| `display_name` | `str` | Human-readable name shown in the UI and API responses |
| `description` | `str` | Brief description of what this agent does |
| `system_prompt` | `str` | The LLM system prompt that defines the agent's persona, capabilities, and output format. This is the most important field — it determines how the agent reasons and what it produces. |
| `tools` | `list[str]` | Tool names this agent can use (e.g., `["pubmed", "semantic_scholar"]`). Must match keys in the tool registry. Available tools: `pubmed`, `semantic_scholar`, `uniprot`, `kegg`, `reactome`, `mygene`, `chembl`, `clinicaltrials`, `esm`, `python_repl` |
| `kg_write_permissions` | `list[NodeType]` | Which node types this agent is allowed to create. Enforced during KG writes. |
| `kg_edge_permissions` | `list[EdgeRelationType]` | Which edge/relation types this agent is allowed to create. |
| `requires_yami` | `bool` | Whether this agent needs the Yami/ESM interface for protein structure prediction. If `True` and Yami is unavailable, the agent will error on Yami calls. |
| `falsification_protocol` | `str` | Instructions for how this agent should attempt to falsify its own claims. Injected into the falsification prompt. Empty string means no self-falsification. |
| `max_iterations` | `int` | Maximum number of investigation iterations (maps to `max_turns / 2` in constraints). Default: `10`. |
| `timeout_seconds` | `int` | Wall-clock timeout for the entire agent execution. Default: `300` (5 minutes). |

### System Prompt Best Practices

1. **Define the persona** — "You are a [specialist] with expertise in [domain]."
2. **List numbered steps** — Give the agent a clear investigation procedure (5-7 steps).
3. **Specify output format** — "Output structured JSON with [NODE_TYPE] nodes and [RELATION] edges."
4. **Set evidence requirements** — "Every claim must cite a [source type]."
5. **Define boundaries** — "You may NOT [action]. You are the [role], not the [other role]."

---

## Dynamic Specs (`AgentSpec`)

The orchestrator can create agents **without pre-defined templates** using `AgentSpec`. This is used for dynamic orchestration where the orchestrator LLM decides what role an agent should play.

### `AgentSpec` Fields

| Field | Type | Description |
|-------|------|-------------|
| `role` | `str` | Free-text role description (e.g., "Epigenetics specialist focusing on histone modifications") |
| `instructions` | `str` | Detailed instructions for the agent's investigation |
| `tools` | `list[str]` | Tool names to make available |
| `constraints` | `AgentConstraints` | Resource limits (see below) |
| `parent_agent_id` | `str \| None` | ID of the parent agent (for sub-agent spawning) |
| `hypothesis_branch` | `str` | Which hypothesis branch this agent is investigating |
| `agent_type_hint` | `AgentType \| None` | Optional hint to use a specialized subclass |
| `system_prompt` | `str` | Override system prompt (takes precedence over template) |
| `kg_write_permissions` | `list[NodeType]` | Override KG node permissions |
| `kg_edge_permissions` | `list[EdgeRelationType]` | Override KG edge permissions |
| `falsification_protocol` | `str` | Override falsification instructions |

### `AgentConstraints` Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_turns` | `int` | `200` | Maximum multi-turn loop iterations |
| `token_budget` | `int` | `50,000` | Per-agent token limit |
| `timeout_seconds` | `int` | `300` | Wall-clock timeout |
| `max_llm_calls` | `int` | `20` | Maximum LLM API calls |

### Resolution Priority

When both a template and a spec are provided, fields resolve with this priority:

```
spec override → template default → BaseAgentImpl fallback
```

This is implemented via `@property` accessors on `BaseAgentImpl`:

- `effective_system_prompt` — spec → template → empty
- `effective_kg_write_permissions` — spec → template → all types
- `effective_kg_edge_permissions` — spec → template → all types
- `effective_falsification_protocol` — spec → template → empty
- `effective_constraints` — spec → derived from template → defaults

### Example: Creating an Agent from Spec

```python
from agents.factory import create_agent_from_spec
from core.models import AgentSpec, AgentConstraints, AgentType, NodeType

spec = AgentSpec(
    role="Epigenetics specialist",
    instructions="Investigate histone modification patterns in the context of the research question.",
    tools=["pubmed", "semantic_scholar"],
    constraints=AgentConstraints(max_turns=15, token_budget=30000),
    agent_type_hint=AgentType.LITERATURE_ANALYST,  # reuse LiteratureAnalyst subclass
    kg_write_permissions=[NodeType.GENE, NodeType.PROTEIN, NodeType.MECHANISM],
)

agent = create_agent_from_spec(spec, llm=llm, kg=kg, tools=tool_instances)
result = await agent.execute(task)
```

If `agent_type_hint` is `None`, a generic `BaseAgentImpl` is created that runs the multi-turn loop using only the spec's role and instructions.

---

## Existing Agent Reference

| Agent Type | Class | File | Tools | Requires Yami | Key Capability |
|------------|-------|------|-------|---------------|----------------|
| `literature_analyst` | `LiteratureAnalystAgent` | `agents/literature_analyst.py` | pubmed, semantic_scholar | No | Extracts biological claims from publications |
| `protein_engineer` | `ProteinEngineerAgent` | `agents/protein_engineer.py` | uniprot, esm | **Yes** | Protein structure prediction, domain analysis |
| `genomics_mapper` | `GenomicsMapperAgent` | `agents/genomics_mapper.py` | mygene, kegg | No | Gene-pathway mapping, regulatory networks |
| `pathway_analyst` | `PathwayAnalystAgent` | `agents/pathway_analyst.py` | kegg, reactome | No | Signaling cascades, pathway cross-talk |
| `drug_hunter` | `DrugHunterAgent` | `agents/drug_hunter.py` | chembl, clinicaltrials | No | Drug-target binding, repurposing opportunities |
| `clinical_analyst` | `ClinicalAnalystAgent` | `agents/clinical_analyst.py` | clinicaltrials, pubmed | No | Clinical trial analysis, failure analysis |
| `scientific_critic` | `ScientificCriticAgent` | `agents/scientific_critic.py` | pubmed, semantic_scholar | No | Systematic falsification of KG edges |
| `experiment_designer` | `ExperimentDesignerAgent` | `agents/experiment_designer.py` | *(none — reasoning only)* | No | Designs experiments to resolve uncertainty |
| `tool_creator` | `ToolCreatorAgent` | `agents/tool_creator.py` | pubmed, semantic_scholar, python_repl | No | Discovers and creates new tool wrappers (STELLA-inspired) |

### Example: LiteratureAnalystAgent

**File:** `backend/agents/literature_analyst.py`

```python
class LiteratureAnalystAgent(BaseAgentImpl):
    agent_type = AgentType.LITERATURE_ANALYST

    async def _investigate(self, task, kg_context):
        return await self._multi_turn_investigate(
            task, kg_context,
            max_turns=8,
            investigation_focus=(
                "Search PubMed and Semantic Scholar for papers relevant to the "
                "research question. Extract biological entities and their "
                "relationships from paper abstracts. Every claim must cite a "
                "PMID or DOI."
            ),
        )
```

This is the canonical pattern: a thin subclass that sets `agent_type` and delegates to `_multi_turn_investigate()` with a domain-specific `investigation_focus` string.

### Example: ProteinEngineerAgent

**File:** `backend/agents/protein_engineer.py`

```python
class ProteinEngineerAgent(BaseAgentImpl):
    agent_type = AgentType.PROTEIN_ENGINEER

    async def _investigate(self, task, kg_context):
        return await self._multi_turn_investigate(
            task, kg_context,
            investigation_focus=(
                "Look up protein sequences and annotations from UniProt. "
                "Predict protein structure using Yami (ESMFold) for sequences "
                "≤400 residues. Identify functional domains, active sites, and "
                "binding interfaces. Map protein-protein interactions. Include "
                "pLDDT scores as confidence metrics."
            ),
        )
```

Note: This agent's template has `requires_yami=True`, meaning it expects the Yami/ESM interface to be available for structural predictions.

---

## The Multi-Turn Investigation Loop

The `_multi_turn_investigate()` method in `BaseAgentImpl` is the core execution engine. Understanding it helps you write better `investigation_focus` strings.

### Loop Structure

1. **Planning phase** — The LLM receives the task, KG context, available tools, and KG permissions. It creates a 3-8 step investigation plan wrapped in `<think>` tags.

2. **Execution loop** (up to `max_turns` iterations) — Each turn, the LLM can:
   - `<think>reasoning</think>` — Internal reasoning (no side effects)
   - `<tool>tool_name:{"arg": "value"}</tool>` — Call an external tool
   - `<execute>python_code</execute>` — Run Python in the sandboxed REPL
   - `<answer>{...}</answer>` — Provide the final structured answer (terminates the loop)

3. **Budget enforcement** — The loop tracks token usage and remaining turns. When budget is low, urgency messages nudge the agent toward answering.

4. **Answer parsing** — The `<answer>` JSON is parsed into `KGNode` and `KGEdge` objects, which are then written to the knowledge graph.

### Virtual KG Tools

During the multi-turn loop, agents can also use virtual KG tools:
- `kg_get_recent_edges` — Get recently added edges
- `kg_get_node` — Look up a specific node
- `kg_search_nodes` — Search nodes by name/type

These are handled internally by `_execute_kg_tool()` without going through the external tool registry.

---

## Checklist for Adding a New Agent

- [ ] Add enum value to `AgentType` in `core/models.py`
- [ ] Create `AgentTemplate` in `agents/templates.py` with system prompt, tools, KG permissions, and falsification protocol
- [ ] Register template in `AGENT_TEMPLATES` dict
- [ ] Create agent subclass in `agents/<your_agent>.py` overriding `_investigate()`
- [ ] Import and register in `_AGENT_CLASS_MAP` in `agents/factory.py`
- [ ] Add tests in `backend/tests/`
- [ ] Verify the agent appears in `GET /api/v1/templates`
