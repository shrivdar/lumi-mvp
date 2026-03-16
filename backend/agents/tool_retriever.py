"""ToolRetriever — LLM-based dynamic tool selection for agents.

Given a task prompt and hypothesis, selects the N most relevant tools from all
available tools in the registry. Uses a fast model (Sonnet) for cost efficiency.

Replaces static tool assignment in SwarmComposer.
"""

from __future__ import annotations

from typing import Any

import structlog

from core.config import settings
from core.models import ToolRegistryEntry

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Tool capability descriptions — richer than registry entries, includes
# example use cases so the LLM can match tasks to tools accurately.
# ---------------------------------------------------------------------------

TOOL_CAPABILITIES: dict[str, dict[str, Any]] = {
    "pubmed": {
        "description": "Search PubMed/MEDLINE for biomedical publications by keyword, MeSH term, or PMID.",
        "capabilities": [
            "keyword search across biomedical literature",
            "fetch paper abstracts, authors, MeSH terms",
            "citation chasing (cited_by)",
        ],
        "example_tasks": [
            "Find papers about BRCA1 mutations in breast cancer",
            "Search for clinical evidence of drug efficacy",
            "Look up counter-evidence for a biological claim",
        ],
        "category": "literature",
    },
    "semantic_scholar": {
        "description": "Search Semantic Scholar for academic papers with citation graphs and influence scores.",
        "capabilities": [
            "semantic search across scientific literature",
            "citation count and influence metrics",
            "paper metadata and abstracts",
        ],
        "example_tasks": [
            "Find highly cited papers on immune checkpoint inhibitors",
            "Discover recent publications on a protein target",
            "Get citation context for a specific claim",
        ],
        "category": "literature",
    },
    "uniprot": {
        "description": "Query UniProt for protein sequences, annotations, domains, and functional data.",
        "capabilities": [
            "protein sequence retrieval",
            "functional annotations and GO terms",
            "domain architecture and active sites",
            "protein-protein interaction data",
        ],
        "example_tasks": [
            "Get the sequence and domains of B7-H3 (CD276)",
            "Find post-translational modifications of a kinase",
            "Look up protein function and subcellular localization",
        ],
        "category": "protein",
    },
    "esm": {
        "description": "ESM-2/ESMFold for protein embeddings, structure prediction, and fitness estimation.",
        "capabilities": [
            "protein embedding generation",
            "3D structure prediction (ESMFold)",
            "mutation fitness scoring",
        ],
        "example_tasks": [
            "Predict the structure of a novel protein variant",
            "Generate embeddings for protein similarity analysis",
            "Score the fitness impact of point mutations",
        ],
        "category": "protein",
    },
    "kegg": {
        "description": "KEGG pathway database — pathway lookups, gene-pathway mappings, metabolic maps.",
        "capabilities": [
            "pathway information retrieval",
            "gene-to-pathway mapping",
            "metabolic and signaling pathway analysis",
        ],
        "example_tasks": [
            "Find pathways involving PI3K/AKT signaling",
            "Map a gene to its biological pathways",
            "Get pathway components and interactions",
        ],
        "category": "pathway",
    },
    "reactome": {
        "description": "Reactome pathway analysis — reaction-level pathway data, cross-references.",
        "capabilities": [
            "detailed pathway reaction data",
            "pathway hierarchy and cross-references",
            "event-based biological process data",
        ],
        "example_tasks": [
            "Get detailed reactions in the NF-kB signaling pathway",
            "Find upstream regulators of apoptosis",
            "Cross-reference pathway data with KEGG",
        ],
        "category": "pathway",
    },
    "mygene": {
        "description": "MyGene.info — gene annotation, orthologs, expression, and cross-database mapping.",
        "capabilities": [
            "gene annotation and metadata",
            "ortholog mapping across species",
            "gene-disease associations",
            "Entrez/Ensembl/HGNC ID conversion",
        ],
        "example_tasks": [
            "Get gene annotations and expression data for EGFR",
            "Map gene symbols to Entrez IDs",
            "Find disease associations for a gene list",
        ],
        "category": "genomics",
    },
    "chembl": {
        "description": "ChEMBL — drug/compound search, bioactivity data, target-drug mappings.",
        "capabilities": [
            "compound search by name or target",
            "bioactivity assay data (IC50, Ki, EC50)",
            "drug approval status and clinical phase",
            "target-compound interaction data",
        ],
        "example_tasks": [
            "Find inhibitors targeting B7-H3 with IC50 data",
            "Look up clinical phase for checkpoint inhibitors",
            "Search for compounds with activity against a kinase",
        ],
        "category": "drug",
    },
    "clinicaltrials": {
        "description": "ClinicalTrials.gov — search clinical trials by condition, intervention, status.",
        "capabilities": [
            "trial search by disease/drug/biomarker",
            "trial design, phase, and enrollment data",
            "trial outcomes and status tracking",
        ],
        "example_tasks": [
            "Find phase III trials for NSCLC immunotherapy",
            "Look up trial results for a specific drug",
            "Search for recruiting trials targeting B7-H3",
        ],
        "category": "clinical",
    },
    "slack": {
        "description": "Slack notifications for human-in-the-loop (HITL) decisions.",
        "capabilities": [
            "send notification to HITL channel",
            "request human expert input",
        ],
        "example_tasks": [
            "Notify the team about a high-uncertainty finding",
            "Request expert review of a controversial claim",
        ],
        "category": "communication",
    },
    "python_repl": {
        "description": "Sandboxed Python REPL for code execution, data analysis, and tool testing.",
        "capabilities": [
            "execute Python code in a sandboxed Docker container",
            "persistent namespace across code blocks",
            "data analysis and transformation",
        ],
        "example_tasks": [
            "Parse and analyze API response data",
            "Test a generated tool wrapper function",
            "Run data validation checks",
        ],
        "category": "compute",
    },
}

# Default number of tools to select
DEFAULT_TOP_K = 4


class ToolRetriever:
    """LLM-based dynamic tool selector.

    Given a task description and optional hypothesis context, asks an LLM
    to pick the N most relevant tools from the full registry. Designed to
    use a fast model (Sonnet) for cost efficiency.

    Usage::

        retriever = ToolRetriever(llm=llm_client, tool_entries=registry.list_tools())
        selected = await retriever.select_tools(
            task="Find papers about BRCA1 mutations",
            hypothesis="BRCA1 loss-of-function drives breast cancer",
            top_k=3,
        )
    """

    def __init__(
        self,
        *,
        llm: Any,  # LLMClient
        tool_entries: list[ToolRegistryEntry] | None = None,
    ) -> None:
        self.llm = llm
        self._tool_entries = {e.name: e for e in (tool_entries or [])}

    def _build_tool_catalog(self) -> str:
        """Build a formatted catalog of available tools for the LLM prompt."""
        lines: list[str] = []
        for name, entry in sorted(self._tool_entries.items()):
            if not entry.enabled:
                continue
            caps = TOOL_CAPABILITIES.get(name, {})
            desc = caps.get("description", entry.description)
            capabilities = caps.get("capabilities", [])
            examples = caps.get("example_tasks", [])

            lines.append(f"### {name}")
            lines.append(f"Description: {desc}")
            if capabilities:
                lines.append("Capabilities:")
                for c in capabilities:
                    lines.append(f"  - {c}")
            if examples:
                lines.append("Example tasks:")
                for ex in examples:
                    lines.append(f"  - {ex}")
            lines.append("")

        return "\n".join(lines)

    async def select_tools(
        self,
        task: str,
        hypothesis: str = "",
        top_k: int = DEFAULT_TOP_K,
        agent_type: str = "",
    ) -> list[str]:
        """Select the top_k most relevant tools for the given task.

        Returns a list of tool names (strings) ordered by relevance.
        Falls back to heuristic selection if LLM call fails.
        """
        available_tools = [n for n, e in self._tool_entries.items() if e.enabled]
        if len(available_tools) <= top_k:
            return available_tools

        catalog = self._build_tool_catalog()
        prompt = self._build_selection_prompt(task, hypothesis, agent_type, top_k, catalog)

        try:
            response = await self.llm.query(
                prompt,
                system_prompt=(
                    "You are a tool selection engine for a biomedical research platform. "
                    "Given a research task, select the most relevant tools from the catalog. "
                    "Respond with ONLY a JSON array of tool name strings, ordered by relevance. "
                    "Be precise — only select tools whose capabilities directly match the task."
                ),
                model=settings.llm_cheap_model,
            )
            from core.llm import LLMClient
            parsed = LLMClient.parse_json(response)
            if not isinstance(parsed, list):
                parsed = parsed.get("tools", []) if isinstance(parsed, dict) else []

            # Validate and filter to known tools
            selected = [
                str(t).strip()
                for t in parsed
                if str(t).strip() in self._tool_entries
                and self._tool_entries[str(t).strip()].enabled
            ]
            selected = selected[:top_k]

            if selected:
                logger.info(
                    "tool_retriever.selected",
                    task_preview=task[:100],
                    tools=selected,
                    top_k=top_k,
                )
                return selected

        except Exception as exc:
            logger.warning("tool_retriever.llm_failed", error=str(exc))

        # Fallback: heuristic selection
        return self._heuristic_select(task, hypothesis, top_k)

    def _build_selection_prompt(
        self,
        task: str,
        hypothesis: str,
        agent_type: str,
        top_k: int,
        catalog: str,
    ) -> str:
        lines = [
            "## Task",
            task,
        ]
        if hypothesis:
            lines.extend(["", "## Hypothesis Context", hypothesis])
        if agent_type:
            lines.extend(["", f"## Agent Type: {agent_type}"])
        lines.extend([
            "",
            f"## Available Tools (select up to {top_k})",
            catalog,
            "",
            f"Select the {top_k} most relevant tools for this task.",
            "Return a JSON array of tool name strings.",
        ])
        return "\n".join(lines)

    def _heuristic_select(
        self,
        task: str,
        hypothesis: str,
        top_k: int,
    ) -> list[str]:
        """Keyword-based fallback when LLM selection fails."""
        text = (task + " " + hypothesis).lower()

        # Score each tool based on keyword matches
        scores: dict[str, float] = {}
        keyword_map: dict[str, list[str]] = {
            "pubmed": ["paper", "publication", "literature", "study", "evidence", "abstract", "pubmed"],
            "semantic_scholar": ["citation", "paper", "literature", "academic", "influence"],
            "uniprot": ["protein", "sequence", "domain", "uniprot", "annotation", "amino acid"],
            "esm": ["structure", "fold", "esm", "embedding", "mutation", "fitness", "plddt"],
            "kegg": ["pathway", "kegg", "signaling", "metabolic", "cascade"],
            "reactome": ["pathway", "reaction", "reactome", "signaling", "process"],
            "mygene": ["gene", "expression", "ortholog", "genomic", "entrez", "ensembl"],
            "chembl": ["drug", "compound", "inhibitor", "ic50", "bioactivity", "chembl", "molecule"],
            "clinicaltrials": ["clinical", "trial", "phase", "patient", "efficacy", "nct"],
            "slack": ["notify", "human", "expert", "hitl", "review"],
            "python_repl": ["code", "execute", "repl", "test", "analyze", "compute", "script"],
        }

        for tool_name, keywords in keyword_map.items():
            if tool_name not in self._tool_entries:
                continue
            if not self._tool_entries[tool_name].enabled:
                continue
            score = sum(1.0 for kw in keywords if kw in text)
            if score > 0:
                scores[tool_name] = score

        # Sort by score descending, take top_k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [name for name, _ in ranked[:top_k]]

        # If we don't have enough, pad with pubmed and semantic_scholar
        fallback_order = ["pubmed", "semantic_scholar", "mygene", "kegg", "chembl"]
        for fb in fallback_order:
            if len(selected) >= top_k:
                break
            if fb not in selected and fb in self._tool_entries and self._tool_entries[fb].enabled:
                selected.append(fb)

        logger.info("tool_retriever.heuristic_fallback", tools=selected[:top_k])
        return selected[:top_k]
