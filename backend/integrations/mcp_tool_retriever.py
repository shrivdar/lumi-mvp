"""MCP Tool Retriever — dynamic tool selection for agent swarms.

Given a research query, selects the most relevant MCP tools from the registry
and returns them in a format agents can use. Works with both native tools
(via IntegrationsRegistry) and MCP tools (via MCPServerManager).
"""

from __future__ import annotations

import structlog

from core.models import ToolRegistryEntry, ToolSourceType
from core.tool_registry import InMemoryToolRegistry

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Category relevance scoring — maps research keywords to tool categories
# ---------------------------------------------------------------------------

KEYWORD_CATEGORY_MAP: dict[str, list[str]] = {
    # Genomics keywords
    "gene": ["genomics", "gene_expression", "pathway_analysis"],
    "genome": ["genomics", "variant_analysis"],
    "variant": ["variant_analysis", "genomics"],
    "mutation": ["variant_analysis", "genomics"],
    "snp": ["variant_analysis", "genomics"],
    "chromosome": ["genomics"],
    "sequence": ["genomics", "protein_analysis", "computation"],
    "transcript": ["genomics", "gene_expression"],

    # Protein keywords
    "protein": ["protein_analysis", "structural_biology", "proteomics"],
    "structure": ["structural_biology"],
    "domain": ["protein_analysis", "structural_biology"],
    "fold": ["structural_biology"],
    "binding": ["structural_biology", "drug_discovery"],
    "interaction": ["network_analysis", "protein_analysis"],
    "ppi": ["network_analysis"],

    # Drug keywords
    "drug": ["drug_discovery", "safety_toxicology", "regulatory_data"],
    "compound": ["drug_discovery", "chemistry"],
    "molecule": ["drug_discovery", "chemistry"],
    "target": ["drug_discovery", "network_analysis"],
    "pharmacology": ["drug_discovery", "safety_toxicology"],
    "admet": ["safety_toxicology"],
    "toxicity": ["safety_toxicology"],
    "docking": ["drug_discovery", "structural_biology"],
    "smiles": ["chemistry", "drug_discovery"],

    # Clinical keywords
    "clinical": ["clinical_data", "regulatory_data"],
    "trial": ["clinical_data"],
    "patient": ["clinical_data"],
    "adverse": ["safety_toxicology", "clinical_data"],
    "fda": ["regulatory_data", "clinical_data"],

    # Pathway keywords
    "pathway": ["pathway_analysis"],
    "signaling": ["pathway_analysis", "network_analysis"],
    "network": ["network_analysis"],
    "enrichment": ["pathway_analysis", "ontology_annotation"],
    "go": ["ontology_annotation"],
    "ontology": ["ontology_annotation"],

    # Expression keywords
    "expression": ["gene_expression"],
    "rna": ["gene_expression", "genomics"],
    "transcriptome": ["gene_expression"],
    "single-cell": ["gene_expression"],
    "scrna": ["gene_expression"],
    "tissue": ["gene_expression"],

    # Disease keywords
    "disease": ["ontology_annotation", "clinical_data"],
    "phenotype": ["ontology_annotation"],
    "cancer": ["variant_analysis", "gene_expression"],
    "tumor": ["variant_analysis", "gene_expression"],

    # Epigenetics keywords
    "epigenetic": ["epigenetics"],
    "methylation": ["epigenetics"],
    "chromatin": ["epigenetics"],
    "histone": ["epigenetics"],
    "regulatory": ["epigenetics", "regulatory_data"],
    "enhancer": ["epigenetics"],
    "promoter": ["epigenetics"],
    "mirna": ["epigenetics"],

    # Chemistry keywords
    "chemical": ["chemistry", "drug_discovery"],
    "metabolite": ["metabolomics"],
    "lipid": ["metabolomics"],

    # Literature
    "paper": ["literature_search"],
    "publication": ["literature_search"],
    "literature": ["literature_search"],
    "citation": ["literature_search"],
}


class MCPToolRetriever:
    """Selects relevant MCP tools for a given research context.

    Used by SwarmComposer to assign tools to agents based on:
    1. Keyword matching from the research query
    2. Agent specialization (agent_type → relevant categories)
    3. Explicit tool requests from the user

    Usage::

        retriever = MCPToolRetriever(registry)
        tools = retriever.retrieve_for_query("TP53 mutations in NSCLC", max_tools=20)
        tools = retriever.retrieve_for_agent("literature_agent", query="B7-H3")
        tools = retriever.retrieve_for_categories(["drug_discovery", "clinical_data"])
    """

    def __init__(self, registry: InMemoryToolRegistry) -> None:
        self._registry = registry

    def retrieve_for_query(
        self,
        query: str,
        *,
        max_tools: int = 30,
        source_types: list[ToolSourceType] | None = None,
        exclude_categories: list[str] | None = None,
    ) -> list[ToolRegistryEntry]:
        """Retrieve tools relevant to a free-text research query."""
        category_scores = self._score_categories(query)
        if not category_scores:
            # No keyword matches — return a broad selection
            return self._broad_selection(max_tools, source_types)

        # Sort categories by relevance score
        ranked_categories = sorted(category_scores.items(), key=lambda x: -x[1])
        exclude = set(exclude_categories or [])

        selected: list[ToolRegistryEntry] = []
        seen: set[str] = set()

        for category, _score in ranked_categories:
            if category in exclude:
                continue
            tools = self._registry.list_tools(category=category)
            if source_types:
                tools = [t for t in tools if t.source_type in source_types]
            for tool in tools:
                if tool.name not in seen and len(selected) < max_tools:
                    selected.append(tool)
                    seen.add(tool.name)

        logger.debug(
            "tools_retrieved",
            query=query[:50],
            categories=[c for c, _ in ranked_categories[:5]],
            tool_count=len(selected),
        )
        return selected

    def retrieve_for_agent(
        self,
        agent_type: str,
        query: str = "",
        *,
        max_tools: int = 15,
    ) -> list[ToolRegistryEntry]:
        """Retrieve tools relevant to a specific agent type."""
        agent_categories = AGENT_CATEGORY_MAP.get(agent_type, [])

        # Start with agent's primary categories
        selected: list[ToolRegistryEntry] = []
        seen: set[str] = set()

        for category in agent_categories:
            tools = self._registry.list_tools(category=category)
            for tool in tools:
                if tool.name not in seen and len(selected) < max_tools:
                    selected.append(tool)
                    seen.add(tool.name)

        # If query provided, augment with query-relevant tools
        if query and len(selected) < max_tools:
            query_tools = self.retrieve_for_query(query, max_tools=max_tools - len(selected))
            for tool in query_tools:
                if tool.name not in seen and len(selected) < max_tools:
                    selected.append(tool)
                    seen.add(tool.name)

        return selected

    def retrieve_for_categories(
        self,
        categories: list[str],
        *,
        max_tools: int = 50,
        source_types: list[ToolSourceType] | None = None,
    ) -> list[ToolRegistryEntry]:
        """Retrieve all tools from specific categories."""
        selected: list[ToolRegistryEntry] = []
        seen: set[str] = set()

        for category in categories:
            tools = self._registry.list_tools(category=category)
            if source_types:
                tools = [t for t in tools if t.source_type in source_types]
            for tool in tools:
                if tool.name not in seen and len(selected) < max_tools:
                    selected.append(tool)
                    seen.add(tool.name)

        return selected

    def retrieve_by_name(self, tool_names: list[str]) -> list[ToolRegistryEntry]:
        """Retrieve specific tools by name."""
        tools = []
        for name in tool_names:
            tool = self._registry.get_tool(name)
            if tool:
                tools.append(tool)
        return tools

    def get_tool_summary(self, tools: list[ToolRegistryEntry]) -> str:
        """Generate a compact text summary of tools for LLM prompts."""
        lines = []
        by_category: dict[str, list[ToolRegistryEntry]] = {}
        for tool in tools:
            by_category.setdefault(tool.category, []).append(tool)

        for category, entries in sorted(by_category.items()):
            lines.append(f"\n## {category.replace('_', ' ').title()}")
            for tool in entries:
                caps = ", ".join(tool.capabilities[:3]) if tool.capabilities else ""
                source = tool.source_type.value
                lines.append(f"  - [{source}] {tool.name}: {tool.description[:60]}... [{caps}]")

        return f"Available tools ({len(tools)}):\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _score_categories(self, query: str) -> dict[str, float]:
        """Score categories based on keyword matches in the query."""
        query_lower = query.lower()
        scores: dict[str, float] = {}
        for keyword, categories in KEYWORD_CATEGORY_MAP.items():
            if keyword in query_lower:
                for i, cat in enumerate(categories):
                    # First category in list gets higher weight
                    weight = 1.0 / (i + 1)
                    scores[cat] = scores.get(cat, 0) + weight
        return scores

    def _broad_selection(
        self,
        max_tools: int,
        source_types: list[ToolSourceType] | None,
    ) -> list[ToolRegistryEntry]:
        """Return a broad selection when no specific categories match."""
        tools = self._registry.list_tools()
        if source_types:
            tools = [t for t in tools if t.source_type in source_types]
        return tools[:max_tools]


# ---------------------------------------------------------------------------
# Agent type → category mapping
# ---------------------------------------------------------------------------

AGENT_CATEGORY_MAP: dict[str, list[str]] = {
    "literature_agent": ["literature_search", "web_search"],
    "protein_agent": ["protein_analysis", "structural_biology", "proteomics", "network_analysis"],
    "genomics_agent": ["genomics", "variant_analysis", "gene_expression"],
    "pathway_agent": ["pathway_analysis", "network_analysis", "ontology_annotation"],
    "drug_agent": ["drug_discovery", "chemistry", "safety_toxicology", "regulatory_data"],
    "clinical_agent": ["clinical_data", "regulatory_data", "safety_toxicology"],
    "critic_agent": ["literature_search", "computation"],
    "experiment_agent": ["computation", "gene_expression", "structural_biology"],
}
