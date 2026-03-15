"""Tests for MCPToolRetriever — query-based tool selection."""

from __future__ import annotations

import pytest

from core.models import ToolRegistryEntry, ToolSourceType
from core.tool_registry import InMemoryToolRegistry
from integrations.mcp_tool_retriever import MCPToolRetriever


def _build_registry() -> InMemoryToolRegistry:
    """Build a test registry with tools across multiple categories."""
    registry = InMemoryToolRegistry()
    tools = [
        ToolRegistryEntry(name="pubmed", description="Search PubMed", source_type=ToolSourceType.NATIVE, category="literature_search"),
        ToolRegistryEntry(name="ncbi_gene", description="NCBI Gene search", source_type=ToolSourceType.MCP, category="genomics", mcp_server="genomics"),
        ToolRegistryEntry(name="ensembl", description="Ensembl gene lookup", source_type=ToolSourceType.MCP, category="genomics", mcp_server="genomics"),
        ToolRegistryEntry(name="clinvar", description="ClinVar variant search", source_type=ToolSourceType.MCP, category="variant_analysis", mcp_server="genomics"),
        ToolRegistryEntry(name="pdb_search", description="PDB structure search", source_type=ToolSourceType.MCP, category="structural_biology", mcp_server="protein-structure"),
        ToolRegistryEntry(name="pubchem", description="PubChem compound search", source_type=ToolSourceType.MCP, category="drug_discovery", mcp_server="drug-discovery"),
        ToolRegistryEntry(name="openfda", description="OpenFDA adverse events", source_type=ToolSourceType.MCP, category="clinical_data", mcp_server="drug-discovery"),
        ToolRegistryEntry(name="string_db", description="STRING interactions", source_type=ToolSourceType.MCP, category="network_analysis", mcp_server="pathway-network"),
        ToolRegistryEntry(name="quickgo", description="QuickGO annotations", source_type=ToolSourceType.MCP, category="ontology_annotation", mcp_server="pathway-network"),
        ToolRegistryEntry(name="gtex", description="GTEx expression", source_type=ToolSourceType.MCP, category="gene_expression", mcp_server="gene-expression"),
        ToolRegistryEntry(name="encode", description="ENCODE epigenomics", source_type=ToolSourceType.MCP, category="epigenetics", mcp_server="regulatory-epigenomics"),
        ToolRegistryEntry(name="rdkit", description="RDKit cheminformatics", source_type=ToolSourceType.CONTAINER, category="chemistry"),
    ]
    for t in tools:
        registry.register(t)
    return registry


class TestMCPToolRetriever:
    def test_retrieve_gene_query(self) -> None:
        registry = _build_registry()
        retriever = MCPToolRetriever(registry)
        tools = retriever.retrieve_for_query("TP53 gene mutations in cancer")
        names = {t.name for t in tools}
        # Should include genomics and variant tools
        assert "ncbi_gene" in names
        assert "clinvar" in names

    def test_retrieve_drug_query(self) -> None:
        registry = _build_registry()
        retriever = MCPToolRetriever(registry)
        tools = retriever.retrieve_for_query("drug compound binding affinity")
        names = {t.name for t in tools}
        assert "pubchem" in names

    def test_retrieve_protein_structure_query(self) -> None:
        registry = _build_registry()
        retriever = MCPToolRetriever(registry)
        tools = retriever.retrieve_for_query("protein structure binding domain")
        names = {t.name for t in tools}
        assert "pdb_search" in names

    def test_retrieve_for_agent(self) -> None:
        registry = _build_registry()
        retriever = MCPToolRetriever(registry)
        tools = retriever.retrieve_for_agent("genomics_agent")
        names = {t.name for t in tools}
        assert "ncbi_gene" in names
        assert "clinvar" in names
        assert "gtex" in names

    def test_retrieve_for_agent_with_query(self) -> None:
        registry = _build_registry()
        retriever = MCPToolRetriever(registry)
        tools = retriever.retrieve_for_agent("literature_agent", query="drug interaction")
        names = {t.name for t in tools}
        assert "pubmed" in names  # from agent type
        assert "pubchem" in names  # from query

    def test_retrieve_for_categories(self) -> None:
        registry = _build_registry()
        retriever = MCPToolRetriever(registry)
        tools = retriever.retrieve_for_categories(["epigenetics", "chemistry"])
        names = {t.name for t in tools}
        assert "encode" in names
        assert "rdkit" in names
        assert "pubmed" not in names

    def test_retrieve_by_name(self) -> None:
        registry = _build_registry()
        retriever = MCPToolRetriever(registry)
        tools = retriever.retrieve_by_name(["pubmed", "pdb_search", "nonexistent"])
        assert len(tools) == 2

    def test_max_tools_limit(self) -> None:
        registry = _build_registry()
        retriever = MCPToolRetriever(registry)
        tools = retriever.retrieve_for_query("gene variant mutation protein drug", max_tools=3)
        assert len(tools) <= 3

    def test_source_type_filter(self) -> None:
        registry = _build_registry()
        retriever = MCPToolRetriever(registry)
        tools = retriever.retrieve_for_query("chemical compound", source_types=[ToolSourceType.CONTAINER])
        names = {t.name for t in tools}
        assert "rdkit" in names
        # MCP tools should be excluded
        assert "pubchem" not in names

    def test_get_tool_summary(self) -> None:
        registry = _build_registry()
        retriever = MCPToolRetriever(registry)
        tools = retriever.retrieve_for_query("gene")
        summary = retriever.get_tool_summary(tools)
        assert "Available tools" in summary
        assert "Genomics" in summary

    def test_empty_query_returns_broad_selection(self) -> None:
        registry = _build_registry()
        retriever = MCPToolRetriever(registry)
        tools = retriever.retrieve_for_query("xyz_no_keywords_match_123")
        # Should still return tools (broad selection)
        assert len(tools) > 0
