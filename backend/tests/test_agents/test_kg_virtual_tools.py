"""Tests for KG virtual tools — kg_add_node and kg_add_edge."""

from __future__ import annotations

import json

import pytest

from agents.base import BaseAgentImpl
from agents.templates import get_template
from core.models import (
    AgentTask,
    AgentType,
    EdgeConfidence,
    EdgeRelationType,
    EvidenceSource,
    EvidenceSourceType,
    KGEdge,
    KGNode,
    NodeType,
)
from tests.test_agents.conftest import MockLLMClient
from world_model.knowledge_graph import InMemoryKnowledgeGraph


# ---------------------------------------------------------------------------
# Concrete subclass for testing
# ---------------------------------------------------------------------------


class StubAgent(BaseAgentImpl):
    """Minimal agent for testing KG virtual tools."""

    agent_type = AgentType.LITERATURE_ANALYST

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def _investigate(self, task, kg_context):
        return {
            "nodes": [],
            "edges": [],
            "summary": "Stub investigation",
            "reasoning_trace": "stub",
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kg() -> InMemoryKnowledgeGraph:
    return InMemoryKnowledgeGraph(graph_id="test-kg-virtual-tools")


@pytest.fixture()
def agent(kg: InMemoryKnowledgeGraph) -> StubAgent:
    template = get_template(AgentType.LITERATURE_ANALYST)
    llm = MockLLMClient()
    a = StubAgent(
        agent_id="agent-test-001",
        template=template,
        llm=llm,
        kg=kg,
    )
    # Simulate execute() setting _current_task
    a._current_task = AgentTask(
        task_id="task-001",
        research_id="research-001",
        agent_type=AgentType.LITERATURE_ANALYST,
        agent_id="agent-test-001",
        hypothesis_branch="h-test-branch",
        instruction="Test instruction",
        context={},
    )
    return a


@pytest.fixture()
def sample_task() -> AgentTask:
    return AgentTask(
        task_id="task-001",
        research_id="research-001",
        agent_type=AgentType.LITERATURE_ANALYST,
        agent_id="agent-test-001",
        hypothesis_branch="h-test-branch",
        instruction="Test instruction",
        context={},
    )


# ---------------------------------------------------------------------------
# Tests: kg_add_node
# ---------------------------------------------------------------------------


class TestKgAddNode:
    """Tests for the kg_add_node virtual tool."""

    def test_add_node_basic(self, agent: StubAgent, kg: InMemoryKnowledgeGraph):
        """Adding a node via virtual tool creates it in the KG."""
        result_str = agent._execute_kg_add_node({
            "name": "BRCA1",
            "type": "GENE",
            "description": "Breast cancer gene 1",
            "confidence": 0.9,
        })
        result = json.loads(result_str)

        assert result["status"] == "created"
        assert result["name"] == "BRCA1"
        assert result["type"] == "GENE"
        assert result["node_id"]

        # Verify node is in KG
        node = kg.get_node_by_name("BRCA1")
        assert node is not None
        assert node.name == "BRCA1"
        assert node.type == NodeType.GENE

        # Verify node is tracked incrementally
        assert len(agent._incremental_nodes) == 1
        assert agent._incremental_nodes[0].name == "BRCA1"

        # Verify node is tracked in _nodes_added (via write_node)
        assert len(agent._nodes_added) == 1

    def test_add_node_with_properties(self, agent: StubAgent, kg: InMemoryKnowledgeGraph):
        """Node properties are preserved."""
        agent._execute_kg_add_node({
            "name": "TP53",
            "type": "GENE",
            "description": "Tumor protein p53",
            "properties": {"chromosome": "17p13.1", "aliases": ["p53"]},
            "confidence": 0.95,
        })

        node = kg.get_node_by_name("TP53")
        assert node is not None
        assert node.properties.get("chromosome") == "17p13.1"

    def test_add_node_duplicate_in_kg(self, agent: StubAgent, kg: InMemoryKnowledgeGraph):
        """Adding a node that already exists in KG returns already_exists."""
        # First add
        agent._execute_kg_add_node({"name": "BRCA1", "type": "GENE"})

        # Second add — should detect duplicate
        result_str = agent._execute_kg_add_node({"name": "BRCA1", "type": "GENE"})
        result = json.loads(result_str)

        assert result["status"] == "already_exists"
        assert result["name"] == "BRCA1"

        # Only one node should exist
        assert len(agent._incremental_nodes) == 1

    def test_add_node_duplicate_case_insensitive(self, agent: StubAgent, kg: InMemoryKnowledgeGraph):
        """Duplicate detection is case-insensitive for incremental nodes."""
        agent._execute_kg_add_node({"name": "BRCA1", "type": "GENE"})

        result_str = agent._execute_kg_add_node({"name": "brca1", "type": "GENE"})
        result = json.loads(result_str)

        assert result["status"] == "already_exists"

    def test_add_node_missing_name(self, agent: StubAgent):
        """Missing name returns an error."""
        result_str = agent._execute_kg_add_node({"type": "GENE"})
        result = json.loads(result_str)

        assert "error" in result

    def test_add_node_invalid_type_defaults(self, agent: StubAgent, kg: InMemoryKnowledgeGraph):
        """Invalid node type defaults to GENE."""
        agent._execute_kg_add_node({
            "name": "SomeEntity",
            "type": "INVALID_TYPE",
        })

        node = kg.get_node_by_name("SomeEntity")
        assert node is not None
        assert node.type == NodeType.GENE

    def test_add_node_sets_hypothesis_branch(self, agent: StubAgent, kg: InMemoryKnowledgeGraph):
        """Node gets hypothesis_branch from current task."""
        agent._execute_kg_add_node({"name": "EGFR", "type": "GENE"})

        node = kg.get_node_by_name("EGFR")
        assert node is not None
        assert node.hypothesis_branch == "h-test-branch"

    def test_add_node_sets_agent_provenance(self, agent: StubAgent, kg: InMemoryKnowledgeGraph):
        """Node gets agent_id provenance."""
        agent._execute_kg_add_node({"name": "KRAS", "type": "GENE"})

        node = kg.get_node_by_name("KRAS")
        assert node is not None
        assert node.created_by == "agent-test-001"


# ---------------------------------------------------------------------------
# Tests: kg_add_edge
# ---------------------------------------------------------------------------


class TestKgAddEdge:
    """Tests for the kg_add_edge virtual tool."""

    def _add_two_nodes(self, agent: StubAgent):
        """Helper to add source and target nodes."""
        agent._execute_kg_add_node({"name": "BRCA1", "type": "GENE"})
        agent._execute_kg_add_node({"name": "Breast Cancer", "type": "DISEASE"})

    def test_add_edge_basic(self, agent: StubAgent, kg: InMemoryKnowledgeGraph):
        """Adding an edge between existing nodes works."""
        self._add_two_nodes(agent)

        result_str = agent._execute_kg_add_edge({
            "source": "BRCA1",
            "target": "Breast Cancer",
            "relation": "ASSOCIATED_WITH",
            "confidence": 0.85,
            "claim": "BRCA1 mutations increase breast cancer risk",
        })
        result = json.loads(result_str)

        assert result["status"] == "created"
        assert result["source"] == "BRCA1"
        assert result["target"] == "Breast Cancer"
        assert result["confidence"] == 0.85

        # Verify edge is tracked incrementally
        assert len(agent._incremental_edges) == 1

        # Verify edge is tracked in _edges_added (via write_edge)
        assert len(agent._edges_added) == 1

    def test_add_edge_missing_source(self, agent: StubAgent, kg: InMemoryKnowledgeGraph):
        """Adding edge with non-existent source returns error."""
        agent._execute_kg_add_node({"name": "Breast Cancer", "type": "DISEASE"})

        result_str = agent._execute_kg_add_edge({
            "source": "NONEXISTENT",
            "target": "Breast Cancer",
            "relation": "ASSOCIATED_WITH",
        })
        result = json.loads(result_str)

        assert "error" in result
        assert "NONEXISTENT" in result["error"]

    def test_add_edge_missing_target(self, agent: StubAgent, kg: InMemoryKnowledgeGraph):
        """Adding edge with non-existent target returns error."""
        agent._execute_kg_add_node({"name": "BRCA1", "type": "GENE"})

        result_str = agent._execute_kg_add_edge({
            "source": "BRCA1",
            "target": "NONEXISTENT",
            "relation": "ASSOCIATED_WITH",
        })
        result = json.loads(result_str)

        assert "error" in result
        assert "NONEXISTENT" in result["error"]

    def test_add_edge_missing_names(self, agent: StubAgent):
        """Missing source/target names returns error."""
        result_str = agent._execute_kg_add_edge({
            "relation": "ASSOCIATED_WITH",
        })
        result = json.loads(result_str)

        assert "error" in result

    def test_add_edge_with_evidence_strings(self, agent: StubAgent, kg: InMemoryKnowledgeGraph):
        """Evidence can be provided as a list of strings."""
        self._add_two_nodes(agent)

        agent._execute_kg_add_edge({
            "source": "BRCA1",
            "target": "Breast Cancer",
            "relation": "ASSOCIATED_WITH",
            "confidence": 0.8,
            "evidence": ["PMID:12345", "PMID:67890"],
        })

        assert len(agent._incremental_edges) == 1
        edge = agent._incremental_edges[0]
        assert len(edge.evidence) == 2

    def test_add_edge_with_evidence_dicts(self, agent: StubAgent, kg: InMemoryKnowledgeGraph):
        """Evidence can be provided as a list of dicts."""
        self._add_two_nodes(agent)

        agent._execute_kg_add_edge({
            "source": "BRCA1",
            "target": "Breast Cancer",
            "relation": "ASSOCIATED_WITH",
            "confidence": 0.8,
            "evidence": [
                {"source_type": "PUBMED", "source_id": "PMID:12345", "claim": "Strong association"},
            ],
        })

        assert len(agent._incremental_edges) == 1
        edge = agent._incremental_edges[0]
        assert edge.evidence[0].source_type == EvidenceSourceType.PUBMED

    def test_add_edge_invalid_relation_defaults(self, agent: StubAgent, kg: InMemoryKnowledgeGraph):
        """Invalid relation type defaults to ASSOCIATED_WITH."""
        self._add_two_nodes(agent)

        agent._execute_kg_add_edge({
            "source": "BRCA1",
            "target": "Breast Cancer",
            "relation": "INVALID_RELATION",
            "confidence": 0.7,
        })

        edge = agent._incremental_edges[0]
        assert edge.relation == EdgeRelationType.ASSOCIATED_WITH

    def test_add_edge_sets_hypothesis_branch(self, agent: StubAgent, kg: InMemoryKnowledgeGraph):
        """Edge gets hypothesis_branch from current task."""
        self._add_two_nodes(agent)

        agent._execute_kg_add_edge({
            "source": "BRCA1",
            "target": "Breast Cancer",
            "relation": "ASSOCIATED_WITH",
        })

        edge = agent._incremental_edges[0]
        assert edge.hypothesis_branch == "h-test-branch"


# ---------------------------------------------------------------------------
# Tests: Virtual tool dispatch via _execute_kg_tool
# ---------------------------------------------------------------------------


class TestKgToolDispatch:
    """Tests that kg_add_node/kg_add_edge are dispatched correctly."""

    def test_dispatch_kg_add_node(self, agent: StubAgent):
        """kg_add_node is dispatched through _execute_kg_tool."""
        result_str = agent._execute_kg_tool("kg_add_node", {
            "name": "BRCA2",
            "type": "GENE",
        })
        result = json.loads(result_str)
        assert result["status"] == "created"

    def test_dispatch_kg_add_edge(self, agent: StubAgent):
        """kg_add_edge is dispatched through _execute_kg_tool."""
        agent._execute_kg_tool("kg_add_node", {"name": "A", "type": "GENE"})
        agent._execute_kg_tool("kg_add_node", {"name": "B", "type": "GENE"})

        result_str = agent._execute_kg_tool("kg_add_edge", {
            "source": "A",
            "target": "B",
            "relation": "ASSOCIATED_WITH",
        })
        result = json.loads(result_str)
        assert result["status"] == "created"


# ---------------------------------------------------------------------------
# Tests: Compile methods merge incremental entities
# ---------------------------------------------------------------------------


class TestCompileMerge:
    """Tests that _compile_answer and _compile_from_observations merge incremental entities."""

    def test_compile_from_observations_returns_incremental(
        self, agent: StubAgent, sample_task: AgentTask
    ):
        """_compile_from_observations returns incrementally added nodes/edges."""
        agent._execute_kg_add_node({"name": "BRCA1", "type": "GENE"})
        agent._execute_kg_add_node({"name": "Breast Cancer", "type": "DISEASE"})
        agent._execute_kg_add_edge({
            "source": "BRCA1",
            "target": "Breast Cancer",
            "relation": "ASSOCIATED_WITH",
        })

        result = agent._compile_from_observations([], [], sample_task)

        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1
        assert "2 nodes" in result["summary"]
        assert "1 edges" in result["summary"]

    def test_compile_answer_merges_incremental_and_final(self, agent: StubAgent):
        """_compile_answer merges incremental nodes with final answer nodes, deduplicating."""
        # Add a node incrementally
        agent._execute_kg_add_node({"name": "BRCA1", "type": "GENE"})

        # Simulate an answer JSON that includes BRCA1 (duplicate) and TP53 (new)
        answer_json = json.dumps({
            "entities": [
                {"name": "BRCA1", "type": "GENE", "description": "dup", "confidence": 0.9},
                {"name": "TP53", "type": "GENE", "description": "new", "confidence": 0.8},
            ],
            "relationships": [],
            "summary": "Test summary",
            "reasoning_trace": "Test trace",
        })

        result = agent._compile_answer(answer_json, [], [])

        # Should have 2 nodes: BRCA1 (incremental) + TP53 (from answer), not 3
        node_names = [n.name for n in result["nodes"]]
        assert len(result["nodes"]) == 2
        assert "BRCA1" in node_names
        assert "TP53" in node_names

    def test_compile_answer_no_incremental(self, agent: StubAgent):
        """_compile_answer works normally when no incremental nodes exist."""
        answer_json = json.dumps({
            "entities": [
                {"name": "EGFR", "type": "GENE", "description": "test", "confidence": 0.7},
            ],
            "relationships": [],
            "summary": "Test",
            "reasoning_trace": "Test",
        })

        result = agent._compile_answer(answer_json, [], [])
        assert len(result["nodes"]) == 1
        assert result["nodes"][0].name == "EGFR"


# ---------------------------------------------------------------------------
# Tests: Tool descriptions include new tools
# ---------------------------------------------------------------------------


class TestToolDescriptions:
    """Tests that tool descriptions include kg_add_node and kg_add_edge."""

    def test_descriptions_include_kg_add_node(self, agent: StubAgent):
        desc = agent._build_tool_descriptions()
        assert "kg_add_node" in desc

    def test_descriptions_include_kg_add_edge(self, agent: StubAgent):
        desc = agent._build_tool_descriptions()
        assert "kg_add_edge" in desc

    def test_descriptions_include_incremental_hint(self, agent: StubAgent):
        desc = agent._build_tool_descriptions()
        assert "incrementally" in desc
