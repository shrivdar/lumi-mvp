"""Tests for multi-turn agent investigation loop — XML parsing, observation history, budget limits."""

from __future__ import annotations

import json

import pytest

from agents.base import BaseAgentImpl
from agents.templates import get_template
from core.models import (
    AgentType,
    TurnType,
)
from tests.test_agents.conftest import MockLLMClient

# ---------------------------------------------------------------------------
# Concrete subclass for multi-turn testing
# ---------------------------------------------------------------------------


class MultiTurnStubAgent(BaseAgentImpl):
    """Agent that delegates to _multi_turn_investigate for testing."""

    agent_type = AgentType.LITERATURE_ANALYST

    async def _investigate(self, task, kg_context):
        return await self._multi_turn_investigate(
            task, kg_context, investigation_focus="Test investigation",
        )


# ---------------------------------------------------------------------------
# XML Tag Parsing
# ---------------------------------------------------------------------------


class TestXMLParsing:
    """Test _extract_tag and _parse_agent_response."""

    def test_extract_think_tag(self):
        text = "Some preamble <think>my reasoning here</think> trailing"
        result = BaseAgentImpl._extract_tag(text, "think")
        assert result == "my reasoning here"

    def test_extract_tool_tag(self):
        text = '<tool>pubmed:{"action": "search", "query": "BRCA1"}</tool>'
        result = BaseAgentImpl._extract_tag(text, "tool")
        assert result == 'pubmed:{"action": "search", "query": "BRCA1"}'

    def test_extract_execute_tag(self):
        text = "<execute>print('hello')</execute>"
        result = BaseAgentImpl._extract_tag(text, "execute")
        assert result == "print('hello')"

    def test_extract_answer_tag(self):
        text = '<answer>{"summary": "done"}</answer>'
        result = BaseAgentImpl._extract_tag(text, "answer")
        assert result == '{"summary": "done"}'

    def test_extract_missing_tag_returns_none(self):
        text = "No tags here"
        assert BaseAgentImpl._extract_tag(text, "think") is None

    def test_extract_multiline_tag(self):
        text = "<think>\nLine 1\nLine 2\nLine 3\n</think>"
        result = BaseAgentImpl._extract_tag(text, "think")
        assert "Line 1" in result
        assert "Line 3" in result

    def test_parse_response_answer_priority(self, agent_kg):
        """Answer tag should take priority over tool tag."""
        template = get_template(AgentType.LITERATURE_ANALYST)
        llm = MockLLMClient()
        agent = MultiTurnStubAgent(template=template, llm=llm, kg=agent_kg)

        action_type, content, think = agent._parse_agent_response(
            '<think>reasoning</think><answer>{"done": true}</answer>'
        )
        assert action_type == "answer"
        assert content == '{"done": true}'
        assert think == "reasoning"

    def test_parse_response_tool(self, agent_kg):
        template = get_template(AgentType.LITERATURE_ANALYST)
        llm = MockLLMClient()
        agent = MultiTurnStubAgent(template=template, llm=llm, kg=agent_kg)

        action_type, content, think = agent._parse_agent_response(
            '<think>let me search</think><tool>pubmed:{"query": "test"}</tool>'
        )
        assert action_type == "tool"
        assert "pubmed" in content
        assert think == "let me search"

    def test_parse_response_think_only(self, agent_kg):
        template = get_template(AgentType.LITERATURE_ANALYST)
        llm = MockLLMClient()
        agent = MultiTurnStubAgent(template=template, llm=llm, kg=agent_kg)

        action_type, content, think = agent._parse_agent_response("just thinking")
        assert action_type == "think"
        assert content == "just thinking"


# ---------------------------------------------------------------------------
# Multi-turn Investigation Loop
# ---------------------------------------------------------------------------


class TestMultiTurnLoop:
    """Test the _multi_turn_investigate method end-to-end."""

    @pytest.mark.asyncio
    async def test_basic_three_turn_investigation(self, agent_kg, mock_tools, sample_task):
        """Agent can execute 3+ turns, observe results, and produce answer."""
        llm = MockLLMClient(responses=[
            # Turn 0: plan
            "<think>1. Search pubmed\n2. Extract entities\n3. Answer</think>",
            # Turn 1: tool call
            '<tool>pubmed:{"action": "search", "query": "BRCA1", "max_results": 5}</tool>',
            # Turn 2: another tool call
            '<tool>semantic_scholar:{"action": "search", "query": "BRCA1 cancer"}</tool>',
            # Turn 3: answer
            '<answer>' + json.dumps({
                "entities": [
                    {"name": "BRCA1", "type": "GENE", "description": "DNA repair gene"},
                    {"name": "Breast Cancer", "type": "DISEASE", "description": "Cancer of breast tissue"},
                ],
                "relationships": [
                    {"source": "BRCA1", "target": "Breast Cancer",
                     "relation": "ASSOCIATED_WITH", "confidence": 0.8,
                     "claim": "BRCA1 mutations increase BC risk"},
                ],
                "summary": "Found BRCA1-BC association in literature.",
            }) + '</answer>',
        ])

        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = MultiTurnStubAgent(
            template=template, llm=llm, kg=agent_kg, tools=mock_tools,
        )

        result = await agent.execute(sample_task)

        assert result.success is True
        assert len(result.turns) >= 4  # plan + 2 tools + answer
        assert len(result.nodes_added) == 2
        assert len(result.edges_added) == 1

        # Verify turn types
        assert result.turns[0].turn_type == TurnType.THINK
        assert result.turns[1].turn_type == TurnType.TOOL_CALL
        assert result.turns[2].turn_type == TurnType.TOOL_CALL
        assert result.turns[3].turn_type == TurnType.ANSWER

    @pytest.mark.asyncio
    async def test_observation_history_grows(self, agent_kg, mock_tools, sample_task):
        """Each turn's result should be visible to subsequent turns."""
        llm = MockLLMClient(responses=[
            "<think>Plan</think>",
            '<tool>pubmed:{"action": "search", "query": "test"}</tool>',
            '<answer>' + json.dumps({
                "entities": [], "relationships": [],
                "summary": "Done",
            }) + '</answer>',
        ])

        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = MultiTurnStubAgent(
            template=template, llm=llm, kg=agent_kg, tools=mock_tools,
        )

        result = await agent.execute(sample_task)

        # Tool call turn should have execution_result populated
        tool_turn = [t for t in result.turns if t.turn_type == TurnType.TOOL_CALL]
        assert len(tool_turn) == 1
        assert tool_turn[0].execution_result != ""
        # The result should contain data from the mock pubmed tool
        assert "results" in tool_turn[0].execution_result or "BRCA1" in tool_turn[0].execution_result

    @pytest.mark.asyncio
    async def test_budget_limit_enforced(self, agent_kg, sample_task):
        """Agent should stop after max_turns even without <answer>."""
        # All responses are just thinking — never provides an answer
        llm = MockLLMClient(responses=[
            "<think>Planning...</think>",
            "<think>Still thinking turn 1...</think>",
            "<think>Still thinking turn 2...</think>",
            "<think>Still thinking turn 3...</think>",
            "<think>Still thinking turn 4...</think>",
            "<think>Still thinking turn 5...</think>",
        ])

        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = MultiTurnStubAgent(
            template=template, llm=llm, kg=agent_kg, tools={},
        )

        # Use a small max_turns via monkey-patching the method
        async def limited_investigate(task, kg_context):
            return await agent._multi_turn_investigate(
                task, kg_context, max_turns=3, investigation_focus="Test",
            )

        agent._investigate = limited_investigate
        result = await agent.execute(sample_task)

        assert result.success is True
        # Should have 4 turns: plan + 3 execution turns
        assert len(result.turns) == 4
        assert "turn limit" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_tool_error_shown_to_agent(self, agent_kg, sample_task):
        """When a tool call fails, the error should be in observations."""
        llm = MockLLMClient(responses=[
            "<think>Plan</think>",
            # Call a non-existent tool
            '<tool>nonexistent_tool:{"query": "test"}</tool>',
            # Agent sees error and provides answer
            '<answer>' + json.dumps({
                "entities": [], "relationships": [],
                "summary": "Tool failed, could not complete investigation.",
            }) + '</answer>',
        ])

        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = MultiTurnStubAgent(
            template=template, llm=llm, kg=agent_kg, tools={},
        )

        result = await agent.execute(sample_task)

        assert result.success is True
        # The tool error should be captured in execution_result
        tool_turn = [t for t in result.turns if t.turn_type == TurnType.TOOL_CALL]
        assert len(tool_turn) == 1
        assert "Error" in tool_turn[0].execution_result

    @pytest.mark.asyncio
    async def test_code_execution_not_available(self, agent_kg, sample_task):
        """Code execution should report unavailability when no REPL tool."""
        llm = MockLLMClient(responses=[
            "<think>Plan</think>",
            "<execute>print('hello world')</execute>",
            '<answer>' + json.dumps({
                "entities": [], "relationships": [],
                "summary": "Code execution not available.",
            }) + '</answer>',
        ])

        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = MultiTurnStubAgent(
            template=template, llm=llm, kg=agent_kg, tools={},
        )

        result = await agent.execute(sample_task)

        code_turn = [t for t in result.turns if t.turn_type == TurnType.CODE_EXECUTION]
        assert len(code_turn) == 1
        assert "not available" in code_turn[0].execution_result.lower()

    @pytest.mark.asyncio
    async def test_kg_virtual_tools(self, seeded_kg, sample_task):
        """KG virtual tools (get_recent_edges, etc.) should work."""
        llm = MockLLMClient(responses=[
            "<think>Get KG state</think>",
            '<tool>kg_get_recent_edges:{"n": 5}</tool>',
            '<tool>kg_get_weakest_edges:{"n": 5}</tool>',
            '<tool>kg_get_orphan_nodes:{}</tool>',
            '<answer>' + json.dumps({
                "entities": [], "relationships": [],
                "summary": "Analyzed KG state.",
            }) + '</answer>',
        ])

        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = MultiTurnStubAgent(
            template=template, llm=llm, kg=seeded_kg, tools={},
        )

        result = await agent.execute(sample_task)

        assert result.success is True
        tool_turns = [t for t in result.turns if t.turn_type == TurnType.TOOL_CALL]
        assert len(tool_turns) == 3

        # Recent edges should include the seeded edges
        assert "BRCA1" in tool_turns[0].execution_result or "ASSOCIATED_WITH" in tool_turns[0].execution_result
        # Weakest edges should include the weak edge
        assert "e-brca1-bc" in tool_turns[1].execution_result or "0.4" in tool_turns[1].execution_result

    @pytest.mark.asyncio
    async def test_kg_update_edge_confidence_tool(self, seeded_kg, sample_task):
        """kg_update_edge_confidence should modify edge in KG."""
        original = seeded_kg.get_edge("e-brca1-bc")
        assert original is not None
        orig_confidence = original.confidence.overall

        llm = MockLLMClient(responses=[
            "<think>Update edge</think>",
            '<tool>kg_update_edge_confidence:'
            '{"edge_id": "e-brca1-bc", "confidence": 0.25, '
            '"reason": "Test adjustment"}</tool>',
            '<answer>' + json.dumps({
                "entities": [], "relationships": [],
                "summary": "Updated edge confidence.",
            }) + '</answer>',
        ])

        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = MultiTurnStubAgent(
            template=template, llm=llm, kg=seeded_kg, tools={},
        )

        result = await agent.execute(sample_task)

        assert result.success is True
        assert "e-brca1-bc" in result.edges_updated

        updated = seeded_kg.get_edge("e-brca1-bc")
        # KG recalculates overall confidence internally — just verify it changed
        assert updated.confidence.overall != orig_confidence
        assert updated.confidence.overall < orig_confidence

    @pytest.mark.asyncio
    async def test_parse_nodes_from_answer(self, agent_kg):
        """Test node parsing from answer entities."""
        template = get_template(AgentType.LITERATURE_ANALYST)
        llm = MockLLMClient()
        agent = MultiTurnStubAgent(template=template, llm=llm, kg=agent_kg)

        entities = [
            {"name": "TP53", "type": "GENE", "description": "Tumor suppressor",
             "confidence": 0.95, "evidence_source": "MYGENE", "evidence_id": "7157"},
            {"name": "Lung Cancer", "type": "DISEASE", "description": "NSCLC",
             "confidence": 0.9, "external_ids": {"mesh": "D002289"}},
        ]

        nodes = agent._parse_nodes_from_answer(entities)
        assert len(nodes) == 2

        from core.models import EvidenceSourceType, NodeType
        assert nodes[0].name == "TP53"
        assert nodes[0].type == NodeType.GENE
        assert nodes[0].confidence == 0.95
        assert nodes[0].sources[0].source_type == EvidenceSourceType.MYGENE

        assert nodes[1].name == "Lung Cancer"
        assert nodes[1].type == NodeType.DISEASE
        assert nodes[1].external_ids == {"mesh": "D002289"}

    @pytest.mark.asyncio
    async def test_parse_edges_from_answer_resolves_kg_names(self, seeded_kg):
        """Edge parsing should resolve entity names from both answer and KG."""
        template = get_template(AgentType.LITERATURE_ANALYST)
        llm = MockLLMClient()
        agent = MultiTurnStubAgent(template=template, llm=llm, kg=seeded_kg)

        from core.models import KGNode, NodeType
        # Create a new node that's only in the answer
        new_node = KGNode(type=NodeType.DRUG, name="Tamoxifen", confidence=0.8)
        answer_nodes = [new_node]

        relationships = [
            # Source from answer, target from KG (seeded_kg has "Breast Cancer")
            {"source": "Tamoxifen", "target": "Breast Cancer",
             "relation": "TREATS", "confidence": 0.85,
             "claim": "Tamoxifen treats breast cancer"},
        ]

        edges = agent._parse_edges_from_answer(relationships, answer_nodes)
        assert len(edges) == 1
        assert edges[0].source_id == new_node.id
        # Target should be resolved from seeded KG
        bc_node = seeded_kg.get_node_by_name("Breast Cancer")
        assert edges[0].target_id == bc_node.id

    @pytest.mark.asyncio
    async def test_think_with_tool_both_recorded(self, agent_kg, mock_tools, sample_task):
        """When response has both <think> and <tool>, both should be in observations."""
        llm = MockLLMClient(responses=[
            "<think>Plan</think>",
            '<think>Let me search for papers</think><tool>pubmed:{"action": "search", "query": "test"}</tool>',
            '<answer>{"entities": [], "relationships": [], "summary": "done"}</answer>',
        ])

        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = MultiTurnStubAgent(
            template=template, llm=llm, kg=agent_kg, tools=mock_tools,
        )

        result = await agent.execute(sample_task)

        # Turn 1 is a tool call, but think content should also be captured
        assert result.turns[1].turn_type == TurnType.TOOL_CALL
        assert len(result.turns) == 3

    @pytest.mark.asyncio
    async def test_token_budget_stops_agent(self, agent_kg, sample_task):
        """Agent should stop when token budget is exhausted."""
        llm = MockLLMClient(responses=[
            "<think>Planning...</think>",
            "<think>Thinking turn 1...</think>",
            "<think>Thinking turn 2...</think>",
            "<think>Thinking turn 3...</think>",
            "<think>Thinking turn 4...</think>",
        ])
        # Simulate high token usage
        llm._token_summary = {"total_tokens": 60_000}

        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = MultiTurnStubAgent(
            template=template, llm=llm, kg=agent_kg, tools={},
        )

        async def budget_investigate(task, kg_context):
            return await agent._multi_turn_investigate(
                task, kg_context,
                max_turns=50,
                token_budget=50_000,
                investigation_focus="Test",
            )

        agent._investigate = budget_investigate
        result = await agent.execute(sample_task)

        # Hard kill: token budget exceeded → success=False
        assert result.success is False
        assert any("TOKEN_BUDGET_HARD_KILL" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_observation_compression(self, agent_kg):
        """_compress_observations should summarize old turns."""
        llm = MockLLMClient(responses=[
            "Compressed summary of earlier observations.",
        ])

        template = get_template(AgentType.LITERATURE_ANALYST)
        agent = MultiTurnStubAgent(
            template=template, llm=llm, kg=agent_kg, tools={},
        )

        # Build a large observation history (needs >5000 estimated tokens in old part)
        observations = [f"[TOOL turn {i}] result " + "x" * 2000 for i in range(20)]

        compressed = await agent._compress_observations(observations, keep_recent=5)

        # Should have: 1 compressed block + 5 recent
        assert len(compressed) <= 6
        assert "COMPRESSED HISTORY" in compressed[0]
