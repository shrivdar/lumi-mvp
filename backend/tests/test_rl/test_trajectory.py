"""Tests for RL trajectory collector, format models, and SFT pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from core.models import (
    AgentResult,
    AgentTask,
    AgentTurn,
    AgentType,
    EdgeConfidence,
    EdgeRelationType,
    KGEdge,
    KGNode,
    NodeType,
    TurnType,
)
from rl.sft_pipeline import SFTPipeline
from rl.trajectory_collector import TrajectoryCollector, _parse_tool_action
from rl.trajectory_format import (
    CodeExecRecord,
    KGMutationRecord,
    ToolCallRecord,
    Trajectory,
    Turn,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_task(task_id: str = "task-1") -> AgentTask:
    return AgentTask(
        task_id=task_id,
        research_id="research-1",
        agent_type=AgentType.LITERATURE_ANALYST,
        agent_id="agent-lit-1",
        hypothesis_branch="h-test",
        instruction="Investigate BRCA1 role in breast cancer",
        context={"query": "BRCA1 breast cancer"},
    )


def _make_result(
    task_id: str = "task-1",
    success: bool = True,
    n_turns: int = 3,
) -> AgentResult:
    turns = []
    for i in range(n_turns):
        if i == 0:
            turns.append(AgentTurn(
                turn_number=i,
                turn_type=TurnType.THINK,
                input_prompt="Plan...",
                raw_response="<think>I will search PubMed for BRCA1...</think>",
                parsed_action="I will search PubMed for BRCA1...",
                tokens_used=100,
                duration_ms=500,
            ))
        elif i < n_turns - 1:
            turns.append(AgentTurn(
                turn_number=i,
                turn_type=TurnType.TOOL_CALL,
                input_prompt="Execute next step...",
                raw_response='<tool>pubmed_search:{"query": "BRCA1"}</tool>',
                parsed_action='pubmed_search:{"query": "BRCA1"}',
                execution_result="Found 42 papers on BRCA1...",
                tokens_used=200,
                duration_ms=1500,
            ))
        else:
            turns.append(AgentTurn(
                turn_number=i,
                turn_type=TurnType.ANSWER,
                input_prompt="Provide final answer...",
                raw_response="<answer>BRCA1 is associated with breast cancer...</answer>",
                parsed_action="BRCA1 is associated with breast cancer...",
                tokens_used=150,
                duration_ms=800,
            ))

    nodes = [
        KGNode(
            id="n-brca1",
            type=NodeType.GENE,
            name="BRCA1",
            confidence=0.95,
            created_by="agent-lit-1",
        ),
    ]

    edges = [
        KGEdge(
            id="e-1",
            source_id="n-brca1",
            target_id="n-brca",
            relation=EdgeRelationType.ASSOCIATED_WITH,
            confidence=EdgeConfidence(overall=0.9),
            created_by="agent-lit-1",
        ),
    ]

    return AgentResult(
        task_id=task_id,
        agent_id="agent-lit-1",
        agent_type=AgentType.LITERATURE_ANALYST,
        nodes_added=nodes,
        edges_added=edges,
        nodes_updated=["n-tp53"],
        edges_updated=["e-old"],
        summary="BRCA1 is a key gene in breast cancer susceptibility.",
        turns=turns,
        token_usage={"input": 300, "output": 150},
        duration_ms=2800,
        llm_calls=3,
        llm_tokens_used=450,
        success=success,
    )


# ---------------------------------------------------------------------------
# Trajectory Format Tests
# ---------------------------------------------------------------------------


class TestTrajectoryFormat:
    def test_trajectory_roundtrip(self):
        """Trajectory serializes to JSON and back."""
        traj = Trajectory(
            task_id="t1",
            agent_type="literature_analyst",
            instruction="test",
            turns=[
                Turn(turn_number=0, role="assistant", content="hello", turn_type="think"),
            ],
            reward=1.0,
            success=True,
        )
        json_str = traj.model_dump_json()
        restored = Trajectory.model_validate_json(json_str)
        assert restored.task_id == "t1"
        assert restored.reward == 1.0
        assert len(restored.turns) == 1

    def test_tool_call_record(self):
        tc = ToolCallRecord(
            tool_name="pubmed_search",
            arguments={"query": "BRCA1"},
            result="Found papers",
            duration_ms=100,
        )
        assert tc.tool_name == "pubmed_search"
        assert tc.error is None

    def test_code_exec_record(self):
        ce = CodeExecRecord(
            code="print('hello')",
            output="hello",
            duration_ms=50,
        )
        assert ce.code == "print('hello')"

    def test_kg_mutation_record(self):
        m = KGMutationRecord(
            operation="add_node",
            entity_id="n-1",
            entity_type="GENE",
            details={"name": "BRCA1"},
        )
        assert m.operation == "add_node"


# ---------------------------------------------------------------------------
# Trajectory Collector Tests
# ---------------------------------------------------------------------------


class TestTrajectoryCollector:
    def test_collect_successful(self, tmp_path: Path):
        collector = TrajectoryCollector(output_dir=tmp_path)
        task = _make_task()
        result = _make_result()

        traj = collector.collect(task, result)

        assert traj.task_id == "task-1"
        assert traj.reward == 1.0
        assert traj.success is True
        assert len(traj.turns) == 3
        assert traj.agent_type == "literature_analyst"
        assert traj.instruction == "Investigate BRCA1 role in breast cancer"
        assert collector.buffered_count == 1

    def test_collect_failed(self, tmp_path: Path):
        collector = TrajectoryCollector(output_dir=tmp_path)
        task = _make_task()
        result = _make_result(success=False)

        traj = collector.collect(task, result)

        assert traj.reward == 0.0
        assert traj.success is False

    def test_collect_custom_reward(self, tmp_path: Path):
        collector = TrajectoryCollector(output_dir=tmp_path)
        task = _make_task()
        result = _make_result()

        traj = collector.collect(task, result, reward=0.75)

        assert traj.reward == 0.75

    def test_collect_reward_fn(self, tmp_path: Path):
        def reward_fn(task, result):
            return 0.5 if result.success else 0.0

        collector = TrajectoryCollector(output_dir=tmp_path, reward_fn=reward_fn)
        task = _make_task()
        result = _make_result()

        traj = collector.collect(task, result)
        assert traj.reward == 0.5

    def test_kg_mutations_extracted(self, tmp_path: Path):
        collector = TrajectoryCollector(output_dir=tmp_path)
        task = _make_task()
        result = _make_result()

        traj = collector.collect(task, result)

        # 1 node added + 1 edge added + 1 node updated + 1 edge updated = 4
        assert len(traj.kg_mutations) == 4
        ops = [m.operation for m in traj.kg_mutations]
        assert "add_node" in ops
        assert "add_edge" in ops
        assert "update_node" in ops
        assert "update_edge" in ops

    def test_tool_call_turns(self, tmp_path: Path):
        collector = TrajectoryCollector(output_dir=tmp_path)
        task = _make_task()
        result = _make_result()

        traj = collector.collect(task, result)

        tool_turns = [t for t in traj.turns if t.tool_calls]
        assert len(tool_turns) == 1
        assert tool_turns[0].tool_calls[0].tool_name == "pubmed_search"
        assert tool_turns[0].tool_calls[0].arguments == {"query": "BRCA1"}

    def test_flush_writes_jsonl(self, tmp_path: Path):
        collector = TrajectoryCollector(output_dir=tmp_path, benchmark_run_id="bench-1")
        task = _make_task()
        result = _make_result()
        collector.collect(task, result)

        out_path = collector.flush()

        assert out_path.exists()
        assert out_path.suffix == ".jsonl"
        assert "bench-1" in out_path.name

        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert data["task_id"] == "task-1"
        assert data["reward"] == 1.0

        # Buffer should be cleared
        assert collector.buffered_count == 0

    def test_flush_multiple(self, tmp_path: Path):
        collector = TrajectoryCollector(output_dir=tmp_path)

        for i in range(5):
            task = _make_task(task_id=f"task-{i}")
            result = _make_result(task_id=f"task-{i}")
            collector.collect(task, result)

        out_path = collector.flush()
        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_flush_empty(self, tmp_path: Path):
        collector = TrajectoryCollector(output_dir=tmp_path)
        out_path = collector.flush()
        assert not out_path.exists()  # "empty" sentinel path

    def test_sub_agent_trajectories(self, tmp_path: Path):
        collector = TrajectoryCollector(output_dir=tmp_path)
        task = _make_task()

        # Result with a sub-agent result
        sub_result = _make_result(task_id="sub-task-1", n_turns=2)
        sub_result.agent_type = AgentType.PROTEIN_ENGINEER
        result = _make_result()
        result.sub_agent_results = [sub_result]

        collector.collect(task, result)

        # Should have 2 trajectories: parent + sub-agent
        assert collector.buffered_count == 2
        types = [t.agent_type for t in collector.trajectories]
        assert "literature_analyst" in types
        assert "protein_engineer" in types


# ---------------------------------------------------------------------------
# Tool action parser
# ---------------------------------------------------------------------------


class TestParseToolAction:
    def test_valid_json(self):
        name, args = _parse_tool_action('pubmed_search:{"query": "BRCA1"}')
        assert name == "pubmed_search"
        assert args == {"query": "BRCA1"}

    def test_invalid_json(self):
        name, args = _parse_tool_action("pubmed_search:not json")
        assert name == "pubmed_search"
        assert args == {"raw": "not json"}

    def test_no_colon(self):
        name, args = _parse_tool_action("pubmed_search")
        assert name == "pubmed_search"
        assert args == {}


# ---------------------------------------------------------------------------
# SFT Pipeline Tests
# ---------------------------------------------------------------------------


class TestSFTPipeline:
    def test_filter_by_reward(self, tmp_path: Path):
        pipeline = SFTPipeline(reward_threshold=1.0)

        trajectories = [
            Trajectory(task_id="t1", agent_type="lit", reward=1.0, success=True),
            Trajectory(task_id="t2", agent_type="lit", reward=0.0, success=False),
            Trajectory(task_id="t3", agent_type="lit", reward=0.5, success=True),
        ]

        filtered = pipeline.filter(trajectories)
        assert len(filtered) == 1
        assert filtered[0].task_id == "t1"

    def test_filter_lower_threshold(self):
        pipeline = SFTPipeline(reward_threshold=0.5)

        trajectories = [
            Trajectory(task_id="t1", agent_type="lit", reward=1.0, success=True),
            Trajectory(task_id="t2", agent_type="lit", reward=0.0, success=False),
            Trajectory(task_id="t3", agent_type="lit", reward=0.5, success=True),
        ]

        filtered = pipeline.filter(trajectories)
        assert len(filtered) == 2

    def test_rejection_sampling(self):
        pipeline = SFTPipeline()

        trajectories = [
            # Task A: 3 attempts, 1 success
            Trajectory(task_id="A", agent_type="lit", reward=0.0, success=False,
                       turns=[Turn(turn_number=i, role="assistant") for i in range(5)]),
            Trajectory(task_id="A", agent_type="lit", reward=1.0, success=True,
                       turns=[Turn(turn_number=i, role="assistant") for i in range(3)]),
            Trajectory(task_id="A", agent_type="lit", reward=0.0, success=False,
                       turns=[Turn(turn_number=i, role="assistant") for i in range(4)]),
            # Task B: 2 attempts, 0 success
            Trajectory(task_id="B", agent_type="lit", reward=0.0, success=False),
            Trajectory(task_id="B", agent_type="lit", reward=0.0, success=False),
            # Task C: 1 attempt, 1 success
            Trajectory(task_id="C", agent_type="lit", reward=1.0, success=True,
                       turns=[Turn(turn_number=i, role="assistant") for i in range(2)]),
        ]

        selected = pipeline.rejection_sample(trajectories)

        assert len(selected) == 2  # A and C, not B
        task_ids = {t.task_id for t in selected}
        assert task_ids == {"A", "C"}

        # Should pick shortest successful trajectory for A
        a_traj = next(t for t in selected if t.task_id == "A")
        assert len(a_traj.turns) == 3

    def test_format_conversation(self):
        pipeline = SFTPipeline(include_thinking=False)

        traj = Trajectory(
            task_id="t1",
            agent_type="literature_analyst",
            instruction="Find BRCA1 info",
            turns=[
                Turn(turn_number=0, role="assistant", content="thinking...", turn_type="think"),
                Turn(
                    turn_number=1,
                    role="assistant",
                    content='<tool>pubmed_search:{"q":"BRCA1"}</tool>',
                    turn_type="tool_call",
                    tool_calls=[ToolCallRecord(
                        tool_name="pubmed_search",
                        arguments={"q": "BRCA1"},
                        result="Found 42 papers",
                    )],
                ),
                Turn(
                    turn_number=2,
                    role="assistant",
                    content="BRCA1 is associated with breast cancer",
                    turn_type="answer",
                ),
            ],
            reward=1.0,
            success=True,
        )

        messages = pipeline.format_conversation(traj)

        # System + tool_call assistant + tool result + answer assistant = 4
        # (think is excluded)
        assert messages[0]["role"] == "system"
        assert "literature_analyst" in messages[0]["content"]
        assert "Find BRCA1 info" in messages[0]["content"]

        # Think turn is excluded
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "assistant" in roles
        assert "tool" in roles

    def test_format_conversation_with_thinking(self):
        pipeline = SFTPipeline(include_thinking=True)

        traj = Trajectory(
            task_id="t1",
            agent_type="lit",
            instruction="test",
            turns=[
                Turn(turn_number=0, role="assistant", content="thinking...", turn_type="think"),
                Turn(turn_number=1, role="assistant", content="answer", turn_type="answer"),
            ],
            reward=1.0,
        )

        messages = pipeline.format_conversation(traj)
        # System + think + answer = 3
        assert len(messages) == 3

    def test_load_and_roundtrip(self, tmp_path: Path):
        """Write trajectories, load them, filter, and export."""
        collector = TrajectoryCollector(output_dir=tmp_path)

        for i in range(3):
            task = _make_task(task_id=f"task-{i}")
            result = _make_result(task_id=f"task-{i}", success=(i != 1))
            collector.collect(task, result)

        jsonl_path = collector.flush()

        pipeline = SFTPipeline(reward_threshold=1.0)
        loaded = pipeline.load_trajectories(jsonl_path)
        assert len(loaded) == 3

        filtered = pipeline.filter(loaded)
        assert len(filtered) == 2  # task-0 and task-2

        out = pipeline.export_jsonl(filtered, tmp_path / "sft_data.jsonl")
        assert out.exists()

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert "messages" in record
            assert record["reward"] == 1.0

    def test_load_directory(self, tmp_path: Path):
        # Create two JSONL files
        for batch in range(2):
            collector = TrajectoryCollector(output_dir=tmp_path)
            for i in range(3):
                task = _make_task(task_id=f"task-{batch}-{i}")
                result = _make_result(task_id=f"task-{batch}-{i}")
                collector.collect(task, result)
            collector.flush(filename=f"batch_{batch}.jsonl")

        pipeline = SFTPipeline()
        all_trajs = pipeline.load_directory(tmp_path)
        assert len(all_trajs) == 6

    def test_load_missing_file(self, tmp_path: Path):
        pipeline = SFTPipeline()
        loaded = pipeline.load_trajectories(tmp_path / "nonexistent.jsonl")
        assert loaded == []

    def test_max_turns(self):
        pipeline = SFTPipeline(max_turns=1)

        traj = Trajectory(
            task_id="t1",
            agent_type="lit",
            instruction="test",
            turns=[
                Turn(turn_number=0, role="assistant", content="turn 0", turn_type="think"),
                Turn(turn_number=1, role="assistant", content="turn 1", turn_type="answer"),
                Turn(turn_number=2, role="assistant", content="turn 2", turn_type="answer"),
            ],
            reward=1.0,
        )

        messages = pipeline.format_conversation(traj)
        # System + turns 0 and 1 (think excluded by default) → system + turn 1 only
        # Turn 2 excluded by max_turns=1
        contents = [m["content"] for m in messages if m["role"] == "assistant"]
        assert "turn 2" not in contents
