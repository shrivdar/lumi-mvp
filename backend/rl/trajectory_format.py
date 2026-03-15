"""Pydantic models for RL trajectory data.

Captures multi-turn agent sessions as structured training data:
task → turns (role/content/tool_calls/code_executions) → outcome.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _uuid() -> str:
    return str(uuid.uuid4())


class ToolCallRecord(BaseModel):
    """A single tool invocation within a turn."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    duration_ms: int = 0
    error: str | None = None


class CodeExecRecord(BaseModel):
    """A code execution within a turn."""

    code: str
    output: str = ""
    duration_ms: int = 0
    error: str | None = None


class KGMutationRecord(BaseModel):
    """A knowledge-graph mutation (node or edge add/update) recorded in a trajectory."""

    operation: str  # "add_node", "add_edge", "update_node", "update_edge"
    entity_id: str
    entity_type: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class Turn(BaseModel):
    """A single turn in a multi-turn agent trajectory."""

    turn_number: int
    role: str  # "assistant", "tool", "system"
    content: str = ""
    turn_type: str = ""  # "think", "tool_call", "code_execution", "answer"
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    code_executions: list[CodeExecRecord] = Field(default_factory=list)
    tokens_used: int = 0
    duration_ms: int = 0
    timestamp: datetime = Field(default_factory=_utcnow)


class Trajectory(BaseModel):
    """A complete agent trajectory — one task from start to finish.

    This is the unit of training data for SFT / RL.
    """

    trajectory_id: str = Field(default_factory=_uuid)
    task_id: str
    research_id: str = ""
    agent_type: str
    agent_id: str = ""
    hypothesis_branch: str | None = None

    # The instruction that kicked off the agent
    instruction: str = ""
    context: dict[str, Any] = Field(default_factory=dict)

    # Multi-turn trace
    turns: list[Turn] = Field(default_factory=list)

    # Outcome
    final_answer: str = ""
    reward: float = 0.0  # 0.0 = failure, 1.0 = full success
    success: bool = False

    # KG mutations produced
    kg_mutations: list[KGMutationRecord] = Field(default_factory=list)

    # Metrics
    token_usage: dict[str, int] = Field(default_factory=dict)
    wall_time_ms: int = 0
    llm_calls: int = 0
    total_tokens: int = 0

    # Metadata
    benchmark_run_id: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
