"""Data models for the benchmark suite."""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _uuid() -> str:
    return str(uuid.uuid4())


class BenchmarkSuite(enum.StrEnum):
    BIOMNI_EVAL1 = "biomni_eval1"
    BIXBENCH = "bixbench"
    LAB_BENCH_DBQA = "lab_bench_dbqa"
    LAB_BENCH_SEQQA = "lab_bench_seqqa"
    LAB_BENCH_LITQA2 = "lab_bench_litqa2"


class RunMode(enum.StrEnum):
    ZERO_SHOT = "zero_shot"
    YOHAS_FULL = "yohas_full"
    CODE_FIRST = "code_first"


class InstanceStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class BenchmarkInstance(BaseModel):
    """A single benchmark task instance."""

    id: str = Field(default_factory=_uuid)
    suite: BenchmarkSuite
    instance_id: str  # original ID from the benchmark dataset
    question: str
    context: str = ""
    choices: list[str] = Field(default_factory=list)
    ground_truth: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    category: str = ""  # sub-category within the benchmark


class TrialResult(BaseModel):
    """Result of a single trial within a multi-trial evaluation."""

    trial_number: int
    predicted: str = ""
    correct: bool = False
    score: float = 0.0
    tokens_used: int = 0
    latency_ms: int = 0
    turns: int = 0
    tools_used: list[str] = Field(default_factory=list)
    reasoning_trace: str = ""
    hint_injected: str = ""


class InstanceResult(BaseModel):
    """Result of running a single benchmark instance."""

    instance_id: str
    suite: BenchmarkSuite
    mode: RunMode
    predicted: str = ""
    ground_truth: str = ""
    correct: bool = False
    score: float = 0.0  # for partial credit
    tokens_used: int = 0
    latency_ms: int = 0
    turns: int = 0
    tools_used: list[str] = Field(default_factory=list)
    reasoning_trace: str = ""
    error: str | None = None
    status: InstanceStatus = InstanceStatus.COMPLETED
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None
    # Multi-trial fields
    trial_results: list[TrialResult] = Field(default_factory=list)
    best_trial: int = 0  # 0 = single trial, 1-indexed for multi-trial


class TrajectoryStep(BaseModel):
    """A single step in an agent trajectory (for RL training)."""

    step: int
    action_type: str  # "think", "tool_call", "answer"
    action: str  # the action content
    observation: str = ""
    reward: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trajectory(BaseModel):
    """Full trajectory for a benchmark instance (for RL training data)."""

    instance_id: str
    suite: BenchmarkSuite
    mode: RunMode
    steps: list[TrajectoryStep] = Field(default_factory=list)
    total_reward: float = 0.0
    correct: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class SuiteResults(BaseModel):
    """Aggregate results for a benchmark suite."""

    suite: BenchmarkSuite
    mode: RunMode
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    avg_tokens: float = 0.0
    avg_latency_ms: float = 0.0
    avg_turns: float = 0.0
    by_category: dict[str, dict[str, float]] = Field(default_factory=dict)
    results: list[InstanceResult] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None


class BenchmarkReport(BaseModel):
    """Full comparison report across all suites and modes."""

    id: str = Field(default_factory=_uuid)
    suites: list[SuiteResults] = Field(default_factory=list)
    baselines: dict[str, dict[str, float]] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=_utcnow)
    markdown: str = ""
