"""Benchmark test fixtures."""

from __future__ import annotations

import pytest

from benchmarks.adapters import (
    BiomniEval1Adapter,
    BixBenchAdapter,
    LABBenchDbQAAdapter,
    LABBenchLitQA2Adapter,
    LABBenchSeqQAAdapter,
)
from benchmarks.evaluator import BenchmarkEvaluator
from benchmarks.models import RunMode
from benchmarks.trajectory_store import TrajectoryStore


@pytest.fixture()
def biomni_adapter():
    return BiomniEval1Adapter()


@pytest.fixture()
def bixbench_adapter():
    return BixBenchAdapter()


@pytest.fixture()
def labench_dbqa_adapter():
    return LABBenchDbQAAdapter()


@pytest.fixture()
def labench_seqqa_adapter():
    return LABBenchSeqQAAdapter()


@pytest.fixture()
def labench_litqa2_adapter():
    return LABBenchLitQA2Adapter()


@pytest.fixture()
def zero_shot_evaluator():
    return BenchmarkEvaluator(mode=RunMode.ZERO_SHOT)


@pytest.fixture()
def yohas_full_evaluator():
    return BenchmarkEvaluator(mode=RunMode.YOHAS_FULL)


@pytest.fixture()
def trajectory_store(tmp_path):
    return TrajectoryStore(output_dir=tmp_path / "trajectories")
