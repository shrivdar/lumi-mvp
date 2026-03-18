"""Tests for benchmark dataset adapters."""

from __future__ import annotations

import json

import pytest

from benchmarks.adapters import ADAPTERS, get_adapter
from benchmarks.models import BenchmarkInstance, BenchmarkSuite


class TestBiomniEval1Adapter:
    def test_load_synthetic_instances(self, biomni_adapter):
        instances = biomni_adapter.load_instances(limit=10)
        assert len(instances) == 10
        assert all(isinstance(i, BenchmarkInstance) for i in instances)
        assert all(i.suite == BenchmarkSuite.BIOMNI_EVAL1 for i in instances)

    def test_instance_has_required_fields(self, biomni_adapter):
        instances = biomni_adapter.load_instances(limit=1)
        inst = instances[0]
        assert inst.instance_id
        assert inst.question
        assert inst.ground_truth
        assert inst.category

    def test_full_count(self, biomni_adapter):
        instances = biomni_adapter.load_instances()
        assert len(instances) == 433

    def test_categories_distributed(self, biomni_adapter):
        instances = biomni_adapter.load_instances()
        categories = {i.category for i in instances}
        assert len(categories) >= 5  # multiple categories

    def test_load_from_jsonl(self, biomni_adapter, tmp_path):
        """Test loading from actual JSONL file."""
        data_dir = tmp_path / "data" / "benchmarks" / "biomni_eval1"
        data_dir.mkdir(parents=True)
        jsonl = data_dir / "instances.jsonl"
        rows = [
            {"id": "test_001", "question": "What is BRCA1?", "answer": "tumor suppressor", "category": "gene"},
            {"id": "test_002", "question": "What does TP53 do?", "answer": "cell cycle", "category": "gene"},
        ]
        jsonl.write_text("\n".join(json.dumps(r) for r in rows))

        # Patch the DATA_DIR
        import benchmarks.adapters as adapters_mod

        original = adapters_mod.DATA_DIR
        adapters_mod.DATA_DIR = tmp_path / "data" / "benchmarks"
        try:
            instances = biomni_adapter.load_instances()
            assert len(instances) == 2
            assert instances[0].instance_id == "test_001"
            assert instances[0].ground_truth == "tumor suppressor"
        finally:
            adapters_mod.DATA_DIR = original


class TestBixBenchAdapter:
    def test_load_synthetic(self, bixbench_adapter):
        instances = bixbench_adapter.load_instances(limit=5)
        assert len(instances) == 5
        assert all(i.suite == BenchmarkSuite.BIXBENCH for i in instances)

    def test_full_count(self, bixbench_adapter):
        instances = bixbench_adapter.load_instances()
        assert len(instances) == 205


class TestLABBenchAdapters:
    def test_dbqa_load(self, labench_dbqa_adapter):
        instances = labench_dbqa_adapter.load_instances(limit=5)
        assert len(instances) == 5
        assert all(i.suite == BenchmarkSuite.LAB_BENCH_DBQA for i in instances)

    def test_seqqa_load(self, labench_seqqa_adapter):
        instances = labench_seqqa_adapter.load_instances(limit=5)
        assert len(instances) == 5
        assert all(i.suite == BenchmarkSuite.LAB_BENCH_SEQQA for i in instances)

    def test_litqa2_load(self, labench_litqa2_adapter):
        instances = labench_litqa2_adapter.load_instances(limit=5)
        assert len(instances) == 5
        assert all(i.suite == BenchmarkSuite.LAB_BENCH_LITQA2 for i in instances)

    def test_dbqa_full_count(self, labench_dbqa_adapter):
        instances = labench_dbqa_adapter.load_instances()
        assert len(instances) >= 100  # 100 synthetic or 520 real

    def test_seqqa_full_count(self, labench_seqqa_adapter):
        instances = labench_seqqa_adapter.load_instances()
        assert len(instances) >= 100  # 100 synthetic or 600 real

    def test_litqa2_full_count(self, labench_litqa2_adapter):
        instances = labench_litqa2_adapter.load_instances()
        assert len(instances) >= 100  # 100 synthetic or 199 real


class TestAdapterRegistry:
    def test_all_suites_have_adapters(self):
        for suite in BenchmarkSuite:
            adapter = get_adapter(suite)
            assert adapter.suite == suite

    def test_registry_complete(self):
        assert len(ADAPTERS) == len(BenchmarkSuite)

    def test_invalid_suite_raises(self):
        with pytest.raises(ValueError, match="No adapter"):
            get_adapter("nonexistent")  # type: ignore[arg-type]
