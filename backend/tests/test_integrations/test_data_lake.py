"""Tests for data_lake context provider."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import integrations.data_lake as data_lake_mod
from integrations.data_lake import (
    _DATASET_META,
    data_lake_context,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the module-level cache before each test."""
    data_lake_mod._cached_context = None
    yield
    data_lake_mod._cached_context = None


@pytest.fixture()
def fake_data_dir(tmp_path: Path) -> Path:
    """Create a fake data directory with manifest and parquet stubs."""
    # Create parquet stub files for a few datasets
    for name in ["gene_ontology", "msigdb", "chembl"]:
        meta = _DATASET_META[name]
        parquet_file = tmp_path / meta["file"]
        parquet_file.parent.mkdir(parents=True, exist_ok=True)
        parquet_file.write_bytes(b"PAR1stub")  # just needs to exist

    # Write manifest
    manifest = {
        "gene_ontology": {
            "parquet": str(tmp_path / "gene_ontology/go_annotations.parquet"),
            "rows": 500000, "cols": 17, "error": None,
        },
        "msigdb": {
            "parquet": str(tmp_path / "msigdb/msigdb_hallmark.parquet"),
            "rows": 25000, "cols": 3, "error": None,
        },
        "chembl": {
            "parquet": str(tmp_path / "chembl/chembl_activities.parquet"),
            "rows": 2000000, "cols": 19, "error": None,
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDataLakeContext:
    def test_returns_empty_when_no_data_dir(self) -> None:
        with patch("integrations.data_lake._resolve_data_dir", return_value=None):
            assert data_lake_context() == ""

    def test_returns_empty_when_no_datasets(self, tmp_path: Path) -> None:
        # Empty data dir, no manifest, no files
        with patch("integrations.data_lake._resolve_data_dir", return_value=tmp_path):
            assert data_lake_context() == ""

    def test_includes_available_datasets(self, fake_data_dir: Path) -> None:
        with patch("integrations.data_lake._resolve_data_dir", return_value=fake_data_dir):
            ctx = data_lake_context()

        assert "## Local Data Lake" in ctx
        assert "gene_ontology" in ctx
        assert "msigdb" in ctx
        assert "chembl" in ctx
        assert "500,000 rows" in ctx
        assert "pd.read_parquet" in ctx

    def test_excludes_missing_datasets(self, fake_data_dir: Path) -> None:
        with patch("integrations.data_lake._resolve_data_dir", return_value=fake_data_dir):
            ctx = data_lake_context()

        # These datasets have no files and no manifest entry
        assert "dbsnp" not in ctx
        assert "omim" not in ctx

    def test_excludes_manifest_only_datasets(self, tmp_path: Path) -> None:
        """Datasets with manifest entries but no parquet file should be excluded."""
        # Write manifest with entry for clinvar but don't create the file
        manifest = {
            "clinvar": {"rows": 100000, "cols": 10, "error": None},
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        with patch("integrations.data_lake._resolve_data_dir", return_value=tmp_path):
            ctx = data_lake_context()

        assert "clinvar" not in ctx

    def test_includes_example_queries(self, fake_data_dir: Path) -> None:
        with patch("integrations.data_lake._resolve_data_dir", return_value=fake_data_dir):
            ctx = data_lake_context()

        assert "db_object_symbol" in ctx  # gene_ontology example
        assert "gene_set" in ctx  # msigdb example

    def test_includes_usage_notes(self, fake_data_dir: Path) -> None:
        with patch("integrations.data_lake._resolve_data_dir", return_value=fake_data_dir):
            ctx = data_lake_context()

        assert "YOHAS_DATA_PATH" in ctx
        assert "READ-ONLY" in ctx

    def test_caches_result(self, fake_data_dir: Path) -> None:
        with patch("integrations.data_lake._resolve_data_dir", return_value=fake_data_dir):
            ctx1 = data_lake_context()
            # Second call should use cache — even if we change the mock
        with patch("integrations.data_lake._resolve_data_dir", return_value=None):
            ctx2 = data_lake_context()

        assert ctx1 == ctx2
        assert ctx1 != ""

    def test_handles_corrupt_manifest(self, tmp_path: Path) -> None:
        # Create a parquet file but corrupt manifest
        meta = _DATASET_META["gene_ontology"]
        parquet_file = tmp_path / meta["file"]
        parquet_file.parent.mkdir(parents=True, exist_ok=True)
        parquet_file.write_bytes(b"PAR1stub")
        (tmp_path / "manifest.json").write_text("not json{{{")

        with patch("integrations.data_lake._resolve_data_dir", return_value=tmp_path):
            ctx = data_lake_context()

        # Should still work (file exists, just no row counts)
        assert "gene_ontology" in ctx

    def test_dataset_meta_covers_all_download_datasets(self) -> None:
        """Verify _DATASET_META has entries for all datasets in the downloader."""
        expected = {
            "gene_ontology", "msigdb", "clinvar", "gwas_catalog", "drugbank",
            "uniprot", "reactome", "chembl", "omim", "dbsnp", "ensembl",
        }
        assert set(_DATASET_META.keys()) == expected


class TestDataLakeContextContent:
    def test_key_columns_are_strings(self) -> None:
        for name, meta in _DATASET_META.items():
            assert isinstance(meta["key_columns"], list), f"{name} key_columns should be a list"
            for col in meta["key_columns"]:
                assert isinstance(col, str), f"{name} has non-string key column: {col}"

    def test_all_datasets_have_required_fields(self) -> None:
        required = {"description", "file", "key_columns", "example_query"}
        for name, meta in _DATASET_META.items():
            assert required <= set(meta.keys()), f"{name} missing fields: {required - set(meta.keys())}"
