"""Integration tests for data lake — agents query parquet via Python REPL."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow.parquet as pq
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


def _write_real_parquet(path: Path, df: pd.DataFrame) -> None:
    """Write an actual parquet file (not a stub)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


@pytest.fixture()
def populated_data_lake(tmp_path: Path) -> Path:
    """Create a data lake with real parquet files containing sample data."""

    # Gene Ontology
    go_df = pd.DataFrame([
        {"db_object_symbol": "TP53", "go_id": "GO:0006915", "aspect": "P",
         "evidence_code": "IDA", "db_object_name": "Tumor protein p53"},
        {"db_object_symbol": "BRCA1", "go_id": "GO:0006281", "aspect": "P",
         "evidence_code": "IDA", "db_object_name": "BRCA1 DNA repair associated"},
        {"db_object_symbol": "EGFR", "go_id": "GO:0007169", "aspect": "P",
         "evidence_code": "TAS", "db_object_name": "Epidermal growth factor receptor"},
    ])
    _write_real_parquet(tmp_path / "gene_ontology/go_annotations.parquet", go_df)

    # ClinVar
    cv_df = pd.DataFrame([
        {"GeneSymbol": "BRCA2", "ClinicalSignificance": "Pathogenic",
         "PhenotypeList": "Breast cancer", "Type": "single nucleotide variant",
         "Assembly": "GRCh38"},
        {"GeneSymbol": "BRCA2", "ClinicalSignificance": "Benign",
         "PhenotypeList": "not specified", "Type": "single nucleotide variant",
         "Assembly": "GRCh38"},
        {"GeneSymbol": "TP53", "ClinicalSignificance": "Pathogenic",
         "PhenotypeList": "Li-Fraumeni syndrome", "Type": "single nucleotide variant",
         "Assembly": "GRCh38"},
    ])
    _write_real_parquet(tmp_path / "clinvar/clinvar_summary.parquet", cv_df)

    # UniProt
    up_df = pd.DataFrame([
        {"Entry": "Q5T4S7", "Gene Names": "CD276 B7H3", "Protein names": "CD276 antigen",
         "Function [CC]": "Immune checkpoint", "Sequence": "MLRR..."},
        {"Entry": "P04637", "Gene Names": "TP53", "Protein names": "Cellular tumor antigen p53",
         "Function [CC]": "Tumor suppressor", "Sequence": "MEEP..."},
    ])
    _write_real_parquet(tmp_path / "uniprot/uniprot_human_reviewed.parquet", up_df)

    # ChEMBL
    ch_df = pd.DataFrame([
        {"compound_chembl_id": "CHEMBL25", "compound_name": "Aspirin",
         "target_chembl_id": "CHEMBL220", "target_name": "Cyclooxygenase-2",
         "standard_type": "IC50", "standard_value": 50.0,
         "standard_units": "nM", "pchembl_value": 7.3},
        {"compound_chembl_id": "CHEMBL941", "compound_name": "Erlotinib",
         "target_chembl_id": "CHEMBL203", "target_name": "EGFR",
         "standard_type": "IC50", "standard_value": 2.0,
         "standard_units": "nM", "pchembl_value": 8.7},
    ])
    _write_real_parquet(tmp_path / "chembl/chembl_activities.parquet", ch_df)

    # GWAS Catalog
    gwas_df = pd.DataFrame([
        {"MAPPED_GENE": "TP53", "DISEASE/TRAIT": "Lung cancer",
         "P-VALUE": 1e-10, "OR or BETA": 1.5, "SNPS": "rs1042522"},
        {"MAPPED_GENE": "BRCA1", "DISEASE/TRAIT": "Breast cancer",
         "P-VALUE": 5e-20, "OR or BETA": 2.1, "SNPS": "rs1799950"},
    ])
    _write_real_parquet(tmp_path / "gwas_catalog/gwas_associations.parquet", gwas_df)

    # Manifest with column schemas
    manifest: dict = {}
    for name in ["gene_ontology", "clinvar", "uniprot", "chembl", "gwas_catalog"]:
        meta = _DATASET_META[name]
        pq_path = tmp_path / meta["file"]
        schema = pq.read_schema(pq_path)
        pq_meta = pq.read_metadata(pq_path)
        manifest[name] = {
            "parquet": str(pq_path),
            "rows": pq_meta.num_rows,
            "cols": pq_meta.num_columns,
            "error": None,
            "columns": [
                {"name": schema.field(i).name, "type": str(schema.field(i).type)}
                for i in range(len(schema))
            ],
        }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    return tmp_path


# ---------------------------------------------------------------------------
# Tests: Agent queries real parquet files
# ---------------------------------------------------------------------------


class TestAgentParquetQueries:
    """Simulate the queries an agent would run against the data lake."""

    def test_query_gene_ontology_by_symbol(self, populated_data_lake: Path) -> None:
        """Agent finds GO terms for TP53."""
        data_path = str(populated_data_lake)
        go = pd.read_parquet(f"{data_path}/gene_ontology/go_annotations.parquet")
        result = go[go["db_object_symbol"] == "TP53"]
        assert len(result) == 1
        assert result.iloc[0]["go_id"] == "GO:0006915"

    def test_query_clinvar_pathogenic_variants(self, populated_data_lake: Path) -> None:
        """Agent finds pathogenic variants in BRCA2."""
        data_path = str(populated_data_lake)
        cv = pd.read_parquet(f"{data_path}/clinvar/clinvar_summary.parquet")
        pathogenic = cv[
            (cv["GeneSymbol"] == "BRCA2")
            & (cv["ClinicalSignificance"].str.contains("Pathogenic", na=False))
        ]
        assert len(pathogenic) == 1
        assert "Breast cancer" in pathogenic.iloc[0]["PhenotypeList"]

    def test_query_uniprot_by_gene_name(self, populated_data_lake: Path) -> None:
        """Agent looks up B7-H3 (CD276) in UniProt."""
        data_path = str(populated_data_lake)
        up = pd.read_parquet(f"{data_path}/uniprot/uniprot_human_reviewed.parquet")
        result = up[up["Gene Names"].str.contains("CD276", na=False)]
        assert len(result) == 1
        assert result.iloc[0]["Entry"] == "Q5T4S7"

    def test_query_chembl_by_target_and_potency(self, populated_data_lake: Path) -> None:
        """Agent finds EGFR inhibitors with IC50 < 100 nM."""
        data_path = str(populated_data_lake)
        ch = pd.read_parquet(f"{data_path}/chembl/chembl_activities.parquet")
        result = ch[
            (ch["target_name"].str.contains("EGFR", na=False))
            & (ch["standard_type"] == "IC50")
            & (ch["standard_value"] < 100)
        ]
        assert len(result) == 1
        assert result.iloc[0]["compound_name"] == "Erlotinib"

    def test_query_gwas_by_trait(self, populated_data_lake: Path) -> None:
        """Agent finds GWAS hits for lung cancer."""
        data_path = str(populated_data_lake)
        gwas = pd.read_parquet(f"{data_path}/gwas_catalog/gwas_associations.parquet")
        result = gwas[gwas["DISEASE/TRAIT"].str.contains("lung cancer", case=False, na=False)]
        assert len(result) == 1
        assert result.iloc[0]["MAPPED_GENE"] == "TP53"

    def test_column_selection_works(self, populated_data_lake: Path) -> None:
        """Agent can select specific columns (predicate pushdown pattern)."""
        data_path = str(populated_data_lake)
        up = pd.read_parquet(
            f"{data_path}/uniprot/uniprot_human_reviewed.parquet",
            columns=["Entry", "Gene Names"],
        )
        assert list(up.columns) == ["Entry", "Gene Names"]
        assert len(up) == 2

    def test_cross_dataset_join(self, populated_data_lake: Path) -> None:
        """Agent joins ClinVar + GWAS to find genes with both variants and associations."""
        data_path = str(populated_data_lake)
        cv = pd.read_parquet(f"{data_path}/clinvar/clinvar_summary.parquet")
        gwas = pd.read_parquet(f"{data_path}/gwas_catalog/gwas_associations.parquet")

        clinvar_genes = set(cv["GeneSymbol"].unique())
        gwas_genes = set(gwas["MAPPED_GENE"].unique())
        overlap = clinvar_genes & gwas_genes
        assert "TP53" in overlap
        assert len(overlap) >= 1


class TestDataLakeContextWithRealData:
    """Verify data_lake_context() reflects actual parquet contents."""

    def test_context_reflects_real_row_counts(self, populated_data_lake: Path) -> None:
        with patch("integrations.data_lake._resolve_data_dir", return_value=populated_data_lake):
            ctx = data_lake_context()

        # Should include all 5 datasets we created
        assert "gene_ontology" in ctx
        assert "clinvar" in ctx
        assert "uniprot" in ctx
        assert "chembl" in ctx
        assert "gwas_catalog" in ctx
        # Row counts from manifest
        assert "3 rows" in ctx  # gene_ontology has 3 rows
        assert "2 rows" in ctx  # chembl/uniprot/gwas have 2 rows

    def test_context_includes_column_schema(self, populated_data_lake: Path) -> None:
        with patch("integrations.data_lake._resolve_data_dir", return_value=populated_data_lake):
            ctx = data_lake_context()

        # Should include full column list from manifest
        assert "All columns:" in ctx
        # Check a few columns are listed
        assert "`db_object_symbol`" in ctx  # gene_ontology
        assert "`GeneSymbol`" in ctx  # clinvar

    def test_context_excludes_missing_datasets(self, populated_data_lake: Path) -> None:
        with patch("integrations.data_lake._resolve_data_dir", return_value=populated_data_lake):
            ctx = data_lake_context()

        # These weren't created in the fixture
        assert "dbsnp" not in ctx
        assert "omim" not in ctx
        assert "ensembl" not in ctx
