"""Data lake context provider — reads manifest.json and builds agent-facing descriptions.

The ``data_lake_context()`` function returns a formatted string that agents
can include in their system prompts so they know which datasets are available,
their schemas, row counts, and example access patterns.

Usage::

    from integrations.data_lake import data_lake_context
    ctx = data_lake_context()  # returns "" if manifest not found
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from core.config import settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Dataset metadata — key columns and descriptions for each dataset
# (these are static; the manifest provides row/col counts at runtime)
# ---------------------------------------------------------------------------

_DATASET_META: dict[str, dict[str, Any]] = {
    "gene_ontology": {
        "description": "Gene Ontology annotations (human, GAF 2.2) — gene-to-GO-term mappings with evidence codes",
        "file": "gene_ontology/go_annotations.parquet",
        "key_columns": ["db_object_symbol", "go_id", "aspect", "evidence_code", "db_object_name"],
        "example_query": (
            "go = pd.read_parquet(f'{data_path}/gene_ontology/go_annotations.parquet')\n"
            "# Find GO terms for TP53\n"
            "go[go['db_object_symbol'] == 'TP53'][['go_id', 'aspect', 'db_object_name']]"
        ),
    },
    "msigdb": {
        "description": "MSigDB gene sets — Hallmark + C2 curated pathway gene sets",
        "file": "msigdb/msigdb_hallmark.parquet",
        "key_columns": ["gene_set", "description", "gene_symbol"],
        "example_query": (
            "msig = pd.read_parquet(f'{data_path}/msigdb/msigdb_hallmark.parquet')\n"
            "# Find gene sets containing BRCA1\n"
            "msig[msig['gene_symbol'] == 'BRCA1']['gene_set'].unique()"
        ),
    },
    "clinvar": {
        "description": "ClinVar variant summary — clinical significance of human genetic variants",
        "file": "clinvar/clinvar_summary.parquet",
        "key_columns": ["GeneSymbol", "ClinicalSignificance", "PhenotypeList", "Type", "Assembly"],
        "example_query": (
            "cv = pd.read_parquet(f'{data_path}/clinvar/clinvar_summary.parquet')\n"
            "# Pathogenic variants in BRCA2\n"
            "cv[(cv['GeneSymbol'] == 'BRCA2') & (cv['ClinicalSignificance'].str.contains('Pathogenic', na=False))]"
        ),
    },
    "gwas_catalog": {
        "description": "GWAS Catalog — genome-wide association study results (SNP-trait associations)",
        "file": "gwas_catalog/gwas_associations.parquet",
        "key_columns": ["MAPPED_GENE", "DISEASE/TRAIT", "P-VALUE", "OR or BETA", "SNPS"],
        "example_query": (
            "gwas = pd.read_parquet(f'{data_path}/gwas_catalog/gwas_associations.parquet')\n"
            "# GWAS hits for lung cancer\n"
            "gwas[gwas['DISEASE/TRAIT'].str.contains('lung cancer', case=False, na=False)]"
        ),
    },
    "drugbank": {
        "description": "DrugBank open vocabulary — drug names, identifiers, CAS numbers, and types",
        "file": "drugbank/drugbank_vocabulary.parquet",
        "key_columns": ["drugbank_id", "name", "type", "cas_number", "unii"],
        "example_query": (
            "db = pd.read_parquet(f'{data_path}/drugbank/drugbank_vocabulary.parquet')\n"
            "# Search for drugs by name\n"
            "db[db['name'].str.contains('imatinib', case=False, na=False)]"
        ),
    },
    "uniprot": {
        "description": (
            "UniProt human reviewed (Swiss-Prot) — protein sequences, functions, domains, cross-references"
        ),
        "file": "uniprot/uniprot_human_reviewed.parquet",
        "key_columns": ["Entry", "Gene Names", "Protein names", "Function [CC]", "Sequence"],
        "example_query": (
            "up = pd.read_parquet(f'{data_path}/uniprot/uniprot_human_reviewed.parquet')\n"
            "# Look up B7-H3 (CD276)\n"
            "up[up['Gene Names'].str.contains('CD276', na=False)][['Entry', 'Gene Names', 'Function [CC]']]"
        ),
    },
    "reactome": {
        "description": "Reactome pathway annotations — UniProt-to-Reactome pathway mappings (human)",
        "file": "reactome/reactome_pathways.parquet",
        "key_columns": ["uniprot_id", "reactome_id", "pathway_name", "evidence_code"],
        "example_query": (
            "rx = pd.read_parquet(f'{data_path}/reactome/reactome_pathways.parquet')\n"
            "# Pathways involving P53 (UniProt P04637)\n"
            "rx[rx['uniprot_id'] == 'P04637'][['reactome_id', 'pathway_name']]"
        ),
    },
    "chembl": {
        "description": (
            "ChEMBL bioactivity data — compound-target binding assays (IC50, Ki, Kd, EC50) for human targets"
        ),
        "file": "chembl/chembl_activities.parquet",
        "key_columns": ["compound_chembl_id", "compound_name", "target_chembl_id", "target_name",
                        "standard_type", "standard_value", "standard_units", "pchembl_value"],
        "example_query": (
            "ch = pd.read_parquet(f'{data_path}/chembl/chembl_activities.parquet')\n"
            "# Find compounds targeting EGFR with IC50 < 100 nM\n"
            "ch[(ch['target_name'].str.contains('EGFR', na=False)) & "
            "(ch['standard_type'] == 'IC50') & (ch['standard_value'] < 100)]"
        ),
    },
    "omim": {
        "description": "OMIM gene-disease mappings — Mendelian Inheritance in Man gene map",
        "file": "omim/omim_genemap.parquet",
        "key_columns": ["mim_number", "gene_symbol", "entrez_gene_id"],
        "example_query": (
            "omim = pd.read_parquet(f'{data_path}/omim/omim_genemap.parquet')\n"
            "# Look up OMIM entries for CFTR\n"
            "omim[omim['gene_symbol'] == 'CFTR']"
        ),
    },
    "dbsnp": {
        "description": "dbSNP common variants — human SNP positions, allele frequencies, and clinical significance",
        "file": "dbsnp/dbsnp_common.parquet",
        "key_columns": ["chrom", "pos", "rsid", "ref", "alt", "freq", "gene_info"],
        "example_query": (
            "snp = pd.read_parquet(f'{data_path}/dbsnp/dbsnp_common.parquet')\n"
            "# Look up a specific SNP\n"
            "snp[snp['rsid'] == 'rs1801133']"
        ),
    },
    "ensembl": {
        "description": "Ensembl gene annotations (GRCh38) — gene/transcript/exon coordinates and identifiers",
        "file": "ensembl/ensembl_genes.parquet",
        "key_columns": ["gene_id", "gene_name", "feature", "start", "end", "seqname", "strand"],
        "example_query": (
            "ens = pd.read_parquet(f'{data_path}/ensembl/ensembl_genes.parquet')\n"
            "# Gene coordinates for KRAS\n"
            "ens[(ens['gene_name'] == 'KRAS') & (ens['feature'] == 'gene')]"
        ),
    },
}


def _resolve_data_dir() -> Path | None:
    """Resolve the data directory from settings, checking for existence."""
    data_path = settings.data_lake_path
    if not data_path:
        return None

    p = Path(data_path)
    if p.is_dir():
        return p

    # Also check relative to project root (common in dev)
    project_root = Path(__file__).resolve().parent.parent.parent
    alt = project_root / "data"
    if alt.is_dir():
        return alt

    return None


def _load_manifest(data_dir: Path) -> dict[str, Any]:
    """Load manifest.json from the data directory."""
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("data_lake_manifest_load_failed", path=str(manifest_path), error=str(exc))
        return {}


_cached_context: str | None = None


def data_lake_context() -> str:
    """Return a formatted string describing all available data lake datasets.

    This string is designed to be injected into agent system prompts so
    agents know what local data is available and how to query it.

    Results are cached after first call (data lake contents don't change at runtime).
    Returns empty string if no data directory or manifest is found.
    """
    global _cached_context
    if _cached_context is not None:
        return _cached_context

    data_dir = _resolve_data_dir()
    if data_dir is None:
        _cached_context = ""
        return ""

    manifest = _load_manifest(data_dir)

    # Even without manifest, we can describe datasets whose parquet files exist
    data_path_var = "os.environ.get('YOHAS_DATA_PATH', '/data')"
    lines = [
        "## Local Data Lake (Parquet Files)",
        "",
        "You have access to a local biomedical data lake via the Python REPL.",
        f"Access pattern: `data_path = {data_path_var}`",
        "",
        "Available datasets:",
        "",
    ]

    available_count = 0

    for name, meta in _DATASET_META.items():
        # Check if the parquet file actually exists on disk
        parquet_path = data_dir / meta["file"]
        manifest_entry = manifest.get(name, {})

        # Only advertise datasets whose parquet file actually exists on disk
        if not parquet_path.exists():
            continue

        # Use manifest for row/col counts if available
        rows = manifest_entry.get("rows", 0)
        cols = manifest_entry.get("cols", 0)

        available_count += 1

        row_info = f"{rows:,} rows × {cols} cols" if rows else "size unknown"

        lines.append(f"### {name}")
        lines.append(f"**{meta['description']}**")
        lines.append(f"File: `{meta['file']}` ({row_info})")
        lines.append(f"Key columns: {', '.join(f'`{c}`' for c in meta['key_columns'])}")

        # Include full column schema from manifest when available
        manifest_cols = manifest_entry.get("columns")
        if manifest_cols:
            col_list = ", ".join(f"`{c['name']}`" for c in manifest_cols)
            lines.append(f"All columns: {col_list}")

        lines.append(f"```python\n{meta['example_query']}\n```")
        lines.append("")

    if available_count == 0:
        _cached_context = ""
        return ""

    lines.extend([
        "### Usage Notes",
        "- All files are Parquet format — use `pd.read_parquet()`",
        "- Data path is available as `YOHAS_DATA_PATH` environment variable",
        "- Files are READ-ONLY — do not attempt to modify them",
        "- For large datasets (ClinVar, ChEMBL, dbSNP), use column selection: "
        "`pd.read_parquet(path, columns=['col1', 'col2'])`",
        "- Use `filters` parameter for predicate pushdown on large files",
    ])

    result = "\n".join(lines)
    _cached_context = result
    return result
