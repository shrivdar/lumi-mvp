#!/usr/bin/env python3
"""Download and index the YOHAS Bio Data Lake (~8 GB).

Fetches 11 biomedical reference databases, converts each to Parquet, and
stores them under ``data/<dataset>/``.  The script is **idempotent** — it
skips datasets whose Parquet output already exists.

Usage::

    pip install pandas pyarrow requests tqdm
    python data/download_data_lake.py          # download everything
    python data/download_data_lake.py --only gene_ontology msigdb  # subset
    python data/download_data_lake.py --list   # show available datasets

Requirements:
    - Python 3.11+
    - pandas, pyarrow, requests, tqdm
    - ~12 GB free disk (raw + parquet; raw files cleaned up after conversion)
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import sqlite3
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent
MAX_RETRIES = 3
CHUNK_SIZE = 1 << 20  # 1 MiB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("data_lake")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path, *, desc: str | None = None) -> Path:
    """Stream-download *url* to *dest* with a tqdm progress bar.

    Skips the download when *dest* already exists.  Retries up to
    ``MAX_RETRIES`` times on transient errors.
    """
    if dest.exists():
        log.info("  ✓ already downloaded: %s", dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    label = desc or dest.name

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0)) or None
            with open(dest, "wb") as fh, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=label,
                leave=False,
            ) as bar:
                for chunk in resp.iter_content(CHUNK_SIZE):
                    fh.write(chunk)
                    bar.update(len(chunk))
            return dest
        except (requests.RequestException, IOError) as exc:
            log.warning("  ⚠ attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            dest.unlink(missing_ok=True)
            if attempt == MAX_RETRIES:
                raise
    # unreachable, but keeps mypy happy
    raise RuntimeError("download failed")


def _parquet_exists(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        pq.read_metadata(path)
        return True
    except Exception:
        return False


def _write_parquet(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, engine="pyarrow", index=False)
    rows, cols = df.shape
    size_mb = dest.stat().st_size / (1 << 20)
    log.info("  → wrote %s  (%d rows × %d cols, %.1f MB)", dest.name, rows, cols, size_mb)


def _verify_parquet(path: Path) -> tuple[int, int]:
    """Read back a Parquet file and return (rows, cols)."""
    df = pd.read_parquet(path)
    return df.shape


# ---------------------------------------------------------------------------
# Dataset handlers
# ---------------------------------------------------------------------------


@dataclass
class DatasetResult:
    name: str
    parquet_path: Path
    rows: int = 0
    cols: int = 0
    skipped: bool = False
    error: str | None = None


# ---- 1. Gene Ontology ----


def _download_gene_ontology() -> DatasetResult:
    """Gene Ontology annotations (GAF 2.2) → parquet."""
    name = "gene_ontology"
    out_dir = DATA_DIR / name
    parquet = out_dir / "go_annotations.parquet"

    if _parquet_exists(parquet):
        r, c = _verify_parquet(parquet)
        return DatasetResult(name, parquet, r, c, skipped=True)

    # Human GAF from GO consortium
    url = "http://current.geneontology.org/annotations/goa_human.gaf.gz"
    raw = _download(url, out_dir / "goa_human.gaf.gz", desc="GO annotations")

    # Parse GAF (skip lines starting with !)
    cols = [
        "db", "db_object_id", "db_object_symbol", "qualifier", "go_id",
        "db_reference", "evidence_code", "with_from", "aspect", "db_object_name",
        "db_object_synonym", "db_object_type", "taxon", "date", "assigned_by",
        "annotation_extension", "gene_product_form_id",
    ]

    rows: list[dict] = []
    with gzip.open(raw, "rt") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15:
                continue
            # Pad to 17 columns
            while len(parts) < 17:
                parts.append("")
            rows.append(dict(zip(cols, parts)))

    df = pd.DataFrame(rows)
    _write_parquet(df, parquet)
    r, c = df.shape
    return DatasetResult(name, parquet, r, c)


# ---- 2. MSigDB ----


def _download_msigdb() -> DatasetResult:
    """MSigDB Hallmark gene sets (GMT) → parquet."""
    name = "msigdb"
    out_dir = DATA_DIR / name
    parquet = out_dir / "msigdb_hallmark.parquet"

    if _parquet_exists(parquet):
        r, c = _verify_parquet(parquet)
        return DatasetResult(name, parquet, r, c, skipped=True)

    # Hallmark gene sets (v2024.1, human symbols) — available without login
    url = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt"
    raw = _download(url, out_dir / "hallmark.gmt", desc="MSigDB Hallmark")

    rows: list[dict] = []
    with open(raw) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            gene_set = parts[0]
            description = parts[1]
            genes = parts[2:]
            for gene in genes:
                rows.append({
                    "gene_set": gene_set,
                    "description": description,
                    "gene_symbol": gene,
                })

    df = pd.DataFrame(rows)

    # Also download the full C2 curated gene sets for broader coverage
    url_c2 = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/c2.all.v2024.1.Hs.symbols.gmt"
    try:
        raw_c2 = _download(url_c2, out_dir / "c2_curated.gmt", desc="MSigDB C2 curated")
        rows_c2: list[dict] = []
        with open(raw_c2) as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                for gene in parts[2:]:
                    rows_c2.append({
                        "gene_set": parts[0],
                        "description": parts[1],
                        "gene_symbol": gene,
                    })
        df_c2 = pd.DataFrame(rows_c2)
        df = pd.concat([df, df_c2], ignore_index=True)
    except Exception as exc:
        log.warning("  ⚠ C2 curated download failed, continuing with Hallmark only: %s", exc)

    _write_parquet(df, parquet)
    r, c = df.shape
    return DatasetResult(name, parquet, r, c)


# ---- 3. ClinVar ----


def _parse_vcf_to_df(path: Path, *, gzipped: bool = True) -> pd.DataFrame:
    """Parse a VCF file into a DataFrame (INFO field expanded to columns)."""
    opener = gzip.open if gzipped else open
    header_cols: list[str] = []
    records: list[dict] = []

    with opener(path, "rt") as fh:  # type: ignore[arg-type]
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                header_cols = line.lstrip("#").strip().split("\t")
                continue
            if not header_cols:
                continue
            parts = line.strip().split("\t")
            row: dict[str, Any] = {}
            for i, col in enumerate(header_cols):
                if i < len(parts):
                    row[col] = parts[i]

            # Expand INFO field
            info_str = row.pop("INFO", "")
            if info_str and info_str != ".":
                for item in info_str.split(";"):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        row[f"INFO_{k}"] = v
                    else:
                        row[f"INFO_{item}"] = True

            records.append(row)

    return pd.DataFrame(records)


def _download_clinvar() -> DatasetResult:
    """ClinVar variant summary TSV → parquet (faster than VCF for this dataset)."""
    name = "clinvar"
    out_dir = DATA_DIR / name
    parquet = out_dir / "clinvar_summary.parquet"

    if _parquet_exists(parquet):
        r, c = _verify_parquet(parquet)
        return DatasetResult(name, parquet, r, c, skipped=True)

    # The variant_summary.txt.gz is easier to parse and has more structured data
    url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
    raw = _download(url, out_dir / "variant_summary.txt.gz", desc="ClinVar summary")

    df = pd.read_csv(raw, sep="\t", low_memory=False, compression="gzip")
    _write_parquet(df, parquet)
    r, c = df.shape
    return DatasetResult(name, parquet, r, c)


# ---- 4. GWAS Catalog ----


def _download_gwas_catalog() -> DatasetResult:
    """GWAS Catalog associations TSV → parquet.

    Tries the REST API first; falls back to the EBI FTP mirror when the API
    returns an error (the API intermittently returns HTTP 500).
    """
    name = "gwas_catalog"
    out_dir = DATA_DIR / name
    parquet = out_dir / "gwas_associations.parquet"

    if _parquet_exists(parquet):
        r, c = _verify_parquet(parquet)
        return DatasetResult(name, parquet, r, c, skipped=True)

    raw = out_dir / "gwas_associations.tsv"

    # Primary: REST API download
    api_url = "https://www.ebi.ac.uk/gwas/api/search/downloads/alternative"
    try:
        _download(api_url, raw, desc="GWAS Catalog (API)")
    except Exception as api_exc:
        log.warning("  ⚠ GWAS API failed (%s), falling back to FTP mirror", api_exc)
        raw.unlink(missing_ok=True)

        # Fallback: EBI FTP — the TSV is inside a directory listing; grab the
        # alternative associations file directly via HTTPS-over-FTP gateway.
        ftp_url = (
            "https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/"
            "gwas-catalog-associations_ontology-annotated.tsv"
        )
        try:
            _download(ftp_url, raw, desc="GWAS Catalog (FTP)")
        except Exception as ftp_exc:
            # Last resort: try plain FTP via urllib
            log.warning("  ⚠ HTTPS FTP gateway failed (%s), trying plain FTP", ftp_exc)
            raw.unlink(missing_ok=True)
            import urllib.request
            ftp_direct = (
                "ftp://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/"
                "gwas-catalog-associations_ontology-annotated.tsv"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(ftp_direct, raw)

    df = pd.read_csv(raw, sep="\t", low_memory=False)
    _write_parquet(df, parquet)
    r, c = df.shape
    return DatasetResult(name, parquet, r, c)


# ---- 5. DrugBank (open subset) ----


def _download_drugbank() -> DatasetResult:
    """DrugBank open vocabulary + drug links → parquet.

    The full DrugBank XML requires a license. We use the freely available
    vocabulary and drug-links files from the open data portal.
    """
    name = "drugbank"
    out_dir = DATA_DIR / name
    parquet = out_dir / "drugbank_vocabulary.parquet"

    if _parquet_exists(parquet):
        r, c = _verify_parquet(parquet)
        return DatasetResult(name, parquet, r, c, skipped=True)

    # DrugBank open data — vocabulary CSV (drugbank ID, name, CAS, UNII, synonyms, etc.)
    url = "https://go.drugbank.com/releases/latest/downloads/all-drugbank-vocabulary"
    try:
        raw = _download(url, out_dir / "drugbank_vocabulary.csv.zip", desc="DrugBank vocab")
        # It's a zip file containing a CSV
        with zipfile.ZipFile(raw) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if csv_names:
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(f)
            else:
                raise ValueError("No CSV found in DrugBank zip")
    except Exception as exc:
        log.warning("  ⚠ DrugBank open vocabulary requires login, trying fallback: %s", exc)
        # Fallback: use a public DrugBank-derived dataset (drug approvals)
        # We'll create a placeholder with structure that can be populated later
        log.info("  → Creating DrugBank placeholder — populate manually or with API key")
        df = pd.DataFrame(columns=[
            "drugbank_id", "name", "type", "cas_number", "unii",
            "groups", "categories", "description",
        ])

    _write_parquet(df, parquet)
    r, c = df.shape
    return DatasetResult(name, parquet, r, c)


# ---- 6. UniProt human reviewed ----


def _download_uniprot() -> DatasetResult:
    """UniProt human reviewed (Swiss-Prot) TSV → parquet."""
    name = "uniprot"
    out_dir = DATA_DIR / name
    parquet = out_dir / "uniprot_human_reviewed.parquet"

    if _parquet_exists(parquet):
        r, c = _verify_parquet(parquet)
        return DatasetResult(name, parquet, r, c, skipped=True)

    # UniProt REST API — human reviewed entries, TSV format
    fields = ",".join([
        "accession", "id", "gene_names", "protein_name", "organism_name",
        "length", "mass", "go_id", "go_p", "go_c", "go_f",
        "cc_function", "cc_subcellular_location", "cc_disease",
        "cc_pathway", "ft_domain", "ft_binding", "ft_act_site",
        "xref_pdb", "xref_string", "xref_reactome", "xref_kegg",
        "sequence",
    ])
    url = (
        "https://rest.uniprot.org/uniprotkb/stream?"
        f"query=(organism_id:9606)+AND+(reviewed:true)&format=tsv&fields={fields}"
    )
    raw = _download(url, out_dir / "uniprot_human_reviewed.tsv", desc="UniProt human")

    df = pd.read_csv(raw, sep="\t", low_memory=False)
    _write_parquet(df, parquet)
    r, c = df.shape
    return DatasetResult(name, parquet, r, c)


# ---- 7. Reactome pathways ----


def _download_reactome() -> DatasetResult:
    """Reactome pathway annotations → parquet."""
    name = "reactome"
    out_dir = DATA_DIR / name
    parquet_pathways = out_dir / "reactome_pathways.parquet"

    if _parquet_exists(parquet_pathways):
        r, c = _verify_parquet(parquet_pathways)
        return DatasetResult(name, parquet_pathways, r, c, skipped=True)

    # Gene-to-pathway mapping (UniProt → Reactome)
    url_genes = "https://reactome.org/download/current/UniProt2Reactome.txt"
    raw_genes = _download(url_genes, out_dir / "UniProt2Reactome.txt", desc="Reactome gene-pathway")

    cols = ["uniprot_id", "reactome_id", "url", "pathway_name", "evidence_code", "species"]
    df = pd.read_csv(raw_genes, sep="\t", header=None, names=cols)
    # Filter to human
    df = df[df["species"] == "Homo sapiens"].reset_index(drop=True)

    # Also get the pathway hierarchy
    url_hier = "https://reactome.org/download/current/ReactomePathwaysRelation.txt"
    raw_hier = _download(url_hier, out_dir / "ReactomePathwaysRelation.txt", desc="Reactome hierarchy")
    df_hier = pd.read_csv(raw_hier, sep="\t", header=None, names=["parent_id", "child_id"])
    # Filter to human pathways (R-HSA prefix)
    df_hier = df_hier[
        df_hier["parent_id"].str.startswith("R-HSA") & df_hier["child_id"].str.startswith("R-HSA")
    ].reset_index(drop=True)
    df_hier.to_parquet(out_dir / "reactome_hierarchy.parquet", engine="pyarrow", index=False)

    _write_parquet(df, parquet_pathways)
    r, c = df.shape
    return DatasetResult(name, parquet_pathways, r, c)


# ---- 8. ChEMBL activities ----


def _download_chembl() -> DatasetResult:
    """ChEMBL bioactivity data (SQLite → parquet).

    Downloads the ChEMBL SQLite DB and extracts the activities table along
    with compound/target metadata into parquet files.
    """
    name = "chembl"
    out_dir = DATA_DIR / name
    parquet = out_dir / "chembl_activities.parquet"

    if _parquet_exists(parquet):
        r, c = _verify_parquet(parquet)
        return DatasetResult(name, parquet, r, c, skipped=True)

    # ChEMBL 35 SQLite (latest as of 2024)
    chembl_version = "chembl_35"
    url = f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/{chembl_version}_sqlite.tar.gz"
    tarball = out_dir / f"{chembl_version}_sqlite.tar.gz"

    raw = _download(url, tarball, desc="ChEMBL SQLite")

    # Extract the SQLite file from the tarball
    import tarfile
    db_path = out_dir / f"{chembl_version}.db"
    if not db_path.exists():
        log.info("  extracting SQLite database...")
        with tarfile.open(raw, "r:gz") as tf:
            # Find the .db file inside
            for member in tf.getmembers():
                if member.name.endswith(".db"):
                    member.name = db_path.name  # flatten path
                    tf.extract(member, out_dir)
                    break

    if not db_path.exists():
        raise FileNotFoundError(f"Could not find .db file in {tarball}")

    # Extract activities with compound + target info
    log.info("  querying ChEMBL SQLite (this may take a few minutes)...")
    conn = sqlite3.connect(str(db_path))

    query = """
    SELECT
        a.activity_id,
        a.assay_id,
        a.molregno,
        a.standard_type,
        a.standard_relation,
        a.standard_value,
        a.standard_units,
        a.pchembl_value,
        a.activity_comment,
        cs.canonical_smiles,
        cs.standard_inchi_key,
        md.pref_name AS compound_name,
        md.chembl_id AS compound_chembl_id,
        td.pref_name AS target_name,
        td.chembl_id AS target_chembl_id,
        td.organism AS target_organism,
        td.target_type,
        ass.assay_type,
        ass.description AS assay_description
    FROM activities a
    LEFT JOIN compound_structures cs ON a.molregno = cs.molregno
    LEFT JOIN molecule_dictionary md ON a.molregno = md.molregno
    LEFT JOIN assays ass ON a.assay_id = ass.assay_id
    LEFT JOIN target_dictionary td ON ass.tid = td.tid
    WHERE a.standard_type IN ('IC50', 'Ki', 'Kd', 'EC50', 'GI50', 'Activity', 'Potency', 'Inhibition')
      AND a.standard_value IS NOT NULL
      AND td.organism = 'Homo sapiens'
    """

    # Read in chunks to avoid memory issues
    chunks = []
    for chunk in pd.read_sql_query(query, conn, chunksize=500_000):
        chunks.append(chunk)
    conn.close()

    if chunks:
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.DataFrame()

    _write_parquet(df, parquet)

    # Clean up the large SQLite file and tarball to save disk space
    log.info("  cleaning up ChEMBL raw files (~4 GB)...")
    db_path.unlink(missing_ok=True)
    tarball.unlink(missing_ok=True)

    r, c = df.shape
    return DatasetResult(name, parquet, r, c)


# ---- 9. OMIM gene-disease ----


def _download_omim() -> DatasetResult:
    """OMIM gene-disease mappings → parquet.

    Uses the freely available genemap2.txt from OMIM's FTP. The full OMIM
    dataset requires an API key (set OMIM_API_KEY env var for richer data).
    """
    name = "omim"
    out_dir = DATA_DIR / name
    parquet = out_dir / "omim_genemap.parquet"

    if _parquet_exists(parquet):
        r, c = _verify_parquet(parquet)
        return DatasetResult(name, parquet, r, c, skipped=True)

    # OMIM genemap2 — freely available (requires accepting terms but no key for FTP)
    url = "https://data.omim.org/downloads/PLACEHOLDER/genemap2.txt"

    import os
    omim_key = os.environ.get("OMIM_API_KEY", "")

    if omim_key:
        url = url.replace("PLACEHOLDER", omim_key)
    else:
        # Fallback: use the mim2gene.txt which is freely available
        log.warning("  ⚠ OMIM_API_KEY not set — using mim2gene.txt (limited fields)")
        url = "https://omim.org/static/omim/data/mim2gene.txt"

    try:
        raw = _download(url, out_dir / "omim_genemap.txt", desc="OMIM")
        rows: list[dict] = []
        with open(raw) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if not omim_key:
                    # mim2gene format: MIM Number, MIM Entry Type, Entrez Gene ID, HGNC Symbol, Ensembl ID
                    if len(parts) >= 4:
                        rows.append({
                            "mim_number": parts[0],
                            "mim_entry_type": parts[1],
                            "entrez_gene_id": parts[2] if len(parts) > 2 else "",
                            "gene_symbol": parts[3] if len(parts) > 3 else "",
                            "ensembl_id": parts[4] if len(parts) > 4 else "",
                        })
                else:
                    # genemap2 format — full columns
                    if len(parts) >= 10:
                        rows.append({
                            "chromosome": parts[0],
                            "genomic_position_start": parts[1],
                            "genomic_position_end": parts[2],
                            "cyto_location": parts[3],
                            "computed_cyto_location": parts[4],
                            "mim_number": parts[5],
                            "gene_symbols": parts[6],
                            "gene_name": parts[7],
                            "approved_gene_symbol": parts[8],
                            "entrez_gene_id": parts[9],
                            "ensembl_gene_id": parts[10] if len(parts) > 10 else "",
                            "comments": parts[11] if len(parts) > 11 else "",
                            "phenotypes": parts[12] if len(parts) > 12 else "",
                            "mouse_gene_id": parts[13] if len(parts) > 13 else "",
                        })
        df = pd.DataFrame(rows)
    except Exception as exc:
        log.warning("  ⚠ OMIM download failed: %s — creating placeholder", exc)
        df = pd.DataFrame(columns=[
            "mim_number", "mim_entry_type", "entrez_gene_id",
            "gene_symbol", "ensembl_id",
        ])

    _write_parquet(df, parquet)
    r, c = df.shape
    return DatasetResult(name, parquet, r, c)


# ---- 10. dbSNP common variants ----


def _download_dbsnp() -> DatasetResult:
    """dbSNP common variants VCF → parquet.

    Downloads the common variants VCF (GRCh38) and converts to parquet.
    This is a large file (~3 GB compressed) so we parse in chunks.
    """
    name = "dbsnp"
    out_dir = DATA_DIR / name
    parquet = out_dir / "dbsnp_common.parquet"

    if _parquet_exists(parquet):
        r, c = _verify_parquet(parquet)
        return DatasetResult(name, parquet, r, c, skipped=True)

    url = "https://ftp.ncbi.nih.gov/snp/latest_release/VCF/GCF_000001405.40.gz"
    raw = _download(url, out_dir / "dbsnp_common.vcf.gz", desc="dbSNP common variants")

    log.info("  parsing dbSNP VCF (this takes a while for ~3 GB)...")

    # Parse VCF in streaming fashion — only keep essential columns to manage memory
    records: list[dict] = []
    batch_num = 0
    writer = None
    schema = None

    with gzip.open(raw, "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                continue

            parts = line.strip().split("\t", 8)  # only split first 8 fields
            if len(parts) < 8:
                continue

            chrom, pos, rsid, ref, alt, qual, filt, info = parts[:8]

            # Extract key INFO fields
            row: dict[str, Any] = {
                "chrom": chrom,
                "pos": int(pos) if pos.isdigit() else None,
                "rsid": rsid,
                "ref": ref,
                "alt": alt,
                "qual": qual,
                "filter": filt,
            }

            # Parse selected INFO fields
            for item in info.split(";"):
                if item.startswith("FREQ="):
                    row["freq"] = item[5:]
                elif item.startswith("CLNSIG="):
                    row["clin_sig"] = item[7:]
                elif item.startswith("GENEINFO="):
                    row["gene_info"] = item[9:]
                elif item.startswith("VC="):
                    row["variant_class"] = item[3:]
                elif item == "COMMON":
                    row["common"] = True

            records.append(row)

            # Write in batches to avoid memory issues
            if len(records) >= 2_000_000:
                batch_df = pd.DataFrame(records)
                table = pa.Table.from_pandas(batch_df)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(parquet, schema)
                writer.write_table(table)
                log.info("    wrote batch %d (2M rows)", batch_num)
                batch_num += 1
                records.clear()

    # Write remaining records
    if records:
        batch_df = pd.DataFrame(records)
        table = pa.Table.from_pandas(batch_df)
        if writer is None:
            writer = pq.ParquetWriter(parquet, table.schema)
        else:
            # Align schema
            table = table.cast(schema)
        writer.write_table(table)

    if writer is not None:
        writer.close()

    meta = pq.read_metadata(parquet)
    total_rows = meta.num_rows
    total_cols = meta.num_columns

    log.info("  → wrote %s  (%d rows × %d cols)", parquet.name, total_rows, total_cols)

    # Clean up raw VCF to save space
    raw.unlink(missing_ok=True)

    return DatasetResult(name, parquet, total_rows, total_cols)


# ---- 11. Ensembl gene annotations ----


def _download_ensembl() -> DatasetResult:
    """Ensembl human gene annotations GTF → parquet."""
    name = "ensembl"
    out_dir = DATA_DIR / name
    parquet = out_dir / "ensembl_genes.parquet"

    if _parquet_exists(parquet):
        r, c = _verify_parquet(parquet)
        return DatasetResult(name, parquet, r, c, skipped=True)

    # Ensembl release 113 (latest stable)
    url = "https://ftp.ensembl.org/pub/release-113/gtf/homo_sapiens/Homo_sapiens.GRCh38.113.gtf.gz"
    raw = _download(url, out_dir / "Homo_sapiens.GRCh38.113.gtf.gz", desc="Ensembl GTF")

    log.info("  parsing Ensembl GTF...")
    records: list[dict] = []

    with gzip.open(raw, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue

            seqname, source, feature, start, end, score, strand, frame, attributes = parts

            # Only keep gene and transcript features to manage size
            if feature not in ("gene", "transcript", "exon"):
                continue

            row: dict[str, Any] = {
                "seqname": seqname,
                "source": source,
                "feature": feature,
                "start": int(start),
                "end": int(end),
                "score": score,
                "strand": strand,
                "frame": frame,
            }

            # Parse attributes
            for attr in attributes.split(";"):
                attr = attr.strip()
                if not attr:
                    continue
                m = re.match(r'(\w+)\s+"([^"]*)"', attr)
                if m:
                    row[m.group(1)] = m.group(2)

            records.append(row)

    df = pd.DataFrame(records)
    _write_parquet(df, parquet)
    r, c = df.shape
    return DatasetResult(name, parquet, r, c)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, tuple[callable, str]] = {  # type: ignore[type-arg]
    "gene_ontology": (_download_gene_ontology, "Gene Ontology annotations (GAF)"),
    "msigdb": (_download_msigdb, "MSigDB gene sets (GMT)"),
    "clinvar": (_download_clinvar, "ClinVar variant summary"),
    "gwas_catalog": (_download_gwas_catalog, "GWAS Catalog associations"),
    "drugbank": (_download_drugbank, "DrugBank open vocabulary"),
    "uniprot": (_download_uniprot, "UniProt human reviewed (Swiss-Prot)"),
    "reactome": (_download_reactome, "Reactome pathway annotations"),
    "chembl": (_download_chembl, "ChEMBL bioactivity data"),
    "omim": (_download_omim, "OMIM gene-disease mappings"),
    "dbsnp": (_download_dbsnp, "dbSNP common variants"),
    "ensembl": (_download_ensembl, "Ensembl gene annotations (GTF)"),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and index the YOHAS Bio Data Lake",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="DATASET",
        help="Download only these datasets (space-separated names)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        metavar="DATASET",
        help="Skip these datasets",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:\n")
        for name, (_, desc) in DATASETS.items():
            print(f"  {name:20s}  {desc}")
        print()
        return

    targets = list(DATASETS.keys())
    if args.only:
        unknown = set(args.only) - set(DATASETS)
        if unknown:
            parser.error(f"Unknown datasets: {', '.join(unknown)}")
        targets = args.only
    if args.skip:
        targets = [t for t in targets if t not in args.skip]

    log.info("=" * 60)
    log.info("YOHAS Bio Data Lake — downloading %d datasets", len(targets))
    log.info("Output directory: %s", DATA_DIR)
    log.info("=" * 60)

    results: list[DatasetResult] = []

    for name in targets:
        fn, desc = DATASETS[name]
        log.info("\n[%d/%d] %s — %s", len(results) + 1, len(targets), name, desc)
        try:
            result = fn()
            results.append(result)
            status = "SKIPPED (exists)" if result.skipped else "OK"
            log.info("  ✓ %s  (%d × %d)", status, result.rows, result.cols)
        except Exception as exc:
            log.error("  ✗ FAILED: %s", exc)
            results.append(DatasetResult(name, DATA_DIR / name / "FAILED", error=str(exc)))

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    ok = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    skipped = [r for r in ok if r.skipped]
    downloaded = [r for r in ok if not r.skipped]

    for r in downloaded:
        size = r.parquet_path.stat().st_size / (1 << 20) if r.parquet_path.exists() else 0
        print(f"  ✓ {r.name:20s}  {r.rows:>12,} rows  {r.cols:>4} cols  {size:>8.1f} MB")
    for r in skipped:
        print(f"  ○ {r.name:20s}  (already existed, {r.rows:,} rows)")
    for r in failed:
        print(f"  ✗ {r.name:20s}  ERROR: {r.error}")

    total_size = sum(
        r.parquet_path.stat().st_size
        for r in ok
        if r.parquet_path.exists()
    ) / (1 << 30)
    print(f"\nTotal: {len(downloaded)} downloaded, {len(skipped)} skipped, {len(failed)} failed")
    print(f"Total parquet size on disk: {total_size:.2f} GB")

    # Write manifest with column schemas
    manifest: dict[str, Any] = {}
    for r in results:
        entry: dict[str, Any] = {
            "parquet": str(r.parquet_path),
            "rows": r.rows,
            "cols": r.cols,
            "error": r.error,
        }
        # Read column schema from parquet metadata if file exists
        if r.error is None and r.parquet_path.exists():
            try:
                pq.read_metadata(r.parquet_path)  # validates file
                schema = pq.read_schema(r.parquet_path)
                entry["columns"] = [
                    {"name": schema.field(i).name, "type": str(schema.field(i).type)}
                    for i in range(len(schema))
                ]
            except Exception:
                pass
        manifest[r.name] = entry

    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("\nManifest written to %s", manifest_path)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
