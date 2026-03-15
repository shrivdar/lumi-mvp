# YOHAS Bio Data Lake

Reference biomedical databases pre-indexed as Parquet files for fast local
querying by YOHAS agents.

## Quick Start

```bash
pip install pandas pyarrow requests tqdm
python data/download_data_lake.py            # download all (~8 GB)
python data/download_data_lake.py --only gene_ontology msigdb  # subset
python data/download_data_lake.py --list     # show available datasets
python data/download_data_lake.py --skip chembl dbsnp  # skip large ones
```

The script is **idempotent** — re-running it skips datasets whose Parquet
output already exists.

## Datasets

| # | Dataset | Dir | Approx Size | Format | Key Columns |
|---|---------|-----|-------------|--------|-------------|
| 1 | **Gene Ontology** | `gene_ontology/` | ~200 MB | GAF 2.2 → Parquet | `db_object_symbol`, `go_id`, `aspect`, `evidence_code`, `qualifier` |
| 2 | **MSigDB** | `msigdb/` | ~50 MB | GMT → Parquet | `gene_set`, `description`, `gene_symbol` |
| 3 | **ClinVar** | `clinvar/` | ~1.5 GB | TSV → Parquet | `GeneSymbol`, `ClinicalSignificance`, `PhenotypeList`, `Assembly`, `Type` |
| 4 | **GWAS Catalog** | `gwas_catalog/` | ~100 MB | TSV → Parquet | `MAPPED_GENE`, `DISEASE/TRAIT`, `P-VALUE`, `OR or BETA`, `SNP_ID_CURRENT` |
| 5 | **DrugBank** | `drugbank/` | ~50 MB | CSV → Parquet | `drugbank_id`, `name`, `type`, `cas_number`, `groups` |
| 6 | **UniProt** | `uniprot/` | ~500 MB | TSV → Parquet | `Entry`, `Gene Names`, `Protein names`, `Function [CC]`, `Sequence` |
| 7 | **Reactome** | `reactome/` | ~100 MB | TSV → Parquet | `uniprot_id`, `reactome_id`, `pathway_name`, `evidence_code` |
| 8 | **ChEMBL** | `chembl/` | ~2 GB | SQLite → Parquet | `compound_chembl_id`, `target_chembl_id`, `standard_type`, `standard_value`, `pchembl_value`, `canonical_smiles` |
| 9 | **OMIM** | `omim/` | ~50 MB | TSV → Parquet | `mim_number`, `gene_symbol`, `entrez_gene_id`, `ensembl_id` |
| 10 | **dbSNP** | `dbsnp/` | ~3 GB | VCF → Parquet | `chrom`, `pos`, `rsid`, `ref`, `alt`, `gene_info`, `freq`, `variant_class` |
| 11 | **Ensembl** | `ensembl/` | ~500 MB | GTF → Parquet | `seqname`, `feature`, `gene_id`, `gene_name`, `gene_biotype`, `start`, `end` |

## Column Details

### Gene Ontology (`go_annotations.parquet`)
- `db` — Source database (e.g., UniProtKB)
- `db_object_id` — UniProt accession
- `db_object_symbol` — Gene symbol
- `qualifier` — Relation (enables, part_of, involved_in)
- `go_id` — Gene Ontology term ID (GO:XXXXXXX)
- `evidence_code` — Evidence code (IDA, IPI, TAS, etc.)
- `aspect` — Ontology: P (biological process), F (molecular function), C (cellular component)
- `db_object_name` — Full protein name
- **Update frequency**: Monthly

### MSigDB (`msigdb_hallmark.parquet`)
- `gene_set` — Name of the gene set (e.g., HALLMARK_APOPTOSIS)
- `description` — URL or brief description
- `gene_symbol` — HGNC gene symbol in the set
- Includes Hallmark (H) and Curated (C2) collections
- **Update frequency**: Annually

### ClinVar (`clinvar_summary.parquet`)
- `GeneSymbol` — Associated gene
- `ClinicalSignificance` — Pathogenic, Benign, VUS, etc.
- `PhenotypeList` — Associated phenotypes/diseases
- `Type` — Variant type (single nucleotide variant, deletion, etc.)
- `Assembly` — Genome assembly (GRCh37, GRCh38)
- `ReviewStatus` — Star rating of clinical review
- **Update frequency**: Monthly

### GWAS Catalog (`gwas_associations.parquet`)
- `MAPPED_GENE` — Gene(s) mapped to the variant
- `DISEASE/TRAIT` — Associated trait or disease
- `P-VALUE` — Association p-value
- `OR or BETA` — Effect size
- `SNP_ID_CURRENT` — dbSNP rsID
- `RISK ALLELE FREQUENCY` — Frequency of risk allele
- **Update frequency**: Weekly

### DrugBank (`drugbank_vocabulary.parquet`)
- `drugbank_id` — DrugBank accession (DB00001)
- `name` — Drug name
- `type` — Small molecule, Biotech, etc.
- `cas_number` — CAS registry number
- `groups` — Approved, Experimental, Investigational, etc.
- **Note**: Full dataset requires DrugBank license; open vocabulary used by default
- **Update frequency**: Annually

### UniProt (`uniprot_human_reviewed.parquet`)
- `Entry` — UniProt accession
- `Gene Names` — Associated gene names
- `Protein names` — Full protein name(s)
- `Function [CC]` — Functional description
- `Subcellular location [CC]` — Where the protein is found
- `Involvement in disease` — Disease associations
- `Sequence` — Full amino acid sequence
- `Cross-reference (PDB/STRING/Reactome/KEGG)` — External DB links
- **Update frequency**: Monthly (Swiss-Prot releases)

### Reactome (`reactome_pathways.parquet`, `reactome_hierarchy.parquet`)
- `uniprot_id` — UniProt accession
- `reactome_id` — Reactome stable ID (R-HSA-XXXXXX)
- `pathway_name` — Human-readable pathway name
- `evidence_code` — Evidence type (TAS, IEA)
- Hierarchy file: `parent_id` → `child_id` relationships
- **Update frequency**: Quarterly

### ChEMBL (`chembl_activities.parquet`)
- `compound_chembl_id` / `compound_name` — Compound identifiers
- `target_chembl_id` / `target_name` — Target identifiers
- `standard_type` — Measurement type (IC50, Ki, Kd, EC50)
- `standard_value` / `standard_units` — Measured activity value
- `pchembl_value` — Normalized -log10 activity (comparable across types)
- `canonical_smiles` — SMILES structure
- `target_type` — SINGLE PROTEIN, PROTEIN COMPLEX, etc.
- Filtered to: human targets, activity types IC50/Ki/Kd/EC50/GI50/Activity/Potency/Inhibition
- **Update frequency**: Biannually

### OMIM (`omim_genemap.parquet`)
- `mim_number` — OMIM entry number
- `gene_symbol` — HGNC gene symbol
- `mim_entry_type` — Gene, Phenotype, etc.
- `entrez_gene_id` — NCBI Gene ID
- `ensembl_id` — Ensembl gene ID
- **Note**: Full genemap2 available with `OMIM_API_KEY` env var
- **Update frequency**: Monthly

### dbSNP (`dbsnp_common.parquet`)
- `chrom` — Chromosome
- `pos` — Position (GRCh38)
- `rsid` — dbSNP rsID
- `ref` / `alt` — Reference and alternate alleles
- `gene_info` — Gene symbol and ID
- `freq` — Allele frequencies across populations
- `variant_class` — SNV, DIV, MNV, etc.
- **Update frequency**: Biannually

### Ensembl (`ensembl_genes.parquet`)
- `seqname` — Chromosome/scaffold
- `feature` — gene, transcript, or exon
- `start` / `end` — Genomic coordinates (GRCh38)
- `strand` — + or -
- `gene_id` — Ensembl gene ID (ENSG...)
- `gene_name` — HGNC symbol
- `gene_biotype` — protein_coding, lncRNA, etc.
- `transcript_id` — Ensembl transcript ID (ENST...)
- **Update frequency**: Quarterly (Ensembl releases)

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OMIM_API_KEY` | Optional | OMIM API key for full genemap2 download (free for academic use) |

## Disk Space

| Phase | Space Needed |
|-------|-------------|
| Download (raw + parquet) | ~12 GB |
| After cleanup | ~8 GB (parquet only) |

Large raw files (dbSNP VCF, ChEMBL SQLite) are automatically deleted after
Parquet conversion to save space.

## Loading Data

```python
import pandas as pd

# Load a dataset
go = pd.read_parquet("data/gene_ontology/go_annotations.parquet")
clinvar = pd.read_parquet("data/clinvar/clinvar_summary.parquet")

# Filter examples
apoptosis_genes = go[go["go_id"] == "GO:0006915"]["db_object_symbol"].unique()
pathogenic = clinvar[clinvar["ClinicalSignificance"].str.contains("Pathogenic", na=False)]

# Load manifest for metadata
import json
with open("data/manifest.json") as f:
    manifest = json.load(f)
```
