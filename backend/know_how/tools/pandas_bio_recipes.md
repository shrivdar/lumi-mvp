# Pandas Bio Recipes

## Purpose

Common pandas patterns for manipulating biological data — gene expression matrices, variant tables, association results, and clinical data.

## Recipes

### Load and Clean Gene Expression Matrix

```python
import pandas as pd

# Load expression matrix (genes x samples)
expr = pd.read_csv("expression.tsv", sep="\t", index_col=0)

# Remove low-expression genes (< 1 CPM in at least 20% of samples)
min_samples = int(0.2 * expr.shape[1])
expr = expr[expr.gt(1).sum(axis=1) >= min_samples]

# Log2 transform (add pseudocount to avoid log(0))
expr_log = (expr + 1).apply(pd.np.log2)

# Z-score normalize per gene (row-wise)
expr_z = expr_log.sub(expr_log.mean(axis=1), axis=0).div(expr_log.std(axis=1), axis=0)
```

### Parse GWAS Summary Statistics

```python
# Standard column names vary — normalize first
col_map = {"#CHROM": "chr", "POS": "pos", "ID": "snp", "P": "pval",
           "BETA": "beta", "SE": "se", "A1": "effect_allele", "A2": "other_allele"}
gwas = pd.read_csv("gwas_sumstats.tsv", sep="\t").rename(columns=col_map)

# Filter to genome-wide significant
sig = gwas[gwas["pval"] < 5e-8].sort_values("pval")

# Add -log10(p) for plotting
gwas["neglog10p"] = -gwas["pval"].apply(pd.np.log10)

# LD clumping proxy: remove SNPs within 500kb of a more significant SNP
sig_sorted = sig.sort_values("pval").copy()
keep = []
for _, row in sig_sorted.iterrows():
    if not any((abs(row["pos"] - k["pos"]) < 500_000) and (row["chr"] == k["chr"]) for k in keep):
        keep.append(row)
lead_snps = pd.DataFrame(keep)
```

### Differential Expression Analysis Helper

```python
from scipy import stats

def diff_expression(expr_df, group_a_cols, group_b_cols):
    """Simple t-test differential expression between two groups."""
    results = []
    for gene in expr_df.index:
        a_vals = expr_df.loc[gene, group_a_cols]
        b_vals = expr_df.loc[gene, group_b_cols]
        t_stat, pval = stats.ttest_ind(a_vals, b_vals)
        log2fc = b_vals.mean() - a_vals.mean()  # if log-transformed
        results.append({"gene": gene, "log2fc": log2fc, "pval": pval, "t_stat": t_stat})
    result_df = pd.DataFrame(results)
    # Benjamini-Hochberg FDR correction
    from statsmodels.stats.multitest import multipletests
    result_df["fdr"] = multipletests(result_df["pval"], method="fdr_bh")[1]
    return result_df.sort_values("pval")
```

### Merge Gene Annotations with Results

```python
# Merge gene info from MyGene/Ensembl with analysis results
gene_info = pd.read_csv("gene_annotations.tsv", sep="\t")  # symbol, entrez_id, chr, start, end, description
results = pd.read_csv("analysis_results.tsv", sep="\t")

merged = results.merge(gene_info, left_on="gene", right_on="symbol", how="left")

# Flag genes in specific pathways
pathway_genes = {"KEGG_MAPK": ["BRAF", "KRAS", "MAP2K1", "MAPK1"]}
for pathway, genes in pathway_genes.items():
    merged[pathway] = merged["gene"].isin(genes)
```

### Clinical Data Survival Analysis Prep

```python
# Prepare clinical data for survival analysis
clinical = pd.read_csv("clinical.tsv", sep="\t")

# Convert dates to durations
clinical["os_days"] = (pd.to_datetime(clinical["date_death"].fillna(clinical["date_last_contact"]))
                       - pd.to_datetime(clinical["date_diagnosis"])).dt.days
clinical["os_event"] = clinical["date_death"].notna().astype(int)

# Stratify by biomarker
clinical["biomarker_high"] = clinical["biomarker_value"] > clinical["biomarker_value"].median()
```

### Variant Frequency Table Processing

```python
# Process gnomAD-style variant frequency data
variants = pd.read_csv("variants.tsv", sep="\t")

# Filter to rare variants (AF < 1%)
rare = variants[variants["AF"] < 0.01]

# Annotate variant consequence severity
severity_order = ["frameshift", "nonsense", "splice_donor", "splice_acceptor",
                  "missense", "inframe_indel", "synonymous", "intronic", "intergenic"]
variants["severity_rank"] = variants["consequence"].map({v: i for i, v in enumerate(severity_order)})
variants = variants.sort_values("severity_rank")

# Count variants per gene by consequence
gene_burden = variants.groupby(["gene", "consequence"]).size().unstack(fill_value=0)
```

## Common Pitfalls

- **Mixed types in columns**: Gene names like "MARCH1" get auto-converted to dates in Excel-exported CSVs. Use `dtype=str` for gene columns.
- **Missing values**: `NaN` handling differs between `dropna()`, `fillna()`, and boolean indexing. Be explicit.
- **Memory**: Large expression matrices (>50k genes x >1000 samples) may need chunked processing or `float32` dtype.
- **Index alignment**: Pandas aligns on index during operations — ensure indexes match or use `.values` for raw arrays.
