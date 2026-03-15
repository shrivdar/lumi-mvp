# GWAS Analysis Protocol

## Purpose

Guide agents through genome-wide association study interpretation, from raw summary statistics to prioritized candidate genes with mechanistic hypotheses.

## Step-by-Step Protocol

### 1. Evaluate Study Quality

- Check sample size (N > 10,000 for common variants; meta-analyses preferred).
- Confirm genome-wide significance threshold (p < 5e-8) was used.
- Look for genomic inflation factor (lambda GC ~1.0 indicates proper control).
- Check for population stratification correction (principal components, mixed models).

### 2. Identify Lead SNPs and Loci

- Extract SNPs passing genome-wide significance.
- Group SNPs into independent loci using linkage disequilibrium (LD) clumping (r2 < 0.1, 500 kb window).
- For each locus, identify the lead SNP (lowest p-value).
- Record effect size (beta or OR), allele frequencies, and direction of effect.

### 3. Fine-Mapping

- Apply fine-mapping methods (FINEMAP, SuSiE, CAVIAR) to identify credible sets.
- A 95% credible set should contain the causal variant with 95% probability.
- Prioritize variants in credible sets with posterior inclusion probability (PIP) > 0.5.
- Cross-reference with functional annotations (ENCODE, Roadmap Epigenomics).

### 4. Variant-to-Gene Mapping

- **Proximity**: Nearest gene is often NOT causal — use with caution.
- **eQTL colocalization**: Use coloc or eCAVIAR to test if GWAS and eQTL signals share a causal variant. Colocalization PP4 > 0.8 is strong evidence.
- **Chromatin interaction**: Use Hi-C, promoter-capture Hi-C, or Activity-by-Contact (ABC) models to link enhancer variants to target genes.
- **Coding variants**: Missense/nonsense variants in LD with lead SNP are high-confidence causal candidates.
- **Rare variant burden**: Check if gene-level rare variant tests (SKAT, burden tests) support the same gene.

### 5. Candidate Gene Prioritization

Rank genes using multiple evidence streams:
- eQTL colocalization strength (PP4).
- Protein-protein interaction network proximity to known disease genes.
- Gene expression in disease-relevant tissues (GTEx).
- Mouse knockout phenotypes (MGI, IMPC).
- Constraint metrics (pLI, LOEUF) — highly constrained genes are more likely disease-relevant.
- Druggability assessment (existing compounds targeting the protein).

### 6. Pathway and Network Analysis

- Run gene-set enrichment (GSEA, MAGMA) on full GWAS summary statistics.
- Test enrichment of candidate genes in KEGG, Reactome, GO pathways.
- Build protein interaction subnetworks around candidates.
- Identify hub genes and pathway convergence points.

### 7. Cross-Trait Analysis

- Check LD Score regression for genetic correlations with related traits.
- Perform Mendelian randomization if causal inference is needed.
- Look for pleiotropic loci affecting multiple phenotypes.

## Common Pitfalls

- **Winner's curse**: Effect sizes from discovery GWAS are inflated; always check replication.
- **Population specificity**: LD patterns differ across ancestries — fine-mapping in one population may not transfer.
- **Nearest gene fallacy**: The closest gene to a lead SNP is causal only ~30-50% of the time.
- **Ignoring non-coding mechanisms**: Most GWAS hits are in regulatory regions, not coding sequences.
- **Single-SNP focus**: Consider haplotype effects and epistasis for complex loci.
- **Confounding by LD**: Two signals in LD may represent one causal variant — always perform conditional analysis.

## Key Databases

- **GWAS Catalog** (EBI): curated association database.
- **Open Targets Genetics**: variant-to-gene mapping, colocalization.
- **GTEx**: tissue-specific gene expression and eQTLs.
- **gnomAD**: allele frequencies and constraint metrics.
- **LDlink**: LD calculations across populations.

## Output Expectations

For each GWAS locus, the agent should produce:
1. Lead SNP with effect size and p-value.
2. Fine-mapped credible set (if data available).
3. Top 3 candidate genes with evidence summary.
4. Relevant pathways and biological processes.
5. Confidence score reflecting evidence strength.
