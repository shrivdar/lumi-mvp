# Gene-Disease Association Protocol

## Purpose

Guide agents through evaluating and scoring gene-disease associations, distinguishing Mendelian from complex disease genetics, and building an evidence hierarchy for KG edges.

## Step-by-Step Protocol

### 1. Evidence Hierarchy

Rank evidence sources from strongest to weakest:

1. **Mendelian genetics** (highest): Pathogenic variants in the gene cause the disease in families. Sources: ClinVar, OMIM, ClinGen.
2. **GWAS with fine-mapping**: Genome-wide significant association with credible set containing coding or regulatory variants affecting the gene. Source: GWAS Catalog, Open Targets.
3. **Mendelian randomization**: Causal inference from genetic instruments. Requires valid instrument assumptions (relevance, independence, exclusion restriction).
4. **Rare variant burden tests**: Gene-level association from exome/genome sequencing (SKAT, burden test p < 2.5e-6 for exome-wide significance).
5. **Expression studies**: Differential expression in disease tissue (with replication). Consider direction and effect size.
6. **Animal models**: Gene knockout/knockin recapitulates disease phenotype. Cross-species conservation increases confidence.
7. **Pathway membership**: Gene participates in a disease-relevant pathway. Weak evidence alone — requires additional support.
8. **Literature co-occurrence**: Gene and disease mentioned together in publications. Lowest tier — may reflect publication bias.

### 2. Mendelian Disease Assessment

For suspected Mendelian associations:
- Check OMIM for known gene-disease relationships and inheritance pattern.
- Verify in ClinGen Gene-Disease Validity framework (Definitive, Strong, Moderate, Limited, Disputed, Refuted).
- For novel associations, apply ClinGen criteria:
  - Genetic evidence: segregation in families, de novo occurrences, case-control enrichment.
  - Experimental evidence: functional studies, animal models, rescue experiments.
- Record inheritance pattern: AD, AR, XL, XD, mitochondrial.
- Note penetrance and expressivity — incomplete penetrance reduces confidence.

### 3. Complex Disease Assessment

For common/complex disease associations:
- Start with GWAS evidence — check GWAS Catalog and Open Targets.
- Apply variant-to-gene mapping (see GWAS Analysis Protocol).
- Assess effect size and population attributable risk.
- Check for gene-environment interactions that modify risk.
- Look for convergence: multiple independent signals pointing to the same gene or pathway.
- Distinguish genetic correlation from causation — use MR where possible.

### 4. Cross-Referencing Evidence Streams

- **Convergence scoring**: Count how many independent evidence streams support the association.
  - 1 stream: Low confidence (0.2-0.4).
  - 2 streams: Moderate confidence (0.4-0.6).
  - 3+ streams: High confidence (0.6-0.9).
- **Contradiction detection**: Look for evidence AGAINST the association:
  - Large GWAS with no signal at the locus.
  - Animal model knockout with no phenotype.
  - Failed clinical trials targeting the gene product.
- Weight recent, large, well-designed studies over older, smaller ones.

### 5. Building KG Edges

For each gene-disease association, create edges with:
- **Relation type**: ASSOCIATED_WITH (genetic), CAUSES (Mendelian), RISK_OF (complex).
- **Confidence**: Based on evidence hierarchy and convergence scoring.
- **Evidence sources**: List every supporting study with PMID, effect size, sample size.
- **Directionality**: Specify gain-of-function vs. loss-of-function where known.
- **Population specificity**: Note if the association is population-specific.

### 6. Distinguishing Correlation from Causation

- Genetic association alone does not prove causation.
- Causal evidence requires:
  - Mendelian randomization with valid instruments.
  - Gene knockout/knockin phenocopy.
  - Drug targeting the gene product treats the disease.
  - Segregation in families (Mendelian).
- Label edges appropriately: CORRELATES_WITH vs. CAUSES.

## Common Pitfalls

- **Gene vs. locus**: A GWAS hit at a locus does not implicate the nearest gene — always do variant-to-gene mapping.
- **Publication bias**: Positive associations are published more often; absence of evidence in literature is not evidence of absence.
- **Pleiotropy confusion**: A gene associated with a comorbid trait may appear disease-associated through confounding.
- **Ancestry bias**: Most GWAS are in European populations — allele frequencies and LD patterns differ across ancestries.
- **Ignoring negative evidence**: Failed replication attempts are as informative as positive results.
- **ClinVar over-reliance**: ClinVar contains submitter assertions of variable quality — check review status and evidence.

## Key Databases

- **OMIM**: Mendelian disease-gene relationships.
- **ClinGen**: Gene-disease validity curation.
- **ClinVar**: Variant-disease assertions.
- **GWAS Catalog**: Complex disease associations.
- **Open Targets**: Integrated gene-disease scoring.
- **DisGeNET**: Text-mined and curated gene-disease associations.

## Output Expectations

For each gene-disease association, produce:
1. Gene symbol, disease name, and MIM/EFO identifiers.
2. Evidence tier classification (Mendelian vs. complex).
3. Evidence summary with convergence count.
4. Confidence score with justification.
5. Direction of effect (gain/loss of function, risk allele).
6. Contradicting evidence, if any.
