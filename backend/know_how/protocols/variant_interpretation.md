# Variant Interpretation Protocol

## Purpose

Guide agents through systematic variant classification following ACMG/AMP guidelines, pathogenicity scoring, and evidence assembly for clinical-grade KG edges.

## Step-by-Step Protocol

### 1. Variant Identification

- Standardize variant notation using HGVS nomenclature (e.g., NM_000546.6:c.743G>A, p.Arg248Gln).
- Record: chromosome, position (GRCh38), reference/alternate allele, gene, transcript.
- Determine variant type: SNV, indel, structural variant, copy number variant.
- Check variant is correctly mapped — liftover errors between genome builds are common.

### 2. Population Frequency Assessment

- Query gnomAD for allele frequency across populations.
- ACMG criteria:
  - **BA1 (Benign standalone)**: AF > 5% in any population — variant is almost certainly benign.
  - **BS1 (Benign strong)**: AF greater than expected for the disorder. Threshold depends on disease prevalence and penetrance.
  - **PM2 (Pathogenic moderate)**: Absent from gnomAD or extremely rare (AF < 0.01% for dominant, < 0.1% for recessive).
- Always check population-specific frequencies — a variant rare in Europeans may be common in Africans.
- For recessive diseases, consider carrier frequency and Hardy-Weinberg expectations.

### 3. Computational Predictions

- Run multiple in silico predictors:
  - **CADD** (Combined Annotation Dependent Depletion): Score > 20 = top 1% most deleterious; > 30 = top 0.1%.
  - **REVEL**: Ensemble method for missense variants. Score > 0.5 suggests pathogenic; > 0.75 strong evidence.
  - **SpliceAI**: Splice-altering predictions. Delta score > 0.5 indicates likely splice effect.
  - **SIFT**: < 0.05 = damaging (based on conservation).
  - **PolyPhen-2**: Probably damaging, possibly damaging, benign.
  - **AlphaMissense**: Deep learning pathogenicity predictor. Score > 0.564 = likely pathogenic.
- ACMG criteria:
  - **PP3**: Multiple predictors agree on damaging impact.
  - **BP4**: Multiple predictors agree on benign impact.
- Never rely on a single predictor — concordance across methods strengthens evidence.

### 4. Functional Evidence

- Search literature and ClinVar for functional studies:
  - **PS3 (Pathogenic strong)**: Well-established functional assay shows damaging effect (e.g., enzymatic activity loss, protein mislocalization, impaired binding).
  - **BS3 (Benign strong)**: Functional assay shows no damaging effect.
- Evaluate assay quality:
  - Validated assays with known pathogenic/benign controls are strongest.
  - Overexpression assays may not reflect physiological conditions.
  - Patient-derived cells are more reliable than cell-line models.
- ESM-2 fitness predictions via Yami can supplement but not replace wet-lab functional data.

### 5. Segregation and De Novo Evidence

- **PS2 (Pathogenic strong)**: Confirmed de novo variant in a patient with no family history (maternity/paternity confirmed).
- **PP1 (Pathogenic supporting)**: Co-segregation with disease in affected family members.
  - Strength increases with number of meioses (3+ meioses = strong evidence).
- **BS4 (Benign strong)**: Lack of segregation with disease in affected family members.
- De novo variants in highly constrained genes (pLI > 0.9) are more likely pathogenic.

### 6. ACMG Classification

Apply the 28 ACMG/AMP criteria and combine:

| Classification | Criteria Required |
|---|---|
| **Pathogenic** | 1 Very Strong + 1 Strong; OR 2 Strong; OR 1 Strong + 3 Supporting |
| **Likely Pathogenic** | 1 Very Strong + 1 Moderate; OR 1 Strong + 1-2 Moderate; OR 1 Strong + 2 Supporting |
| **Uncertain Significance (VUS)** | Does not meet criteria for either pathogenic or benign |
| **Likely Benign** | 1 Strong + 1 Supporting benign |
| **Benign** | 1 Standalone benign (BA1); OR 2 Strong benign |

- **PVS1 (Pathogenic very strong)**: Null variant (nonsense, frameshift, canonical splice site) in a gene where loss-of-function is a known mechanism. Apply PVS1 decision tree for strength modifiers.
- Always document which specific criteria are applied and at what strength.

### 7. ClinVar and Database Cross-Reference

- Check ClinVar for existing classifications.
- Assess ClinVar entry quality:
  - **4-star review**: Expert panel consensus — treat as authoritative.
  - **3-star**: Reviewed by expert panel.
  - **1-2 star**: Single submitter or conflicting — use with caution.
  - **0-star**: No assertion criteria — do not use as evidence.
- Check LOVD, HGMD (if available), and gene-specific databases.
- Note classification conflicts — these are genuine uncertainty.

### 8. Building KG Edges

- Create VARIANT_OF edge from variant node to gene node.
- Create ASSOCIATED_WITH or CAUSES edge from variant to disease.
- Set confidence based on ACMG classification:
  - Pathogenic: 0.9-1.0
  - Likely Pathogenic: 0.7-0.9
  - VUS: 0.3-0.5
  - Likely Benign: 0.1-0.3
  - Benign: 0.0-0.1
- Include all ACMG criteria codes in evidence sources.

## Common Pitfalls

- **Over-reliance on in silico tools**: Computational predictions alone are supporting evidence (PP3/BP4), never sufficient for classification.
- **Ignoring population-specific data**: A variant classified as VUS in European studies may have clear frequency data in African or Asian populations.
- **PVS1 misapplication**: Not all truncating variants are pathogenic — check if the gene is haploinsufficient, if the variant escapes NMD, and if the truncated domain is critical.
- **Functional study quality**: Not all published functional data meets ACMG PS3/BS3 criteria — assess assay validity.
- **ClinVar version drift**: ClinVar is updated monthly; a VUS may be reclassified.
- **Mosaic variants**: Somatic variants may be present at low allele fractions and missed by standard calling.

## Key Databases

- **ClinVar**: Variant-disease assertions with review status.
- **gnomAD**: Population allele frequencies.
- **CADD/REVEL/SpliceAI**: In silico pathogenicity predictors.
- **InterVar**: Automated ACMG classification tool.
- **ClinGen**: Curated gene-disease validity and variant curation guidelines.
- **LOVD**: Gene-specific variant databases.

## Output Expectations

For each variant interpretation, produce:
1. Standardized HGVS notation (coding and protein level).
2. Population frequency summary across populations.
3. In silico prediction concordance.
4. ACMG criteria applied with evidence codes.
5. Final classification: Pathogenic / Likely Pathogenic / VUS / Likely Benign / Benign.
6. Confidence score and key evidence sources.
