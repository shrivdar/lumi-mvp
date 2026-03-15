# Protein Function Analysis Protocol

## Purpose

Guide agents through protein functional characterization including domain analysis, evolutionary conservation, structure prediction, and functional annotation for knowledge graph construction.

## Step-by-Step Protocol

### 1. Sequence Retrieval and Annotation

- Retrieve protein sequence from UniProt using accession or gene name.
- Record: length, molecular weight, subcellular localization, post-translational modifications.
- Check UniProt annotation quality: reviewed (Swiss-Prot) entries are curated; unreviewed (TrEMBL) are automated.
- Identify signal peptides, transmembrane domains, and transit peptides.
- Note isoforms — alternative splicing may produce functionally distinct variants.

### 2. Domain Analysis

- Run InterProScan or query InterPro for domain architecture.
- For each domain:
  - Record domain family (Pfam), boundaries, and E-value.
  - Look up domain function in Pfam/InterPro annotations.
  - Check if the domain is a known drug target or interaction interface.
- Identify disordered regions using IUPred or MobiDB — intrinsically disordered regions often mediate protein-protein interactions.
- Map domains to known 3D structures in PDB.

### 3. Conservation Analysis

- Retrieve orthologs using OrthoDB, Ensembl Compara, or NCBI HomoloGene.
- Perform multiple sequence alignment (MSA) with MAFFT or Clustal Omega.
- Calculate per-residue conservation scores (ConSurf, Shannon entropy).
- Highly conserved residues are functionally important:
  - Active site residues: nearly invariant across species.
  - Structural core: conserved hydrophobic residues maintaining fold.
  - Binding interfaces: moderately conserved, may show co-evolution.
- Map disease-associated variants to conservation landscape — pathogenic variants cluster in conserved regions.

### 4. Structure Prediction and Analysis

- Check PDB for experimental structures (X-ray, cryo-EM, NMR).
- If no experimental structure: use AlphaFold DB (pre-computed) or run ESMFold via Yami.
- Evaluate prediction quality:
  - **pLDDT > 90**: Very high confidence — reliable for atomic-level analysis.
  - **pLDDT 70-90**: High confidence — fold is correct, side-chain positions approximate.
  - **pLDDT 50-70**: Low confidence — may be disordered or flexible.
  - **pLDDT < 50**: Very low confidence — likely disordered; do not interpret structurally.
- Identify:
  - Active/binding sites from structure-function annotations.
  - Protein-protein interaction interfaces (large, flat, hydrophobic surfaces).
  - Allosteric sites (cavities distal to active site).
  - Druggable pockets using fpocket or SiteMap.

### 5. Functional Classification

- Assign Gene Ontology (GO) terms:
  - Molecular Function: what the protein does (e.g., kinase activity, DNA binding).
  - Biological Process: what pathway/process it participates in.
  - Cellular Component: where it acts.
- Check enzyme classification (EC number) if applicable.
- Map to Reactome reactions for mechanistic context.
- Identify protein families and superfamilies for evolutionary context.

### 6. Interaction Network

- Query STRING for high-confidence (>0.7) protein-protein interactions.
- Distinguish interaction types: physical binding, co-expression, genetic interaction, pathway co-membership.
- Identify protein complexes the protein participates in (CORUM database).
- Map post-translational modification networks: phosphorylation, ubiquitination, SUMOylation.
- Record interaction partners as KG edges with INTERACTS_WITH or BINDS_TO relations.

### 7. Variant Impact Assessment

- Map known variants from ClinVar, gnomAD, and disease-specific databases.
- Predict variant impact using:
  - **SIFT**: Sequence-based; predicts tolerated vs. damaging.
  - **PolyPhen-2**: Structure-aware; predicts benign vs. probably damaging.
  - **CADD**: Integrative score combining multiple annotations.
  - **ESM-1v / ESM-2**: Deep learning fitness predictions via Yami.
- Variants in conserved domains, active sites, or interaction interfaces are higher risk.
- Create VARIANT_OF or MUTANT_OF edges for disease-associated variants.

## Common Pitfalls

- **Sequence vs. structure vs. function**: High sequence similarity does not guarantee identical function — verify with functional annotations.
- **AlphaFold over-confidence**: pLDDT is per-residue, not per-domain; a high average can mask poorly predicted regions.
- **Moonlighting proteins**: Some proteins have entirely different functions in different cellular compartments.
- **Disorder ≠ unimportant**: Intrinsically disordered regions are critical for signaling and regulation.
- **Interaction database incompleteness**: STRING covers ~60% of human protein interactions; absence of an interaction is not proof of non-interaction.
- **Ignoring isoforms**: The canonical UniProt sequence may not be the disease-relevant isoform.

## Key Databases

- **UniProt**: Protein sequences, annotations, and cross-references.
- **PDB**: Experimental 3D structures.
- **AlphaFold DB**: Predicted structures for most human proteins.
- **InterPro/Pfam**: Domain classification and functional annotation.
- **STRING**: Protein-protein interaction network.
- **ConSurf**: Evolutionary conservation analysis.

## Output Expectations

For each protein analysis, produce:
1. Domain architecture diagram (domain names, boundaries, confidence).
2. Key functional sites (active site, binding sites, PTM sites).
3. Conservation summary with disease-variant mapping.
4. Structure quality assessment (experimental vs. predicted, pLDDT).
5. Top interaction partners with evidence confidence.
6. Functional classification (GO terms, EC number, pathway membership).
