# Drug Repurposing Protocol

## Purpose

Guide agents through systematic identification of existing drugs that could be repurposed for new indications, from target identification through binding analysis to ADMET evaluation.

## Step-by-Step Protocol

### 1. Target Identification and Validation

- Start from disease biology: identify dysregulated genes/proteins from GWAS, expression studies, or KG edges.
- Validate target druggability using:
  - Protein family (kinases, GPCRs, ion channels are highly druggable).
  - Structural data availability (crystal structures, cryo-EM, AlphaFold models).
  - Genetic evidence strength (Mendelian disease genes, causal GWAS hits with MR support).
- Check Open Targets for target-disease association scores.
- Prioritize targets with genetic support AND existing chemical matter.

### 2. Compound Identification

- Search ChEMBL for compounds with activity against validated targets:
  - Filter by activity type (IC50, Ki, Kd, EC50).
  - Apply potency cutoffs (IC50 < 1 uM for initial hits, < 100 nM for leads).
  - Check selectivity panels for off-target activity.
- Search DrugBank for approved drugs targeting the protein family.
- Identify drugs approved for other indications that hit the same target.
- Check patent status and clinical development stage.

### 3. Binding Analysis

- Retrieve or predict binding mode:
  - Known co-crystal structures from PDB.
  - Docking studies using predicted binding site (from AlphaFold or homology model).
  - Binding site comparison across protein family members.
- Evaluate binding characteristics:
  - Competitive vs. allosteric mechanism.
  - Reversible vs. covalent binding.
  - Selectivity profile across family members.
- Flag compounds with promiscuous binding (PAINS filters).

### 4. ADMET Assessment

- **Absorption**: Lipinski's Rule of 5, molecular weight, LogP, polar surface area (PSA).
- **Distribution**: Plasma protein binding, volume of distribution, blood-brain barrier penetration (if CNS indication).
- **Metabolism**: CYP450 inhibition/induction risk (especially CYP3A4, CYP2D6), metabolic stability.
- **Excretion**: Renal vs. hepatic clearance, half-life (target: suitable for once/twice daily dosing).
- **Toxicity**: hERG channel inhibition (cardiac risk), Ames test prediction, hepatotoxicity signals.
- Use known clinical safety data from the drug's approved indication as a starting advantage.

### 5. Clinical Evidence Mining

- Search ClinicalTrials.gov for:
  - Trials of the drug in the new indication (may already be in progress).
  - Trials of related drugs in the same target class.
  - Failed trials — analyze failure reasons (efficacy vs. safety vs. enrollment).
- Mine adverse event databases (FAERS, VigiBase) for unexpected beneficial effects.
- Search literature for case reports of off-label efficacy.

### 6. Mechanistic Validation

- Map drug-target-disease pathway using KG edges.
- Verify the drug's mechanism addresses the disease biology:
  - Does inhibiting the target reverse the disease phenotype?
  - Are there compensatory pathways that could reduce efficacy?
  - Is the target expression/activity altered in the disease state?
- Check for pharmacogenomic interactions affecting the target population.

### 7. Repurposing Prioritization

Score candidates on:
- **Genetic evidence** (0-3): GWAS support, Mendelian genetics, MR evidence.
- **Chemical evidence** (0-3): Potency, selectivity, structural data.
- **Clinical evidence** (0-3): Safety record, off-label reports, related trials.
- **Practical feasibility** (0-3): Patent status, formulation, dose prediction.
- Total score > 8/12 warrants further investigation.

## Common Pitfalls

- **Target vs. disease complexity**: A single target is rarely sufficient for complex diseases; consider combination strategies.
- **Dose translation**: Approved dose for one indication may not achieve therapeutic exposure for the new indication.
- **Indication bias**: Drugs failed for efficacy in one disease may still work in a related but distinct condition.
- **Ignoring pharmacokinetics**: A compound active in vitro may not achieve tissue-level concentrations in vivo.
- **Patent traps**: Repurposing approved drugs may face method-of-use patent barriers.
- **Safety recontextualization**: Acceptable side effects for cancer may be unacceptable for chronic conditions.

## Key Databases

- **ChEMBL**: Bioactivity data for compounds and targets.
- **DrugBank**: Drug-target interactions, approved drugs.
- **Open Targets**: Target-disease associations with genetic evidence.
- **ClinicalTrials.gov**: Trial data for clinical evidence mining.
- **PubChem**: Chemical properties and bioassay data.

## Output Expectations

For each repurposing candidate, produce:
1. Drug name, current indication, and mechanism of action.
2. Target-disease evidence summary with confidence.
3. Binding analysis (potency, selectivity, mechanism).
4. ADMET profile highlights and concerns.
5. Clinical feasibility assessment with prioritization score.
