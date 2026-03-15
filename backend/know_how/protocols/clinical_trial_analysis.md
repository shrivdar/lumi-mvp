# Clinical Trial Analysis Protocol

## Purpose

Guide agents through interpreting clinical trial design, endpoints, statistical results, and failure analysis for evidence-based knowledge graph construction.

## Step-by-Step Protocol

### 1. Trial Identification and Classification

- Search ClinicalTrials.gov by condition, intervention, or NCT number.
- Classify by phase:
  - **Phase I**: Safety, dose-finding. N = 20-80. Outcomes: MTD, DLT, PK parameters.
  - **Phase II**: Efficacy signal, dose optimization. N = 100-300. Outcomes: ORR, PFS, biomarker response.
  - **Phase III**: Confirmatory efficacy. N = 300-3000+. Outcomes: OS, PFS, primary endpoint.
  - **Phase IV**: Post-marketing surveillance. Focus: rare adverse events, long-term safety.
- Note trial status: Recruiting, Active, Completed, Terminated, Withdrawn, Suspended.
- Terminated/Withdrawn trials are high-value — analyze reasons.

### 2. Study Design Assessment

- **Randomization**: Randomized controlled trials (RCTs) are gold standard. Non-randomized trials are hypothesis-generating only.
- **Blinding**: Double-blind > single-blind > open-label. Open-label introduces bias in subjective endpoints.
- **Control arm**: Placebo-controlled vs. active comparator vs. standard of care. Active comparator trials are more clinically relevant.
- **Crossover design**: Allows within-patient comparison but complicates long-term endpoint analysis.
- **Adaptive design**: Bayesian or response-adaptive randomization — note modified statistical framework.
- **Sample size justification**: Check power calculation. Underpowered studies produce unreliable results.

### 3. Endpoint Interpretation

- **Primary endpoint**: The single outcome the trial is powered to detect. All other analyses are exploratory.
- **Hard endpoints**: Overall survival (OS), major adverse cardiac events (MACE) — gold standard but require long follow-up.
- **Surrogate endpoints**:
  - FDA-accepted surrogates: PFS in oncology, HbA1c in diabetes, viral load in HIV.
  - Validate surrogate-outcome relationship before trusting.
  - Response rate (ORR, CR) is a surrogate — may not correlate with survival.
- **Patient-reported outcomes (PROs)**: Quality of life, symptom scores. Important for benefit-risk assessment.
- **Composite endpoints**: Combine multiple events (e.g., death + MI + stroke). Check individual component contributions — a composite driven by the least serious component is misleading.

### 4. Statistical Analysis

- **Significance threshold**: p < 0.05 for primary endpoint (two-sided). Some trials use p < 0.025 (one-sided) — note this.
- **Hazard ratio (HR)**: For time-to-event endpoints. HR < 1 favors treatment. HR confidence interval crossing 1.0 = not significant.
- **Confidence intervals**: More informative than p-values alone. Narrow CI around a clinically meaningful effect = strong evidence.
- **Number needed to treat (NNT)**: NNT < 10 is generally clinically meaningful for serious outcomes.
- **Subgroup analyses**: Pre-specified subgroups are informative; post-hoc subgroups are hypothesis-generating only. Beware of multiplicity.
- **Intention-to-treat (ITT) vs. per-protocol**: ITT is primary (conservative). Per-protocol may inflate efficacy if dropouts are non-random.
- **Multiplicity adjustment**: If multiple primary endpoints or interim analyses, check for alpha spending (Lan-DeMets, O'Brien-Fleming).

### 5. Safety Assessment

- Distinguish adverse events (AEs) from adverse drug reactions (ADRs) — AEs include all events, including background rate.
- Focus on:
  - **Grade 3-4 AEs**: Severe/life-threatening events.
  - **Serious adverse events (SAEs)**: Hospitalization, disability, death.
  - **AEs leading to discontinuation**: Indicates tolerability.
  - **Treatment-related mortality**: Deaths attributed to the intervention.
- Compare AE rates between treatment and control arms — absolute difference matters more than relative.
- Check for dose-dependent safety signals.

### 6. Failure Analysis

For failed/terminated trials:
- **Lack of efficacy**: Was the biological hypothesis wrong, or was the trial poorly designed?
  - Check dose/exposure — was the target adequately engaged?
  - Check patient selection — was the biomarker-defined population appropriate?
  - Check endpoint — was the endpoint sensitive enough to detect benefit?
- **Safety**: What specific toxicities led to termination?
  - On-target toxicity (expected from mechanism) vs. off-target toxicity (unexpected).
  - Reversible vs. irreversible adverse effects.
- **Operational**: Slow enrollment, protocol amendments, COVID impact — not biological failure.
- Record failure mode as KG evidence with EVIDENCE_AGAINST relation.

### 7. Evidence Synthesis

- For drugs with multiple trials: perform informal meta-analysis.
  - Consistent results across trials = high confidence.
  - Contradictory results = examine differences in design, population, dose.
- Map trial results to KG edges:
  - Positive trial: TREATS edge with high confidence.
  - Negative trial: EVIDENCE_AGAINST edge.
  - Mixed results: Moderate confidence with uncertainty annotation.

## Common Pitfalls

- **Phase II optimism**: Phase II response rates are almost always higher than Phase III — do not extrapolate.
- **Surrogate endpoint trap**: Improvements in surrogates do not always translate to clinical benefit (e.g., antiarrhythmics reducing PVCs but increasing mortality).
- **Ignoring control arm**: A treatment with 40% response rate seems impressive until the control arm has 35%.
- **Cherry-picking subgroups**: Subgroup "benefit" in an overall negative trial is almost never real. Require replication.
- **Confusing statistical and clinical significance**: A p-value of 0.001 with HR 0.98 is statistically significant but clinically meaningless.
- **Survivor bias in safety data**: Patients who drop out early due to toxicity are underrepresented in long-term safety analyses.

## Key Databases

- **ClinicalTrials.gov**: Trial registry and results database.
- **FDA Drugs@FDA**: Approval packages with full review documents.
- **EMA EPAR**: European assessment reports.
- **PubMed**: Published trial results and meta-analyses.
- **FAERS**: FDA Adverse Event Reporting System.

## Output Expectations

For each trial analysis, produce:
1. Trial summary (NCT, phase, design, N, status).
2. Primary endpoint result with effect size and CI.
3. Key safety findings.
4. Design quality assessment (blinding, randomization, power).
5. Failure analysis (for negative/terminated trials).
6. Confidence score for treatment efficacy claim.
