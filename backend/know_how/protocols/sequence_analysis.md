# Sequence Analysis Protocol

## Purpose

Guide agents through biological sequence analysis including alignment, motif finding, homology searches, and primer design for molecular biology investigations.

## Step-by-Step Protocol

### 1. Sequence Retrieval and Validation

- Retrieve sequences from authoritative databases:
  - **Nucleotide**: NCBI Nucleotide/RefSeq (NM_, NR_ accessions), Ensembl.
  - **Protein**: UniProt (Swiss-Prot preferred), NCBI Protein (NP_ accessions).
  - **Genome**: NCBI Assembly, UCSC Genome Browser (GRCh38/hg38 for human).
- Validate sequence:
  - Check for ambiguous bases (N) or non-standard amino acids (X, B, Z).
  - Verify length matches expected gene/protein size.
  - Confirm correct transcript variant/isoform.
- Record metadata: organism, gene name, accession, sequence version, length.

### 2. Sequence Alignment

- **Pairwise alignment**:
  - Global (Needleman-Wunsch): When sequences are similar length and expected to align end-to-end.
  - Local (Smith-Waterman): When searching for conserved regions within divergent sequences.
  - BLAST (heuristic): For database searches — fast but may miss distant homologs.
- **Multiple sequence alignment (MSA)**:
  - MAFFT: Fast, accurate for up to ~10,000 sequences. Use L-INS-i for <200 sequences (highest accuracy).
  - Clustal Omega: Good for large alignments (>10,000 sequences).
  - MUSCLE: Fast, good for protein alignments.
- **Alignment quality assessment**:
  - Check for misaligned regions (especially around indels).
  - Remove poorly aligned columns using trimAl or Gblocks.
  - Verify alignment makes biological sense (conserved domains should align).

### 3. Homology Search (BLAST)

- Choose appropriate BLAST program:
  - **blastn**: Nucleotide vs. nucleotide. Use megablast for near-identical sequences, dc-megablast for cross-species.
  - **blastp**: Protein vs. protein. Standard for identifying homologs.
  - **blastx**: Translated nucleotide vs. protein. For finding protein homologs of a DNA sequence.
  - **tblastn**: Protein vs. translated nucleotide. For finding genomic regions encoding a protein.
  - **PSI-BLAST**: Iterated search for remote homologs. More sensitive than blastp for divergent sequences.
- Interpret results:
  - **E-value**: Expected number of hits by chance. E < 1e-5 for reliable homology; E < 1e-50 for close homologs.
  - **Percent identity**: >30% for proteins suggests homology; <25% is the "twilight zone" — interpret cautiously.
  - **Query coverage**: Should be >70% for confident homology assignment. Low coverage may indicate domain-level similarity only.
  - **Bit score**: Higher is better; independent of database size (unlike E-value).

### 4. Motif and Domain Finding

- **Sequence motifs**:
  - MEME Suite: De novo motif discovery in a set of related sequences.
  - FIMO: Scan sequences for known motif occurrences.
  - JASPAR/TRANSFAC: Transcription factor binding site motifs.
- **Protein domains**:
  - InterProScan: Comprehensive domain search across Pfam, PROSITE, SMART, CDD.
  - CD-Search (NCBI): Quick conserved domain identification.
  - Pfam: Hidden Markov Model-based domain classification.
- **Signal sequences**:
  - SignalP: Signal peptide prediction.
  - TMHMM/DeepTMHMM: Transmembrane helix prediction.
  - NLS prediction: Nuclear localization signals.
- Record all identified motifs/domains with positions, E-values, and functional annotations.

### 5. Phylogenetic Analysis

- Build phylogenetic tree from MSA:
  - **Neighbor-joining**: Fast, approximate. Good for initial exploration.
  - **Maximum likelihood (ML)**: More accurate. Use RAxML or IQ-TREE.
  - **Bayesian**: Most statistically rigorous. Use MrBayes. Computationally expensive.
- Model selection: Use ModelTest-NG or IQ-TREE model finder for appropriate substitution model.
- Assess tree support:
  - Bootstrap values > 70 indicate reliable branches.
  - Bayesian posterior probability > 0.95 indicates strong support.
- Root the tree using an outgroup or midpoint rooting.

### 6. Primer Design

- Use Primer3 or NCBI Primer-BLAST:
  - **Tm (melting temperature)**: 55-65C, difference between forward/reverse < 3C.
  - **Length**: 18-25 nucleotides.
  - **GC content**: 40-60%.
  - **3' end**: Should end in G or C (GC clamp) but avoid runs of >3 G/C.
  - **Self-complementarity**: Minimize to prevent hairpins and primer-dimers.
- **Specificity**: BLAST primers against the genome to verify unique binding.
- **For RT-qPCR**: Design primers spanning exon-exon junctions to avoid genomic DNA amplification.
- **Product size**: 100-300 bp for qPCR; 200-1000 bp for cloning; adjust for application.
- Record: primer sequence, Tm, GC%, product size, specificity check results.

### 7. Codon Usage and Optimization

- For recombinant protein expression:
  - Check codon adaptation index (CAI) for the target organism.
  - Identify rare codons that may cause translational pausing.
  - Optimize codons for expression host while maintaining mRNA structure.
- Avoid: cryptic splice sites, internal poly-A signals, restriction sites needed for cloning.

## Common Pitfalls

- **BLAST E-value depends on database size**: The same alignment has different E-values in nr vs. Swiss-Prot. Compare bit scores for cross-database comparison.
- **Percent identity is misleading for short alignments**: 90% identity over 20 amino acids is not significant.
- **MSA quality affects everything downstream**: Bad alignment produces wrong phylogenies, wrong conservation scores, and wrong motifs.
- **Composition bias**: Low-complexity regions (e.g., polyQ tracts) produce spurious BLAST hits. Use SEG/DUST filtering.
- **Primer specificity in gene families**: Highly similar paralogs may produce off-target amplification — always check in silico PCR.
- **Ignoring strand orientation**: Confusing sense/antisense strand is a common error in primer design and BLAST interpretation.

## Key Tools

- **BLAST/NCBI**: Sequence similarity search.
- **MAFFT/Clustal Omega**: Multiple sequence alignment.
- **MEME Suite**: Motif discovery and scanning.
- **InterProScan**: Domain and motif annotation.
- **Primer3/Primer-BLAST**: Primer design and specificity check.
- **IQ-TREE/RAxML**: Phylogenetic tree construction.

## Output Expectations

For each sequence analysis task, produce:
1. Sequence metadata (accession, organism, length, type).
2. Top homologs with E-values and percent identity.
3. Domain architecture with boundaries and functions.
4. Conservation-notable residues or regions.
5. Primers (if applicable) with Tm, specificity, and product size.
