# Pathway Analysis Protocol

## Purpose

Guide agents through biological pathway enrichment analysis, network topology assessment, and cross-pathway interaction mapping for knowledge graph construction.

## Step-by-Step Protocol

### 1. Gene Set Preparation

- Collect the gene set from the research context (GWAS hits, differentially expressed genes, KG neighbors).
- Use standard gene identifiers (Entrez IDs, HGNC symbols) — convert if needed using MyGene.
- Remove duplicates and verify gene symbols are current (not retired/merged).
- For GWAS: use MAGMA or H-MAGMA for gene-level association scores before enrichment.

### 2. Enrichment Analysis

- **Over-Representation Analysis (ORA)**:
  - Apply Fisher's exact test or hypergeometric test.
  - Background set must match the experiment (all genes tested, not all genes in genome).
  - Multiple testing correction: Benjamini-Hochberg FDR < 0.05.
  - Report enrichment fold-change alongside p-value — statistically significant but low fold-change is biologically weak.
- **Gene Set Enrichment Analysis (GSEA)**:
  - Preferred when you have a ranked gene list (by p-value, fold-change, or association score).
  - Does not require arbitrary significance cutoff.
  - Report normalized enrichment score (NES), p-value, and FDR.
  - Leading edge genes are the key drivers — add these to the KG.
- **Pathway databases to query**:
  - KEGG: metabolic and signaling pathways.
  - Reactome: detailed mechanistic pathways.
  - GO Biological Process: broader functional categories.
  - WikiPathways: community-curated disease pathways.

### 3. Network Topology Analysis

For pathway networks built in the KG:
- **Degree centrality**: Nodes with many connections are pathway hubs. High-degree nodes are essential but may be too generic.
- **Betweenness centrality**: Nodes bridging separate pathway modules. These are potential therapeutic targets — disrupting them affects multiple processes.
- **Closeness centrality**: Nodes that can quickly propagate signals. Relevant for signaling cascades.
- **Clustering coefficient**: High clustering indicates tightly regulated modules. Low clustering in a disease context may indicate network disruption.
- **Shortest path analysis**: Distance between disease genes and drug targets through the pathway network.

### 4. Cross-Pathway Interactions

- Identify shared components between enriched pathways (pathway cross-talk).
- Map upstream regulators that control multiple enriched pathways.
- Look for feedback loops: pathway A activates pathway B, which inhibits pathway A.
- Identify convergence points where multiple dysregulated pathways intersect — these are high-value therapeutic targets.
- Record cross-talk edges in the KG with UPSTREAM_OF/DOWNSTREAM_OF relations.

### 5. Pathway Contextualization

- Check tissue-specific pathway activity using GTEx or Human Protein Atlas.
- Verify pathway relevance to the disease context — a pathway enriched in your gene set may not be active in the disease tissue.
- Compare pathway activity between disease and healthy states using expression data.
- Consider pathway dynamics: some pathways are active only during specific cellular states (e.g., cell cycle, stress response).

### 6. Building KG Pathway Nodes

For each significant pathway:
- Create PATHWAY node with KEGG/Reactome ID, name, and description.
- Add MEMBER_OF edges from gene/protein nodes to pathway node.
- Add PARTICIPATES_IN edges for catalytic/regulatory roles.
- Add UPSTREAM_OF/DOWNSTREAM_OF edges for pathway cross-talk.
- Set confidence based on enrichment strength (FDR, fold-change).

## Common Pitfalls

- **Annotation bias**: Well-studied genes appear in more pathways; enrichment may reflect study bias, not biology.
- **Redundant pathways**: Many pathway databases have overlapping entries (e.g., "MAPK signaling" appears in KEGG, Reactome, and GO with different gene sets). Deduplicate or use pathway clustering.
- **Background set mismatch**: Using "all human genes" as background when only 20,000 were tested inflates significance.
- **Ignoring pathway size**: Very large pathways (>500 genes) are enriched by chance more often. Very small pathways (<5 genes) have unstable statistics.
- **Static view**: Pathway databases represent canonical pathways; disease states may rewire pathway topology.
- **Over-interpretation of topology**: Centrality metrics depend on network completeness — missing interactions distort results.

## Key Databases

- **KEGG**: Metabolic and signaling pathway maps.
- **Reactome**: Detailed mechanistic pathway models.
- **Gene Ontology (GO)**: Biological process, molecular function, cellular component.
- **STRING**: Protein-protein interaction network with confidence scores.
- **WikiPathways**: Community-curated disease-specific pathways.

## Output Expectations

For each pathway analysis, produce:
1. Ranked list of enriched pathways with FDR and fold-change.
2. Leading edge / driver genes for top pathways.
3. Network topology metrics for KG pathway subgraph.
4. Cross-pathway interaction map.
5. Therapeutic target candidates from network analysis.
