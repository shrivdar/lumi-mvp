"""Tool Catalog — 150+ tool definitions for YOHAS agent assignment.

Three source types:
1. NATIVE — implemented in integrations/ (PubMed, ChEMBL, etc.)
2. MCP — external MCP servers discovered via MCPClient
3. CONTAINER — sandboxed compute containers (RDKit, BLAST, etc.)

The catalog is the single source of truth for tool discovery.
SwarmComposer queries this catalog when assigning tools to agents.
"""

from __future__ import annotations

from core.models import ToolRegistryEntry, ToolSourceType

# ---------------------------------------------------------------------------
# Categories used for LLM-driven tool selection
# ---------------------------------------------------------------------------

CATEGORY_LITERATURE = "literature_search"
CATEGORY_PROTEIN = "protein_analysis"
CATEGORY_GENOMICS = "genomics"
CATEGORY_PATHWAY = "pathway_analysis"
CATEGORY_DRUG = "drug_discovery"
CATEGORY_CLINICAL = "clinical_data"
CATEGORY_STRUCTURE = "structural_biology"
CATEGORY_EXPRESSION = "gene_expression"
CATEGORY_VARIANT = "variant_analysis"
CATEGORY_ONTOLOGY = "ontology_annotation"
CATEGORY_CHEMISTRY = "chemistry"
CATEGORY_IMAGING = "imaging"
CATEGORY_NETWORK = "network_analysis"
CATEGORY_EPIGENETICS = "epigenetics"
CATEGORY_PHYLOGENETICS = "phylogenetics"
CATEGORY_METABOLOMICS = "metabolomics"
CATEGORY_PROTEOMICS = "proteomics"
CATEGORY_SAFETY = "safety_toxicology"
CATEGORY_REGULATORY = "regulatory_data"
CATEGORY_WEB = "web_search"
CATEGORY_COMPUTE = "computation"
CATEGORY_COMMUNICATION = "communication"


def build_catalog() -> list[ToolRegistryEntry]:
    """Return the full tool catalog (150+ entries)."""
    entries: list[ToolRegistryEntry] = []
    entries.extend(_native_tools())
    entries.extend(_mcp_literature_tools())
    entries.extend(_mcp_protein_tools())
    entries.extend(_mcp_genomics_tools())
    entries.extend(_mcp_pathway_tools())
    entries.extend(_mcp_drug_tools())
    entries.extend(_mcp_clinical_tools())
    entries.extend(_mcp_structure_tools())
    entries.extend(_mcp_expression_tools())
    entries.extend(_mcp_variant_tools())
    entries.extend(_mcp_ontology_tools())
    entries.extend(_mcp_chemistry_tools())
    entries.extend(_mcp_network_tools())
    entries.extend(_mcp_epigenetics_tools())
    entries.extend(_mcp_metabolomics_tools())
    entries.extend(_mcp_proteomics_tools())
    entries.extend(_mcp_safety_tools())
    entries.extend(_mcp_regulatory_tools())
    entries.extend(_mcp_web_tools())
    entries.extend(_container_tools())
    return entries


def _e(
    name: str,
    description: str,
    category: str,
    source_type: ToolSourceType = ToolSourceType.MCP,
    *,
    enabled: bool = True,
    capabilities: list[str] | None = None,
) -> ToolRegistryEntry:
    return ToolRegistryEntry(
        name=name,
        description=description,
        source_type=source_type,
        category=category,
        enabled=enabled,
        capabilities=capabilities or [],
    )


# ===================================================================
# NATIVE TOOLS (10) — implemented in integrations/
# ===================================================================

def _native_tools() -> list[ToolRegistryEntry]:
    return [
        _e("pubmed", "Search PubMed for biomedical literature by keyword, MeSH term, or PMID. Returns titles, abstracts, authors, citation counts.", CATEGORY_LITERATURE, ToolSourceType.NATIVE, capabilities=["search", "fetch_abstract", "mesh_lookup", "citation_count"]),
        _e("semantic_scholar", "Search Semantic Scholar for papers, citations, references, influential citations, and author profiles.", CATEGORY_LITERATURE, ToolSourceType.NATIVE, capabilities=["search", "paper_details", "citations", "references", "author_search"]),
        _e("uniprot", "Search UniProt for protein sequences, annotations, domains, GO terms, disease associations, and cross-references.", CATEGORY_PROTEIN, ToolSourceType.NATIVE, capabilities=["search", "fetch_entry", "sequence", "features", "cross_references"]),
        _e("esm", "Run ESM-2 protein language model for embeddings, fitness prediction, and structure prediction via ESMFold.", CATEGORY_STRUCTURE, ToolSourceType.NATIVE, capabilities=["embeddings", "fitness", "fold", "contact_map"]),
        _e("kegg", "Query KEGG for pathways, compounds, reactions, diseases, genes, and pathway maps.", CATEGORY_PATHWAY, ToolSourceType.NATIVE, capabilities=["search", "pathway_details", "pathway_genes", "compound_details", "reaction_details"]),
        _e("reactome", "Search Reactome for curated biological pathways, reactions, and interactors in human and model organisms.", CATEGORY_PATHWAY, ToolSourceType.NATIVE, capabilities=["search", "pathway_details", "pathway_participants", "pathway_diagram"]),
        _e("mygene", "Query MyGene.info for gene annotations, orthologs, GO terms, pathway memberships, and expression data.", CATEGORY_GENOMICS, ToolSourceType.NATIVE, capabilities=["search", "gene_info", "orthologs", "go_terms", "pathway_membership"]),
        _e("chembl", "Search ChEMBL for bioactive compounds, targets, assays, mechanisms of action, and drug indications.", CATEGORY_DRUG, ToolSourceType.NATIVE, capabilities=["search", "compound_details", "target_details", "assay_data", "mechanism_of_action"]),
        _e("clinicaltrials", "Search ClinicalTrials.gov for clinical trials by condition, intervention, sponsor, phase, and status.", CATEGORY_CLINICAL, ToolSourceType.NATIVE, capabilities=["search", "trial_details", "enrollment", "outcomes", "eligibility"]),
        _e("slack", "Send messages and files to Slack channels for human-in-the-loop communication during research.", CATEGORY_COMMUNICATION, ToolSourceType.NATIVE, capabilities=["send_message", "upload_file", "create_thread"]),
        # --- New native tools (tool expansion) ---
        _e("opentargets", "Query Open Targets for target-disease associations, evidence, and prioritization scores.", CATEGORY_DRUG, ToolSourceType.NATIVE, capabilities=["target_disease", "evidence_search", "drug_mechanism", "tractability"]),
        _e("clinvar", "Search ClinVar for clinical significance of human genomic variants.", CATEGORY_VARIANT, ToolSourceType.NATIVE, capabilities=["variant_search", "clinical_significance", "condition_links", "submission_data"]),
        _e("gtex", "Query GTEx for gene expression across human tissues with eQTL data.", CATEGORY_EXPRESSION, ToolSourceType.NATIVE, capabilities=["tissue_expression", "eqtl", "splicing", "sample_info"]),
        _e("gnomad", "Query gnomAD for population allele frequencies and variant constraint metrics.", CATEGORY_VARIANT, ToolSourceType.NATIVE, capabilities=["variant_lookup", "allele_frequency", "constraint_metrics", "structural_variants"]),
        _e("hpo", "Search Human Phenotype Ontology for standardized phenotype terms and gene-phenotype links.", CATEGORY_ONTOLOGY, ToolSourceType.NATIVE, capabilities=["term_search", "gene_to_phenotype", "disease_to_phenotype"]),
        _e("omim", "Search Online Mendelian Inheritance in Man for gene-disease relationships and phenotype descriptions.", CATEGORY_GENOMICS, ToolSourceType.NATIVE, capabilities=["search", "gene_phenotype", "allelic_variants"]),
        _e("biogrid", "Search BioGRID for curated protein-protein, genetic, and chemical interactions.", CATEGORY_NETWORK, ToolSourceType.NATIVE, capabilities=["search", "interactions", "chemical_associations"]),
        _e("depmap", "Query DepMap for cancer dependency data — gene essentiality across cancer cell lines.", CATEGORY_EXPRESSION, ToolSourceType.NATIVE, capabilities=["gene_dependency", "cell_line_info", "crispr_data", "drug_sensitivity"]),
        _e("cellxgene", "Search CZ CELLxGENE for single-cell RNA-seq datasets and cell type annotations.", CATEGORY_EXPRESSION, ToolSourceType.NATIVE, capabilities=["study_search", "cell_types", "gene_expression", "cluster_data"]),
        _e("string_db", "Query STRING for known and predicted protein-protein interactions with confidence scores.", CATEGORY_NETWORK, ToolSourceType.NATIVE, capabilities=["interaction_partners", "network", "enrichment", "confidence_scores"]),
    ]


# ===================================================================
# MCP TOOLS — external MCP servers (~100 tools)
# ===================================================================

# --- Literature & Knowledge ---

def _mcp_literature_tools() -> list[ToolRegistryEntry]:
    return [
        _e("arxiv", "Search arXiv preprints by keyword, author, or category. Returns papers in physics, CS, biology, and quantitative biology.", CATEGORY_LITERATURE, capabilities=["search", "fetch_paper", "fetch_pdf"]),
        _e("biorxiv", "Search bioRxiv and medRxiv preprints for the latest unpublished biomedical research.", CATEGORY_LITERATURE, capabilities=["search", "fetch_paper", "doi_lookup"]),
        _e("europe_pmc", "Search Europe PMC for open-access biomedical literature, annotations, and full-text mining.", CATEGORY_LITERATURE, capabilities=["search", "full_text", "annotations", "citations"]),
        _e("google_scholar", "Search Google Scholar for academic papers, citation metrics, and related articles.", CATEGORY_LITERATURE, capabilities=["search", "citation_count", "related_articles"]),
        _e("crossref", "Look up DOIs, metadata, and citation links via the Crossref API.", CATEGORY_LITERATURE, capabilities=["doi_lookup", "metadata", "citation_links", "funder_search"]),
        _e("openalex", "Query OpenAlex for open scholarly metadata — works, authors, institutions, venues, and concepts.", CATEGORY_LITERATURE, capabilities=["search_works", "search_authors", "concepts", "institutions"]),
        _e("unpaywall", "Find open-access versions of papers using the Unpaywall API.", CATEGORY_LITERATURE, capabilities=["oa_lookup", "pdf_url"]),
        _e("scite", "Search scite.ai for smart citations — see how papers cite each other (supporting, contradicting, mentioning).", CATEGORY_LITERATURE, capabilities=["smart_citations", "search", "citation_context"]),
        _e("dimensions", "Search Dimensions for publications, grants, patents, clinical trials, and policy documents.", CATEGORY_LITERATURE, capabilities=["search", "grants", "patents", "datasets"]),
        _e("lens_org", "Search Lens.org for patents, scholarly works, and biological sequences across global patent databases.", CATEGORY_LITERATURE, capabilities=["patent_search", "scholarly_search", "sequence_search"]),
    ]


def _mcp_protein_tools() -> list[ToolRegistryEntry]:
    return [
        _e("pdb", "Search the Protein Data Bank for experimentally determined 3D structures of proteins, nucleic acids, and complexes.", CATEGORY_STRUCTURE, capabilities=["search", "fetch_structure", "ligand_info", "resolution_filter"]),
        _e("alphafold_db", "Retrieve AlphaFold predicted structures from the EBI AlphaFold Protein Structure Database.", CATEGORY_STRUCTURE, capabilities=["fetch_prediction", "plddt_scores", "pae_matrix"]),
        _e("interpro", "Classify proteins into families, predict domains and functional sites via InterPro.", CATEGORY_PROTEIN, capabilities=["search", "domain_annotation", "family_classification", "go_annotation"]),
        _e("pfam", "Search Pfam for protein family HMM profiles, domain architectures, and clan memberships.", CATEGORY_PROTEIN, capabilities=["search", "domain_search", "clan_info", "alignment"]),
        # string_db → now NATIVE (see _native_tools)
        # biogrid → now NATIVE (see _native_tools)
        _e("intact", "Query IntAct for molecular interaction data with full experimental evidence details.", CATEGORY_NETWORK, capabilities=["search", "interactions", "evidence_details"]),
        _e("disprot", "Search DisProt for intrinsically disordered protein regions and their functional annotations.", CATEGORY_PROTEIN, capabilities=["search", "disorder_regions", "function_annotation"]),
        _e("proteomes_uniprot", "Search UniProt Proteomes for complete proteome data of organisms.", CATEGORY_PROTEOMICS, capabilities=["proteome_search", "organism_proteome", "reference_proteome"]),
        _e("elm", "Search the Eukaryotic Linear Motif resource for short linear motifs in protein sequences.", CATEGORY_PROTEIN, capabilities=["motif_search", "instance_search", "class_info"]),
    ]


def _mcp_genomics_tools() -> list[ToolRegistryEntry]:
    return [
        _e("ensembl", "Query Ensembl for gene models, transcripts, regulatory features, and comparative genomics.", CATEGORY_GENOMICS, capabilities=["gene_lookup", "transcript_info", "regulatory_features", "comparative_genomics", "variant_effect"]),
        _e("ncbi_gene", "Search NCBI Gene for gene summaries, RefSeq, nomenclature, and gene-disease links.", CATEGORY_GENOMICS, capabilities=["search", "gene_info", "refseq", "gene_disease"]),
        _e("ucsc_genome", "Query the UCSC Genome Browser for genomic tracks, annotations, and conservation scores.", CATEGORY_GENOMICS, capabilities=["track_data", "annotations", "conservation", "blat_search"]),
        _e("gencode", "Access GENCODE gene annotations — comprehensive reference gene sets for human and mouse.", CATEGORY_GENOMICS, capabilities=["gene_annotations", "transcript_models", "pseudogenes"]),
        _e("hgnc", "Search HUGO Gene Nomenclature Committee for official gene symbols and nomenclature.", CATEGORY_GENOMICS, capabilities=["symbol_lookup", "search", "previous_symbols"]),
        _e("ncbi_blast", "Run BLAST sequence similarity searches against NCBI nucleotide and protein databases.", CATEGORY_GENOMICS, capabilities=["blastn", "blastp", "blastx", "tblastn"]),
        _e("decipher", "Query DECIPHER for genomic variants associated with developmental disorders.", CATEGORY_VARIANT, capabilities=["variant_search", "patient_data", "syndrome_search"]),
        # omim → now NATIVE (see _native_tools)
    ]


def _mcp_pathway_tools() -> list[ToolRegistryEntry]:
    return [
        _e("wikipathways", "Search WikiPathways for community-curated biological pathways in GPML format.", CATEGORY_PATHWAY, capabilities=["search", "pathway_details", "pathway_genes", "pathway_image"]),
        _e("signor", "Query SIGNOR for causal signaling interactions between biological entities.", CATEGORY_PATHWAY, capabilities=["search", "signaling_interactions", "pathway_data"]),
        _e("panther", "Search PANTHER for protein classification, pathways, and gene list analysis.", CATEGORY_PATHWAY, capabilities=["gene_list_analysis", "pathway_search", "protein_classification"]),
        _e("biocyc", "Query BioCyc for metabolic pathways, reactions, and compounds across organisms.", CATEGORY_PATHWAY, capabilities=["pathway_search", "reaction_search", "compound_search", "organism_db"]),
        _e("pathway_commons", "Search Pathway Commons for integrated biological pathway data from multiple databases.", CATEGORY_PATHWAY, capabilities=["search", "pathway_data", "interactions", "neighborhood"]),
        _e("msigdb", "Query the Molecular Signatures Database for gene sets used in GSEA and pathway analysis.", CATEGORY_PATHWAY, capabilities=["gene_set_search", "hallmark_sets", "curated_sets", "go_sets"]),
        _e("david", "Run DAVID functional annotation and enrichment analysis on gene lists.", CATEGORY_ONTOLOGY, capabilities=["functional_annotation", "enrichment", "clustering"]),
        _e("enrichr", "Run Enrichr gene set enrichment analysis across multiple gene set libraries.", CATEGORY_ONTOLOGY, capabilities=["enrichment_analysis", "library_search", "combined_score"]),
    ]


def _mcp_drug_tools() -> list[ToolRegistryEntry]:
    return [
        _e("drugbank", "Search DrugBank for comprehensive drug data — targets, pharmacology, interactions, and ADMET.", CATEGORY_DRUG, capabilities=["search", "drug_details", "targets", "interactions", "pharmacology"]),
        _e("pubchem", "Search PubChem for chemical structures, bioassays, compound properties, and biological activities.", CATEGORY_CHEMISTRY, capabilities=["compound_search", "bioassay_search", "structure_search", "property_lookup"]),
        _e("zinc", "Search ZINC database for commercially available compounds for virtual screening.", CATEGORY_DRUG, capabilities=["compound_search", "vendor_info", "property_filter"]),
        _e("bindingdb", "Search BindingDB for measured binding affinities between proteins and drug-like molecules.", CATEGORY_DRUG, capabilities=["search", "binding_data", "target_search", "ki_kd_ic50"]),
        _e("dgidb", "Query DGIdb for drug-gene interactions from multiple curated sources.", CATEGORY_DRUG, capabilities=["interaction_search", "gene_search", "drug_search", "interaction_types"]),
        # opentargets → now NATIVE (see _native_tools)
        _e("pharmgkb", "Search PharmGKB for pharmacogenomics data — drug-gene associations and clinical annotations.", CATEGORY_DRUG, capabilities=["search", "drug_label", "clinical_annotation", "variant_annotation"]),
        _e("ttd", "Search Therapeutic Target Database for drug targets, diseases, and pathway information.", CATEGORY_DRUG, capabilities=["target_search", "drug_search", "disease_search"]),
        _e("stitch", "Query STITCH for known and predicted chemical-protein interactions with confidence scores.", CATEGORY_DRUG, capabilities=["interaction_search", "network", "confidence_scores"]),
        _e("chembl_webresource", "Access ChEMBL web resource client for programmatic compound/target/assay queries.", CATEGORY_DRUG, capabilities=["compound_query", "target_query", "assay_query", "activity_query"]),
        _e("guide_to_pharmacology", "Search IUPHAR/BPS Guide to Pharmacology for receptor/ligand data and drug targets.", CATEGORY_DRUG, capabilities=["target_search", "ligand_search", "interaction_data"]),
    ]


def _mcp_clinical_tools() -> list[ToolRegistryEntry]:
    return [
        _e("aact", "Query AACT (Aggregate Analysis of ClinicalTrials.gov) for structured trial data analysis.", CATEGORY_CLINICAL, capabilities=["trial_search", "outcome_data", "eligibility", "facility_search"]),
        _e("fda_drug_labels", "Search FDA drug labels (DailyMed/OpenFDA) for prescribing information and safety data.", CATEGORY_CLINICAL, capabilities=["label_search", "adverse_events", "drug_interactions"]),
        _e("openfda", "Query openFDA for drug adverse events, recalls, labeling, and device reports.", CATEGORY_CLINICAL, capabilities=["adverse_events", "drug_recall", "device_events", "label_search"]),
        _e("who_ictrp", "Search WHO ICTRP for international clinical trial registrations across all registries.", CATEGORY_CLINICAL, capabilities=["search", "registry_lookup", "trial_record"]),
        _e("ctgov_v2", "Query ClinicalTrials.gov V2 API for enhanced trial metadata and result summaries.", CATEGORY_CLINICAL, capabilities=["search", "study_fields", "results", "statistics"]),
        _e("mesh", "Search MeSH (Medical Subject Headings) vocabulary for controlled biomedical terminology.", CATEGORY_ONTOLOGY, capabilities=["term_search", "tree_navigation", "descriptor_details"]),
        _e("snomed_ct", "Look up SNOMED CT clinical terms, hierarchies, and mappings.", CATEGORY_ONTOLOGY, capabilities=["concept_search", "hierarchy", "mappings"]),
        _e("icd", "Search ICD-10/ICD-11 classification for diseases and health conditions.", CATEGORY_ONTOLOGY, capabilities=["code_search", "hierarchy", "description"]),
    ]


def _mcp_structure_tools() -> list[ToolRegistryEntry]:
    return [
        _e("rcsb_pdb", "Advanced queries to RCSB PDB — search by sequence, structure, chemical component, and annotations.", CATEGORY_STRUCTURE, capabilities=["advanced_search", "sequence_search", "structure_download", "assembly_info"]),
        _e("modbase", "Retrieve comparative protein structure models from ModBase.", CATEGORY_STRUCTURE, capabilities=["model_search", "model_details", "quality_scores"]),
        _e("swiss_model", "Search SWISS-MODEL repository for homology models and run template-based modeling.", CATEGORY_STRUCTURE, capabilities=["model_search", "template_search", "qmean_scores"]),
        _e("pdbe", "Query PDBe for enhanced PDB entries with quality metrics, validation, and functional annotations.", CATEGORY_STRUCTURE, capabilities=["entry_details", "quality_metrics", "ligand_info", "topology"]),
        _e("uniprot_3d", "Search UniProt 3D cross-references for structural coverage of protein sequences.", CATEGORY_STRUCTURE, capabilities=["structure_coverage", "domain_3d", "binding_sites"]),
    ]


def _mcp_expression_tools() -> list[ToolRegistryEntry]:
    return [
        _e("geo", "Search NCBI GEO for gene expression datasets, series, samples, and platforms.", CATEGORY_EXPRESSION, capabilities=["search", "dataset_details", "sample_data", "platform_info"]),
        _e("expression_atlas", "Query Expression Atlas (EMBL-EBI) for baseline and differential gene expression.", CATEGORY_EXPRESSION, capabilities=["search", "baseline_expression", "differential_expression", "experiment_details"]),
        _e("human_protein_atlas", "Search Human Protein Atlas for protein expression in tissues, cells, and subcellular locations.", CATEGORY_EXPRESSION, capabilities=["tissue_expression", "cell_expression", "subcellular", "pathology"]),
        # gtex → now NATIVE (see _native_tools)
        # depmap → now NATIVE (see _native_tools)
        _e("ccle", "Search Cancer Cell Line Encyclopedia for genomic/transcriptomic data across 1000+ cell lines.", CATEGORY_EXPRESSION, capabilities=["expression_data", "mutation_data", "drug_response", "cell_line_info"]),
        _e("tcga", "Query TCGA for multi-omics cancer genomics data — mutations, expression, methylation, CNV.", CATEGORY_EXPRESSION, capabilities=["mutation_data", "expression_data", "methylation", "cnv", "clinical_data"]),
        _e("single_cell_portal", "Search Broad Single Cell Portal for scRNA-seq datasets and cell type annotations.", CATEGORY_EXPRESSION, capabilities=["study_search", "cell_types", "gene_expression", "cluster_data"]),
        # cellxgene → now NATIVE (see _native_tools)
    ]


def _mcp_variant_tools() -> list[ToolRegistryEntry]:
    return [
        # clinvar → now NATIVE (see _native_tools)
        _e("dbsnp", "Look up dbSNP for reference SNP clusters, allele frequencies, and variant annotations.", CATEGORY_VARIANT, capabilities=["rs_lookup", "allele_frequency", "functional_annotation"]),
        # gnomad → now NATIVE (see _native_tools)
        _e("cosmic", "Search COSMIC for somatic mutations in cancer — mutation frequency, gene census, and signatures.", CATEGORY_VARIANT, capabilities=["mutation_search", "gene_census", "mutation_signatures", "cancer_type"]),
        _e("cbioportal", "Query cBioPortal for cancer genomics data — mutations, CNA, expression across studies.", CATEGORY_VARIANT, capabilities=["study_search", "mutation_data", "cna_data", "clinical_data"]),
        _e("lovd", "Search LOVD (Leiden Open Variation Database) for gene-specific variant databases.", CATEGORY_VARIANT, capabilities=["variant_search", "gene_database", "phenotype_data"]),
        _e("mutalyzer", "Use Mutalyzer for HGVS nomenclature checking and variant description normalization.", CATEGORY_VARIANT, capabilities=["name_check", "position_converter", "syntax_check"]),
        _e("vep", "Run Ensembl Variant Effect Predictor to annotate variant consequences.", CATEGORY_VARIANT, capabilities=["variant_annotation", "consequence_prediction", "regulatory_annotation"]),
        _e("cadd", "Query CADD scores for predicting deleteriousness of SNVs and indels.", CATEGORY_VARIANT, capabilities=["score_lookup", "annotation", "phred_score"]),
    ]


def _mcp_ontology_tools() -> list[ToolRegistryEntry]:
    return [
        _e("gene_ontology", "Search Gene Ontology for biological process, molecular function, and cellular component terms.", CATEGORY_ONTOLOGY, capabilities=["term_search", "annotation_search", "enrichment", "slim"]),
        # hpo → now NATIVE (see _native_tools)
        _e("mondo", "Search Mondo Disease Ontology for harmonized disease terms across databases.", CATEGORY_ONTOLOGY, capabilities=["disease_search", "hierarchy", "mappings", "gene_associations"]),
        _e("ols", "Search EBI Ontology Lookup Service for terms across 200+ biomedical ontologies.", CATEGORY_ONTOLOGY, capabilities=["search", "term_details", "hierarchy", "mappings"]),
        _e("bioportal", "Search BioPortal for ontology terms, mappings, and annotations across 900+ ontologies.", CATEGORY_ONTOLOGY, capabilities=["search", "class_details", "mappings", "recommender"]),
        _e("disgenet", "Query DisGeNET for gene-disease associations with evidence scores and source annotations.", CATEGORY_ONTOLOGY, capabilities=["gene_disease", "variant_disease", "disease_search", "evidence_scores"]),
    ]


def _mcp_chemistry_tools() -> list[ToolRegistryEntry]:
    return [
        _e("chemspider", "Search ChemSpider for chemical structures, identifiers, and data sources.", CATEGORY_CHEMISTRY, capabilities=["search", "structure_lookup", "identifiers"]),
        _e("chebi", "Search ChEBI for chemical entities of biological interest — names, structures, ontology.", CATEGORY_CHEMISTRY, capabilities=["search", "entity_details", "ontology", "relationships"]),
        _e("surechembl", "Search SureChEMBL for chemical structures extracted from patents.", CATEGORY_CHEMISTRY, capabilities=["patent_search", "compound_search", "structure_extraction"]),
        _e("metabolights", "Search MetaboLights for metabolomics experiments and reference metabolite spectra.", CATEGORY_METABOLOMICS, capabilities=["study_search", "compound_search", "spectra"]),
    ]


def _mcp_network_tools() -> list[ToolRegistryEntry]:
    return [
        _e("ndex", "Search NDEx for biological network models — signaling, regulatory, and metabolic networks.", CATEGORY_NETWORK, capabilities=["network_search", "network_details", "neighborhood_query"]),
        _e("signaling_gateway", "Query the Signaling Gateway for curated signal transduction pathway data.", CATEGORY_NETWORK, capabilities=["pathway_search", "molecule_search", "interaction_data"]),
        _e("mentha", "Query mentha for protein-protein interaction data aggregated from multiple databases.", CATEGORY_NETWORK, capabilities=["interaction_search", "network", "reliability_score"]),
        _e("reactome_fi", "Query Reactome Functional Interaction network for predicted and known gene interactions.", CATEGORY_NETWORK, capabilities=["interaction_search", "network_clustering", "pathway_enrichment"]),
    ]


def _mcp_epigenetics_tools() -> list[ToolRegistryEntry]:
    return [
        _e("encode", "Query ENCODE for epigenomic data — ChIP-seq, ATAC-seq, DNase-seq, and histone modifications.", CATEGORY_EPIGENETICS, capabilities=["experiment_search", "file_search", "annotation_data", "biosample_info"]),
        _e("roadmap_epigenomics", "Search Roadmap Epigenomics for chromatin state maps and DNA methylation across tissues.", CATEGORY_EPIGENETICS, capabilities=["chromatin_states", "methylation_data", "tissue_data"]),
        _e("cistrome", "Query Cistrome for ChIP-seq and ATAC-seq datasets and transcription factor binding data.", CATEGORY_EPIGENETICS, capabilities=["tf_binding", "chromatin_accessibility", "peak_data"]),
        _e("ewas_catalog", "Search EWAS Catalog for epigenome-wide association study results and CpG annotations.", CATEGORY_EPIGENETICS, capabilities=["ewas_results", "cpg_annotation", "trait_search"]),
    ]


def _mcp_metabolomics_tools() -> list[ToolRegistryEntry]:
    return [
        _e("hmdb", "Search Human Metabolome Database for metabolite data — concentrations, spectra, pathways, and diseases.", CATEGORY_METABOLOMICS, capabilities=["metabolite_search", "spectra", "pathway_association", "disease_association"]),
        _e("lipidmaps", "Search LIPID MAPS for lipid classification, structures, and mass spectrometry data.", CATEGORY_METABOLOMICS, capabilities=["lipid_search", "classification", "ms_data"]),
        _e("metlin", "Search METLIN for metabolite identification using MS/MS spectra.", CATEGORY_METABOLOMICS, capabilities=["spectral_search", "metabolite_id", "fragmentation"]),
    ]


def _mcp_proteomics_tools() -> list[ToolRegistryEntry]:
    return [
        _e("pride", "Search PRIDE proteomics data repository for mass spectrometry datasets.", CATEGORY_PROTEOMICS, capabilities=["project_search", "peptide_search", "spectrum_data"]),
        _e("phosphosite", "Search PhosphoSitePlus for post-translational modification data — phosphorylation, ubiquitination, etc.", CATEGORY_PROTEOMICS, capabilities=["ptm_search", "site_data", "kinase_substrate"]),
        _e("peptide_atlas", "Search PeptideAtlas for observed peptides from mass spectrometry experiments.", CATEGORY_PROTEOMICS, capabilities=["peptide_search", "protein_coverage", "sample_info"]),
        _e("massive", "Search MassIVE for shared mass spectrometry datasets and spectral libraries.", CATEGORY_PROTEOMICS, capabilities=["dataset_search", "spectral_library", "reanalysis"]),
    ]


def _mcp_safety_tools() -> list[ToolRegistryEntry]:
    return [
        _e("tox21", "Search Tox21 for high-throughput toxicology screening data across 10,000+ chemicals.", CATEGORY_SAFETY, capabilities=["chemical_search", "assay_data", "activity_summary"]),
        _e("ctd", "Query Comparative Toxicogenomics Database for chemical-gene-disease relationships.", CATEGORY_SAFETY, capabilities=["chemical_gene", "chemical_disease", "gene_disease", "pathway_association"]),
        _e("toxcast", "Query ToxCast for in vitro bioactivity data from EPA chemical screening.", CATEGORY_SAFETY, capabilities=["chemical_search", "assay_data", "bioactivity"]),
        _e("faers", "Search FDA Adverse Event Reporting System for drug safety signal data.", CATEGORY_SAFETY, capabilities=["adverse_event_search", "drug_search", "signal_detection"]),
        _e("sider", "Search SIDER for drug side effect data mined from package inserts.", CATEGORY_SAFETY, capabilities=["side_effect_search", "drug_search", "frequency_data"]),
    ]


def _mcp_regulatory_tools() -> list[ToolRegistryEntry]:
    return [
        _e("fda_orange_book", "Search FDA Orange Book for approved drug products, patents, and exclusivity.", CATEGORY_REGULATORY, capabilities=["product_search", "patent_search", "exclusivity"]),
        _e("ema_medicines", "Search EMA for European medicines — marketing authorizations, EPARs, and safety data.", CATEGORY_REGULATORY, capabilities=["medicine_search", "epar_search", "safety_referrals"]),
        _e("dailymed", "Search DailyMed for FDA-approved drug labeling and medication guides.", CATEGORY_REGULATORY, capabilities=["label_search", "spl_data", "drug_interactions"]),
    ]


def _mcp_web_tools() -> list[ToolRegistryEntry]:
    return [
        _e("tavily", "AI-optimized web search for research queries — returns structured, relevant results.", CATEGORY_WEB, capabilities=["search", "extract", "summarize"]),
        _e("brave_search", "Search the web via Brave Search API for academic and general results.", CATEGORY_WEB, capabilities=["web_search", "news_search"]),
        _e("exa", "Search with Exa (neural search) for semantically relevant web pages and papers.", CATEGORY_WEB, capabilities=["search", "find_similar", "contents"]),
        _e("perplexity", "Query Perplexity AI for summarized answers with citations.", CATEGORY_WEB, capabilities=["ask", "search"]),
        _e("wikipedia", "Search and retrieve Wikipedia articles for background context.", CATEGORY_WEB, capabilities=["search", "page_content", "summary"]),
    ]


# ===================================================================
# CONTAINER TOOLS (~40) — sandboxed compute
# ===================================================================

def _container_tools() -> list[ToolRegistryEntry]:
    return [
        # Cheminformatics
        _e("rdkit_compute", "Run RDKit cheminformatics — molecular descriptors, fingerprints, substructure search, SMILES processing.", CATEGORY_CHEMISTRY, ToolSourceType.CONTAINER, capabilities=["descriptors", "fingerprints", "substructure_search", "similarity", "smiles_to_mol"]),
        _e("openbabel", "Convert between chemical file formats and compute molecular properties using Open Babel.", CATEGORY_CHEMISTRY, ToolSourceType.CONTAINER, capabilities=["format_conversion", "property_calculation", "fingerprint"]),

        # Structural biology
        _e("pymol_render", "Generate protein structure visualizations and surface renderings using PyMOL.", CATEGORY_STRUCTURE, ToolSourceType.CONTAINER, capabilities=["structure_render", "surface_render", "alignment_viz", "electrostatics"]),
        _e("chimerax_render", "Render molecular structures with UCSF ChimeraX for publication-quality images.", CATEGORY_STRUCTURE, ToolSourceType.CONTAINER, capabilities=["structure_render", "density_map", "morph"]),
        _e("fpocket", "Run fpocket for protein pocket detection and druggability assessment.", CATEGORY_STRUCTURE, ToolSourceType.CONTAINER, capabilities=["pocket_detection", "druggability", "descriptor_calculation"]),
        _e("p2rank", "Run P2Rank for machine-learning based protein binding site prediction.", CATEGORY_STRUCTURE, ToolSourceType.CONTAINER, capabilities=["binding_site_prediction", "ligandability"]),

        # Molecular dynamics / docking
        _e("autodock_vina", "Run AutoDock Vina for molecular docking — predict binding poses and affinities.", CATEGORY_DRUG, ToolSourceType.CONTAINER, capabilities=["dock", "score", "pose_prediction"]),
        _e("gnina", "Run GNINA deep-learning molecular docking for improved binding affinity prediction.", CATEGORY_DRUG, ToolSourceType.CONTAINER, capabilities=["dock", "score", "rescore"]),
        _e("diffdock", "Run DiffDock generative docking — diffusion-based blind molecular docking.", CATEGORY_DRUG, ToolSourceType.CONTAINER, capabilities=["blind_dock", "pose_generation", "confidence_score"]),

        # ADMET / pharmacokinetics
        _e("admet_ai", "Predict ADMET properties using ML models — absorption, distribution, metabolism, excretion, toxicity.", CATEGORY_SAFETY, ToolSourceType.CONTAINER, capabilities=["absorption", "distribution", "metabolism", "excretion", "toxicity"]),
        _e("pkcsm", "Predict pharmacokinetic and toxicity properties using pkCSM models.", CATEGORY_SAFETY, ToolSourceType.CONTAINER, capabilities=["pharmacokinetics", "toxicity_prediction", "drug_likeness"]),
        _e("swissadme", "Compute molecular properties, drug-likeness, and pharmacokinetics via SwissADME models.", CATEGORY_DRUG, ToolSourceType.CONTAINER, capabilities=["drug_likeness", "lipinski", "boiled_egg", "bioavailability"]),

        # Sequence analysis
        _e("blast_local", "Run local BLAST for sequence similarity searches against custom databases.", CATEGORY_GENOMICS, ToolSourceType.CONTAINER, capabilities=["blastn", "blastp", "makeblastdb"]),
        _e("hmmer", "Run HMMER for sequence search against profile HMMs — protein family identification.", CATEGORY_PROTEIN, ToolSourceType.CONTAINER, capabilities=["hmmsearch", "hmmscan", "jackhmmer"]),
        _e("clustal_omega", "Run Clustal Omega for multiple sequence alignment.", CATEGORY_GENOMICS, ToolSourceType.CONTAINER, capabilities=["msa", "profile_alignment", "guide_tree"]),
        _e("muscle", "Run MUSCLE for fast and accurate multiple sequence alignment.", CATEGORY_GENOMICS, ToolSourceType.CONTAINER, capabilities=["msa", "profile_alignment"]),
        _e("mafft", "Run MAFFT for multiple sequence alignment with various algorithms.", CATEGORY_GENOMICS, ToolSourceType.CONTAINER, capabilities=["msa", "progressive", "iterative"]),

        # Variant effect prediction
        _e("polyphen2", "Run PolyPhen-2 to predict impact of amino acid substitutions on protein function.", CATEGORY_VARIANT, ToolSourceType.CONTAINER, capabilities=["variant_prediction", "structural_analysis", "conservation"]),
        _e("sift", "Run SIFT to predict whether amino acid substitution affects protein function.", CATEGORY_VARIANT, ToolSourceType.CONTAINER, capabilities=["variant_prediction", "conservation_score"]),
        _e("deepsea", "Run DeepSEA for predicting chromatin effects of noncoding variants.", CATEGORY_VARIANT, ToolSourceType.CONTAINER, capabilities=["noncoding_prediction", "chromatin_effect", "tf_binding"]),
        _e("spliceai", "Run SpliceAI for predicting splicing effects of genetic variants.", CATEGORY_VARIANT, ToolSourceType.CONTAINER, capabilities=["splice_prediction", "delta_score"]),

        # Statistics / analysis
        _e("scipy_stats", "Run statistical tests and analysis — t-test, ANOVA, Fisher's exact, enrichment, regression.", CATEGORY_COMPUTE, ToolSourceType.CONTAINER, capabilities=["statistical_test", "regression", "survival_analysis", "enrichment"]),
        _e("scanpy", "Run Scanpy for single-cell RNA-seq analysis — clustering, differential expression, trajectory.", CATEGORY_EXPRESSION, ToolSourceType.CONTAINER, capabilities=["preprocessing", "clustering", "de_analysis", "trajectory"]),
        _e("deseq2", "Run DESeq2 for differential gene expression analysis from RNA-seq count data.", CATEGORY_EXPRESSION, ToolSourceType.CONTAINER, capabilities=["differential_expression", "normalization", "shrinkage"]),
        _e("gsea_compute", "Run Gene Set Enrichment Analysis (GSEA) on ranked gene lists.", CATEGORY_PATHWAY, ToolSourceType.CONTAINER, capabilities=["gsea", "leading_edge", "enrichment_plot"]),

        # Network analysis
        _e("cytoscape_analysis", "Run network analysis algorithms — centrality, clustering, community detection, shortest paths.", CATEGORY_NETWORK, ToolSourceType.CONTAINER, capabilities=["centrality", "clustering", "community_detection", "shortest_path", "enrichment"]),

        # Molecular generation
        _e("reinvent", "Run REINVENT for de novo molecular design using reinforcement learning.", CATEGORY_DRUG, ToolSourceType.CONTAINER, capabilities=["generate", "optimize", "score"]),
        _e("molgen", "Generate molecular structures with desired properties using generative models.", CATEGORY_DRUG, ToolSourceType.CONTAINER, capabilities=["generate", "property_optimization", "scaffold_hopping"]),

        # Phylogenetics
        _e("raxml", "Run RAxML for maximum likelihood phylogenetic tree estimation.", CATEGORY_PHYLOGENETICS, ToolSourceType.CONTAINER, capabilities=["tree_inference", "bootstrap", "model_selection"]),
        _e("iqtree", "Run IQ-TREE for phylogenetic analysis with ultrafast bootstrap and model selection.", CATEGORY_PHYLOGENETICS, ToolSourceType.CONTAINER, capabilities=["tree_inference", "bootstrap", "model_test"]),

        # Image analysis
        _e("cellprofiler", "Run CellProfiler for automated cell image analysis and feature extraction.", CATEGORY_IMAGING, ToolSourceType.CONTAINER, capabilities=["cell_segmentation", "feature_extraction", "measurement"]),
        _e("bioformats", "Convert and read microscopy images using Bio-Formats.", CATEGORY_IMAGING, ToolSourceType.CONTAINER, capabilities=["format_conversion", "metadata_extraction"]),

        # Protein ML
        _e("colabfold", "Run ColabFold (MMseqs2 + AlphaFold2) for fast protein structure prediction.", CATEGORY_STRUCTURE, ToolSourceType.CONTAINER, capabilities=["structure_prediction", "msa_generation", "model_quality"]),
        _e("proteinmpnn", "Run ProteinMPNN for protein sequence design given a backbone structure.", CATEGORY_STRUCTURE, ToolSourceType.CONTAINER, capabilities=["sequence_design", "fixed_positions", "multichain"]),
        _e("esmfold_local", "Run ESMFold locally for single-sequence protein structure prediction.", CATEGORY_STRUCTURE, ToolSourceType.CONTAINER, capabilities=["structure_prediction", "plddt_scores"]),
        _e("ankh", "Run Ankh protein language model for embeddings and downstream prediction tasks.", CATEGORY_PROTEIN, ToolSourceType.CONTAINER, capabilities=["embeddings", "fine_tuning"]),
    ]


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------

_CATALOG_CACHE: list[ToolRegistryEntry] | None = None


def get_catalog() -> list[ToolRegistryEntry]:
    """Return the cached tool catalog."""
    global _CATALOG_CACHE
    if _CATALOG_CACHE is None:
        _CATALOG_CACHE = build_catalog()
    return _CATALOG_CACHE


def get_catalog_summary() -> str:
    """Return a compact text summary for LLM prompts."""
    catalog = get_catalog()
    lines = []
    by_category: dict[str, list[ToolRegistryEntry]] = {}
    for entry in catalog:
        by_category.setdefault(entry.category, []).append(entry)

    for category, entries in sorted(by_category.items()):
        lines.append(f"\n## {category.replace('_', ' ').title()} ({len(entries)} tools)")
        for e in entries:
            caps = ", ".join(e.capabilities[:4]) if e.capabilities else ""
            lines.append(f"  - {e.name}: {e.description[:80]}... [{caps}]")

    return f"YOHAS Tool Catalog — {len(catalog)} tools\n" + "\n".join(lines)


def get_tools_for_category(category: str) -> list[ToolRegistryEntry]:
    """Filter catalog by category."""
    return [e for e in get_catalog() if e.category == category]
