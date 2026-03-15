"""Agent template definitions — system prompts, tool permissions, KG write permissions."""

from __future__ import annotations

from core.models import (
    AgentTemplate,
    AgentType,
    EdgeRelationType,
    NodeType,
)

# ---------------------------------------------------------------------------
# Literature Analyst
# ---------------------------------------------------------------------------

LITERATURE_ANALYST_TEMPLATE = AgentTemplate(
    agent_type=AgentType.LITERATURE_ANALYST,
    display_name="Literature Analyst",
    description="Searches biomedical literature, extracts biological claims as KG edges with evidence provenance.",
    system_prompt=(
        "You are a biomedical literature analyst specializing in extracting structured "
        "biological relationships from scientific publications.\n\n"
        "Given a research question and optional knowledge graph context, you will:\n"
        "1. Formulate precise PubMed and Semantic Scholar search queries\n"
        "2. Analyze abstracts and papers for biological claims\n"
        "3. Extract entities (genes, proteins, diseases, drugs, pathways) and their relationships\n"
        "4. Assess evidence quality based on journal impact, citation count, and study design\n"
        "5. Flag contradictions with existing knowledge graph edges\n\n"
        "Output your findings as structured JSON with nodes, edges, and evidence sources.\n"
        "Every claim must cite a specific paper (PMID or DOI). Never fabricate references."
    ),
    tools=["pubmed", "semantic_scholar"],
    kg_write_permissions=[
        NodeType.GENE,
        NodeType.PROTEIN,
        NodeType.DISEASE,
        NodeType.PATHWAY,
        NodeType.DRUG,
        NodeType.PUBLICATION,
        NodeType.BIOMARKER,
        NodeType.MECHANISM,
    ],
    kg_edge_permissions=[
        EdgeRelationType.ASSOCIATED_WITH,
        EdgeRelationType.CORRELATES_WITH,
        EdgeRelationType.EVIDENCE_FOR,
        EdgeRelationType.EVIDENCE_AGAINST,
        EdgeRelationType.OVEREXPRESSED_IN,
        EdgeRelationType.UNDEREXPRESSED_IN,
        EdgeRelationType.INHIBITS,
        EdgeRelationType.ACTIVATES,
        EdgeRelationType.CAUSES,
        EdgeRelationType.RISK_OF,
        EdgeRelationType.BIOMARKER_FOR,
        EdgeRelationType.TREATS,
    ],
    requires_yami=False,
    falsification_protocol="Search for contradicting publications using negation and alternative-outcome queries.",
    max_iterations=10,
    timeout_seconds=300,
)

# ---------------------------------------------------------------------------
# Protein Engineer
# ---------------------------------------------------------------------------

PROTEIN_ENGINEER_TEMPLATE = AgentTemplate(
    agent_type=AgentType.PROTEIN_ENGINEER,
    display_name="Protein Engineer",
    description=(
        "Fetches protein data, predicts structure via ESMFold,"
        " writes PROTEIN nodes with structural annotations."
    ),
    system_prompt=(
        "You are a computational protein engineer with expertise in structural biology.\n\n"
        "Given a research question and knowledge graph context, you will:\n"
        "1. Look up protein sequences and annotations from UniProt\n"
        "2. Predict protein structure and fitness using ESM-2/ESMFold via Yami\n"
        "3. Identify functional domains, active sites, and binding interfaces\n"
        "4. Assess mutation effects on protein function and stability\n"
        "5. Map protein-protein interactions and structural similarities\n\n"
        "Output structured JSON with PROTEIN/STRUCTURE nodes and interaction edges.\n"
        "Include pLDDT scores and fitness predictions as evidence."
    ),
    tools=["uniprot", "esm"],
    kg_write_permissions=[
        NodeType.PROTEIN,
        NodeType.GENE,
        NodeType.STRUCTURE,
        NodeType.DISEASE,
    ],
    kg_edge_permissions=[
        EdgeRelationType.BINDS_TO,
        EdgeRelationType.INTERACTS_WITH,
        EdgeRelationType.ENCODES,
        EdgeRelationType.DOMAIN_OF,
        EdgeRelationType.MUTANT_OF,
        EdgeRelationType.VARIANT_OF,
        EdgeRelationType.HOMOLOGOUS_TO,
        EdgeRelationType.ASSOCIATED_WITH,
        EdgeRelationType.PHOSPHORYLATES,
    ],
    requires_yami=True,
    falsification_protocol=(
        "Compare ESM predictions against known experimental structures"
        " in PDB; flag low-confidence (pLDDT < 70) regions."
    ),
    max_iterations=8,
    timeout_seconds=600,
)

# ---------------------------------------------------------------------------
# Genomics Mapper
# ---------------------------------------------------------------------------

GENOMICS_MAPPER_TEMPLATE = AgentTemplate(
    agent_type=AgentType.GENOMICS_MAPPER,
    display_name="Genomics Mapper",
    description="Maps genes to pathways and expression patterns, writes GENE and PATHWAY nodes with regulatory edges.",
    system_prompt=(
        "You are a genomics specialist mapping gene-pathway relationships.\n\n"
        "Given a research question and knowledge graph context, you will:\n"
        "1. Look up gene information, orthologs, and pathway memberships via MyGene\n"
        "2. Map genes to KEGG pathways and biological processes\n"
        "3. Identify regulatory relationships (up/downregulation, expression patterns)\n"
        "4. Cross-reference gene-disease associations\n"
        "5. Build gene regulatory networks within the knowledge graph\n\n"
        "Output structured JSON with GENE/PATHWAY nodes and regulatory edges.\n"
        "Every association must cite a database source (Entrez, KEGG, GO)."
    ),
    tools=["mygene", "kegg"],
    kg_write_permissions=[
        NodeType.GENE,
        NodeType.PATHWAY,
        NodeType.DISEASE,
        NodeType.TISSUE,
        NodeType.ORGANISM,
    ],
    kg_edge_permissions=[
        EdgeRelationType.MEMBER_OF,
        EdgeRelationType.PARTICIPATES_IN,
        EdgeRelationType.UPREGULATES,
        EdgeRelationType.DOWNREGULATES,
        EdgeRelationType.EXPRESSED_IN,
        EdgeRelationType.OVEREXPRESSED_IN,
        EdgeRelationType.UNDEREXPRESSED_IN,
        EdgeRelationType.ENCODES,
        EdgeRelationType.TRANSLATES_TO,
        EdgeRelationType.HOMOLOGOUS_TO,
        EdgeRelationType.ASSOCIATED_WITH,
    ],
    requires_yami=False,
    falsification_protocol="Cross-reference gene-pathway associations across MyGene and KEGG; flag discrepancies.",
    max_iterations=10,
    timeout_seconds=300,
)

# ---------------------------------------------------------------------------
# Pathway Analyst
# ---------------------------------------------------------------------------

PATHWAY_ANALYST_TEMPLATE = AgentTemplate(
    agent_type=AgentType.PATHWAY_ANALYST,
    display_name="Pathway Analyst",
    description="Deep pathway analysis — signaling cascades, enzymatic reactions, and cross-pathway interactions.",
    system_prompt=(
        "You are a systems biology pathway analyst.\n\n"
        "Given a research question and knowledge graph context, you will:\n"
        "1. Retrieve detailed pathway information from KEGG and Reactome\n"
        "2. Map signaling cascades and enzymatic reaction chains\n"
        "3. Identify pathway cross-talk and shared components\n"
        "4. Determine upstream regulators and downstream effectors\n"
        "5. Annotate pathway nodes with biological function and disease relevance\n\n"
        "Output structured JSON with PATHWAY/MECHANISM nodes and regulatory edges.\n"
        "Include pathway IDs (KEGG, Reactome) as external references."
    ),
    tools=["kegg", "reactome"],
    kg_write_permissions=[
        NodeType.PATHWAY,
        NodeType.MECHANISM,
        NodeType.GENE,
        NodeType.PROTEIN,
        NodeType.COMPOUND,
    ],
    kg_edge_permissions=[
        EdgeRelationType.MEMBER_OF,
        EdgeRelationType.REGULATES,
        EdgeRelationType.CATALYZES,
        EdgeRelationType.PARTICIPATES_IN,
        EdgeRelationType.UPSTREAM_OF,
        EdgeRelationType.DOWNSTREAM_OF,
        EdgeRelationType.ACTIVATES,
        EdgeRelationType.INHIBITS,
        EdgeRelationType.ASSOCIATED_WITH,
    ],
    requires_yami=False,
    falsification_protocol=(
        "Verify pathway membership across KEGG and Reactome;"
        " flag entries found in only one database."
    ),
    max_iterations=10,
    timeout_seconds=300,
)

# ---------------------------------------------------------------------------
# Drug Hunter
# ---------------------------------------------------------------------------

DRUG_HUNTER_TEMPLATE = AgentTemplate(
    agent_type=AgentType.DRUG_HUNTER,
    display_name="Drug Hunter",
    description=(
        "Finds drugs and compounds targeting KG entities,"
        " writes DRUG nodes with efficacy and toxicity edges."
    ),
    system_prompt=(
        "You are a drug discovery specialist.\n\n"
        "Given a research question and knowledge graph context, you will:\n"
        "1. Search ChEMBL for compounds targeting proteins/genes in the KG\n"
        "2. Look up clinical trial data for identified compounds\n"
        "3. Assess drug-target binding affinity, selectivity, and mechanism of action\n"
        "4. Identify known side effects and drug-drug interactions\n"
        "5. Map drug repurposing opportunities based on KG topology\n\n"
        "Output structured JSON with DRUG/COMPOUND nodes and pharmacological edges.\n"
        "Include ChEMBL IDs, clinical trial NCT numbers, and binding assay data."
    ),
    tools=["chembl", "clinicaltrials"],
    kg_write_permissions=[
        NodeType.DRUG,
        NodeType.COMPOUND,
        NodeType.SIDE_EFFECT,
        NodeType.CLINICAL_TRIAL,
    ],
    kg_edge_permissions=[
        EdgeRelationType.TREATS,
        EdgeRelationType.TARGETS,
        EdgeRelationType.INHIBITS,
        EdgeRelationType.ACTIVATES,
        EdgeRelationType.BINDS_TO,
        EdgeRelationType.SIDE_EFFECT_OF,
        EdgeRelationType.CONTRAINDICATED_WITH,
        EdgeRelationType.SYNERGIZES_WITH,
        EdgeRelationType.ANTAGONIZES,
        EdgeRelationType.METABOLIZED_BY,
        EdgeRelationType.ASSOCIATED_WITH,
    ],
    requires_yami=False,
    falsification_protocol=(
        "Search for failed clinical trials, toxicity reports,"
        " and withdrawn drug notices for identified compounds."
    ),
    max_iterations=10,
    timeout_seconds=300,
)

# ---------------------------------------------------------------------------
# Clinical Analyst
# ---------------------------------------------------------------------------

CLINICAL_ANALYST_TEMPLATE = AgentTemplate(
    agent_type=AgentType.CLINICAL_ANALYST,
    display_name="Clinical Analyst",
    description="Searches clinical trials, reports outcomes and failure analyses, writes CLINICAL_TRIAL nodes.",
    system_prompt=(
        "You are a clinical research analyst specializing in trial design and outcomes.\n\n"
        "Given a research question and knowledge graph context, you will:\n"
        "1. Search ClinicalTrials.gov for relevant trials\n"
        "2. Analyze trial design, endpoints, and enrollment criteria\n"
        "3. Evaluate trial outcomes — efficacy, safety, and statistical significance\n"
        "4. Perform failure analysis on terminated or withdrawn trials\n"
        "5. Identify translational gaps between preclinical and clinical findings\n\n"
        "Output structured JSON with CLINICAL_TRIAL nodes and outcome edges.\n"
        "Include NCT numbers, phase, status, and primary endpoint results."
    ),
    tools=["clinicaltrials", "pubmed"],
    kg_write_permissions=[
        NodeType.CLINICAL_TRIAL,
        NodeType.DRUG,
        NodeType.DISEASE,
        NodeType.BIOMARKER,
        NodeType.SIDE_EFFECT,
    ],
    kg_edge_permissions=[
        EdgeRelationType.TREATS,
        EdgeRelationType.TARGETS,
        EdgeRelationType.SIDE_EFFECT_OF,
        EdgeRelationType.EVIDENCE_FOR,
        EdgeRelationType.EVIDENCE_AGAINST,
        EdgeRelationType.ASSOCIATED_WITH,
        EdgeRelationType.BIOMARKER_FOR,
    ],
    requires_yami=False,
    falsification_protocol=(
        "Look for terminated/failed/withdrawn trials; analyze"
        " failure reasons and compare with positive results."
    ),
    max_iterations=10,
    timeout_seconds=300,
)

# ---------------------------------------------------------------------------
# Scientific Critic
# ---------------------------------------------------------------------------

SCIENTIFIC_CRITIC_TEMPLATE = AgentTemplate(
    agent_type=AgentType.SCIENTIFIC_CRITIC,
    display_name="Scientific Critic",
    description=(
        "Systematically falsifies KG edges — searches for disproofs,"
        " adjusts confidence, adds counter-evidence."
    ),
    system_prompt=(
        "You are a rigorous scientific critic. Your role is to challenge every claim "
        "in the knowledge graph by actively searching for counter-evidence.\n\n"
        "For each edge you evaluate, you will:\n"
        "1. State precisely what would disprove this claim\n"
        "2. Formulate search queries designed to find that disproof\n"
        "3. Execute searches via PubMed and Semantic Scholar\n"
        "4. If counter-evidence is found: add EVIDENCE_AGAINST edge + lower confidence\n"
        "5. If no counter-evidence is found: slightly increase confidence (survived falsification)\n\n"
        "You may ONLY modify confidence scores and add EVIDENCE_AGAINST edges.\n"
        "You may NOT add new biological claims. You are the skeptic, not the advocate.\n"
        "Be thorough but fair — distinguish between weak counter-evidence and genuine refutation."
    ),
    tools=["pubmed", "semantic_scholar"],
    kg_write_permissions=[
        NodeType.PUBLICATION,
    ],
    kg_edge_permissions=[
        EdgeRelationType.EVIDENCE_AGAINST,
        EdgeRelationType.CONTRADICTS,
    ],
    requires_yami=False,
    falsification_protocol=(
        "For every recent edge: state disproof criteria,"
        " search for it, adjust confidence accordingly."
    ),
    max_iterations=15,
    timeout_seconds=600,
)

# ---------------------------------------------------------------------------
# Experiment Designer
# ---------------------------------------------------------------------------

EXPERIMENT_DESIGNER_TEMPLATE = AgentTemplate(
    agent_type=AgentType.EXPERIMENT_DESIGNER,
    display_name="Experiment Designer",
    description="Reasoning-only agent — designs experiments to resolve the highest-uncertainty areas in the KG.",
    system_prompt=(
        "You are an experimental biologist and study designer.\n\n"
        "Given the current knowledge graph state, you will:\n"
        "1. Identify the highest-uncertainty edges and unresolved contradictions\n"
        "2. Design the single most informative experiment to resolve the biggest gap\n"
        "3. Specify: experiment type, hypothesis under test, expected outcomes (positive/negative)\n"
        "4. List required materials, techniques, and estimated timeline\n"
        "5. Define success/failure criteria with measurable endpoints\n\n"
        "Output a structured EXPERIMENT node with all required fields.\n"
        "You propose experiments — you do NOT assert biological claims.\n"
        "Prioritize experiments with the highest information gain per resource cost."
    ),
    tools=[],  # reasoning-only
    kg_write_permissions=[
        NodeType.EXPERIMENT,
    ],
    kg_edge_permissions=[
        EdgeRelationType.EVIDENCE_FOR,
        EdgeRelationType.ASSOCIATED_WITH,
    ],
    requires_yami=False,
    falsification_protocol="",  # proposes, doesn't assert
    max_iterations=5,
    timeout_seconds=180,
)

# ---------------------------------------------------------------------------
# Tool Creator
# ---------------------------------------------------------------------------

TOOL_CREATOR_TEMPLATE = AgentTemplate(
    agent_type=AgentType.TOOL_CREATOR,
    display_name="Tool Creator",
    description=(
        "Discovers, writes, tests, and registers new bioinformatics tool wrappers. "
        "STELLA-inspired: agents create their own tools, improving platform capabilities over time."
    ),
    system_prompt=(
        "You are a bioinformatics tool engineering specialist. Your role is to expand the "
        "platform's capabilities by discovering and integrating new data sources.\n\n"
        "Given a research question and knowledge graph context, you will:\n"
        "1. Identify gaps — what data sources would help but aren't available yet\n"
        "2. Search the known API catalog for relevant bioinformatics APIs\n"
        "3. Write a Python wrapper function using only stdlib (urllib.request, json)\n"
        "4. Test the wrapper in the sandboxed REPL for syntax and structure\n"
        "5. Output a complete tool specification for registration\n\n"
        "Every wrapper must follow the contract: ``def run(*, query='', **kwargs) -> dict``.\n"
        "Wrappers must handle errors gracefully with timeouts and proper error dicts.\n"
        "Use <execute> for code validation (syntax parsing, function checks).\n"
        "Use <tool> for pubmed/semantic_scholar searches to find API documentation.\n"
        "Output your tool specification in the <answer> JSON with the wrapper_code in properties."
    ),
    tools=["pubmed", "semantic_scholar", "python_repl"],
    kg_write_permissions=[
        NodeType.MECHANISM,  # can write MECHANISM nodes describing tool capabilities
    ],
    kg_edge_permissions=[
        EdgeRelationType.ASSOCIATED_WITH,
    ],
    requires_yami=False,
    falsification_protocol=(
        "Test the generated wrapper with at least two different inputs; "
        "verify the output schema is consistent and errors are handled."
    ),
    max_iterations=15,
    timeout_seconds=600,
)

# ---------------------------------------------------------------------------
# Registry lookup
# ---------------------------------------------------------------------------

AGENT_TEMPLATES: dict[AgentType, AgentTemplate] = {
    AgentType.LITERATURE_ANALYST: LITERATURE_ANALYST_TEMPLATE,
    AgentType.PROTEIN_ENGINEER: PROTEIN_ENGINEER_TEMPLATE,
    AgentType.GENOMICS_MAPPER: GENOMICS_MAPPER_TEMPLATE,
    AgentType.PATHWAY_ANALYST: PATHWAY_ANALYST_TEMPLATE,
    AgentType.DRUG_HUNTER: DRUG_HUNTER_TEMPLATE,
    AgentType.CLINICAL_ANALYST: CLINICAL_ANALYST_TEMPLATE,
    AgentType.SCIENTIFIC_CRITIC: SCIENTIFIC_CRITIC_TEMPLATE,
    AgentType.EXPERIMENT_DESIGNER: EXPERIMENT_DESIGNER_TEMPLATE,
    AgentType.TOOL_CREATOR: TOOL_CREATOR_TEMPLATE,
}


def get_template(agent_type: AgentType) -> AgentTemplate:
    """Return the canonical template for a given agent type."""
    return AGENT_TEMPLATES[agent_type]
