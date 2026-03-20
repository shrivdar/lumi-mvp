"""YOHAS-powered benchmark evaluation using the full architecture.

Runs each benchmark question through a 6-phase pipeline that mirrors the
core YOHAS research loop — Hypothesize, Investigate, Code Review, Science
Review, Synthesize, Falsify — adapted for single-question answering.

The dual-review validation (phases 3-4) is inspired by K-Dense's approach
to BixBench: two review agents validate every result before synthesis.
The Code Review agent checks technical correctness while the Science Review
agent validates methodology and biological reasoning.

Usage from other benchmark runners::

    from yohas_bench_eval import evaluate_with_yohas, YOHASBenchResult

    result = await evaluate_with_yohas(
        question="Which gene is in DisGeNET but NOT in OMIM for ...",
        choices=[("A", "GeneX"), ("B", "GeneY"), ...],
        correct_letter="B",
        llm=llm_client,
        subtask="dga_task",
    )
    print(result.predicted, result.confidence)

Each evaluation uses 5-7 LLM calls (vs 1 for zero-shot) but should be
significantly more accurate thanks to multi-hypothesis investigation,
dual-review validation, and active falsification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("yohas_bench")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HypothesisEvidence:
    """Evidence collected for a single answer hypothesis."""

    answer: str  # the answer letter (A, B, C, ...)
    answer_text: str  # full answer text
    supporting_evidence: list[str] = field(default_factory=list)
    counter_evidence: list[str] = field(default_factory=list)
    confidence: float = 0.5
    db_results: list[dict] = field(default_factory=list)
    code_results: list[str] = field(default_factory=list)


@dataclass
class YOHASBenchResult:
    """Result from YOHAS-powered evaluation."""

    predicted: str  # final answer letter
    confidence: float
    hypotheses: list[HypothesisEvidence]
    reasoning: str
    tokens_used: int
    duration_ms: int
    phases_completed: list[str]  # which phases ran successfully


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def _extract_json_from_response(text: str) -> Any | None:
    """Try to extract a JSON array or object from an LLM response."""
    # Try ```json ... ``` blocks first
    m = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try raw JSON array
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # Try raw JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _extract_answer_letter(text: str) -> str:
    """Extract a single answer letter from LLM text."""
    # <answer>X</answer>
    m = re.search(r"<answer>\s*\(?([A-Ha-h])\)?\s*</answer>", text)
    if m:
        return m.group(1).upper()

    # "final answer is X", "Answer: X"
    patterns = [
        r"(?:final\s+)?answer\s*(?:is|:)\s*\(?([A-Ha-h])\)?",
        r"(?:select|choose|pick)\s+\(?([A-Ha-h])\)?",
        r"(?:confirmed|confirm)\s*:?\s*\(?([A-Ha-h])\)?",
        r"\b([A-Ha-h])\b\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()

    # Last standalone letter A-H
    matches = re.findall(r"\b([A-H])\b", text)
    if matches:
        return matches[-1]

    return ""


def _format_choices_block(choices: list[tuple[str, str]]) -> str:
    """Format choices as a readable list."""
    return "\n".join(f"  ({letter}) {text}" for letter, text in choices)


def _get_hypothesis(
    hypotheses: list[HypothesisEvidence], letter: str,
) -> HypothesisEvidence | None:
    """Find a hypothesis by its answer letter."""
    for h in hypotheses:
        if h.answer == letter:
            return h
    return None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

REFUSE_PHRASES = (
    "insufficient information",
    "cannot determine",
    "not enough information",
    "unable to determine",
)


def _is_refuse_choice(text: str) -> bool:
    """Check if a choice text is a refusal / insufficient-info option."""
    lower = text.lower().strip()
    return any(phrase in lower for phrase in REFUSE_PHRASES)


def _detect_databases_in_question(question: str) -> list[str]:
    """Detect which databases a question is asking about."""
    q_lower = question.lower()
    detected: list[str] = []
    db_keywords = {
        "disgenet": "DisGeNET",
        "omim": "OMIM",
        "uniprot": "UniProt",
        "clinvar": "ClinVar",
        "kegg": "KEGG",
        "ensembl": "Ensembl",
        "ncbi gene": "NCBI Gene",
        "ncbi": "NCBI",
        "gnomad": "gnomAD",
        "mirbase": "miRBase",
        "mirtarbase": "miRTarBase",
        "gtrd": "GTRD",
        "msigdb": "MSigDB",
        "cosmic": "COSMIC",
        "reactome": "Reactome",
        "intact": "IntAct",
        "biogrid": "BioGRID",
        "string": "STRING",
        "dbsnp": "dbSNP",
        "chembl": "ChEMBL",
    }
    for kw, name in db_keywords.items():
        if kw in q_lower:
            detected.append(name)
    return detected


def _query_databases_for_choices(
    question: str,
    choices: list[tuple[str, str]],
    subtask: str,
) -> str:
    """Query real databases for each answer choice BEFORE LLM calls.

    Returns a formatted string of database results to inject into prompts.
    This is the key optimization: giving the LLM real data instead of
    relying on its training knowledge for obscure database entries.

    Strategy: be SELECTIVE. Only query when the subtask/question clearly maps
    to a database we can query. Noisy/irrelevant results hurt more than help.
    """
    from bench_helpers import (
        query_gene_disease_association,
        query_gene_location,
        query_protein_function,
        query_variant_significance,
        compare_databases,
    )

    db_context_parts: list[str] = []
    q_lower = question.lower()
    detected_dbs = _detect_databases_in_question(question)

    # Extract entity names from non-refuse choices
    choice_entities = []
    for letter, text in choices:
        if not _is_refuse_choice(text):
            choice_entities.append((letter, text.strip()))

    def _safe_call(fn, *args, **kwargs):
        """Call a DB helper, return result or empty dict on failure."""
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return {"error": str(e)}

    def _summarize_gene_disease(result: dict) -> str:
        """Extract a concise summary from gene-disease query results."""
        gene = result.get("gene", "")
        diseases = result.get("diseases_found", [])
        if not diseases:
            return f"'{gene}': No disease associations found in NCBI/UniProt/ClinVar"
        sources = set()
        disease_names = []
        for d in diseases[:5]:
            src = d.get("source", "")
            if src:
                sources.add(src)
            dname = d.get("disease", "")
            if dname and len(dname) < 100:
                disease_names.append(dname)
        return (
            f"'{gene}': Found in {', '.join(sorted(sources))}. "
            f"Associations: {'; '.join(disease_names[:3])}"
        )

    def _summarize_gene_location(result: dict) -> str:
        """Extract concise location info."""
        gene = result.get("gene", "")
        chrom = result.get("chromosome", "")
        band = result.get("band", "")
        if not chrom and not band:
            return f"'{gene}': Location not found"
        loc = f"chr{chrom}" if chrom else ""
        if band:
            loc += f" ({band})" if loc else band
        return f"'{gene}': {loc}"

    def _summarize_variant(result: dict) -> str:
        """Extract concise variant info."""
        variant = result.get("variant", "")
        sig = result.get("significance", "")
        cond = result.get("condition", "")
        if not sig or sig == "Not found in ClinVar":
            return f"'{variant}': Not found in ClinVar"
        parts = [f"'{variant}': {sig}"]
        if cond:
            parts.append(f"({cond})")
        return " ".join(parts)

    def _summarize_protein(result: dict) -> str:
        """Extract concise protein function info."""
        name = result.get("name", "")
        func = result.get("function", "")
        if not func or func == "Not found":
            return f"'{name}': Not found in UniProt"
        return f"'{name}': {func[:200]}"

    # ---- Subtask-specific routing (be selective!) ----

    # Gene-disease association tasks (dga_task)
    # dga_task asks about DisGeNET vs OMIM — these are specialized databases
    # that cannot be directly queried via free public APIs. Our proxy queries
    # (NCBI, UniProt, ClinVar) return similar data for all choices and add noise.
    # Skip DB queries for dga_task and rely on improved prompting instead.
    if subtask in ("dga_task", "dga_task-v1-public"):
        pass  # No DB queries — LLM knowledge is more reliable here

    elif subtask in ("gene_disease_task",) and "DisGeNET" not in detected_dbs and "OMIM" not in detected_dbs:
        for letter, entity in choice_entities:
            result = _safe_call(query_gene_disease_association, entity)
            db_context_parts.append(
                f"  ({letter}) {_summarize_gene_disease(result)}"
            )

    # Gene location tasks
    elif subtask == "gene_location_task" or "chromosome" in q_lower or "locus" in q_lower or "cytogenetic" in q_lower:
        for letter, entity in choice_entities:
            result = _safe_call(query_gene_location, entity)
            db_context_parts.append(
                f"  ({letter}) {_summarize_gene_location(result)}"
            )

    # Variant tasks
    elif subtask == "variant_task" or "ClinVar" in detected_dbs or ("variant" in q_lower and ("pathogenic" in q_lower or "benign" in q_lower)):
        for letter, entity in choice_entities:
            result = _safe_call(query_variant_significance, entity)
            db_context_parts.append(
                f"  ({letter}) {_summarize_variant(result)}"
            )

    # miRNA target tasks — query PubMed for evidence
    elif subtask == "mirna_target_task" or "miRTarBase" in detected_dbs or ("mirna" in q_lower and "target" in q_lower):
        try:
            from run_labbench import query_pubmed
        except ImportError:
            query_pubmed = None
        if query_pubmed:
            for letter, entity in choice_entities:
                result = _safe_call(query_pubmed, f"miRNA target {entity}", 3)
                abstracts = result[0].get("abstracts", "")[:300] if result else "No results"
                db_context_parts.append(
                    f"  ({letter}) PubMed search 'miRNA target {entity}': {abstracts}"
                )

    # Transcription factor binding site tasks
    elif subtask == "tfbs_task" or "GTRD" in detected_dbs or "transcription factor" in q_lower:
        try:
            from run_labbench import query_pubmed
        except ImportError:
            query_pubmed = None
        if query_pubmed:
            for letter, entity in choice_entities:
                result = _safe_call(query_pubmed, f"transcription factor binding {entity}", 3)
                abstracts = result[0].get("abstracts", "")[:300] if result else "No results"
                db_context_parts.append(
                    f"  ({letter}) PubMed search 'TF binding {entity}': {abstracts}"
                )

    # Protein/UniProt tasks
    elif "UniProt" in detected_dbs or subtask in ("protein_task",):
        for letter, entity in choice_entities:
            result = _safe_call(query_protein_function, entity)
            db_context_parts.append(
                f"  ({letter}) {_summarize_protein(result)}"
            )

    # For questions about specific databases we CAN query, do a general lookup
    elif detected_dbs and "DisGeNET" not in detected_dbs and "OMIM" not in detected_dbs:
        for letter, entity in choice_entities:
            result = _safe_call(query_gene_disease_association, entity)
            db_context_parts.append(
                f"  ({letter}) {_summarize_gene_disease(result)}"
            )

    # No DB queries for unrecognized subtasks or DisGeNET/OMIM — avoid noise
    # (the LLM's training knowledge is better than irrelevant proxy DB results)

    if db_context_parts:
        db_names = ", ".join(detected_dbs) if detected_dbs else "biological databases"
        return (
            f"\n=== DATABASE LOOKUP RESULTS ({db_names}) ===\n"
            f"Live query results for each answer choice:\n"
            + "\n".join(db_context_parts)
            + "\n\nNOTE: These are real-time lookups via NCBI, UniProt, Ensembl, "
            "and ClinVar APIs. An entity with 'Not found' likely has limited "
            "presence in that database. Use this to distinguish between choices.\n"
            "=== END DATABASE RESULTS ===\n"
        )
    return ""


async def _query_databases_for_choices_async(
    question: str,
    choices: list[tuple[str, str]],
    subtask: str,
) -> str:
    """Run DB queries in a thread pool to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return await loop.run_in_executor(
            pool, _query_databases_for_choices, question, choices, subtask,
        )


def _build_evidence_prompt(
    question: str,
    choices: list[tuple[str, str]],
    subtask: str,
    db_helpers_available: bool,
    db_context: str = "",
) -> str:
    """Build the Phase 2 evidence-collection prompt."""
    choices_block = _format_choices_block(choices)

    db_hint = ""
    if db_helpers_available:
        db_hint = (
            "\n\nYou have access to biological databases (UniProt, NCBI Gene, "
            "KEGG, ClinVar, PubMed, Ensembl, STRING). Use your deep knowledge "
            "of what these databases contain to evaluate each choice."
        )

    subtask_hint = ""
    if subtask:
        subtask_hint = f"\n\nQuestion domain/subtask: {subtask}"

    db_results_section = ""
    if db_context:
        db_results_section = f"""
{db_context}
Use the database results above as ADDITIONAL evidence to cross-check your knowledge.
If a choice entity was NOT found in a relevant database, that is significant evidence.
"""

    return f"""You are an expert biomedical researcher evaluating a multiple-choice question.
For EACH answer choice, provide evidence supporting or contradicting it.

Question: {question}

Answer choices:
{choices_block}
{subtask_hint}{db_hint}{db_results_section}

IMPORTANT: "Insufficient information" is almost never the correct answer (<10% of questions).
Prefer a specific, evidence-backed answer.

For each choice, analyze:
1. What biological facts support this being the correct answer?
2. What biological facts contradict this being correct?
3. How confident are you (0.0 to 1.0)?

Return your analysis as a JSON array (one object per choice):
```json
[
  {{"answer": "A", "evidence_for": "...", "evidence_against": "...", "confidence": 0.8}},
  {{"answer": "B", "evidence_for": "...", "evidence_against": "...", "confidence": 0.3}},
  ...
]
```

Be thorough and specific. Reference the database query results above when available.
Cross-check against actual database entries, gene functions, pathway memberships,
variant classifications, or published findings."""


def _build_independent_verifier_prompt(
    question: str,
    choices: list[tuple[str, str]],
    subtask: str,
    db_context: str = "",
) -> str:
    """Build the Phase 2 independent-verifier prompt.

    This agent answers from scratch, without seeing the evidence collector's
    output, to provide an independent perspective.
    """
    choices_block = _format_choices_block(choices)

    db_section = ""
    if db_context:
        db_section = f"""
{db_context}
"""

    return f"""You are an expert biomedical scientist. Answer this multiple-choice question
using your deep knowledge of biological databases and molecular biology.

Question: {question}

Choices:
{choices_block}
{db_section}
Think step by step:
1. What specific database(s) or biological domain is this question about?
2. What do you know about each entity/gene/protein/variant mentioned?
3. Which answer is most consistent with known database entries?

IMPORTANT: "Insufficient information" is almost never correct. Commit to a specific answer.

Provide your reasoning, then state your answer on the last line as:
Answer: X
where X is a single letter."""


def _build_synthesis_prompt(
    question: str,
    choices: list[tuple[str, str]],
    hypotheses: list[HypothesisEvidence],
    verifier_answer: str,
    verifier_reasoning: str,
) -> str:
    """Build the Phase 3 synthesis prompt."""
    choices_block = _format_choices_block(choices)

    evidence_summary = []
    for h in hypotheses:
        supporting = "; ".join(h.supporting_evidence) if h.supporting_evidence else "None found"
        counter = "; ".join(h.counter_evidence) if h.counter_evidence else "None found"
        evidence_summary.append(
            f"  ({h.answer}) {h.answer_text}\n"
            f"      Evidence FOR: {supporting}\n"
            f"      Evidence AGAINST: {counter}\n"
            f"      Investigator confidence: {h.confidence:.2f}"
        )
    evidence_block = "\n".join(evidence_summary)

    verifier_section = ""
    if verifier_answer:
        verifier_section = (
            f"\n\nIndependent verifier chose: ({verifier_answer})\n"
            f"Verifier reasoning: {verifier_reasoning[:500]}"
        )

    return f"""You are synthesizing evidence from multiple research agents to determine the
correct answer to a biomedical question.

Question: {question}

Choices:
{choices_block}

== Evidence collected by investigator agent ==
{evidence_block}
{verifier_section}

Your task:
1. Weigh the evidence for and against each hypothesis.
2. Consider the agreement or disagreement between the investigator and verifier.
3. If both agents agree, that strongly supports the answer.
4. If they disagree, carefully evaluate which has stronger evidence.
5. Pick the answer with the strongest supporting evidence and weakest counter-evidence.

IMPORTANT: "Insufficient information" is almost never correct. Choose a specific answer.

State your synthesis reasoning, then provide your answer:
<answer>X</answer>
where X is a single letter (A, B, C, D, or E).

Also rate your confidence (0.0-1.0):
<confidence>0.85</confidence>"""


def _build_falsification_prompt(
    question: str,
    choices: list[tuple[str, str]],
    top_answer: str,
    top_answer_text: str,
    hypotheses: list[HypothesisEvidence],
    db_context: str = "",
) -> str:
    """Build the Phase 4 falsification prompt."""
    choices_block = _format_choices_block(choices)

    # Show the runner-up alternatives
    alternatives = []
    for h in hypotheses:
        if h.answer != top_answer and not _is_refuse_choice(h.answer_text):
            alternatives.append(f"  ({h.answer}) {h.answer_text} [confidence: {h.confidence:.2f}]")
    alternatives_block = "\n".join(alternatives) if alternatives else "  (none)"

    db_section = ""
    if db_context:
        db_section = f"""

{db_context}

Use the database results above to check whether the proposed answer is actually
correct according to real database entries. Pay special attention to:
- Whether the proposed entity actually appears in the database the question asks about
- Whether an alternative entity has stronger database evidence
"""

    return f"""You are a scientific critic whose job is to FALSIFY a proposed answer.

Question: {question}

Choices:
{choices_block}

The proposed answer is: ({top_answer}) {top_answer_text}

Alternative candidates:
{alternatives_block}
{db_section}
Your task — ACTIVELY TRY TO PROVE THE PROPOSED ANSWER WRONG:
1. What specific biological facts or database entries would make ({top_answer}) incorrect?
2. Is there a known exception, edge case, or database inconsistency that applies here?
3. Could one of the alternative answers actually be more correct? Why?
4. For "which is in database X but NOT database Y" questions: is this entity actually
   in BOTH databases (making it a distractor, not the answer)?

If you find STRONG counter-evidence:
- State what the counter-evidence is
- State which alternative answer is actually correct
- Respond with: <answer>Y</answer> (the corrected answer)

If the proposed answer SURVIVES falsification:
- Confirm it is robust
- Respond with: <answer>{top_answer}</answer>

Be rigorous. A good scientist tries hard to disprove their hypothesis."""


# ---------------------------------------------------------------------------
# Dual-review validation helpers
# ---------------------------------------------------------------------------


def _extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from LLM output."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return [b.strip() for b in blocks if b.strip()]


def _extract_code_outputs(text: str) -> list[str]:
    """Extract code output sections from LLM output."""
    outputs = re.findall(r"(?:Output|Result|>>>)\s*:?\s*\n?(.*?)(?=\n```|\n##|\Z)", text, re.DOTALL)
    return [o.strip() for o in outputs if o.strip()]


def parse_json_safe(text: str) -> dict:
    """Parse JSON from LLM response, returning empty dict on failure."""
    result = _extract_json_from_response(text)
    if isinstance(result, dict):
        return result
    return {}


async def _code_review(
    query_fn,
    question: str,
    agent_output: str,
    code_blocks: list[str],
    code_outputs: list[str],
) -> dict:
    """Review the technical quality of code execution and results.

    Args:
        query_fn: Async callable wrapping llm.query with token tracking.
        question: The benchmark question text.
        agent_output: The agent's conclusion text.
        code_blocks: List of code snippets the agent executed.
        code_outputs: Corresponding outputs for each code block.

    Returns:
        Dict with keys: approved (bool), issues (list[str]),
        corrected_answer (str|None), confidence (float).
    """
    if not code_blocks:
        # Nothing to review — auto-approve
        return {"approved": True, "issues": [], "corrected_answer": None, "confidence": 1.0}

    code_section = "\n\n".join(
        f"```python\n{c}\n```\nOutput: {o}"
        for c, o in zip(code_blocks, code_outputs or ["(no output)"] * len(code_blocks))
    )

    prompt = f"""You are a senior bioinformatics code reviewer. Review this analysis:

QUESTION: {question}

CODE EXECUTED:
{code_section}

AGENT'S CONCLUSION: {agent_output[:1500]}

Review checklist:
1. Does the code correctly load and parse data?
2. Are the right variables/columns being analyzed?
3. Is the statistical test appropriate for this question type?
4. Are there any off-by-one errors or indexing issues?
5. Does the output match what the agent claims?
6. Are edge cases handled (missing data, NaN values, empty results)?

Return JSON only (no other text):
{{"approved": true, "issues": [], "corrected_answer": null, "confidence": 0.95}}

If problems found:
{{"approved": false, "issues": ["specific problem 1", "specific problem 2"], "corrected_answer": "X", "confidence": 0.4}}"""

    resp = await query_fn(prompt, max_tokens=500)
    result = parse_json_safe(resp.text)
    # Ensure expected keys exist
    result.setdefault("approved", True)
    result.setdefault("issues", [])
    result.setdefault("corrected_answer", None)
    result.setdefault("confidence", 0.5)
    return result


async def _science_review(
    query_fn,
    question: str,
    choices: list[tuple[str, str]],
    evidence_summary: str,
    proposed_answer: str,
) -> dict:
    """Review scientific methodology and reasoning.

    Args:
        query_fn: Async callable wrapping llm.query with token tracking.
        question: The benchmark question text.
        choices: Answer choices as (letter, text) tuples.
        evidence_summary: Truncated summary of evidence collected.
        proposed_answer: The current best answer letter.

    Returns:
        Dict with keys: approved (bool), scientific_issues (list[str]),
        recommended_answer (str), confidence (float), reasoning (str).
    """
    choices_block = _format_choices_block(choices)

    prompt = f"""You are a PhD-level biomedical scientist reviewing a research analysis.

QUESTION: {question}
CHOICES:
{choices_block}
PROPOSED ANSWER: {proposed_answer}
EVIDENCE COLLECTED: {evidence_summary[:3000]}

Scientific review checklist:
1. Is the biological reasoning sound?
2. Is the proposed answer consistent with known biology?
3. Could any of the other answer choices be more correct?
4. Are there any logical fallacies in the reasoning?
5. Does the evidence actually support the proposed answer?
6. Are there confounding factors not considered?

Return JSON only (no other text):
{{"approved": true, "scientific_issues": [], "recommended_answer": "{proposed_answer}", "confidence": 0.9, "reasoning": "brief explanation"}}

If problems found:
{{"approved": false, "scientific_issues": ["issue 1"], "recommended_answer": "X", "confidence": 0.6, "reasoning": "why the answer should change"}}"""

    resp = await query_fn(prompt, max_tokens=500)
    result = parse_json_safe(resp.text)
    # Ensure expected keys exist
    result.setdefault("approved", True)
    result.setdefault("scientific_issues", [])
    result.setdefault("recommended_answer", proposed_answer)
    result.setdefault("confidence", 0.5)
    result.setdefault("reasoning", "")
    return result


async def _code_review_with_retry(
    query_fn,
    question: str,
    agent_output: str,
    code_blocks: list[str],
    code_outputs: list[str],
    max_attempts: int = 3,
) -> dict:
    """Run code review with iterative correction on failure.

    If the code review finds issues, asks the LLM to fix them and re-reviews
    up to *max_attempts* times.

    Returns the final review dict (approved may still be False if all retries fail).
    """
    current_blocks = list(code_blocks)
    current_outputs = list(code_outputs)
    current_output_text = agent_output

    for attempt in range(max_attempts):
        review = await _code_review(
            query_fn, question, current_output_text, current_blocks, current_outputs,
        )
        if review.get("approved"):
            review["_attempts"] = attempt + 1
            return review

        # Ask LLM to fix the identified issues
        issues_text = "; ".join(review.get("issues", ["unknown issue"]))
        fix_prompt = (
            f"Your code analysis had these issues: {issues_text}\n\n"
            f"Original question: {question}\n\n"
            f"Fix the problems and provide corrected code and conclusions. "
            f"Be specific about what changed."
        )
        fix_resp = await query_fn(fix_prompt, max_tokens=1500)
        fixed_text = fix_resp.text

        # Extract corrected code blocks and outputs for the next review round
        new_blocks = _extract_code_blocks(fixed_text)
        if new_blocks:
            current_blocks = new_blocks
            current_outputs = _extract_code_outputs(fixed_text) or ["(re-executed)"] * len(new_blocks)
        current_output_text = fixed_text

    # All retries exhausted — return last review
    review["_attempts"] = max_attempts
    return review


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


async def evaluate_with_yohas(
    question: str,
    choices: list[tuple[str, str]],  # [(letter, text), ...]
    correct_letter: str,  # for logging only, not used in evaluation logic
    *,
    llm: Any,  # LLMClient instance
    model: str | None = None,
    db_helpers: dict | None = None,  # pre-built query functions
    code_namespace: dict | None = None,  # for code execution
    subtask: str = "",  # for specialized prompts
    max_agents: int = 3,
    max_turns_per_agent: int = 4,
    enable_falsification: bool = True,
    timeout_seconds: int = 180,
) -> YOHASBenchResult:
    """Run a benchmark question through the full YOHAS architecture.

    Phases:
      1. HYPOTHESIZE — each answer choice becomes a hypothesis
      2. INVESTIGATE — parallel agents collect evidence for/against + verify independently
      3. CODE REVIEW — technical validation of any code executed during investigation
      4. SCIENCE REVIEW — scientific methodology validation of proposed answer
      5. SYNTHESIZE — weigh all evidence + review feedback, pick top answer
      6. FALSIFY — actively try to disprove the top answer; switch if counter-evidence is strong

    Args:
        question: The question text.
        choices: List of (letter, text) tuples for each answer option.
        correct_letter: The correct answer letter (used for logging only).
        llm: An LLMClient instance with an async ``query()`` method.
        model: Model ID to pass to llm.query(). None uses the default.
        db_helpers: Dict of pre-built database query functions (injected into code namespace).
        code_namespace: Dict for sandboxed code execution.
        subtask: Domain hint string (e.g. "dga_task", "gene_location_task").
        max_agents: Max parallel investigation agents (currently 2 are used).
        max_turns_per_agent: Max LLM turns per investigation agent.
        enable_falsification: Whether to run Phase 4.
        timeout_seconds: Total timeout for the entire evaluation.

    Returns:
        YOHASBenchResult with the predicted answer, confidence, evidence, and metadata.
    """
    total_tokens = 0
    start_ms = _now_ms()
    phases: list[str] = []
    all_reasoning: list[str] = []

    # Helper to track tokens
    async def _query(prompt: str, *, system: str = "", max_tokens: int = 4096) -> Any:
        nonlocal total_tokens
        remaining = timeout_seconds - (_now_ms() - start_ms) / 1000
        if remaining <= 5:
            raise asyncio.TimeoutError("YOHAS evaluation timeout")
        resp = await asyncio.wait_for(
            llm.query(
                prompt,
                system_prompt=system,
                max_tokens=max_tokens,
                model=model,
            ),
            timeout=max(remaining, 10),
        )
        total_tokens += resp.call_tokens
        return resp

    try:
        # ================================================================
        # Phase 1: HYPOTHESIZE
        # ================================================================
        hypotheses: list[HypothesisEvidence] = []
        for letter, text in choices:
            hypotheses.append(HypothesisEvidence(
                answer=letter,
                answer_text=text,
            ))
        phases.append("hypothesize")
        all_reasoning.append(
            f"[Phase 1 - Hypothesize] Created {len(hypotheses)} hypotheses "
            f"from answer choices: {', '.join(h.answer for h in hypotheses)}"
        )

        # ================================================================
        # Phase 2: INVESTIGATE (parallel agents)
        # ================================================================

        system_prompt = (
            "You are an expert biomedical researcher with deep knowledge of "
            "biological databases including DisGeNET, OMIM, UniProt, KEGG, "
            "ClinVar, gnomAD, IntAct, BioGRID, NCBI Gene, Ensembl, MSigDB, "
            "miRTarBase, GTRD, dbSNP, Reactome, ChEMBL, and COSMIC."
        )

        # === PRE-LLM DATABASE QUERIES ===
        # Query actual databases BEFORE any LLM calls so we can feed
        # real data into the prompts. This is the key accuracy boost.
        db_context = ""
        try:
            db_context = await _query_databases_for_choices_async(
                question, choices, subtask,
            )
            if db_context:
                logger.info(
                    "DB queries returned %d chars of context for %d choices",
                    len(db_context), len(choices),
                )
                all_reasoning.append(
                    f"[Phase 2 - DB Queries] Retrieved {len(db_context)} chars "
                    f"of real database results before LLM calls"
                )
            else:
                all_reasoning.append(
                    "[Phase 2 - DB Queries] No database results retrieved"
                )
        except Exception as exc:
            logger.warning("Pre-LLM database queries failed: %s", exc)
            all_reasoning.append(
                f"[Phase 2 - DB Queries] FAILED: {exc}"
            )

        # Agent 1: Evidence Collector
        evidence_prompt = _build_evidence_prompt(
            question, choices, subtask, db_helpers is not None,
            db_context=db_context,
        )

        # Agent 2: Independent Verifier
        verifier_prompt = _build_independent_verifier_prompt(
            question, choices, subtask, db_context=db_context,
        )

        # Run both agents in parallel
        evidence_task = asyncio.create_task(
            _query(evidence_prompt, system=system_prompt, max_tokens=4096)
        )
        verifier_task = asyncio.create_task(
            _query(verifier_prompt, system=system_prompt, max_tokens=2048)
        )

        evidence_resp, verifier_resp = await asyncio.gather(
            evidence_task, verifier_task, return_exceptions=True,
        )

        # Process evidence collector results
        if isinstance(evidence_resp, Exception):
            logger.warning("Evidence collector failed: %s", evidence_resp)
            all_reasoning.append(
                f"[Phase 2 - Evidence Collector] FAILED: {evidence_resp}"
            )
        else:
            evidence_text = evidence_resp.text
            all_reasoning.append(
                f"[Phase 2 - Evidence Collector] {evidence_text[:500]}"
            )

            # Parse JSON evidence
            parsed = _extract_json_from_response(evidence_text)
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    letter = str(item.get("answer", "")).upper().strip()
                    h = _get_hypothesis(hypotheses, letter)
                    if h is None:
                        continue
                    evidence_for = str(item.get("evidence_for", ""))
                    evidence_against = str(item.get("evidence_against", ""))
                    if evidence_for:
                        h.supporting_evidence.append(evidence_for)
                    if evidence_against:
                        h.counter_evidence.append(evidence_against)
                    try:
                        h.confidence = float(item.get("confidence", 0.5))
                    except (ValueError, TypeError):
                        pass

        # Process verifier results
        verifier_answer = ""
        verifier_reasoning = ""
        if isinstance(verifier_resp, Exception):
            logger.warning("Verifier failed: %s", verifier_resp)
            all_reasoning.append(
                f"[Phase 2 - Verifier] FAILED: {verifier_resp}"
            )
        else:
            verifier_text = verifier_resp.text
            verifier_reasoning = verifier_text
            verifier_answer = _extract_answer_letter(verifier_text)
            all_reasoning.append(
                f"[Phase 2 - Verifier] Answer: {verifier_answer}. "
                f"Reasoning: {verifier_text[:300]}"
            )

            # Boost confidence for the verifier's chosen answer
            vh = _get_hypothesis(hypotheses, verifier_answer)
            if vh is not None:
                vh.supporting_evidence.append(
                    f"Independent verifier selected this answer"
                )
                # Moderate confidence boost
                vh.confidence = min(1.0, vh.confidence + 0.15)

        phases.append("investigate")

        # ================================================================
        # Phase 3: CODE REVIEW
        # ================================================================
        # Collect any code blocks and outputs from the investigation phase
        investigation_text = ""
        if not isinstance(evidence_resp, Exception):
            investigation_text = evidence_resp.text
        all_code_blocks = _extract_code_blocks(investigation_text)
        all_code_outputs = _extract_code_outputs(investigation_text)

        code_review_result: dict = {"approved": True, "issues": [], "corrected_answer": None, "confidence": 1.0}
        if all_code_blocks:
            try:
                code_review_result = await _code_review_with_retry(
                    _query, question, investigation_text, all_code_blocks, all_code_outputs,
                    max_attempts=3,
                )
                all_reasoning.append(
                    f"[Phase 3 - Code Review] approved={code_review_result.get('approved')}, "
                    f"issues={code_review_result.get('issues', [])}, "
                    f"attempts={code_review_result.get('_attempts', 1)}"
                )
                # If code review suggests a corrected answer, apply it
                corrected = code_review_result.get("corrected_answer")
                if corrected and _get_hypothesis(hypotheses, str(corrected).upper()):
                    ch = _get_hypothesis(hypotheses, str(corrected).upper())
                    if ch is not None:
                        ch.supporting_evidence.append("Code review corrected answer to this choice")
                        ch.confidence = min(1.0, ch.confidence + 0.1)
            except (asyncio.TimeoutError, Exception) as exc:
                logger.warning("Code review failed: %s", exc)
                all_reasoning.append(f"[Phase 3 - Code Review] FAILED: {exc}")
        else:
            all_reasoning.append("[Phase 3 - Code Review] No code blocks found — skipped")

        phases.append("code_review")

        # ================================================================
        # Phase 4: SCIENCE REVIEW
        # ================================================================
        # Determine the current best answer before science review
        pre_review_ranked = sorted(
            [h for h in hypotheses if not _is_refuse_choice(h.answer_text)],
            key=lambda h: h.confidence,
            reverse=True,
        )
        pre_review_answer = pre_review_ranked[0].answer if pre_review_ranked else ""

        # Build a compact evidence summary for the science reviewer
        evidence_summary_parts = []
        for h in hypotheses:
            sup = "; ".join(h.supporting_evidence) if h.supporting_evidence else "None"
            ctr = "; ".join(h.counter_evidence) if h.counter_evidence else "None"
            evidence_summary_parts.append(
                f"({h.answer}) {h.answer_text}: FOR=[{sup}] AGAINST=[{ctr}] conf={h.confidence:.2f}"
            )
        evidence_summary_text = "\n".join(evidence_summary_parts)

        science_review_result: dict = {"approved": True, "scientific_issues": [], "recommended_answer": pre_review_answer, "confidence": 0.5, "reasoning": ""}
        if pre_review_answer:
            try:
                science_review_result = await _science_review(
                    _query, question, choices, evidence_summary_text, pre_review_answer,
                )
                all_reasoning.append(
                    f"[Phase 4 - Science Review] approved={science_review_result.get('approved')}, "
                    f"recommended={science_review_result.get('recommended_answer')}, "
                    f"issues={science_review_result.get('scientific_issues', [])}"
                )
                # If science review recommends a different answer, boost that hypothesis
                rec_answer = str(science_review_result.get("recommended_answer", "")).upper()
                if not science_review_result.get("approved") and rec_answer and rec_answer != pre_review_answer:
                    rh = _get_hypothesis(hypotheses, rec_answer)
                    if rh is not None:
                        rh.supporting_evidence.append(
                            f"Science review recommended this answer: {science_review_result.get('reasoning', '')[:200]}"
                        )
                        rh.confidence = min(1.0, rh.confidence + 0.2)
                        # Reduce confidence in the originally proposed answer
                        orig_h = _get_hypothesis(hypotheses, pre_review_answer)
                        if orig_h is not None:
                            orig_h.counter_evidence.append(
                                f"Science review rejected: {'; '.join(science_review_result.get('scientific_issues', []))}"
                            )
                            orig_h.confidence = max(0.0, orig_h.confidence - 0.15)
                elif science_review_result.get("approved"):
                    # Boost confidence for the approved answer
                    ah = _get_hypothesis(hypotheses, pre_review_answer)
                    if ah is not None:
                        ah.supporting_evidence.append("Science review approved this answer")
                        ah.confidence = min(1.0, ah.confidence + 0.1)
            except (asyncio.TimeoutError, Exception) as exc:
                logger.warning("Science review failed: %s", exc)
                all_reasoning.append(f"[Phase 4 - Science Review] FAILED: {exc}")
        else:
            all_reasoning.append("[Phase 4 - Science Review] No proposed answer — skipped")

        phases.append("science_review")

        # ================================================================
        # Phase 5: SYNTHESIZE
        # ================================================================
        synthesis_prompt = _build_synthesis_prompt(
            question, choices, hypotheses, verifier_answer, verifier_reasoning,
        )
        synthesis_resp = await _query(
            synthesis_prompt, system=system_prompt, max_tokens=2048,
        )
        synthesis_text = synthesis_resp.text
        all_reasoning.append(
            f"[Phase 5 - Synthesize] {synthesis_text[:500]}"
        )

        # Extract the synthesized answer
        top_answer = _extract_answer_letter(synthesis_text)

        # Extract confidence
        conf_match = re.search(
            r"<confidence>\s*([\d.]+)\s*</confidence>", synthesis_text,
        )
        if conf_match:
            try:
                synth_confidence = float(conf_match.group(1))
                th = _get_hypothesis(hypotheses, top_answer)
                if th is not None:
                    th.confidence = synth_confidence
            except ValueError:
                pass

        # Fallback: if synthesis didn't produce a letter, use highest-confidence hypothesis
        if not top_answer:
            # Sort by confidence, excluding refuse choices
            ranked = sorted(
                [h for h in hypotheses if not _is_refuse_choice(h.answer_text)],
                key=lambda h: h.confidence,
                reverse=True,
            )
            if ranked:
                top_answer = ranked[0].answer
            elif hypotheses:
                top_answer = hypotheses[0].answer

        phases.append("synthesize")

        # ================================================================
        # Phase 6: FALSIFY
        # ================================================================
        if enable_falsification and top_answer:
            top_h = _get_hypothesis(hypotheses, top_answer)
            top_text = top_h.answer_text if top_h else ""

            # Skip falsification for refuse answers (not worth falsifying)
            if not _is_refuse_choice(top_text):
                falsify_prompt = _build_falsification_prompt(
                    question, choices, top_answer, top_text, hypotheses,
                    db_context=db_context,
                )
                falsify_resp = await _query(
                    falsify_prompt, system=system_prompt, max_tokens=2048,
                )
                falsify_text = falsify_resp.text
                all_reasoning.append(
                    f"[Phase 6 - Falsify] {falsify_text[:500]}"
                )

                revised_answer = _extract_answer_letter(falsify_text)
                if revised_answer and revised_answer != top_answer:
                    # The falsifier found counter-evidence and switched the answer
                    logger.info(
                        "Falsification CHANGED answer: %s -> %s",
                        top_answer, revised_answer,
                    )
                    all_reasoning.append(
                        f"[Phase 6 - Falsify] CHANGED answer from "
                        f"{top_answer} to {revised_answer}"
                    )
                    top_answer = revised_answer
                else:
                    all_reasoning.append(
                        f"[Phase 6 - Falsify] CONFIRMED answer {top_answer}"
                    )

                phases.append("falsify")

        # ================================================================
        # Build result
        # ================================================================
        final_h = _get_hypothesis(hypotheses, top_answer)
        final_confidence = final_h.confidence if final_h else 0.5

        return YOHASBenchResult(
            predicted=top_answer,
            confidence=final_confidence,
            hypotheses=hypotheses,
            reasoning="\n\n".join(all_reasoning),
            tokens_used=total_tokens,
            duration_ms=_now_ms() - start_ms,
            phases_completed=phases,
        )

    except asyncio.TimeoutError:
        logger.warning("YOHAS evaluation timed out after %dms", _now_ms() - start_ms)
        # Return best guess so far
        ranked = sorted(
            [h for h in hypotheses if not _is_refuse_choice(h.answer_text)],
            key=lambda h: h.confidence,
            reverse=True,
        )
        fallback = ranked[0].answer if ranked else (hypotheses[0].answer if hypotheses else "")
        return YOHASBenchResult(
            predicted=fallback,
            confidence=0.3,
            hypotheses=hypotheses,
            reasoning="\n\n".join(all_reasoning) + "\n\n[TIMEOUT — returning best guess]",
            tokens_used=total_tokens,
            duration_ms=_now_ms() - start_ms,
            phases_completed=phases,
        )

    except Exception as exc:
        logger.error("YOHAS evaluation failed: %s", exc, exc_info=True)
        return YOHASBenchResult(
            predicted="",
            confidence=0.0,
            hypotheses=hypotheses,
            reasoning="\n\n".join(all_reasoning) + f"\n\n[ERROR: {exc}]",
            tokens_used=total_tokens,
            duration_ms=_now_ms() - start_ms,
            phases_completed=phases,
        )


# ---------------------------------------------------------------------------
# Convenience wrapper for LAB-Bench MCQ format
# ---------------------------------------------------------------------------


async def evaluate_mcq_with_yohas(
    question_text: str,
    correct_answer_text: str,
    distractor_texts: list[str],
    correct_letter: str,
    choices_formatted: list[str],  # ["(A) ...", "(B) ...", ...]
    *,
    llm: Any,
    model: str | None = None,
    subtask: str = "",
    enable_falsification: bool = True,
    timeout_seconds: int = 180,
) -> YOHASBenchResult:
    """Convenience wrapper for LAB-Bench style MCQ evaluation.

    Parses the formatted choices list into (letter, text) tuples and
    delegates to evaluate_with_yohas().
    """
    parsed_choices: list[tuple[str, str]] = []
    for choice_str in choices_formatted:
        m = re.match(r"\(([A-H])\)\s*(.*)", choice_str)
        if m:
            parsed_choices.append((m.group(1), m.group(2)))
        else:
            # Fallback: just use the whole string
            parsed_choices.append(("?", choice_str))

    return await evaluate_with_yohas(
        question=question_text,
        choices=parsed_choices,
        correct_letter=correct_letter,
        llm=llm,
        model=model,
        subtask=subtask,
        enable_falsification=enable_falsification,
        timeout_seconds=timeout_seconds,
    )
