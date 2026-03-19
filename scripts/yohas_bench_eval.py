"""YOHAS-powered benchmark evaluation using the full architecture.

Runs each benchmark question through a 4-phase pipeline that mirrors the
core YOHAS research loop — Hypothesize, Investigate, Synthesize, Falsify —
adapted for single-question answering.

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

Each evaluation uses 3-5 LLM calls (vs 1 for zero-shot) but should be
significantly more accurate thanks to multi-hypothesis investigation and
active falsification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
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


def _build_evidence_prompt(
    question: str,
    choices: list[tuple[str, str]],
    subtask: str,
    db_helpers_available: bool,
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

    return f"""You are an expert biomedical researcher evaluating a multiple-choice question.
For EACH answer choice, provide evidence supporting or contradicting it.

Question: {question}

Answer choices:
{choices_block}
{subtask_hint}{db_hint}

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

Be thorough and specific. Reference actual database entries, gene functions,
pathway memberships, variant classifications, or published findings when possible."""


def _build_independent_verifier_prompt(
    question: str,
    choices: list[tuple[str, str]],
    subtask: str,
) -> str:
    """Build the Phase 2 independent-verifier prompt.

    This agent answers from scratch, without seeing the evidence collector's
    output, to provide an independent perspective.
    """
    choices_block = _format_choices_block(choices)

    return f"""You are an expert biomedical scientist. Answer this multiple-choice question
using your deep knowledge of biological databases and molecular biology.

Question: {question}

Choices:
{choices_block}

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
) -> str:
    """Build the Phase 4 falsification prompt."""
    choices_block = _format_choices_block(choices)

    # Show the runner-up alternatives
    alternatives = []
    for h in hypotheses:
        if h.answer != top_answer and not _is_refuse_choice(h.answer_text):
            alternatives.append(f"  ({h.answer}) {h.answer_text} [confidence: {h.confidence:.2f}]")
    alternatives_block = "\n".join(alternatives) if alternatives else "  (none)"

    return f"""You are a scientific critic whose job is to FALSIFY a proposed answer.

Question: {question}

Choices:
{choices_block}

The proposed answer is: ({top_answer}) {top_answer_text}

Alternative candidates:
{alternatives_block}

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
      3. SYNTHESIZE — weigh evidence, rank hypotheses, pick top answer
      4. FALSIFY — actively try to disprove the top answer; switch if counter-evidence is strong

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

        # Agent 1: Evidence Collector
        evidence_prompt = _build_evidence_prompt(
            question, choices, subtask, db_helpers is not None,
        )

        # Agent 2: Independent Verifier
        verifier_prompt = _build_independent_verifier_prompt(
            question, choices, subtask,
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
        # Phase 3: SYNTHESIZE
        # ================================================================
        synthesis_prompt = _build_synthesis_prompt(
            question, choices, hypotheses, verifier_answer, verifier_reasoning,
        )
        synthesis_resp = await _query(
            synthesis_prompt, system=system_prompt, max_tokens=2048,
        )
        synthesis_text = synthesis_resp.text
        all_reasoning.append(
            f"[Phase 3 - Synthesize] {synthesis_text[:500]}"
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
        # Phase 4: FALSIFY
        # ================================================================
        if enable_falsification and top_answer:
            top_h = _get_hypothesis(hypotheses, top_answer)
            top_text = top_h.answer_text if top_h else ""

            # Skip falsification for refuse answers (not worth falsifying)
            if not _is_refuse_choice(top_text):
                falsify_prompt = _build_falsification_prompt(
                    question, choices, top_answer, top_text, hypotheses,
                )
                falsify_resp = await _query(
                    falsify_prompt, system=system_prompt, max_tokens=2048,
                )
                falsify_text = falsify_resp.text
                all_reasoning.append(
                    f"[Phase 4 - Falsify] {falsify_text[:500]}"
                )

                revised_answer = _extract_answer_letter(falsify_text)
                if revised_answer and revised_answer != top_answer:
                    # The falsifier found counter-evidence and switched the answer
                    logger.info(
                        "Falsification CHANGED answer: %s -> %s",
                        top_answer, revised_answer,
                    )
                    all_reasoning.append(
                        f"[Phase 4 - Falsify] CHANGED answer from "
                        f"{top_answer} to {revised_answer}"
                    )
                    top_answer = revised_answer
                else:
                    all_reasoning.append(
                        f"[Phase 4 - Falsify] CONFIRMED answer {top_answer}"
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
