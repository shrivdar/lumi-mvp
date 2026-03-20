"""CodeAct execution loop for benchmark evaluation.

Implements the Plan-Code-Execute-Observe-Iterate pattern used by
top benchmark systems (Biomni, K-Dense).

The loop:
  Step 1: PLAN   -- Agent receives question + data, produces a plan
  Step 2: CODE   -- Agent writes code to execute step 1 of the plan
  Step 3: EXECUTE -- Code runs, output captured
  Step 4: OBSERVE -- Agent sees the output, decides what to do next
  Step 5: ITERATE -- Agent writes code for next step, or produces final answer
  ... repeat steps 2-5 until answer or max steps

Key design principles (from Biomni / K-Dense):
  1. Short conversation history -- summarize old steps, only show last 3 in full
  2. Plan first -- always start with a plan before coding
  3. Error recovery -- when code fails, agent sees error and fixes it
  4. Force answer -- if max steps reached, force an answer from available evidence
  5. Code-first -- the default action is writing code, not reasoning in text
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CodeActStep:
    """A single step in the CodeAct loop."""

    step_number: int
    action: str  # "plan", "code", "answer", "think"
    content: str  # the plan text, code, or answer
    output: str = ""  # execution output (for code steps)
    error: str = ""  # error message if execution failed
    tokens: int = 0
    duration_ms: int = 0


@dataclass
class CodeActResult:
    """Result of a complete CodeAct evaluation."""

    answer: str
    confidence: float
    steps: list[CodeActStep]
    total_tokens: int
    total_duration_ms: int
    code_executions: int
    code_errors: int
    plan: str


# ---------------------------------------------------------------------------
# History summarisation
# ---------------------------------------------------------------------------


def _summarize_steps(steps: list[CodeActStep]) -> str:
    """Summarize the execution history concisely for the next iteration.

    CRITICAL: Keep this short. Long histories confuse the model.
    Only include the LAST 3 steps in full, summarize earlier ones.
    """
    if len(steps) <= 3:
        parts: list[str] = []
        for s in steps:
            if s.action == "code":
                parts.append(
                    f"[Step {s.step_number}] Code:\n```\n{s.content[:500]}\n```\n"
                    f"Output: {s.output[:500]}"
                )
            elif s.action == "plan":
                parts.append(f"[Plan] {s.content[:300]}")
            elif s.action == "answer":
                parts.append(f"[Answer] {s.content}")
            else:
                parts.append(f"[Think] {s.content[:300]}")
        return "\n\n".join(parts)

    # Summarize older steps
    summary = f"Previous steps (summarized): Executed {len(steps) - 3} steps. "
    code_steps = [s for s in steps[:-3] if s.action == "code"]
    if code_steps:
        summary += f"Ran {len(code_steps)} code blocks. "
        errors = [s for s in code_steps if s.error or "Error" in s.output]
        if errors:
            summary += f"{len(errors)} had errors. "
    summary += "\n\n"

    # Include last 3 steps in full
    for s in steps[-3:]:
        if s.action == "code":
            summary += (
                f"[Step {s.step_number}] Code:\n```\n{s.content[:500]}\n```\n"
                f"Output: {s.output[:500]}\n\n"
            )
        elif s.action == "answer":
            summary += f"[Answer] {s.content}\n\n"
        else:
            summary += f"[{s.action.title()}] {s.content[:300]}\n\n"

    return summary


# ---------------------------------------------------------------------------
# Main CodeAct loop
# ---------------------------------------------------------------------------


async def codeact_evaluate(
    question: str,
    context: str,  # data file listings, capsule info, etc.
    *,
    llm: Any,
    model: str,
    execute_fn: Callable[[str], str],  # code execution function
    max_steps: int = 10,
    max_code_retries: int = 3,
    timeout_seconds: int = 180,
    system_prompt: str = "",
    helper_functions_doc: str = "",  # documentation for pre-built helpers
) -> CodeActResult:
    """Run a CodeAct evaluation loop.

    Parameters
    ----------
    question:
        The question to answer.
    context:
        Supporting context (file listings, data previews, capsule info).
    llm:
        An LLM client with an ``async query(prompt, *, system_prompt, model)``
        method that returns an object with ``.text`` and ``.call_tokens``.
    model:
        Model identifier to pass through to the LLM client.
    execute_fn:
        Synchronous function ``(code: str) -> str`` that executes Python code
        and returns captured stdout/stderr.
    max_steps:
        Maximum number of iterate steps (excluding the initial plan step).
    max_code_retries:
        (Reserved) Max consecutive retries for the same code error.
    timeout_seconds:
        Wall-clock timeout for the entire evaluation.
    system_prompt:
        System prompt passed to the LLM on every call.
    helper_functions_doc:
        Documentation string for pre-built helper functions available in the
        execution namespace. Injected into the plan prompt so the agent knows
        what tools it can call.

    Returns
    -------
    CodeActResult with the final answer and full execution trace.
    """
    steps: list[CodeActStep] = []
    total_tokens = 0
    start = time.monotonic()
    code_executions = 0
    code_errors = 0

    # ---- Step 1: PLAN + first code block ----
    plan_parts = [question]
    if context:
        plan_parts.append(context)
    if helper_functions_doc:
        plan_parts.append(helper_functions_doc)

    plan_parts.append(
        "Create a step-by-step plan to answer this question. For each step:\n"
        "1. State what you need to find out\n"
        "2. What code you'll write to find it\n"
        "3. What you expect the output to look like\n\n"
        "Keep the plan to 3-5 steps maximum. Be specific about which "
        "functions/APIs to use.\n\n"
        "Format:\n"
        "PLAN:\n"
        "Step 1: [description]\n"
        "Step 2: [description]\n"
        "...\n\n"
        "Then immediately write the code for Step 1 in <code> tags:\n"
        "<code>\n"
        "your_code_here\n"
        "</code>"
    )

    plan_prompt = "\n\n".join(plan_parts)

    plan_response = await llm.query(
        plan_prompt, system_prompt=system_prompt, model=model
    )
    total_tokens += plan_response.call_tokens

    plan_text = plan_response.text
    code_match = re.search(r"<code>(.*?)</code>", plan_text, re.DOTALL)

    plan_step = CodeActStep(
        step_number=0,
        action="plan",
        content=plan_text[:1000],
        tokens=plan_response.call_tokens,
    )
    steps.append(plan_step)

    # Execute the first code block if present
    if code_match:
        code = code_match.group(1).strip()
        output = execute_fn(code)
        code_executions += 1
        has_error = "Error" in output or "Traceback" in output
        if has_error:
            code_errors += 1
        steps.append(
            CodeActStep(
                step_number=1,
                action="code",
                content=code,
                output=output[:2000],
                error=output[:500] if has_error else "",
            )
        )

    # ---- Steps 2-N: ITERATE ----
    conversation_summary = _summarize_steps(steps)

    for step_num in range(2, max_steps + 1):
        elapsed = time.monotonic() - start
        if elapsed > timeout_seconds:
            break

        iterate_prompt = (
            f"You are continuing your analysis of this question:\n"
            f"{question}\n\n"
            f"Progress so far:\n"
            f"{conversation_summary}\n\n"
            f"What's your next step? Either:\n"
            f"1. Write more code in <code> tags to continue the analysis\n"
            f"2. Provide your final answer in <answer> tags: "
            f"<answer>YOUR_ANSWER</answer>\n\n"
            f"If your previous code had an error, fix it and try again.\n"
            f"If you have enough information, provide your answer now."
        )

        response = await llm.query(
            iterate_prompt, system_prompt=system_prompt, model=model
        )
        total_tokens += response.call_tokens

        # Check for final answer
        answer_match = re.search(
            r"<answer>(.*?)</answer>", response.text, re.DOTALL
        )
        if answer_match:
            answer = answer_match.group(1).strip()
            steps.append(
                CodeActStep(
                    step_number=step_num,
                    action="answer",
                    content=answer,
                    tokens=response.call_tokens,
                )
            )
            break

        # Check for code
        code_match = re.search(
            r"<code>(.*?)</code>", response.text, re.DOTALL
        )
        if code_match:
            code = code_match.group(1).strip()
            output = execute_fn(code)
            code_executions += 1
            has_error = "Error" in output or "Traceback" in output
            if has_error:
                code_errors += 1
            steps.append(
                CodeActStep(
                    step_number=step_num,
                    action="code",
                    content=code,
                    output=output[:2000],
                    error=output[:500] if has_error else "",
                    tokens=response.call_tokens,
                )
            )
            conversation_summary = _summarize_steps(steps)
        else:
            # Reasoning step (no code, no answer)
            steps.append(
                CodeActStep(
                    step_number=step_num,
                    action="think",
                    content=response.text[:1000],
                    tokens=response.call_tokens,
                )
            )
            conversation_summary = _summarize_steps(steps)

    # ---- Extract final answer ----
    final_answer = ""
    for step in reversed(steps):
        if step.action == "answer":
            final_answer = step.content
            break

    if not final_answer:
        # Force an answer from the agent
        force_prompt = (
            f"Based on your analysis so far:\n"
            f"{conversation_summary}\n\n"
            f"You MUST provide a final answer now. What is your answer?\n"
            f"<answer>YOUR_ANSWER</answer>"
        )
        force_response = await llm.query(
            force_prompt, system_prompt=system_prompt, model=model
        )
        total_tokens += force_response.call_tokens
        answer_match = re.search(
            r"<answer>(.*?)</answer>", force_response.text, re.DOTALL
        )
        if answer_match:
            final_answer = answer_match.group(1).strip()
        else:
            # Last resort: take the last line of the response
            final_answer = force_response.text.strip().split("\n")[-1].strip()

        steps.append(
            CodeActStep(
                step_number=len(steps),
                action="answer",
                content=final_answer,
                tokens=force_response.call_tokens,
            )
        )

    total_duration_ms = int((time.monotonic() - start) * 1000)

    return CodeActResult(
        answer=final_answer,
        confidence=0.8 if code_errors == 0 else 0.5,
        steps=steps,
        total_tokens=total_tokens,
        total_duration_ms=total_duration_ms,
        code_executions=code_executions,
        code_errors=code_errors,
        plan=steps[0].content if steps else "",
    )
