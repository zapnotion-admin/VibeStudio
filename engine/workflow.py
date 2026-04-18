"""
engine/workflow.py
Planner → Coder → Reviewer pipeline.

Used only in Code Mode when core.context.is_task_prompt() returns True.
All calls happen inside PipelineWorker (a QThread) — never from the UI thread.

Stage model assignments:
  Stage 1 (Plan)   — deepseek-reasoner  (or fallback)
  Stage 2 (Code)   — qwen3-coder        (or fallback)
  Stage 3 (Review) — qwen3-coder        (reuses already-warmed model)
  Stage 4 (Retry)  — qwen3-coder        (only if verdict != PASS)

[v3.2] cancel_check: callable injected by PipelineWorker, checked before
every stage transition. Returns True when the user has hit [Stop].
"""

import re
from engine.ollama_client import single_response, ensure_model, resolve_model
from engine.logger import log
from core.config import (
    MODEL_CODER, MODEL_REASONER, MODEL_FALLBACK,
    MAX_CTX_CODER, MAX_CTX_REASONER,
)

CONSTRAINTS = """
CONSTRAINTS (follow always):
- Make minimal changes. Only modify what the task explicitly requires.
- Do not introduce new dependencies unless asked.
- Follow the coding style visible in the provided files.
- Do not modify files outside the stated scope.
"""


def extract_verdict(text: str) -> str:
    """
    Robustly parses the reviewer's VERDICT from unstructured model output.
    Handles markdown bold markers (**PASS**), extra whitespace, and
    mixed-case variants.
    Falls back to NEEDS_CHANGES — safer than PASS — when parsing fails.
    """
    m = re.search(
        r"VERDICT[\s:]+\*{0,2}(PASS|NEEDS_CHANGES|FAIL)\*{0,2}",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    upper = text.upper()
    if "NEEDS_CHANGES" in upper or "NEEDS CHANGES" in upper:
        return "NEEDS_CHANGES"
    if "FAIL" in upper:
        return "FAIL"
    if "PASS" in upper:
        return "PASS"
    return "NEEDS_CHANGES"


def extract_code_blocks(text: str) -> list[str]:
    """Pulls fenced code blocks out of model output for clean display."""
    return re.findall(r"```.*?\n(.*?)```", text, re.DOTALL)


def run_pipeline(
    task: str,
    file_context: str,
    progress_callback=None,
    cancel_check=None,
) -> dict:
    """
    Executes the 3-stage (+ optional retry) pipeline.

    Args:
        task:              The user's raw request string.
        file_context:      Pre-assembled file content string from context.py.
        progress_callback: Called as callback(stage: str, text: str) at each
                           stage. Stage values: status, plan_done, code_done,
                           review_done, retry_done. Matches the mapping in
                           ui/main_window._on_pipeline_progress().
        cancel_check:      [v3.2] Zero-arg callable — returns True when the
                           user has requested cancellation. Checked before
                           every stage transition. Pass lambda: False (or None)
                           to disable cancellation.

    Returns:
        dict with keys: plan, code, review, verdict, final_code.
        Returns {} immediately if cancelled before any stage completes.
    """

    def emit(stage: str, text: str) -> None:
        log(f"[pipeline] stage={stage} len={len(text)}")
        if progress_callback:
            progress_callback(stage, text)

    def is_cancelled() -> bool:
        return cancel_check is not None and cancel_check()

    # Resolve both models up front — guard against missing custom models.
    # resolve_model() checks /api/tags and falls back to qwen3:14b if needed.
    reasoner = resolve_model(MODEL_REASONER, MODEL_FALLBACK)
    coder    = resolve_model(MODEL_CODER,    MODEL_FALLBACK)

    # ------------------------------------------------------------------
    # STAGE 1: PLAN — DeepSeek-R1 analyses the task and produces a plan
    # ------------------------------------------------------------------
    if is_cancelled():
        log("[pipeline] Cancelled before Stage 1")
        return {}

    emit("status", "🧠 Planning with DeepSeek-R1...")
    ensure_model(reasoner)

    files_section = "FILES:\n" + file_context if file_context else "No files provided."
    plan_prompt = f"""
TASK: {task}
{CONSTRAINTS}
{files_section}

Produce:
1. Analysis of the task (2-4 sentences)
2. Numbered execution plan — each step: which file, what change, why
3. Risks or edge cases to watch

Be precise. Another AI will execute this plan literally.
""".strip()

    plan = single_response(reasoner, plan_prompt, num_ctx=MAX_CTX_REASONER)
    emit("plan_done", plan)

    # ------------------------------------------------------------------
    # STAGE 2: CODE — Qwen3 executes the plan
    # ------------------------------------------------------------------
    if is_cancelled():
        log("[pipeline] Cancelled before Stage 2")
        return {}

    emit("status", "⚙️ Writing code with Qwen3...")
    ensure_model(coder)  # Unloads DeepSeek, loads Qwen3 (cache handles no-op if same)

    files_section_optional = "FILES:\n" + file_context if file_context else ""
    code_prompt = f"""
TASK: {task}
{CONSTRAINTS}
PLAN TO EXECUTE:
{plan}
{files_section_optional}

Execute the plan exactly. For each changed file:
1. State the filename
2. Show the complete modified file (or a clearly marked diff block)
3. One-line explanation of what changed

Write production-quality code.
""".strip()

    code = single_response(coder, code_prompt, num_ctx=MAX_CTX_CODER)
    emit("code_done", code)

    # ------------------------------------------------------------------
    # STAGE 3: REVIEW — Qwen3 reviews its own output
    # ------------------------------------------------------------------
    if is_cancelled():
        log("[pipeline] Cancelled before Stage 3")
        return {}

    emit("status", "🔍 Reviewing...")

    review_prompt = f"""
You are a code reviewer. Review this AI-generated code.

ORIGINAL TASK: {task}
PLAN:
{plan}
CODE OUTPUT:
{code}

Check: correctness, logic errors, style consistency, missing error handling, security issues.

Respond in this exact format:
VERDICT: PASS / NEEDS_CHANGES / FAIL
ISSUES: (list, or "None")
SUGGESTIONS: (specific improvements, or "None")
""".strip()

    review  = single_response(coder, review_prompt, num_ctx=MAX_CTX_CODER)
    verdict = extract_verdict(review)
    emit("review_done", f"VERDICT: {verdict}\n\n{review}")

    final_code = code

    # ------------------------------------------------------------------
    # STAGE 4: RETRY — only if reviewer flagged issues
    # ------------------------------------------------------------------
    if verdict in ("NEEDS_CHANGES", "FAIL"):
        if is_cancelled():
            log("[pipeline] Cancelled before Stage 4 (retry)")
            return {}

        emit("status", "🔁 Reviewer flagged issues. Fixing...")

        retry_prompt = f"""
The previous code attempt had issues. Fix ONLY what was flagged.

ORIGINAL TASK: {task}
REVIEWER FEEDBACK:
{review}
ORIGINAL PLAN:
{plan}
PREVIOUS CODE (fix ONLY the flagged issues):
{code}
""".strip()

        final_code = single_response(coder, retry_prompt, num_ctx=MAX_CTX_CODER)
        emit("retry_done", final_code)

    log(f"[pipeline] completed. verdict={verdict}")
    return {
        "plan":       plan,
        "code":       code,
        "review":     review,
        "verdict":    verdict,
        "final_code": final_code,
    }
