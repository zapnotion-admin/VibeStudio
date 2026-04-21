"""
engine/workflow.py
3-stage pipeline with model specialisation:

  Stage 1 (Scan)   — qwen3-coder     Fast file analysis, issue identification
  Stage 2 (Reason) — deepseek-reasoner  Deep reasoning, diagnosis, planning
  Stage 3 (Code)   — qwen3-coder     Code execution from the reasoned plan
  Stage 4 (Review) — qwen3-coder     Final review (same model, already warm)
  Stage 5 (Retry)  — qwen3-coder     Only if reviewer flags issues

VRAM discipline:
  - Only one model loaded at a time.
  - explicit unload_model() call before every model swap.
  - ensure_model() skips warmup if the model is already hot.
  - cancel_check honoured before every stage transition.
"""

import re
from engine.ollama_client import single_response, ensure_model, unload_model, resolve_model
from engine.logger import log
from engine.brief import read_brief, format_brief_for_prompt
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
    return re.findall(r"```.*?\n(.*?)```", text, re.DOTALL)


def _swap_to(model: str, display_name: str, emit, cancel_check) -> bool:
    """
    Unload current model from VRAM then warm up the target model.
    Returns False if cancelled before the swap completes.
    """
    if cancel_check():
        return False
    log(f"[pipeline] VRAM swap → {model}")
    unload_model()          # evict whatever is currently in VRAM
    ensure_model(model)     # load the next model (skipped if already warm)
    return not cancel_check()


def run_pipeline(
    task: str,
    file_context: str,
    project_dir: str = "",
    progress_callback=None,
    cancel_check=None,
) -> dict:
    """
    Executes the 3-model pipeline:
      Scan (Qwen) → Reason (DeepSeek) → Code (Qwen) → Review (Qwen)

    progress_callback(stage, text) stages:
      status, scan_done, plan_done, code_done, review_done, retry_done

    Returns {} if cancelled at any point before completion.
    """

    def emit(stage: str, text: str) -> None:
        log(f"[pipeline] stage={stage} len={len(text)}")
        if progress_callback:
            progress_callback(stage, text)

    def is_cancelled() -> bool:
        return cancel_check is not None and cancel_check()

    coder    = resolve_model(MODEL_CODER,    MODEL_FALLBACK)
    reasoner = resolve_model(MODEL_REASONER, MODEL_FALLBACK)

    files_section = "FILES:\n" + file_context if file_context else "No files provided."

    # Load project brief — gives every stage persistent context across runs
    brief_content = read_brief(project_dir) if project_dir else ""
    brief_section = format_brief_for_prompt(brief_content)

    # Extract explicit target filename from task if specified
    # e.g. "save it as calculator.py", "save as foo.py", "call it bar.py"
    import re as _re
    _explicit = _re.search(
        r"(?:save (?:it )?as|save to|call it|name it|file(?:name)? (?:is )?)\s+([\w\-]+\.\w+)",
        task, _re.IGNORECASE
    ) or _re.search(
        r"([\w\-]+\.(?:py|js|ts|html|css|sh|json|rs|cpp|c|java|go|rb|php))", task
    )
    target_file = _explicit.group(1).strip() if _explicit else None
    target_instruction = (
        f"\nIMPORTANT: The output file MUST be named '{target_file}'. "
        f"Use 'FILE: {target_file}' in your output. Do NOT use any other filename."
    ) if target_file else ""

    # ──────────────────────────────────────────────────────────────────
    # STAGE 1: SCAN — Qwen3 reads the files, identifies issues
    # ──────────────────────────────────────────────────────────────────
    emit("status", "🔍 Scanning files...")
    ensure_model(coder)
    if is_cancelled():
        return {}

    scan_prompt = f"""
TASK: {task}{target_instruction}
{brief_section}{files_section}

You are a code scanner. Do NOT write any fixes yet.

Identify:
1. Which functions / classes are directly relevant to this task
2. Any existing bugs, risks, or edge cases in those areas
3. Dependencies between components that the task will affect

Cross-file consistency checks (do these for every multi-file project):
- Are all defined functions actually called somewhere? Flag dead code.
- Do function signatures match how they are called, including framework callback patterns?
- Are constants, allowed values, or data formats consistent across files?
  (e.g. if file A validates input to a set of chars, file B must produce only those chars)
- Are imports consistent with what each file actually uses?

Output structured findings:
- File: <filename>
- Function/Class: <n>
- Issue: <description>
- Relevant lines: <brief quote or line range>

Mark cross-file issues with [CROSS-FILE]. Be precise and concise.
""".strip()

    scan = single_response(coder, scan_prompt, num_ctx=MAX_CTX_CODER)
    emit("scan_done", scan)

    # ──────────────────────────────────────────────────────────────────
    # STAGE 2: REASON — DeepSeek-R1 reasons over the findings → plan
    # VRAM swap: unload Qwen, load DeepSeek
    # ──────────────────────────────────────────────────────────────────
    emit("status", "🧠 Reasoning...")
    if not _swap_to(reasoner, "DeepSeek-R1", emit, is_cancelled):
        return {}

    reason_prompt = f"""
TASK: {task}{target_instruction}
{brief_section}{CONSTRAINTS}

SCAN FINDINGS (from static analysis):
{scan}

{files_section}

You are a reasoning model. Using the scan findings:
1. Diagnose the root cause of any issues
2. Identify the safest, minimal solution
3. Produce a numbered execution plan — each step: file, change, reason
4. Flag any risks or ordering dependencies

Be thorough. A coding model will execute your plan literally — ambiguity causes bugs.
""".strip()

    plan = single_response(reasoner, reason_prompt, num_ctx=MAX_CTX_REASONER)
    emit("plan_done", plan)

    # ──────────────────────────────────────────────────────────────────
    # STAGE 3: CODE — Qwen3 executes the reasoned plan
    # VRAM swap: unload DeepSeek, load Qwen
    # ──────────────────────────────────────────────────────────────────
    emit("status", "⚙️ Writing code...")
    if not _swap_to(coder, "Qwen3-Coder", emit, is_cancelled):
        return {}

    code_prompt = f"""
TASK: {task}{target_instruction}
{brief_section}{CONSTRAINTS}
EXECUTION PLAN:
{plan}

{files_section}

Execute the plan exactly. Output EVERY changed or created file using this STRICT format:

FILE: relative/path/to/file.py
```python
<complete file contents here>
```

Rules:
- Always use the FILE: line before each code block
- Always show the COMPLETE file — never partial or diff output
- Use the correct language tag (python, js, etc.)
- No explanations outside the FILE blocks
- If creating a new file, use the path relative to the project root

Write production-quality code. Do not deviate from the plan.
""".strip()

    code = single_response(coder, code_prompt, num_ctx=MAX_CTX_CODER)
    emit("code_done", code)

    # ──────────────────────────────────────────────────────────────────
    # STAGE 4: REVIEW — Qwen3 reviews (already warm, no swap needed)
    # ──────────────────────────────────────────────────────────────────
    if is_cancelled():
        return {}

    emit("status", "✅ Reviewing...")

    review_prompt = f"""
You are a code reviewer. Review this AI-generated code.

ORIGINAL TASK: {task}
PLAN:
{plan}
CODE OUTPUT:
{code}

Check ALL of the following:
- Correctness: does the code do what the task asks?
- Logic errors: off-by-one, wrong conditions, incorrect operator precedence
- Function signatures: does every function definition match exactly how it is called?
  Pay special attention to callback/handler functions registered with frameworks
  (e.g. tkinter validatecommand requires specific argument patterns)
- Dead code: is every defined function actually called somewhere? Flag unused functions.
- Cross-file consistency: if multiple files are present, do they agree on data formats,
  allowed values, and interfaces? (e.g. if one file validates input to a set of values,
  the other file must produce/accept exactly those values)
- Missing error handling: unhandled exceptions, missing edge cases
- Security issues: unsafe eval, injection risks, path traversal

Respond in this exact format:
VERDICT: PASS / NEEDS_CHANGES / FAIL
ISSUES: (list each issue on its own line, or "None")
SUGGESTIONS: (specific improvements, or "None")
""".strip()

    review  = single_response(coder, review_prompt, num_ctx=MAX_CTX_CODER)
    verdict = extract_verdict(review)
    emit("review_done", f"VERDICT: {verdict}\n\n{review}")

    final_code = code

    # ──────────────────────────────────────────────────────────────────
    # STAGE 5: RETRY — only if reviewer flagged issues (Qwen already warm)
    # ──────────────────────────────────────────────────────────────────
    if verdict in ("NEEDS_CHANGES", "FAIL"):
        if is_cancelled():
            return {}

        emit("status", "🔁 Fixing reviewer issues...")

        # Extract which files were produced in the code stage so we can tell
        # the retry model exactly what files it must output
        import re as _re2
        code_files = _re2.findall(r"FILE:\s*([^\n]+)\n```", code)
        file_list = "\n".join(f"- FILE: {f.strip()}" for f in code_files) if code_files else "- same files as in the previous code"

        retry_prompt = f"""
The previous code had issues flagged by a reviewer. Produce a complete corrected version.

ORIGINAL TASK: {task}{target_instruction}
REVIEWER FEEDBACK:
{review}
ORIGINAL PLAN:
{plan}
PREVIOUS CODE:
{code}

You MUST output ALL of the following files in their COMPLETE corrected form:
{file_list}

Use this STRICT format for EVERY file — no exceptions:

FILE: filename.py
```python
<complete file contents — every line, no omissions>
```

Rules:
- One FILE: block per file listed above
- Complete file contents only — never partial, never snippets, never diffs
- No explanations, no commentary outside the FILE blocks
- If a file needs no changes, output it anyway unchanged
""".strip()

        final_code = single_response(coder, retry_prompt, num_ctx=MAX_CTX_CODER)
        emit("retry_done", final_code)

    # Validate retry output — must contain structured FILE: blocks matching the code stage.
    # If retry only produced prose/snippets, fall back to the code stage output which
    # we know is complete and correctly formatted.
    if final_code != code:
        import re as _re
        retry_file_count = len(_re.findall(r"FILE:\s*[^\n]+\n```", final_code))
        code_file_count  = len(_re.findall(r"FILE:\s*[^\n]+\n```", code))
        if retry_file_count == 0 or retry_file_count < code_file_count:
            log(f"[pipeline] retry had {retry_file_count} FILE: blocks vs code stage {code_file_count} — falling back")
            final_code = code

    log(f"[pipeline] completed. verdict={verdict}")
    return {
        "scan":       scan,
        "plan":       plan,
        "code":       code,
        "review":     review,
        "verdict":    verdict,
        "final_code": final_code,
    }
