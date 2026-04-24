"""
engine/workflow.py
Agent pipeline: SCAN → REASON → DECOMPOSE → [STEP LOOP] → FINAL REVIEW

Stage model assignments:
  Scan     — coder model    Fast file analysis + cross-file consistency
  Reason   — reasoner model Structured step plan with dependencies
  Execute  — coder model    One step at a time, re-reads files between steps
  Review   — coder model    Final review of all written files
  Retry    — coder model    Only if reviewer flags issues (single-shot fallback)

v3 fixes:
  - SCAN: When no context files, truly skips scan (no model call, no hallucination)
  - SCAN prompt: explicitly asks model to check cross-file references
    (are JS files referenced in HTML? are imports correct?)
  - REASON prompt: must keep all changes in fewest possible files;
    discourages splitting across invented files
  - REASON prompt: for web projects, warns that JS files must be <script>-linked in HTML
  - REVIEW: explicitly checks for unreferenced/orphan files
  - append_run_summary wired after completion
"""

import re
import os
from engine.ollama_client import single_response, ensure_model, unload_model, resolve_model
from engine.logger import log
from engine.brief import read_brief, format_brief_for_prompt, append_run_summary
from engine.project_map import build_project_map_section, update_summaries
from engine.plan_parser import parse_steps, extract_plan_summary
from engine.step_state import StepState
from engine.step_executor import run_steps
from core.config import (
    MODEL_CODER, MODEL_REASONER, MODEL_FALLBACK,
    MAX_CTX_CODER, MAX_CTX_REASONER,
)


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


def _swap_to(model: str, emit, cancel_check) -> bool:
    if cancel_check():
        return False
    log(f"[pipeline] VRAM swap → {model}")
    unload_model()
    ensure_model(model)
    return not cancel_check()


def run_pipeline(
    task:              str,
    file_context:      str,
    project_dir:       str = "",
    context_files:     list = None,
    coder_model:       str = None,
    reasoner_model:    str = None,
    stable_mode:       bool = True,
    progress_callback  = None,
    cancel_check:      object = None,
) -> dict:

    def emit(stage: str, text: str) -> None:
        log(f"[pipeline] stage={stage} len={len(text)}")
        if progress_callback:
            progress_callback(stage, text)

    def is_cancelled() -> bool:
        return cancel_check is not None and cancel_check()

    coder    = resolve_model(coder_model    or MODEL_CODER,    MODEL_FALLBACK)
    reasoner = resolve_model(reasoner_model or MODEL_REASONER, MODEL_FALLBACK)
    log(f"[pipeline] coder={coder}  reasoner={reasoner}  stable={stable_mode}")

    mode_label = "⚡ Stable" if stable_mode else "⚡ Fast"
    emit("status", f"{mode_label} mode  |  Coder: {coder}  |  Reasoner: {reasoner}")

    files_section = "FILES:\n" + file_context if file_context else "No files provided."
    brief_content    = read_brief(project_dir) if project_dir else ""
    brief_section    = format_brief_for_prompt(brief_content)

    project_map_section = build_project_map_section(
        project_dir, exclude_files=context_files or []
    ) if project_dir else ""
    if project_map_section:
        brief_section = brief_section + "\n" + project_map_section + "\n"

    _explicit = re.search(
        r"(?:save (?:it )?as|save to|call it|name it|file(?:name)? (?:is )?)\s+([\w\-]+\.\w+)",
        task, re.IGNORECASE
    ) or re.search(
        r"\b([\w\-]+\.(?:py|js|ts|html|css|sh|json|rs|cpp|c|java|go|rb|php))\b", task
    )
    target_file = _explicit.group(1).strip() if _explicit else None
    target_instruction = (
        f"\nIMPORTANT: The output file MUST be named '{target_file}'. "
        f"Use 'FILE: {target_file}' in your output. Do NOT use any other filename."
    ) if target_file else ""

    # ──────────────────────────────────────────────────────────────────
    # STAGE 1: SCAN
    # Only runs when there are actual files to scan.
    # If skipped, scan = sentinel string — no model call, no hallucination.
    # ──────────────────────────────────────────────────────────────────
    has_files = bool(
        file_context
        and file_context.strip()
        and file_context.strip() != "No files provided."
    )

    if has_files:
        emit("status", "🔍 Scanning files...")
        ensure_model(coder)
        if is_cancelled():
            return {}

        # Build list of context file names for the cross-reference check
        ctx_filenames = [os.path.basename(f) for f in (context_files or [])]
        ctx_list = ", ".join(ctx_filenames) if ctx_filenames else "(none listed)"

        scan_prompt = f"""
TASK: {task}{target_instruction}
{brief_section}{files_section}

You are a code scanner. Your job is to find REAL problems in the existing files above.

IMPORTANT: Check these specific things:
1. Logic bugs visible in the code (wrong variable names, off-by-one errors, etc.)
2. Cross-file references: if file A imports or uses file B, does file B exist and export what's needed?
3. For HTML projects: are all JavaScript files referenced via <script src="..."> tags?
   Context files provided: {ctx_list}
   Flag any JS files that contain game/app logic but are NOT linked in the HTML.
4. Duplicate or contradictory logic across files

Be concise — maximum 15 lines. Only report issues you can see in the actual code.
Format: "File: <name> | Issue: <one line>"
Mark cross-file issues [CROSS-FILE].
If everything looks correct for the task: output "No issues found."
""".strip()

        scan = single_response(coder, scan_prompt, num_ctx=MAX_CTX_CODER)
        emit("scan_done", scan)
    else:
        scan = "(No existing files — creating from scratch)"
        emit("status", "⏭ No files to scan — proceeding to plan...")

    # ──────────────────────────────────────────────────────────────────
    # STAGE 2: REASON
    # ──────────────────────────────────────────────────────────────────
    emit("status", "🧠 Reasoning...")
    if not _swap_to(reasoner, emit, is_cancelled):
        return {}

    # Build the authoritative file list from context (what files actually exist/are relevant)
    ctx_filenames = [os.path.basename(f) for f in (context_files or [])]
    if ctx_filenames:
        file_inventory = (
            "EXISTING FILES IN PROJECT (only modify these unless you must create a new one):\n"
            + "\n".join(f"  - {f}" for f in ctx_filenames)
            + "\n"
        )
    else:
        file_inventory = ""

    reason_prompt = f"""
TASK: {task}{target_instruction}
{brief_section}
{file_inventory}SCAN FINDINGS: {scan}

Break this task into 4-8 steps. Each step = one logical unit of work.
Aim for 30-80 lines of code change per step.

CRITICAL FILE RULES:
- Prefer modifying existing files over creating new ones.
- If you create a new .js file, it MUST be <script src="..."> linked in the HTML in the same plan.
- Do NOT plan steps that write to files that will not be used anywhere.
- Each FILE: must be a file that will actually be part of the running project.
- Consolidate logic: one HTML file + one JS file is better than one HTML + three JS files.

Use EXACTLY this format for EVERY step:

STEP 1: <one clear action sentence>
FILES: <filename.ext>
DEPENDS_ON: none
SUCCESS_CRITERIA: <one observable outcome>

STEP 2: <one clear action sentence>
FILES: <filename.ext>
DEPENDS_ON: STEP 1
SUCCESS_CRITERIA: <one observable outcome>

Rules:
- Output ONLY the STEP blocks — no preamble, no summary
- Each step touches the MINIMUM number of files needed
- New project: step 1 = skeleton only, later steps add features
- SUCCESS_CRITERIA must be specific and observable
""".strip()

    plan = single_response(reasoner, reason_prompt, num_ctx=MAX_CTX_REASONER)
    emit("plan_done", plan)

    # ──────────────────────────────────────────────────────────────────
    # STAGE 3: DECOMPOSE
    # ──────────────────────────────────────────────────────────────────
    steps = parse_steps(plan)

    if not _swap_to(coder, emit, is_cancelled):
        return {}

    if not steps:
        log("[pipeline] No structured steps — falling back to single-step")
        emit("status", "⚙️ Writing code (single pass)...")
        return _single_step_fallback(
            task, plan, file_context, brief_section,
            target_instruction, coder, emit, is_cancelled,
            project_dir, context_files
        )

    emit("status", f"📋 Plan: {len(steps)} steps")

    # ──────────────────────────────────────────────────────────────────
    # STAGE 4: STEP EXECUTION LOOP
    # ──────────────────────────────────────────────────────────────────
    state = StepState(
        project_dir=project_dir,
        task=task,
        steps=steps,
    )

    loop_result = run_steps(
        state             = state,
        task              = task,
        all_context_files = context_files or [],
        project_dir       = project_dir,
        brief_content     = brief_content,
        coder_model       = coder,
        stable_mode       = stable_mode,
        progress_callback = progress_callback,
        cancel_check      = cancel_check,
    )

    if is_cancelled():
        return {}

    completed_files = loop_result.get("completed_files", [])
    diffs           = loop_result.get("diffs", [])
    failed_steps    = loop_result.get("failed_steps", [])

    if diffs:
        emit("code_done", "Files written:\n" + "\n".join(diffs))

    if failed_steps:
        emit("status", f"⚠️ {len(failed_steps)} step(s) skipped: {failed_steps}")

    # ──────────────────────────────────────────────────────────────────
    # STAGE 5: FINAL REVIEW
    # ──────────────────────────────────────────────────────────────────
    if is_cancelled():
        return {}

    emit("status", "✅ Final review...")

    final_file_content = ""
    for fpath in completed_files:
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                rel = os.path.relpath(fpath, project_dir) if project_dir else fpath
                final_file_content += f"\nFILE: {rel}\n```\n{content}\n```\n"
            except Exception:
                pass

    if not final_file_content:
        final_file_content = file_context

    # Build list of all written files for cross-reference check
    written_filenames = [
        os.path.relpath(f, project_dir) if project_dir else f
        for f in completed_files if os.path.exists(f)
    ]
    written_list = "\n".join(f"  - {f}" for f in written_filenames) or "  (none)"

    review_prompt = f"""
You are a code reviewer. Review this AI-generated code.

ORIGINAL TASK: {task}
PLAN:
{extract_plan_summary(steps)}

FILES WRITTEN BY THE PIPELINE:
{written_list}

CODE OUTPUT:
{final_file_content}

Check ALL of the following:
- Correctness: does the code do what the task asks?
- Logic errors: wrong variable names, off-by-one, incorrect conditions
- Function signatures: every definition must match how it is called
- Dead code: every defined function must be called somewhere; flag unused ones
- Orphan files: are ALL written files actually used/referenced?
  For HTML projects: every .js file must have a matching <script src="..."> in the HTML.
  If a .js file exists but is not linked, that is a critical issue.
- Cross-file consistency: files must agree on data formats and interfaces
- Missing error handling: unhandled exceptions, missing edge cases

Respond in this EXACT format:
VERDICT: PASS / NEEDS_CHANGES / FAIL
ISSUES: (list each issue on its own line, or "None")
SUGGESTIONS: (specific improvements, or "None")
""".strip()

    review  = single_response(coder, review_prompt, num_ctx=MAX_CTX_CODER)
    verdict = extract_verdict(review)
    emit("review_done", f"VERDICT: {verdict}\n\n{review}")

    # ──────────────────────────────────────────────────────────────────
    # STAGE 6: RETRY
    # ──────────────────────────────────────────────────────────────────
    final_code = final_file_content

    if verdict in ("NEEDS_CHANGES", "FAIL"):
        if is_cancelled():
            return {}

        emit("status", "🔁 Fixing reviewer issues...")

        code_files = re.findall(r"FILE:\s*([^\n]+)\n```", final_file_content)
        file_list = "\n".join(f"- FILE: {f.strip()}" for f in code_files) if code_files \
                    else "- same files as in the previous code"

        retry_prompt = f"""
The previous code had issues flagged by a reviewer. Produce a complete corrected version.

ORIGINAL TASK: {task}{target_instruction}
REVIEWER FEEDBACK:
{review}
PLAN:
{extract_plan_summary(steps)}
PREVIOUS CODE:
{final_file_content}

You MUST output ALL of the following files in their COMPLETE corrected form:
{file_list}

Use this STRICT format for EVERY file:

FILE: filename.ext
```language
<complete file contents — every line, no omissions>
```

Rules:
- One FILE: block per file listed above
- Complete file contents only — never partial, never snippets
- No explanations outside the FILE blocks
- If a file needs no changes, output it anyway unchanged
- Do NOT create additional files beyond those listed
""".strip()

        retry_output = single_response(coder, retry_prompt, num_ctx=MAX_CTX_CODER)

        import re as _re
        retry_count = len(_re.findall(r"FILE:\s*[^\n]+\n```", retry_output))
        code_count  = len(_re.findall(r"FILE:\s*[^\n]+\n```", final_file_content))
        if retry_count > 0 and retry_count >= code_count:
            final_code = retry_output
            emit("retry_done", retry_output)
        else:
            log(f"[pipeline] retry had {retry_count} FILE: blocks vs {code_count} — keeping original")
            emit("retry_done", "(Retry did not improve output — keeping reviewed version)")

    if completed_files and project_dir:
        update_summaries(project_dir, completed_files)

    if project_dir:
        rel_files = [os.path.relpath(f, project_dir) for f in completed_files if os.path.exists(f)]
        append_run_summary(project_dir, task, rel_files, verdict)

    log(f"[pipeline] completed. verdict={verdict} files={len(completed_files)}")
    return {
        "scan":            scan,
        "plan":            plan,
        "steps":           steps,
        "review":          review,
        "verdict":         verdict,
        "final_code":      final_code,
        "completed_files": completed_files,
        "diffs":           diffs,
    }


def _single_step_fallback(
    task, plan, file_context, brief_section,
    target_instruction, coder, emit, is_cancelled,
    project_dir, context_files
) -> dict:
    emit("status", "⚙️ Writing code...")

    files_section = "FILES:\n" + file_context if file_context else ""

    code_prompt = f"""
TASK: {task}{target_instruction}
EXECUTION PLAN:
{plan}

{files_section}

Execute the plan exactly. Output EVERY changed or created file using this STRICT format:

FILE: relative/path/to/file.ext
```language
<complete file contents here>
```

Rules:
- Always use the FILE: line before each code block
- Always show the COMPLETE file — never partial or diff output
- Use the correct language tag (python, javascript, html, etc.)
- No explanations outside the FILE blocks
- Do NOT create additional files beyond what the plan requires

Write production-quality code. Do not deviate from the plan.
""".strip()

    code = single_response(coder, code_prompt, num_ctx=MAX_CTX_CODER)
    emit("code_done", code)

    review_prompt = f"""
You are a code reviewer. Review this AI-generated code.

ORIGINAL TASK: {task}
PLAN: {plan}
CODE OUTPUT: {code}

Check: correctness, logic errors, dead code, cross-file consistency,
function signatures vs call sites, missing error handling, orphan files
(JS files not referenced via <script> in HTML).

VERDICT: PASS / NEEDS_CHANGES / FAIL
ISSUES: (list, or "None")
SUGGESTIONS: (specific improvements, or "None")
""".strip()

    review  = single_response(coder, review_prompt, num_ctx=MAX_CTX_CODER)
    verdict = extract_verdict(review)
    emit("review_done", f"VERDICT: {verdict}\n\n{review}")

    final_code = code
    if verdict in ("NEEDS_CHANGES", "FAIL") and not is_cancelled():
        emit("status", "🔁 Fixing...")
        code_files = re.findall(r"FILE:\s*([^\n]+)\n```", code)
        file_list = "\n".join(f"- FILE: {f.strip()}" for f in code_files) if code_files else ""

        retry_prompt = f"""
Fix the issues flagged by the reviewer. Output ALL files complete.

TASK: {task}{target_instruction}
FEEDBACK: {review}
PREVIOUS CODE: {code}
REQUIRED FILES: {file_list}

FILE: filename.ext
```language
<complete file>
```
""".strip()
        retry = single_response(coder, retry_prompt, num_ctx=MAX_CTX_CODER)
        import re as _re
        if len(_re.findall(r"FILE:\s*[^\n]+\n```", retry)) > 0:
            final_code = retry
        emit("retry_done", final_code)

    return {
        "scan": "", "plan": plan, "steps": [],
        "review": review, "verdict": verdict,
        "final_code": final_code, "completed_files": [], "diffs": [],
    }
