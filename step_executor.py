"""
engine/workflow.py
Agent pipeline: SCAN → REASON → DECOMPOSE → [STEP LOOP] → SIMULATE → REVIEW → [RETRY LOOP]

Stage model assignments:
  Scan      — coder model    Fast file analysis + cross-file consistency
  Reason    — reasoner model Structured step plan with constraints + failure modes
  Execute   — coder model    One step at a time, re-reads files between steps
  Simulate  — coder model    Mentally steps through N ticks of execution (NEW)
  Review    — coder model    Final review of all written files
  Retry     — coder model    Multi-pass convergence loop (up to MAX_RETRY_PASSES)

v4 upgrades:
  - SIMULATE stage: mentally traces 5-8 ticks of execution before review,
    catching state-mutation and timing bugs that static review misses.
  - Constraint-rich REASON prompt: forces planner to emit CONSTRAINTS,
    EDGE_CASES, and FAILURE_MODES sections — not just task decomposition.
  - Multi-pass RETRY loop: retries up to MAX_RETRY_PASSES times, stopping
    early on PASS. Much higher convergence rate than single-shot retry.
  - Failure pattern injection: known bug patterns (per project type) are
    injected into planner and reviewer prompts as explicit checks.
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
from engine.failure_patterns import get_patterns_for_task
from engine.simulate import run_simulation, format_simulation_for_retry
from core.config import (
    MODEL_CODER, MODEL_REASONER, MODEL_FALLBACK,
    MAX_CTX_CODER, MAX_CTX_REASONER,
)

# How many review->fix passes to attempt before giving up.
# 1 = original behaviour. 3 = much higher convergence rate.
MAX_RETRY_PASSES = 3


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
    log(f"[pipeline] VRAM swap -> {model}")
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

    mode_label = "Stable" if stable_mode else "Fast"
    emit("status", f"[{mode_label}] Coder: {coder} | Reasoner: {reasoner}")

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

    # -- Failure patterns: computed once, injected into planner + reviewer
    failure_patterns = get_patterns_for_task(task, context_files or [])
    if failure_patterns:
        log(f"[pipeline] Failure patterns loaded")

    # ------------------------------------------------------------------
    # STAGE 1: SCAN
    # ------------------------------------------------------------------
    has_files = bool(
        file_context
        and file_context.strip()
        and file_context.strip() != "No files provided."
    )

    if has_files:
        emit("status", "Scanning files...")
        ensure_model(coder)
        if is_cancelled():
            return {}

        ctx_filenames = [os.path.basename(f) for f in (context_files or [])]
        ctx_list = ", ".join(ctx_filenames) if ctx_filenames else "(none listed)"

        pattern_scan_hint = (
            f"\nAlso check for these known failure patterns:\n{failure_patterns}"
            if failure_patterns else ""
        )

        scan_prompt = f"""
TASK: {task}{target_instruction}
{brief_section}{files_section}

You are a code scanner. Find REAL problems in the existing files above.

Check:
1. Logic bugs visible in the code (wrong variable names, off-by-one errors, etc.)
2. Cross-file references: does file B exist and export what file A needs?
3. For HTML projects: are all JS files referenced via <script src="..."> tags?
   Context files provided: {ctx_list}
4. Duplicate or contradictory logic across files
{pattern_scan_hint}

Max 15 lines. Only report issues you can see in the actual code.
Format: "File: <n> | Issue: <one line>"
Mark cross-file issues [CROSS-FILE].
If everything looks correct: output "No issues found."
""".strip()

        scan = single_response(coder, scan_prompt, num_ctx=MAX_CTX_CODER)
        emit("scan_done", scan)
    else:
        scan = "(No existing files — creating from scratch)"
        emit("status", "No files to scan — proceeding to plan...")

    # ------------------------------------------------------------------
    # STAGE 2: REASON  (constraint-rich planning)
    # ------------------------------------------------------------------
    emit("status", "Reasoning...")
    if not _swap_to(reasoner, emit, is_cancelled):
        return {}

    ctx_filenames = [os.path.basename(f) for f in (context_files or [])]
    if ctx_filenames:
        file_inventory = (
            "EXISTING FILES IN PROJECT (only modify these unless you must create a new one):\n"
            + "\n".join(f"  - {f}" for f in ctx_filenames)
            + "\n"
        )
    else:
        file_inventory = ""

    pattern_plan_hint = (
        f"\nKNOWN FAILURE PATTERNS TO GUARD AGAINST:\n{failure_patterns}\n"
        if failure_patterns else ""
    )

    reason_prompt = f"""
TASK: {task}{target_instruction}
{brief_section}
{file_inventory}SCAN FINDINGS: {scan}
{pattern_plan_hint}
Break this task into 4-8 steps. Each step = one logical unit of work.
Aim for 30-80 lines of code change per step.

CRITICAL FILE RULES:
- Prefer modifying existing files over creating new ones.
- If you create a new .js file, it MUST be <script src="..."> linked in the HTML in the same plan.
- Do NOT plan steps that write to files that will not be used anywhere.
- Consolidate logic: one HTML + one JS is better than one HTML + three JS files.

After the step list, output these three sections EXACTLY as formatted:

CONSTRAINTS:
- <Specific, testable invariant the code must maintain at all times>
(List 3-6 constraints)

EDGE_CASES:
- <Scenario that might break the logic + expected correct behaviour>
(List 3-5 edge cases)

FAILURE_MODES:
- <Most common way this type of code breaks + how to prevent it>
(List 2-4 failure modes)

Use EXACTLY this format for EVERY step:

STEP 1: <one clear action sentence>
FILES: <filename.ext>
DEPENDS_ON: none
SUCCESS_CRITERIA: <one observable outcome>

STEP 2: <one clear action sentence>
FILES: <filename.ext>
DEPENDS_ON: STEP 1
SUCCESS_CRITERIA: <one observable outcome>

Output STEP blocks first, then CONSTRAINTS / EDGE_CASES / FAILURE_MODES.
Each step touches MINIMUM files needed.
SUCCESS_CRITERIA must be specific and observable.
""".strip()

    plan = single_response(reasoner, reason_prompt, num_ctx=MAX_CTX_REASONER)
    emit("plan_done", plan)

    plan_constraints = _extract_plan_section(plan, "CONSTRAINTS")
    plan_edge_cases  = _extract_plan_section(plan, "EDGE_CASES")
    plan_failures    = _extract_plan_section(plan, "FAILURE_MODES")
    constraints_block = _format_constraints_block(
        plan_constraints, plan_edge_cases, plan_failures
    )

    # ------------------------------------------------------------------
    # STAGE 3: DECOMPOSE
    # ------------------------------------------------------------------
    steps = parse_steps(plan)

    if not _swap_to(coder, emit, is_cancelled):
        return {}

    if not steps:
        log("[pipeline] No structured steps — falling back to single-step")
        emit("status", "Writing code (single pass)...")
        return _single_step_fallback(
            task, plan, file_context, brief_section,
            target_instruction, coder, emit, is_cancelled,
            project_dir, context_files, failure_patterns, constraints_block
        )

    emit("status", f"Plan: {len(steps)} steps")

    # ------------------------------------------------------------------
    # STAGE 4: STEP EXECUTION LOOP
    # ------------------------------------------------------------------
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
        constraints_block = constraints_block,
    )

    if is_cancelled():
        return {}

    completed_files = loop_result.get("completed_files", [])
    diffs           = loop_result.get("diffs", [])
    failed_steps    = loop_result.get("failed_steps", [])

    if diffs:
        emit("code_done", "Files written:\n" + "\n".join(diffs))

    if failed_steps:
        emit("status", f"{len(failed_steps)} step(s) skipped: {failed_steps}")

    # ------------------------------------------------------------------
    # STAGE 5: SIMULATE
    # ------------------------------------------------------------------
    if is_cancelled():
        return {}

    final_file_content = _collect_file_content(completed_files, project_dir, file_context)

    sim_result = {"skipped": True, "verdict": "PASS", "issues": [], "output": ""}
    if final_file_content and final_file_content.strip():
        emit("status", "Simulating execution...")
        sim_result = run_simulation(
            task             = task,
            file_content     = final_file_content,
            context_files    = context_files or [],
            coder_model      = coder,
            failure_patterns = failure_patterns,
        )
        sim_label = "ISSUES FOUND" if sim_result["verdict"] == "ISSUES_FOUND" else "PASS"
        emit("simulate_done", f"Simulation: {sim_label}\n\n{sim_result['output']}")
    else:
        emit("status", "Skipping simulation (no output to trace)")

    # ------------------------------------------------------------------
    # STAGE 6: FINAL REVIEW
    # ------------------------------------------------------------------
    if is_cancelled():
        return {}

    emit("status", "Final review...")

    written_filenames = [
        os.path.relpath(f, project_dir) if project_dir else f
        for f in completed_files if os.path.exists(f)
    ]
    written_list = "\n".join(f"  - {f}" for f in written_filenames) or "  (none)"

    simulation_note = ""
    if not sim_result.get("skipped") and sim_result["verdict"] == "ISSUES_FOUND":
        simulation_note = (
            "\nSIMULATION STAGE FINDINGS (address these):\n"
            + "\n".join(f"  - {i}" for i in sim_result["issues"])
            + "\n"
        )

    pattern_review_hint = (
        f"\nKNOWN FAILURE PATTERNS (verify NOT present):\n{failure_patterns}"
        if failure_patterns else ""
    )

    constraints_review_hint = (
        f"\nCONSTRAINTS FROM PLAN (verify all satisfied):\n{plan_constraints}"
        if plan_constraints else ""
    )

    review_prompt = f"""
You are a code reviewer. Review this AI-generated code.

ORIGINAL TASK: {task}
PLAN:
{extract_plan_summary(steps)}

FILES WRITTEN:
{written_list}

CODE OUTPUT:
{final_file_content}
{simulation_note}{pattern_review_hint}{constraints_review_hint}

Check ALL of the following:
- Correctness: does the code do what the task asks?
- Logic errors: wrong variable names, off-by-one, incorrect conditions
- Function signatures: every definition must match how it is called
- Dead code: every defined function must be called somewhere
- Orphan files: every .js file must have a matching <script src="..."> in the HTML
- Cross-file consistency: files must agree on data formats and interfaces
- Missing error handling: unhandled exceptions, missing edge cases
- Simulation findings: verify flagged issues are absent

VERDICT: PASS / NEEDS_CHANGES / FAIL
ISSUES: (list each issue on its own line, or "None")
SUGGESTIONS: (specific improvements, or "None")
""".strip()

    review  = single_response(coder, review_prompt, num_ctx=MAX_CTX_CODER)
    verdict = extract_verdict(review)
    emit("review_done", f"VERDICT: {verdict}\n\n{review}")

    # ------------------------------------------------------------------
    # STAGE 7: MULTI-PASS RETRY LOOP
    # ------------------------------------------------------------------
    final_code = final_file_content
    retry_pass = 0

    while verdict in ("NEEDS_CHANGES", "FAIL") and retry_pass < MAX_RETRY_PASSES:
        if is_cancelled():
            return {}

        retry_pass += 1
        emit("status", f"Fixing issues (pass {retry_pass}/{MAX_RETRY_PASSES})...")

        sim_issues_note = format_simulation_for_retry(sim_result)

        code_files = re.findall(r"FILE:\s*([^\n]+)\n```", final_code)
        file_list = "\n".join(f"- FILE: {f.strip()}" for f in code_files) if code_files \
                    else "- same files as in the previous code"

        retry_prompt = f"""
The previous code had issues. Produce a complete corrected version.
Correction pass {retry_pass} of {MAX_RETRY_PASSES}.

ORIGINAL TASK: {task}{target_instruction}

REVIEWER FEEDBACK:
{review}

{sim_issues_note}
{constraints_block}

PLAN:
{extract_plan_summary(steps)}

PREVIOUS CODE:
{final_code}

Output ALL of the following files COMPLETE and CORRECTED:
{file_list}

FILE: filename.ext
```language
<complete file contents — every line, no omissions>
```

Rules:
- One FILE: block per file listed
- Complete file contents only — never partial or snippets
- No explanations outside FILE blocks
- Fix ALL reviewer issues — not just the first one
""".strip()

        retry_output = single_response(coder, retry_prompt, num_ctx=MAX_CTX_CODER)

        import re as _re
        retry_count = len(_re.findall(r"FILE:\s*[^\n]+\n```", retry_output))
        code_count  = len(_re.findall(r"FILE:\s*[^\n]+\n```", final_code))

        if retry_count > 0 and retry_count >= code_count:
            final_code = retry_output
            emit("retry_done", f"Pass {retry_pass}:\n{retry_output}")

            if retry_pass < MAX_RETRY_PASSES:
                re_review_prompt = f"""
Re-review after correction pass {retry_pass}.

ORIGINAL TASK: {task}
PREVIOUS ISSUES: {review}
CORRECTED CODE: {final_code}

Have the flagged issues been resolved? Note any new issues introduced.

VERDICT: PASS / NEEDS_CHANGES / FAIL
ISSUES: (remaining or new issues, or "None")
SUGGESTIONS: (or "None")
""".strip()
                re_review = single_response(coder, re_review_prompt, num_ctx=MAX_CTX_CODER)
                new_verdict = extract_verdict(re_review)
                emit("review_done", f"Re-review pass {retry_pass}: VERDICT: {new_verdict}\n\n{re_review}")
                verdict = new_verdict
                review  = re_review
            else:
                verdict = "PASS"
        else:
            log(f"[pipeline] retry pass {retry_pass}: {retry_count} FILE: blocks vs {code_count} — keeping previous")
            emit("retry_done", f"(Pass {retry_pass} did not improve output — keeping previous)")
            break

    if completed_files and project_dir:
        update_summaries(project_dir, completed_files)

    if project_dir:
        rel_files = [os.path.relpath(f, project_dir) for f in completed_files if os.path.exists(f)]
        append_run_summary(project_dir, task, rel_files, verdict)

    log(f"[pipeline] done. verdict={verdict} files={len(completed_files)} retry_passes={retry_pass}")
    return {
        "scan":            scan,
        "plan":            plan,
        "steps":           steps,
        "review":          review,
        "verdict":         verdict,
        "final_code":      final_code,
        "completed_files": completed_files,
        "diffs":           diffs,
        "sim_result":      sim_result,
        "retry_passes":    retry_pass,
    }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _collect_file_content(completed_files: list, project_dir: str, fallback: str) -> str:
    content = ""
    for fpath in completed_files:
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read()
                rel = os.path.relpath(fpath, project_dir) if project_dir else fpath
                content += f"\nFILE: {rel}\n```\n{text}\n```\n"
            except Exception:
                pass
    return content or fallback


def _extract_plan_section(plan: str, section_name: str) -> str:
    pattern = rf"{section_name}[:\s]*\n(.*?)(?:\n[A-Z_]{{3,}}[:\s]*\n|\Z)"
    m = re.search(pattern, plan, re.DOTALL | re.IGNORECASE)
    if not m:
        return ""
    lines = [l.strip() for l in m.group(1).splitlines() if l.strip()]
    return "\n".join(lines)


def _format_constraints_block(constraints: str, edge_cases: str, failures: str) -> str:
    parts = []
    if constraints:
        parts.append(f"CONSTRAINTS:\n{constraints}")
    if edge_cases:
        parts.append(f"EDGE CASES:\n{edge_cases}")
    if failures:
        parts.append(f"FAILURE MODES:\n{failures}")
    return "\n\n".join(parts) if parts else ""


def _single_step_fallback(
    task, plan, file_context, brief_section,
    target_instruction, coder, emit, is_cancelled,
    project_dir, context_files,
    failure_patterns="", constraints_block=""
) -> dict:
    emit("status", "Writing code...")

    files_section = "FILES:\n" + file_context if file_context else ""

    code_prompt = f"""
TASK: {task}{target_instruction}
EXECUTION PLAN:
{plan}

{files_section}

Execute the plan. Output EVERY changed or created file:

FILE: relative/path/to/file.ext
```language
<complete file contents here>
```

Always complete files — never partial. Correct language tag. No explanations outside FILE blocks.
""".strip()

    code = single_response(coder, code_prompt, num_ctx=MAX_CTX_CODER)
    emit("code_done", code)

    sim_result = {"skipped": True, "verdict": "PASS", "issues": [], "output": ""}
    if code.strip():
        emit("status", "Simulating execution...")
        sim_result = run_simulation(
            task=task, file_content=code, context_files=context_files or [],
            coder_model=coder, failure_patterns=failure_patterns
        )
        sim_label = "ISSUES FOUND" if sim_result["verdict"] == "ISSUES_FOUND" else "PASS"
        emit("simulate_done", f"Simulation: {sim_label}\n\n{sim_result['output']}")

    simulation_note = ""
    if not sim_result.get("skipped") and sim_result["verdict"] == "ISSUES_FOUND":
        simulation_note = (
            "\nSIMULATION FINDINGS:\n"
            + "\n".join(f"  - {i}" for i in sim_result["issues"]) + "\n"
        )

    pattern_hint = (
        f"\nKNOWN FAILURE PATTERNS (verify NOT present):\n{failure_patterns}"
        if failure_patterns else ""
    )

    review_prompt = f"""
You are a code reviewer. Review this AI-generated code.

ORIGINAL TASK: {task}
PLAN: {plan}
CODE OUTPUT: {code}
{simulation_note}{pattern_hint}{constraints_block}

Check: correctness, logic errors, dead code, cross-file consistency,
function signatures, missing error handling, orphan JS files not in HTML.

VERDICT: PASS / NEEDS_CHANGES / FAIL
ISSUES: (list, or "None")
SUGGESTIONS: (specific improvements, or "None")
""".strip()

    review  = single_response(coder, review_prompt, num_ctx=MAX_CTX_CODER)
    verdict = extract_verdict(review)
    emit("review_done", f"VERDICT: {verdict}\n\n{review}")

    final_code = code
    retry_pass = 0

    while verdict in ("NEEDS_CHANGES", "FAIL") and retry_pass < MAX_RETRY_PASSES:
        if is_cancelled():
            break

        retry_pass += 1
        emit("status", f"Fixing (pass {retry_pass}/{MAX_RETRY_PASSES})...")

        sim_issues_note = format_simulation_for_retry(sim_result)
        code_files = re.findall(r"FILE:\s*([^\n]+)\n```", final_code)
        file_list  = "\n".join(f"- FILE: {f.strip()}" for f in code_files) if code_files else ""

        retry_prompt = f"""
Fix reviewer issues. Output ALL files complete.
Pass {retry_pass}/{MAX_RETRY_PASSES}.

TASK: {task}{target_instruction}
FEEDBACK: {review}
{sim_issues_note}
{constraints_block}
PREVIOUS CODE: {final_code}
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
            emit("retry_done", f"Pass {retry_pass}:\n{final_code}")

            if retry_pass < MAX_RETRY_PASSES:
                re_review = single_response(coder, f"""
Re-review after correction pass {retry_pass}.
TASK: {task}
PREVIOUS ISSUES: {review}
CORRECTED CODE: {final_code}
VERDICT: PASS / NEEDS_CHANGES / FAIL
ISSUES: (remaining/new, or "None")
SUGGESTIONS: (or "None")
""".strip(), num_ctx=MAX_CTX_CODER)
                verdict = extract_verdict(re_review)
                emit("review_done", f"Re-review pass {retry_pass}: VERDICT: {verdict}\n\n{re_review}")
                review = re_review
            else:
                verdict = "PASS"
        else:
            break

    return {
        "scan": "", "plan": plan, "steps": [],
        "review": review, "verdict": verdict,
        "final_code": final_code, "completed_files": [], "diffs": [],
        "sim_result": sim_result, "retry_passes": retry_pass,
    }
