"""
engine/step_executor.py
The step execution loop — core of the agent architecture.

For each step:
  1. Read current on-disk file state (file system is the memory)
  2. Build context: full content for step files, interface-only for others
  3. Generate code for this step only
  4. Parse FILE: blocks from output
  5. Validate that written files match planned files (no invented filenames)
  6. Stage the writes (not committed yet)
  7. Verify the step (structural check)
  8. On PASS → commit staged files, advance state
  9. On FAIL → rollback staged, retry with explicit failure reason

v3 fixes:
- File invention guard: rejects output that writes to files not in the step plan.
  This prevents the model from creating rogue files (ensure.js, velocity.js etc)
  when it should be modifying a specific file.
- Retry prompt now explicitly lists the EXACT filenames that must be produced.
- Minimum code length raised to 5 lines.
- Refusal check fires on parsed code content only.
"""

import os
import re
from engine.logger import log
from engine.ollama_client import single_response, ensure_model
from engine.apply_changes import extract_files
from engine.context_manager import (
    build_file_context_for_step,
    read_project_files,
    compute_diff,
)
from engine.plan_parser import steps_to_status_summary, extract_plan_summary
from engine.step_state import StepState
from engine.brief import format_brief_for_prompt
from core.config import MODEL_CODER, MODEL_FALLBACK, MAX_CTX_CODER


_RETRIES_STABLE = 2
_RETRIES_FAST   = 1


def run_steps(
    state:             StepState,
    task:              str,
    all_context_files: list,
    project_dir:       str,
    brief_content:     str,
    coder_model:       str = None,
    stable_mode:       bool = True,
    progress_callback  = None,
    cancel_check       = None,
    constraints_block: str = "",
) -> dict:

    def emit(stage: str, text: str) -> None:
        log(f"[step_executor] {stage}: {text[:80]}")
        if progress_callback:
            progress_callback(stage, text)

    def is_cancelled() -> bool:
        return cancel_check is not None and cancel_check()

    brief_section  = format_brief_for_prompt(brief_content)
    plan_summary   = extract_plan_summary(state.steps)
    completed_files: list = []
    all_diffs:       list = []
    max_retries = _RETRIES_STABLE if stable_mode else _RETRIES_FAST

    from engine.ollama_client import resolve_model
    active_coder = resolve_model(coder_model or MODEL_CODER, MODEL_FALLBACK)
    ensure_model(active_coder)

    while True:
        if is_cancelled():
            state.cancel()
            emit("status", "⛔ Cancelled.")
            break

        next_idx = state.next_pending_index
        if next_idx is None:
            state.complete()
            break

        step = state.steps[next_idx]
        step_num   = step["number"]
        step_total = len(state.steps)
        step_desc  = step["description"]
        step_files = step.get("files", [])

        emit("step_start", f"Step {step_num}/{step_total}: {step_desc}")
        state.begin_step(next_idx)

        if not step_files:
            step_files = [os.path.basename(f) for f in all_context_files]

        all_relevant = list(set(
            [os.path.basename(f) for f in all_context_files] + step_files
        ))

        # Normalised set of allowed output filenames for this step
        allowed_filenames = {f.lower() for f in step_files}

        retry_reason = ""
        success = False

        for attempt in range(max_retries + 1):
            if is_cancelled():
                state.cancel()
                return _result(completed_files, all_diffs, state)

            if attempt > 0:
                emit("status", f"  ↩ Retrying step {step_num} (attempt {attempt + 1}/{max_retries + 1})...")
                state.retry_step()

            current_files = read_project_files(project_dir, all_relevant)

            file_context = build_file_context_for_step(
                step_files=step_files,
                all_project_files=current_files,
            )

            status_summary = steps_to_status_summary(state.steps)

            if retry_reason and attempt > 0:
                retry_note = (
                    f"\n⚠ PREVIOUS ATTEMPT FAILED\n"
                    f"Reason: {retry_reason}\n"
                    f"You MUST fix this specific problem. Do NOT repeat the previous output.\n"
                )
            else:
                retry_note = ""

            sc = step.get("success_criteria", "").strip()
            success_hint = f"\nSuccess looks like: {sc}\n" if sc else ""

            # Build the exact file list the model MUST output
            required_files_str = "\n".join(f"  FILE: {f}" for f in step_files)

            constraints_hint = (
                f"\n{constraints_block}\n" if constraints_block else ""
            )

            prompt = f"""
TASK: {task}
{brief_section}
{constraints_hint}OVERALL PLAN:
{plan_summary}

PROGRESS:
{status_summary}

{file_context}
{retry_note}{success_hint}
EXECUTE THIS STEP ONLY — Step {step_num} of {step_total}:
{step_desc}

You MUST output EXACTLY these files — no others:
{required_files_str}

Output format (one block per file):
FILE: <filename>
```<language>
<complete file contents — every line>
```

Rules:
- Only make the changes required by this step
- Do not implement anything from future steps
- Do not remove anything added by previous steps
- Output the COMPLETE file — no diffs, no snippets, no omissions
- Do NOT create files with different names than listed above
""".strip()

            try:
                raw_output = single_response(
                    active_coder, prompt, num_ctx=MAX_CTX_CODER
                )
            except Exception as e:
                retry_reason = f"Model generation failed: {e}"
                log(f"[step_executor] Generation error: {e}")
                continue

            parsed_files = extract_files(raw_output, task=step_desc)
            if not parsed_files:
                retry_reason = (
                    "No FILE: blocks found in output. "
                    f"You MUST output exactly: {', '.join(step_files)} using the FILE: format."
                )
                log(f"[step_executor] No files parsed from output")
                continue

            # ── File invention guard ──────────────────────────────────
            # Reject outputs that write to files not in this step's plan.
            # This prevents the model from creating new rogue files when
            # it should be modifying specific existing ones.
            if allowed_filenames:
                invented = [
                    f["path"] for f in parsed_files
                    if os.path.basename(f["path"]).lower() not in allowed_filenames
                ]
                if invented:
                    retry_reason = (
                        f"You wrote to unexpected file(s): {', '.join(invented)}. "
                        f"This step ONLY allows writing to: {', '.join(step_files)}. "
                        f"Do not create new files — modify only the files listed."
                    )
                    log(f"[step_executor] File invention rejected: {invented}")
                    continue

            verify_result = _verify_step(
                step_desc    = step_desc,
                output       = raw_output,
                parsed_files = parsed_files,
            )

            if verify_result["pass"]:
                for f in parsed_files:
                    rel_path  = f["path"]
                    full_path = os.path.join(project_dir, rel_path) if project_dir else rel_path
                    old = current_files.get(os.path.basename(rel_path), "")
                    diff = compute_diff(old, f["code"], rel_path)
                    all_diffs.append(diff)
                    if full_path not in completed_files:
                        completed_files.append(full_path)

                for f in parsed_files:
                    rel_path  = f["path"]
                    full_path = os.path.join(project_dir, rel_path) if project_dir else rel_path
                    state.stage_file(full_path, f["code"])

                state.step_success()
                names = [f["path"] for f in parsed_files]
                emit("step_done", f"✓ Step {step_num}: {', '.join(names)}")
                success = True
                break
            else:
                retry_reason = verify_result["reason"]
                log(f"[step_executor] Verify failed: {retry_reason}")

        if not success:
            state.step_failed(retry_reason)
            emit(
                "step_failed",
                f"✗ Step {step_num} failed after {max_retries + 1} attempt(s) — skipping. "
                f"Reason: {retry_reason}"
            )

    return _result(completed_files, all_diffs, state)


def _verify_step(step_desc: str, output: str, parsed_files: list) -> dict:
    """
    Structural verification only.
    1. At least one FILE: block with content
    2. Code not suspiciously short (< 5 lines)
    3. Parsed code doesn't contain refusal text
    """
    if not parsed_files:
        return {"pass": False, "reason": "No FILE: blocks in output"}

    total_lines = sum(f["code"].count("\n") + 1 for f in parsed_files)
    if total_lines < 5:
        return {
            "pass": False,
            "reason": f"Output too short ({total_lines} lines) — likely incomplete"
        }

    code_concat = "\n".join(f["code"] for f in parsed_files).lower()
    refusal_markers = [
        "i cannot complete",
        "i'm unable to",
        "i am unable to",
        "as an ai, i",
    ]
    for marker in refusal_markers:
        if marker in code_concat:
            return {"pass": False, "reason": f"Code block contains refusal text: '{marker}'"}

    return {"pass": True, "reason": ""}


def _result(completed_files: list, diffs: list, state: StepState) -> dict:
    return {
        "completed_files": completed_files,
        "diffs":           diffs,
        "failed_steps":    [s["number"] for s in state.steps if s["status"] == "failed"],
        "completed_steps": [s["number"] for s in state.steps if s["status"] == "complete"],
        "state":           state.state,
    }
