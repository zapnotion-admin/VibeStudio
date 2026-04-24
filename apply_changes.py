"""
engine/simulate.py
Simulation stage — mentally steps through generated code before final review.

This stage asks the coder model to:
  1. Trace through the first N "ticks" or "events" of execution
  2. Track key state variables explicitly
  3. Identify any discrepancies between expected and actual behaviour

This catches a class of bugs that static review misses:
  - State that mutates unexpectedly per tick
  - Event handlers that fire in wrong order
  - Off-by-one in loop/index logic
  - Missing initialisation that only surfaces at runtime

Returns:
    {
        "output":   <raw simulation text from model>,
        "verdict":  "PASS" | "ISSUES_FOUND",
        "issues":   [list of issue strings],
        "skipped":  True if simulation was skipped (no simulatable code)
    }
"""

import re
from engine.logger import log
from engine.ollama_client import single_response
from core.config import MAX_CTX_CODER


# Tags that indicate the code has runtime state worth simulating.
_SIMULATABLE_EXTENSIONS = {".js", ".py", ".ts", ".jsx", ".tsx"}

# Minimum lines of code before simulation is worth running
_MIN_LINES_FOR_SIMULATION = 20


def _is_worth_simulating(file_content: str, context_files: list) -> bool:
    """Quick check: is there enough stateful code to simulate?"""
    if not file_content or not file_content.strip():
        return False
    total_lines = file_content.count("\n")
    if total_lines < _MIN_LINES_FOR_SIMULATION:
        return False
    exts = {
        "." + f.rsplit(".", 1)[-1].lower()
        for f in (context_files or [])
        if "." in f
    }
    return bool(exts & _SIMULATABLE_EXTENSIONS)


def run_simulation(
    task: str,
    file_content: str,
    context_files: list,
    coder_model: str,
    failure_patterns: str = "",
) -> dict:
    """
    Run the simulation stage against the generated code.

    Parameters
    ----------
    task            : original user task
    file_content    : concatenated FILE: blocks from code stage
    context_files   : list of filenames (used for relevance check)
    coder_model     : which model to use for simulation
    failure_patterns: pre-formatted failure pattern hints (from failure_patterns.py)

    Returns
    -------
    dict with keys: output, verdict, issues, skipped
    """
    if not _is_worth_simulating(file_content, context_files):
        log("[simulate] Skipping — code too short or no simulatable files")
        return {"output": "", "verdict": "PASS", "issues": [], "skipped": True}

    pattern_section = (
        f"\n{failure_patterns}\n" if failure_patterns else ""
    )

    prompt = f"""
You are a runtime simulation engine. Your job is to mentally execute the code below
and trace what actually happens — not what was intended.

TASK: {task}
{pattern_section}
CODE:
{file_content}

SIMULATION INSTRUCTIONS:
Step through the first 5-8 "ticks" or key events of execution.
For each tick/event, track ALL state variables that change.
Use this exact format:

TICK 1:
  Input:    <what triggers this tick — timer fires / key pressed / function called>
  State before: <relevant variables and their values>
  Execution: <which lines/branches run>
  State after:  <variables after this tick>
  Notes:    <anything unexpected or suspicious>

TICK 2:
  ... (continue for 5-8 ticks total)

EDGE CASES:
  List 2-3 edge cases you identified during simulation.
  For each: describe what happens and whether the code handles it correctly.

SIMULATION VERDICT:
  PASS — code behaves correctly across all ticks and edge cases
  ISSUES_FOUND — one or more ticks show incorrect behaviour

SIMULATION ISSUES (if ISSUES_FOUND):
  - Issue 1: <specific description with tick number>
  - Issue 2: ...

Keep each tick description concise (3-6 lines). Focus on STATE changes, not code narration.
""".strip()

    log("[simulate] Running simulation stage...")
    output = single_response(coder_model, prompt, num_ctx=MAX_CTX_CODER)

    verdict, issues = _parse_simulation_output(output)
    log(f"[simulate] verdict={verdict} issues={len(issues)}")

    return {
        "output":  output,
        "verdict": verdict,
        "issues":  issues,
        "skipped": False,
    }


def _parse_simulation_output(text: str) -> tuple:
    """Extract verdict and issues list from simulation output."""
    upper = text.upper()

    # Determine verdict
    if "ISSUES_FOUND" in upper or "ISSUES FOUND" in upper:
        verdict = "ISSUES_FOUND"
    elif "PASS" in upper:
        verdict = "PASS"
    else:
        verdict = "PASS"  # Default optimistic if unclear

    # Extract issues
    issues = []
    issue_section = re.search(
        r"SIMULATION ISSUES.*?:(.*?)(?:\n[A-Z]{3,}|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    if issue_section:
        raw = issue_section.group(1)
        for line in raw.splitlines():
            line = line.strip().lstrip("-•*").strip()
            if len(line) > 10:
                issues.append(line)

    return verdict, issues


def format_simulation_for_retry(sim_result: dict) -> str:
    """
    Format simulation issues for injection into the retry prompt.
    Returns empty string if simulation passed or was skipped.
    """
    if sim_result.get("skipped") or sim_result.get("verdict") == "PASS":
        return ""
    issues = sim_result.get("issues", [])
    if not issues:
        return ""
    lines = ["SIMULATION STAGE found runtime issues:"]
    for i in issues:
        lines.append(f"  - {i}")
    return "\n".join(lines)
