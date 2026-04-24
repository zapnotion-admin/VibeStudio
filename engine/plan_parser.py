"""
engine/plan_parser.py
Parses the structured step plan produced by the REASON stage.

Expected format (must match what workflow.py prompts for):

STEP 1: <description>
FILES: <filename>
DEPENDS_ON: none
SUCCESS_CRITERIA: <criteria>

STEP 2: <description>
FILES: <filename>
DEPENDS_ON: STEP 1
SUCCESS_CRITERIA: <criteria>

v2 fixes:
- SUCCESS_CRITERIA parsing made more robust (handles missing field gracefully)
- extract_plan_summary now includes success criteria in the summary shown to executor
- steps_to_status_summary handles all step statuses correctly
"""

import re


def parse_steps(plan_text: str) -> list:
    """
    Parse the REASON output into a list of step dicts.

    Each step dict:
      number:           int
      description:      str
      files:            list[str]
      depends_on:       list[str]   (e.g. ["STEP 1", "STEP 2"])
      success_criteria: str
      status:           "pending" | "in_progress" | "complete" | "failed"
    """
    # Split on STEP N: boundaries
    blocks = re.split(r"(?=STEP\s+\d+\s*:)", plan_text.strip(), flags=re.IGNORECASE)
    steps = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Extract step number and description from first line
        header = re.match(r"STEP\s+(\d+)\s*:\s*(.+)", block, re.IGNORECASE)
        if not header:
            continue

        number      = int(header.group(1))
        description = header.group(2).strip()

        # FILES:
        files_match = re.search(r"FILES?\s*:\s*(.+)", block, re.IGNORECASE)
        files_raw   = files_match.group(1).strip() if files_match else ""
        # Handle comma-separated or space-separated file lists
        files = [f.strip() for f in re.split(r"[,\s]+", files_raw) if f.strip() and "." in f]

        # DEPENDS_ON:
        dep_match  = re.search(r"DEPENDS_ON\s*:\s*(.+)", block, re.IGNORECASE)
        dep_raw    = dep_match.group(1).strip() if dep_match else "none"
        if dep_raw.lower() in ("none", "n/a", "-", ""):
            depends_on = []
        else:
            depends_on = [d.strip().upper() for d in re.split(r"[,\s]+", dep_raw) if d.strip()]

        # SUCCESS_CRITERIA: (optional — missing field is fine)
        sc_match         = re.search(r"SUCCESS_CRITERIA\s*:\s*(.+)", block, re.IGNORECASE)
        success_criteria = sc_match.group(1).strip() if sc_match else ""

        steps.append({
            "number":           number,
            "description":      description,
            "files":            files,
            "depends_on":       depends_on,
            "success_criteria": success_criteria,
            "status":           "pending",
        })

    # Sort by step number (in case the model output them out of order)
    steps.sort(key=lambda s: s["number"])
    return steps


def extract_plan_summary(steps: list) -> str:
    """
    Compact summary of the full plan — injected into every step prompt
    so the model knows the big picture while executing one step.
    Includes success criteria so executor knows what each step should achieve.
    """
    if not steps:
        return "(no structured plan)"

    lines = []
    for s in steps:
        status_icon = {
            "pending":     "○",
            "in_progress": "▶",
            "complete":    "✓",
            "failed":      "✗",
        }.get(s.get("status", "pending"), "○")

        line = f"  {status_icon} Step {s['number']}: {s['description']}"
        if s.get("files"):
            line += f"  [{', '.join(s['files'])}]"
        if s.get("success_criteria"):
            line += f"\n       → {s['success_criteria']}"
        lines.append(line)

    return "\n".join(lines)


def steps_to_status_summary(steps: list) -> str:
    """
    One-liner per step showing current status.
    Shown in every step prompt so the model knows what's already done.
    """
    lines = []
    for s in steps:
        status = s.get("status", "pending")
        icon = {
            "pending":     "○ PENDING",
            "in_progress": "▶ IN PROGRESS",
            "complete":    "✓ DONE",
            "failed":      "✗ FAILED (skipped)",
        }.get(status, "○ PENDING")
        lines.append(f"  Step {s['number']}: {icon} — {s['description']}")
    return "\n".join(lines)
