"""
engine/brief.py
Manages the VIBESTUDIO_BRIEF.md file in the project folder.

The brief is a small structured text file that persists across pipeline runs.
It tells every stage what the project is, what's been decided, and what to avoid.
This prevents the model from rediscovering the same problems every run.

Format (plain markdown, human-editable):
  # Project Brief
  ## Goal
  ## Current State
  ## Decisions Made
  ## Do Not Change
  ## Known Issues
"""

import os
from engine.logger import log

BRIEF_FILENAME = "VIBESTUDIO_BRIEF.md"

BRIEF_TEMPLATE = """\
# Project Brief

## Goal
{goal}

## Current State
{state}

## Decisions Made
{decisions}

## Do Not Change
{frozen}

## Known Issues
{issues}
"""

DEFAULT_BRIEF = BRIEF_TEMPLATE.format(
    goal="(describe what this project is and what it should do)",
    state="(describe the current state of the codebase)",
    decisions="(list architectural/technical decisions already made — e.g. 'using AST-based eval, not eval()')",
    frozen="(list files or functions that should not be modified)",
    issues="(list known issues or limitations to be aware of)",
)


def brief_path(project_dir: str) -> str:
    return os.path.join(project_dir, BRIEF_FILENAME)


def read_brief(project_dir: str) -> str:
    """Returns brief content, or empty string if none exists."""
    path = brief_path(project_dir)
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        log(f"[brief] Loaded from {path} ({len(content)} chars)")
        return content
    except Exception as e:
        log(f"[brief] Failed to read: {e}")
        return ""


def write_brief(project_dir: str, content: str) -> bool:
    """Writes brief content to disk. Returns True on success."""
    path = brief_path(project_dir)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        log(f"[brief] Written to {path}")
        return True
    except Exception as e:
        log(f"[brief] Failed to write: {e}")
        return False


def create_default_brief(project_dir: str, goal: str = "") -> bool:
    """Creates a default brief if one doesn't already exist."""
    path = brief_path(project_dir)
    if os.path.exists(path):
        return True  # Don't overwrite existing brief
    content = BRIEF_TEMPLATE.format(
        goal=goal if goal else "(describe what this project is and what it should do)",
        state="Starting fresh.",
        decisions="(none yet)",
        frozen="(none)",
        issues="(none yet)",
    )
    return write_brief(project_dir, content)


def brief_exists(project_dir: str) -> bool:
    return os.path.exists(brief_path(project_dir))


def format_brief_for_prompt(brief_content: str) -> str:
    """Wraps the brief in a prompt-ready block."""
    if not brief_content.strip():
        return ""
    return (
        "\nPROJECT BRIEF (read this first — it defines the project context and constraints):\n"
        "---\n"
        f"{brief_content.strip()}\n"
        "---\n"
    )


def append_run_summary(project_dir: str, task: str, files_written: list, verdict: str) -> None:
    """
    Appends a summary of a completed pipeline run to the brief.
    Keeps the brief up to date so future runs don't rediscover resolved issues.
    Only appends if a brief exists — never creates one automatically here.
    """
    if not project_dir or not brief_exists(project_dir):
        return

    import datetime
    date = datetime.date.today().isoformat()
    file_list = ", ".join(files_written) if files_written else "none"

    summary = (
        f"\n\n## Run Log — {date}\n"
        f"- Task: {task[:120]}\n"
        f"- Files written: {file_list}\n"
        f"- Verdict: {verdict}\n"
    )

    path = brief_path(project_dir)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(summary)
        log(f"[brief] Appended run summary to {path}")
    except Exception as e:
        log(f"[brief] Failed to append run summary: {e}")
