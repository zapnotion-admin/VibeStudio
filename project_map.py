"""
engine/context_manager.py
Manages context budget for the step execution loop.

Key principle (from architecture review):
  - FULL content only for files being modified in the current step
  - INTERFACE only (signatures) for other project files
  - NO summarised executable code — summaries corrupt invariants
  - Compression triggers at 60% of context limit

Token estimation: ~4 chars per token (conservative estimate for code).
"""

import os
import re
from engine.logger import log
from core.config import MAX_CTX_CODER


# At 60% of limit, switch other files to interface-only mode
COMPRESSION_THRESHOLD = 0.60

# Approximate chars per token for code (conservative)
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def extract_interface(file_content: str, filename: str) -> str:
    """
    Extracts just the public interface from a file:
    - Function/class definitions (signatures only, no bodies)
    - Top-level imports
    - Module docstring

    This is safe to use for context — it preserves exact signatures
    without losing them to summarisation drift.
    """
    lines = file_content.splitlines()
    interface_lines = []
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".py":
        in_class = False
        for line in lines:
            stripped = line.strip()
            # Module docstring (first few lines)
            if len(interface_lines) < 5 and (stripped.startswith('"""') or stripped.startswith("'''")):
                interface_lines.append(line)
                continue
            # Imports
            if stripped.startswith("import ") or stripped.startswith("from "):
                interface_lines.append(line)
                continue
            # Class definitions
            if stripped.startswith("class "):
                interface_lines.append(line)
                in_class = True
                continue
            # Function/method definitions
            if stripped.startswith("def ") or stripped.startswith("async def "):
                interface_lines.append(line)
                interface_lines.append("    ...")  # body placeholder
                continue
            # Top-level constants/assignments (not inside functions)
            if re.match(r"^[A-Z_][A-Z0-9_]*\s*=", stripped):
                interface_lines.append(line)
                continue

        return "\n".join(interface_lines)

    # For non-Python files, return first 30 lines as interface
    return "\n".join(lines[:30]) + ("\n... (truncated)" if len(lines) > 30 else "")


def build_file_context_for_step(
    step_files: list,
    all_project_files: dict,
    budget_tokens: int = None,
) -> str:
    """
    Builds the file context string for a single step execution.

    step_files:        list of filenames being modified in this step
    all_project_files: dict of {filename: content} for all project files
    budget_tokens:     max tokens to use (defaults to 60% of MAX_CTX_CODER)

    Returns a formatted string ready to inject into the step prompt.
    """
    if budget_tokens is None:
        budget_tokens = int(MAX_CTX_CODER * COMPRESSION_THRESHOLD)

    step_files_norm = [os.path.basename(f).lower() for f in step_files]
    sections = []
    tokens_used = 0

    # Pass 1: full content for step files (always included)
    for fname, content in all_project_files.items():
        if os.path.basename(fname).lower() in step_files_norm:
            section = f"FILE: {fname} (CURRENT — modify this)\n{'='*50}\n{content}\n{'='*50}"
            sections.append(("full", fname, section))
            tokens_used += estimate_tokens(section)
            log(f"[context_manager] Full: {fname} ({estimate_tokens(content)} tokens)")

    # Pass 2: interface-only for other files
    remaining_budget = budget_tokens - tokens_used
    for fname, content in all_project_files.items():
        if os.path.basename(fname).lower() in step_files_norm:
            continue  # already included above

        interface = extract_interface(content, fname)
        section = f"FILE: {fname} (interface only — do not modify)\n{'-'*50}\n{interface}\n{'-'*50}"
        cost = estimate_tokens(section)

        if remaining_budget - cost > 0:
            sections.append(("interface", fname, section))
            remaining_budget -= cost
            log(f"[context_manager] Interface: {fname} ({cost} tokens)")
        else:
            # Skip entirely if over budget
            sections.append(("skipped", fname, f"FILE: {fname} — omitted (over context budget)"))
            log(f"[context_manager] Skipped: {fname} (over budget)")

    # Sort: full files first, then interfaces
    sections.sort(key=lambda x: 0 if x[0] == "full" else 1)

    return "\n\n".join(s[2] for s in sections)


def read_project_files(project_dir: str, filenames: list) -> dict:
    """
    Reads the current on-disk state of project files.
    Returns {filename: content} for files that exist.
    Files that don't exist yet return empty string.
    """
    result = {}
    for fname in filenames:
        # Try as absolute path, then relative to project_dir
        path = fname if os.path.isabs(fname) else os.path.join(project_dir, fname)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    result[fname] = f.read()
                log(f"[context_manager] Read: {path} ({len(result[fname])} chars)")
            except Exception as e:
                log(f"[context_manager] Failed to read {path}: {e}")
                result[fname] = ""
        else:
            result[fname] = ""  # File doesn't exist yet — will be created
            log(f"[context_manager] Not found (will create): {path}")
    return result


def compute_diff(old_content: str, new_content: str, filename: str) -> str:
    """
    Produces a human-readable diff of changes made to a file.
    Used for step diff display in the UI.
    """
    if not old_content and new_content:
        line_count = len(new_content.splitlines())
        return f"[NEW FILE] {filename} — {line_count} lines created"

    if old_content == new_content:
        return f"[NO CHANGE] {filename}"

    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()

    added   = len([l for l in new_lines if l not in old_lines])
    removed = len([l for l in old_lines if l not in new_lines])

    return f"[MODIFIED] {filename} — ~{added} lines added, ~{removed} lines removed"
