"""
engine/project_map.py
Manages file_summaries.json — a persistent project map that stores
interface extractions for every file in the project.

This lets the REASON stage see the full project structure without
loading every file into the context window. Critical for large codebases.

Format of file_summaries.json:
{
  "calculator.py": {
    "interface": "class Calculator:\\n  def __init__(self)...\\n  def evaluate(self)...",
    "line_count": 87,
    "updated_at": "2026-04-23T14:00:00"
  }
}
"""

import os
import json
from datetime import datetime
from engine.logger import log
from engine.context_manager import extract_interface

SUMMARIES_FILENAME = "file_summaries.json"


def _summaries_path(project_dir: str) -> str:
    return os.path.join(project_dir, SUMMARIES_FILENAME)


def load_summaries(project_dir: str) -> dict:
    """Load existing summaries, return empty dict if none."""
    path = _summaries_path(project_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"[project_map] Failed to load summaries: {e}")
        return {}


def save_summaries(project_dir: str, summaries: dict) -> None:
    path = _summaries_path(project_dir)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        log(f"[project_map] Saved {len(summaries)} file summaries")
    except Exception as e:
        log(f"[project_map] Failed to save summaries: {e}")


def update_summaries(project_dir: str, written_files: list) -> None:
    """
    Update summaries for files that were just written.
    Called after each successful pipeline run.
    """
    if not project_dir or not written_files:
        return

    summaries = load_summaries(project_dir)

    for fpath in written_files:
        # Normalise to relative path
        rel = os.path.relpath(fpath, project_dir) if os.path.isabs(fpath) else fpath
        abs_path = os.path.join(project_dir, rel) if not os.path.isabs(fpath) else fpath

        if not os.path.exists(abs_path):
            continue
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
            interface = extract_interface(content, rel)
            summaries[rel] = {
                "interface":  interface,
                "line_count": len(content.splitlines()),
                "updated_at": datetime.now().isoformat(),
            }
            log(f"[project_map] Updated summary: {rel} ({len(content.splitlines())} lines)")
        except Exception as e:
            log(f"[project_map] Failed to summarise {rel}: {e}")

    save_summaries(project_dir, summaries)


def build_project_map_section(project_dir: str, exclude_files: list = None) -> str:
    """
    Builds a compact project map string for injection into the REASON prompt.
    Shows all known files with their interfaces.

    exclude_files: files already included in full context (don't summarise those)
    """
    summaries = load_summaries(project_dir)
    if not summaries:
        return ""

    exclude_norm = set()
    if exclude_files:
        for f in exclude_files:
            exclude_norm.add(os.path.basename(f).lower())

    lines = ["PROJECT MAP (existing files — do not recreate unless modifying):"]
    for fname, info in summaries.items():
        if os.path.basename(fname).lower() in exclude_norm:
            continue
        line_count = info.get("line_count", "?")
        interface  = info.get("interface", "").strip()
        lines.append(f"\n--- {fname} ({line_count} lines) ---")
        if interface:
            # Limit interface to first 20 lines to keep it compact
            ilines = interface.splitlines()[:20]
            lines.extend(f"  {l}" for l in ilines)
            if len(interface.splitlines()) > 20:
                lines.append("  ... (truncated)")

    return "\n".join(lines) if len(lines) > 1 else ""
