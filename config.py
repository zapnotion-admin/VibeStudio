"""
core/session.py
Saves and loads chat sessions as JSON files in sessions/.

Autosave (_autosave.json) is overwritten silently after every assistant
response — no versioning, no rotation. Named sessions are saved on user
request via the sidebar. list_sessions() excludes files starting with "_".
"""

import json
import datetime
from pathlib import Path
from core.config import SESSIONS_DIR

AUTOSAVE_PATH = SESSIONS_DIR / "_autosave.json"


def save_session(
    name: str,
    messages: list[dict],
    files: list[str],
    project_dir: str,
) -> Path:
    """
    Saves a named session to sessions/<name>.json.
    Overwrites any existing file with the same name.
    Returns the Path of the written file.
    """
    data = {
        "name":        name,
        "timestamp":   datetime.datetime.now().isoformat(),
        "project_dir": project_dir or "",
        "files":       files,
        "messages":    messages,
    }
    path = SESSIONS_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def autosave(
    messages: list[dict],
    files: list[str],
    project_dir: str,
) -> None:
    """
    Overwrites _autosave.json after every assistant response.
    Called from MainWindow._on_stream_finished and _on_pipeline_finished.
    No versioning — always the latest state.
    """
    save_session("_autosave", messages, files, project_dir)


def load_session(name: str) -> dict:
    """
    Loads and returns the session dict for the given name.
    Raises FileNotFoundError if the session does not exist.
    Raises json.JSONDecodeError if the file is corrupt.
    Callers should handle both.
    """
    path = SESSIONS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def list_sessions() -> list[str]:
    """
    Returns saved session names, newest first (by filename sort descending).
    Excludes autosave and any other files starting with "_".
    Returns an empty list if sessions/ is empty or doesn't exist.
    """
    try:
        return sorted(
            [p.stem for p in SESSIONS_DIR.glob("*.json") if not p.stem.startswith("_")],
            reverse=True,
        )
    except Exception:
        return []


def delete_session(name: str) -> None:
    """
    Deletes sessions/<name>.json if it exists.
    Silent no-op if the file is already gone.
    """
    path = SESSIONS_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
