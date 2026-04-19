"""
engine/apply_changes.py
Parses AI output and writes files to disk safely.

Two-tier parsing strategy:
  1. Structured  — FILE: path\n```\ncode\n```  (preferred, explicit filename)
  2. Fallback    — largest single code block, filename inferred from task/context

The fallback only produces ONE file (the largest block) to avoid the problem
of every code snippet in a prose response being written as a separate file.

Rules enforced:
  - All writes sandboxed inside base_dir (path escape blocked)
  - Directories created automatically
  - Returns list of written paths for UI confirmation
  - Never raises on parse failure — returns empty list
"""

import os
import re
from engine.logger import log

_LANG_EXT = {
    "python": ".py",  "py":    ".py",
    "javascript": ".js", "js": ".js",
    "typescript": ".ts", "ts": ".ts",
    "html":  ".html", "css":  ".css",
    "bash":  ".sh",   "sh":   ".sh",
    "json":  ".json", "yaml": ".yaml", "yml": ".yml",
    "sql":   ".sql",  "rust": ".rs",
    "cpp":   ".cpp",  "c":    ".c",
    "java":  ".java", "go":   ".go",
    "ruby":  ".rb",   "php":  ".php",
}

# Conversational words that appear at the start of prompts but are useless for filenames
_CONV_STRIP = re.compile(
    r"^\s*(?:ok|okay|great|sure|yes|alright|cool|now|so|well|hi|hey|hello|"
    r"thanks|thank you|please|can you|could you|i want|i need|i'd like|"
    r"let's|lets|can we|would you|write me|create me|make me|build me|"
    r"give me|show me)[,.\s]+",
    re.IGNORECASE
)


def _clean_task(task: str) -> str:
    """Strip conversational openers so 'ok, great, write calculator.py' → 'write calculator.py'."""
    cleaned = task.strip()
    for _ in range(5):  # strip up to 5 leading filler phrases
        new = _CONV_STRIP.sub("", cleaned).strip()
        if new == cleaned:
            break
        cleaned = new
    return cleaned


def _infer_filename(task: str, lang: str, context_files: list = None) -> str:
    """
    Derive a sensible filename. Priority:
    1. Explicit name in task ("save as X", "save it as X", bare X.ext mention)
    2. Context filename (if exactly one context file and it makes sense)
    3. First meaningful noun from cleaned task + language extension
    """
    ext = _LANG_EXT.get(lang.lower(), f".{lang.lower()}" if lang else ".txt")
    task_clean = _clean_task(task)

    # Priority 1: explicit filename in task
    explicit = re.search(
        r"(?:save (?:it )?as|save to|call it|name it|file(?:name)? (?:is )?)\s+([\w\-]+\.\w+)",
        task, re.IGNORECASE
    ) or re.search(
        r"\b([\w\-]+\.(?:py|js|ts|html|css|sh|json|rs|cpp|c|java|go|rb|php))\b", task
    )
    if explicit:
        filename = explicit.group(1).strip()
        log(f"[apply_changes] Explicit filename: {filename}")
        return filename

    # Priority 2: single context file — use its name (model was asked to modify it)
    if context_files and len(context_files) == 1:
        ctx_name = os.path.basename(context_files[0])
        if ctx_name and ctx_name != "__init__.py":
            log(f"[apply_changes] Using context filename: {ctx_name}")
            return ctx_name

    # Priority 3: first meaningful noun from cleaned task
    stop = {"a", "an", "the", "simple", "basic", "write", "create", "make",
            "build", "generate", "with", "using", "for", "in", "and", "or",
            "script", "save", "as", "it", "me", "slightly", "more", "complex",
            "gui", "app", "tool", "program", "code", "file", "python", "js",
            "improve", "update", "fix", "add", "support", "way", "any",
            "scientific", "expressions", "algebra", "direction", "want", "use"}
    words = re.findall(r"[a-zA-Z]+", task_clean.lower())
    meaningful = [w for w in words if w not in stop]
    stem = meaningful[0] if meaningful else "output"
    stem = re.sub(r"[^a-z0-9_]", "_", stem)
    filename = f"{stem}{ext}"
    log(f"[apply_changes] Inferred filename: {filename} (stem={stem!r})")
    return filename


def extract_files(text: str, task: str = "", context_files: list = None) -> list:
    """
    Parses AI output and returns a list of {"path": str, "code": str} dicts.

    Tier 1: explicit FILE: blocks — can return multiple files.
    Tier 2: fallback — returns only the LARGEST code block to prevent every
            prose snippet from being written as a separate file.
    """
    files = []

    # Tier 1: explicit FILE: blocks
    structured = re.findall(
        r"FILE:\s*([^\n]+)\n```[^\n]*\n(.*?)```",
        text, re.DOTALL
    )
    for raw_path, code in structured:
        path = raw_path.strip().replace("\\", "/")
        if path:
            files.append({"path": path, "code": code.rstrip()})
            log(f"[apply_changes] Structured: {path} ({len(code)} chars)")

    if files:
        return files

    # Tier 2: fallback — take only the LARGEST block (avoids grabbing every snippet)
    log("[apply_changes] No FILE: blocks — using largest-block fallback")
    blocks = re.findall(r"```([^\n]*)\n(.*?)```", text, re.DOTALL)

    best_lang = ""
    best_code = ""
    for lang, code in blocks:
        code = code.rstrip()
        # Must be substantial — at least 10 lines to count as a real file
        if code.count("\n") >= 10 and len(code) > len(best_code):
            best_lang = lang.strip().lower()
            best_code = code

    if best_code:
        path = _infer_filename(task, best_lang, context_files)
        files.append({"path": path, "code": best_code})
        log(f"[apply_changes] Fallback (largest block): {path} ({len(best_code)} chars)")
    else:
        log("[apply_changes] No substantial code blocks found")

    return files


def write_files(files: list, base_dir: str) -> list:
    """
    Writes each file to disk inside base_dir.
    Blocks path traversal. Creates directories as needed.
    Returns list of successfully written absolute paths.
    """
    base_abs = os.path.abspath(base_dir)
    written = []

    for f in files:
        rel_path = f["path"]
        code     = f["code"]
        full_path = os.path.abspath(os.path.join(base_abs, rel_path))

        if not full_path.startswith(base_abs + os.sep) and full_path != base_abs:
            log(f"[apply_changes] BLOCKED escape: {rel_path}")
            continue

        try:
            parent = os.path.dirname(full_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as fh:
                fh.write(code)
            written.append(full_path)
            log(f"[apply_changes] Wrote: {full_path}")
        except Exception as e:
            log(f"[apply_changes] Failed {full_path}: {e}")

    return written
