"""
engine/failure_patterns.py
Lightweight failure pattern memory.

Injects known bug patterns into planner and reviewer prompts based on
the task and project type. Gives the pipeline "pseudo-experience" without
any training — just structured heuristics derived from common LLM coding
failures.

Usage:
    from engine.failure_patterns import get_patterns_for_task

    section = get_patterns_for_task(task, context_files)
    # Returns a formatted string to inject into planner/reviewer prompts.
    # Empty string if no relevant patterns found.
"""

import re

# ---------------------------------------------------------------------------
# Pattern library
# Each entry:
#   "tags":     list of keywords that trigger this pattern (lowercase)
#   "type":     "logic" | "state" | "timing" | "input" | "render" | "async"
#   "title":    short label
#   "pattern":  the bug that commonly occurs
#   "check":    what to verify to confirm it's NOT present
# ---------------------------------------------------------------------------
_PATTERNS = [
    # ── Game / canvas ────────────────────────────────────────────────
    {
        "tags": ["snake", "game", "canvas", "grid"],
        "type": "state",
        "title": "Per-frame state mutation",
        "pattern": "Food, score, or other state is recomputed or regenerated every tick instead of only on the event that should change it.",
        "check": "State changes (food spawn, score increment) must only occur inside the specific event branch (e.g. collision), not unconditionally in the loop.",
    },
    {
        "tags": ["snake", "game", "canvas", "grid"],
        "type": "input",
        "title": "Double-move per tick",
        "pattern": "Multiple direction changes are processed within a single tick, allowing the snake to reverse into itself.",
        "check": "Direction is applied once per game loop tick. Input queue or cooldown prevents processing more than one direction change per interval.",
    },
    {
        "tags": ["snake", "game", "canvas", "grid"],
        "type": "logic",
        "title": "180-degree reversal",
        "pattern": "Player can press the opposite direction key and the snake collides with itself immediately.",
        "check": "Direction change is rejected if new direction is exactly opposite to current direction.",
    },
    {
        "tags": ["game", "canvas", "setinterval", "requestanimationframe"],
        "type": "timing",
        "title": "Multiple simultaneous game loops",
        "pattern": "A new setInterval or requestAnimationFrame loop is started without clearing the previous one, causing the game to accelerate.",
        "check": "Game loop handle is stored and clearInterval/cancelAnimationFrame is called before any new loop is started.",
    },
    {
        "tags": ["game", "canvas", "collision"],
        "type": "logic",
        "title": "Head included in self-collision check",
        "pattern": "Collision check iterates over all segments including index 0 (the head itself), so the game ends immediately.",
        "check": "Self-collision check explicitly skips index 0, or uses a slice starting from index 1.",
    },
    # ── React / frontend ─────────────────────────────────────────────
    {
        "tags": ["react", "usestate", "component", "hook"],
        "type": "state",
        "title": "Stale closure in event handler",
        "pattern": "Event handler captures an initial value of state via closure and never sees updates, causing logic to operate on stale data.",
        "check": "Event handlers that depend on current state use the functional update form setState(prev => ...) or are re-created with useCallback when dependencies change.",
    },
    {
        "tags": ["react", "useeffect", "fetch", "api"],
        "type": "async",
        "title": "Missing useEffect cleanup / race condition",
        "pattern": "Async fetch inside useEffect sets state after the component unmounts, or an older request resolves after a newer one.",
        "check": "useEffect returns a cleanup function that cancels pending requests (AbortController) or sets an ignore flag.",
    },
    {
        "tags": ["react", "list", "map", "key"],
        "type": "render",
        "title": "Missing or unstable list keys",
        "pattern": "Array .map() renders elements without a stable key prop, or uses array index as key when items can be reordered/removed.",
        "check": "Every list item has a unique, stable key derived from the data (e.g. item.id), not the array index.",
    },
    # ── Python ───────────────────────────────────────────────────────
    {
        "tags": ["python", "flask", "fastapi", "django"],
        "type": "logic",
        "title": "Mutable default argument",
        "pattern": "Function defined with a mutable default (list, dict) that is shared across all calls.",
        "check": "Default values for mutable parameters are None with explicit initialisation inside the function body.",
    },
    {
        "tags": ["python", "async", "asyncio", "await"],
        "type": "async",
        "title": "Blocking call inside async function",
        "pattern": "A synchronous blocking call (file I/O, requests.get, time.sleep) is used inside an async function, blocking the event loop.",
        "check": "All I/O inside async functions uses await with async-native libraries (aiofiles, httpx, asyncio.sleep).",
    },
    # ── JavaScript general ────────────────────────────────────────────
    {
        "tags": ["javascript", "js", "html", "fetch", "async"],
        "type": "async",
        "title": "Unhandled promise rejection",
        "pattern": "async/await or .then() chains lack .catch() or try/catch, silently swallowing errors.",
        "check": "Every async operation has explicit error handling. Fetch calls check response.ok before parsing JSON.",
    },
    {
        "tags": ["javascript", "js", "localstorage", "storage"],
        "type": "logic",
        "title": "Missing JSON parse/stringify for storage",
        "pattern": "Objects stored in localStorage/sessionStorage without JSON.stringify are saved as '[object Object]' and read back incorrectly.",
        "check": "All localStorage writes use JSON.stringify() and all reads use JSON.parse() with a fallback for null.",
    },
    # ── File / IO ─────────────────────────────────────────────────────
    {
        "tags": ["file", "read", "write", "path", "open"],
        "type": "logic",
        "title": "Hardcoded paths",
        "pattern": "Absolute or OS-specific paths are hardcoded, causing failures on different machines or OSes.",
        "check": "Paths use os.path.join / pathlib.Path and are relative to a configurable base, not hardcoded.",
    },
    # ── API / data ────────────────────────────────────────────────────
    {
        "tags": ["api", "json", "response", "parse"],
        "type": "logic",
        "title": "Unchecked API response structure",
        "pattern": "Code assumes the API response always has the expected shape and crashes with KeyError/TypeError on missing fields.",
        "check": "API responses are validated with .get() / optional chaining before access. Missing or error responses are handled explicitly.",
    },
]


def _score_patterns(task: str, context_files: list) -> list:
    """Return patterns sorted by relevance score (highest first)."""
    text = (task + " " + " ".join(context_files or [])).lower()
    # Tokenise to avoid partial matches (e.g. "class" matching "classic")
    tokens = set(re.findall(r"[a-z0-9]+", text))

    scored = []
    for p in _PATTERNS:
        hits = sum(1 for tag in p["tags"] if tag in tokens)
        if hits > 0:
            scored.append((hits, p))

    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored]


def get_patterns_for_task(task: str, context_files: list = None, max_patterns: int = 4) -> str:
    """
    Returns a formatted string of relevant failure patterns to inject into prompts.
    Returns empty string if no patterns match.

    max_patterns: cap to keep prompts from bloating (default 4).
    """
    matches = _score_patterns(task, context_files or [])[:max_patterns]
    if not matches:
        return ""

    lines = ["KNOWN FAILURE PATTERNS — check these explicitly:"]
    for i, p in enumerate(matches, 1):
        lines.append(
            f"\n  [{i}] {p['title']} ({p['type']})\n"
            f"      Risk:  {p['pattern']}\n"
            f"      Check: {p['check']}"
        )
    lines.append("")  # trailing newline
    return "\n".join(lines)
