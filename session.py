"""
core/context.py
File reading, relevance filtering, prompt assembly, and token budgeting.

Injection order (never change this):
  RAG results → file context → conversation history → user message

Truncation order when prompt exceeds MAX_PROMPT_CHARS (never change this):
  oldest history pairs first → then files → never the user message itself
"""

from pathlib import Path
from core.config import MAX_FILE_CHARS, MAX_FILES, MAX_PROMPT_CHARS, TASK_KEYWORDS


def filter_relevant_files(files: list[str], user_message: str) -> list[str]:
    """
    Scores each file by keyword overlap with the user's message.
    Returns up to 5 most relevant files.
    Falls back to first 3 files if no keyword matches at all (e.g. vague
    messages like "what does this do?").

    Only the first 2000 chars of each file are scanned for keywords —
    enough to catch imports, class names, and function signatures without
    reading entire large files.
    """
    if not files:
        return []

    keywords = set(user_message.lower().split())
    ranked = []

    for f in files:
        try:
            content = Path(f).read_text(encoding="utf-8", errors="ignore")[:2000].lower()
            score = sum(1 for k in keywords if k in content)
            ranked.append((score, f))
        except Exception:
            ranked.append((0, f))

    ranked.sort(key=lambda x: x[0], reverse=True)

    if all(score == 0 for score, _ in ranked):
        return [f for _, f in ranked[:3]]
    return [f for _, f in ranked[:5]]


def build_file_context(file_paths: list[str]) -> str:
    """
    Reads files from disk and builds a combined context string.
    Hard cap at MAX_FILE_CHARS total. If the cap is hit, the last file
    is truncated with a visible warning so the model knows context was cut.
    Unreadable files are included as error stubs (never silently dropped).
    """
    parts = []
    total = 0

    for fp in file_paths[:MAX_FILES]:
        try:
            name    = Path(fp).name
            content = Path(fp).read_text(encoding="utf-8", errors="ignore")
            block   = f"=== {name} ===\n{content}\n\n"

            if total + len(block) > MAX_FILE_CHARS:
                remaining = MAX_FILE_CHARS - total
                if remaining > 200:
                    block = block[:remaining] + "\n# [FILE TRUNCATED — context limit reached]\n\n"
                    parts.append(block)
                break  # No room for further files even if truncated

            parts.append(block)
            total += len(block)

        except Exception as e:
            parts.append(f"=== {fp} ===\n# [Could not read file: {e}]\n\n")

    return "".join(parts)


def build_chat_prompt(
    user_message: str,
    file_context: str,
    history: list[dict],
    rag_context: str = "",
) -> str:
    """
    Assembles the full prompt string sent to Ollama.

    Injection order: RAG results → file context → history → user message.
    This ordering is defined in the spec and must not be changed.

    [v3.1] If the assembled prompt exceeds MAX_PROMPT_CHARS, history pairs
    are dropped oldest-first until it fits. File context is never truncated
    here (build_file_context() handles that). The user message is never
    truncated under any circumstances.

    Args:
        user_message:  The current user input (never truncated).
        file_context:  Pre-built string from build_file_context(), or "".
        history:       List of {"role": "user"|"assistant", "content": str}.
                       Capped at last 12 messages (6 exchanges) before assembly.
        rag_context:   Optional RAG results string, injected before file context.
    """
    recent = history[-12:] if len(history) > 12 else history

    if file_context:
        system = (
            "You are an expert software engineer. You can see the user's code files below. "
            "Make minimal, focused changes. Always explain what you are changing and why. "
            "Do not modify files outside the stated scope."
        )
    else:
        system = (
            "You are a helpful, knowledgeable AI assistant. "
            "Be conversational, clear, and direct."
        )

    def _assemble(hist_msgs: list[dict]) -> str:
        history_text = ""
        for msg in hist_msgs:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n\n"

        parts = [f"System: {system}\n"]
        if rag_context:
            parts.append(f"=== Relevant Index Results ===\n{rag_context}\n=== End Index ===\n")
        if file_context:
            parts.append(f"=== Project Files ===\n{file_context}=== End Files ===\n")
        if history_text:
            parts.append(f"=== Previous Conversation ===\n{history_text}=== End History ===\n")
        parts.append(f"User: {user_message}\nAssistant:")
        return "\n".join(parts)

    prompt = _assemble(recent)

    # [v3.1] Trim oldest history pairs until prompt fits within MAX_PROMPT_CHARS.
    # Drop 2 messages at a time (one user + one assistant exchange).
    if len(prompt) > MAX_PROMPT_CHARS:
        trimmed = list(recent)
        while len(prompt) > MAX_PROMPT_CHARS and len(trimmed) >= 2:
            trimmed = trimmed[2:]
            prompt  = _assemble(trimmed)

    return prompt


def is_task_prompt(text: str) -> bool:
    """
    Returns True if the user's message looks like a code task (use pipeline),
    False if it looks like a question (use direct streaming).

    Heuristics (either condition triggers pipeline):
    - Contains a task keyword (add, fix, refactor, implement, etc.)
    - Message is long enough to likely be a detailed specification (>20 words)
    """
    has_keyword = any(k in text.lower() for k in TASK_KEYWORDS)
    is_long     = len(text.split()) > 20
    return has_keyword or is_long
