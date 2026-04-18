from pathlib import Path

APP_DIR      = Path(__file__).parent.parent.resolve()
SESSIONS_DIR = APP_DIR / "sessions"
LOGS_DIR     = APP_DIR / "logs"
SESSIONS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11434"

# Models
MODEL_CODER    = "qwen3-coder"        # Qwen3 14B — writes code
MODEL_REASONER = "deepseek-reasoner"  # DeepSeek-R1 8B — plans, reasons
MODEL_FALLBACK = "qwen3:14b"          # Used if custom model hasn't been created yet

# VRAM / context limits  [perf: reduced from 16384/8192 to stay inside 16GB budget]
MAX_CTX_CODER    = 4096
MAX_CTX_REASONER = 4096
MAX_FILE_CHARS   = 12_000   # Hard cap on total injected file content per prompt
MAX_FILES        = 5        # Max files user can add at once
MAX_PROMPT_CHARS = 24_000   # Hard cap on total assembled prompt size

# CPU thread cap for Ollama inference [perf: prevents starving the rest of the OS]
OLLAMA_NUM_THREAD = 4

# UI Palette — dark theme
PALETTE = {
    "bg":       "#0e0f11",
    "bg2":      "#14161a",
    "bg3":      "#1c1f25",
    "border":   "#2a2d35",
    "accent":   "#4ade80",   # green  — AI responses, OK states
    "accent2":  "#22d3ee",   # cyan   — user messages, links
    "warn":     "#fb923c",   # orange — thinking, warnings
    "err":      "#f87171",   # red    — errors
    "text":     "#e2e8f0",
    "text_dim": "#64748b",
    "user_msg": "#1e293b",
    "ai_msg":   "#0f1a12",
}

# VRAM estimates shown in the model selector
VRAM_ESTIMATES = {
    "qwen3-coder":       "~10 GB",
    "deepseek-reasoner": "~5.7 GB",
    "qwen3:14b":         "~10 GB",
}

# Keywords that trigger the Planner → Coder → Reviewer pipeline
TASK_KEYWORDS = [
    "add", "fix", "refactor", "implement", "create", "update",
    "change", "build", "write", "modify", "delete", "rename",
    "move", "extract", "replace", "debug", "optimise", "optimize",
]
