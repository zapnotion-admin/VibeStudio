import datetime
from pathlib import Path
from core.config import LOGS_DIR

LOG_FILE = LOGS_DIR / "runtime.log"


def log(msg: str) -> None:
    """
    Appends a timestamped message to logs/runtime.log.
    Never raises — logging must never crash the app.
    """
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")
    except Exception:
        pass
