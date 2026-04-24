"""
main.py
Zap CodeForge entry point.

Responsibilities:
  1. Ensure logs/ directory exists
  2. Attempt to reach Ollama (3 tries, starting it if needed)
  3. Launch the Qt application
  4. Show the MainWindow — UI shows an error state if Ollama is unreachable

Health check uses /api/tags exclusively (v3.1 spec).
Ollama is started via subprocess.Popen with CREATE_NO_WINDOW so no
console window flashes on Windows.
"""

import sys
import subprocess
import time
import requests
from pathlib import Path

from PySide6.QtWidgets import QApplication

from core.config import APP_DIR
from engine.logger import log


def init_logging() -> None:
    """Ensure logs/ directory exists before anything else tries to write to it."""
    logs_dir = APP_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)


def ensure_ollama() -> bool:
    """
    Checks that Ollama is reachable via GET /api/tags.
    On each failure, attempts to start the Ollama service and waits 4 seconds.
    Makes up to 3 attempts total.

    Returns True if Ollama responds with HTTP 200, False otherwise.
    Returning False does NOT abort the app — the UI handles the error state.
    """
    for attempt in range(3):
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                log(f"[main] Ollama reachable (attempt {attempt + 1})")
                return True
        except Exception:
            pass

        if attempt < 2:
            log(f"[main] Ollama not reachable — attempting to start (attempt {attempt + 1})")
            try:
                subprocess.Popen(
                    ["ollama", "serve"],
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except FileNotFoundError:
                log("[main] 'ollama' not found on PATH — is it installed?")
            except Exception as e:
                log(f"[main] Failed to start Ollama: {e}")
            time.sleep(4)

    log("[main] Ollama unreachable after 3 attempts — launching UI in degraded state")
    return False


def main() -> None:
    init_logging()
    log("[main] Zap CodeForge starting")

    app = QApplication(sys.argv)
    app.setApplicationName("Zap CodeForge")
    app.setOrganizationName("Zap CodeForge")

    ensure_ollama()  # result handled inside MainWindow via status bar

    from ui.main_window import MainWindow
    window = MainWindow()
    window.show()

    log("[main] Window shown — entering event loop")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
