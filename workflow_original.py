"""
engine/step_state.py
State machine for the step execution loop.

States:
  INIT         → pipeline starting, steps parsed
  STEP_RUNNING → currently executing a step
  STEP_SUCCESS → step passed verification, files committed
  STEP_FAILED  → step failed twice, skipped
  COMPLETE     → all steps done, ready for final review
  CANCELLED    → user stopped the pipeline

Transitions:
  INIT         → STEP_RUNNING  (start first step)
  STEP_RUNNING → STEP_SUCCESS  (step verified OK)
  STEP_RUNNING → STEP_FAILED   (step failed after retry)
  STEP_SUCCESS → STEP_RUNNING  (advance to next step)
  STEP_FAILED  → STEP_RUNNING  (skip to next step)
  STEP_RUNNING → CANCELLED     (user hit stop)
  STEP_SUCCESS → COMPLETE      (no more steps)
  STEP_FAILED  → COMPLETE      (no more steps)

File writes are STAGED then COMMITTED:
  - Staged:    written to temp location
  - Committed: moved to real project path (only after STEP_SUCCESS)
  - Rolled back: temp deleted (on STEP_FAILED)
"""

import json
import os
import shutil
from datetime import datetime
from engine.logger import log

STATE_FILENAME = "step_state.json"


class StepState:
    """
    Manages execution state for the agent loop.
    Persists to step_state.json in the project directory.
    """

    VALID_STATES = {
        "INIT", "STEP_RUNNING", "STEP_SUCCESS",
        "STEP_FAILED", "COMPLETE", "CANCELLED"
    }

    def __init__(self, project_dir: str, task: str, steps: list):
        self.project_dir  = project_dir
        self.task         = task
        self.steps        = steps   # list of step dicts from plan_parser
        self.state        = "INIT"
        self.current_step = 0       # index into self.steps
        self.retry_count  = 0
        self.started_at   = datetime.now().isoformat()
        self._staged: dict = {}     # {filepath: content} — staged but not committed

        self._save()
        log(f"[step_state] Initialised with {len(steps)} steps")

    # ── State transitions ────────────────────────────────────────────

    def begin_step(self, step_index: int) -> None:
        assert self.state in ("INIT", "STEP_SUCCESS", "STEP_FAILED"), \
            f"Cannot begin step from state {self.state}"
        self.current_step = step_index
        self.state = "STEP_RUNNING"
        self.retry_count = 0
        self.steps[step_index]["status"] = "running"
        self._save()
        log(f"[step_state] → STEP_RUNNING (step {step_index + 1})")

    def step_success(self) -> None:
        assert self.state == "STEP_RUNNING", \
            f"Cannot succeed from state {self.state}"
        self.state = "STEP_SUCCESS"
        self.steps[self.current_step]["status"] = "complete"
        self._commit_staged()
        self._save()
        log(f"[step_state] → STEP_SUCCESS (step {self.current_step + 1})")

    def step_failed(self, reason: str = "") -> None:
        assert self.state == "STEP_RUNNING", \
            f"Cannot fail from state {self.state}"
        self.state = "STEP_FAILED"
        self.steps[self.current_step]["status"] = "failed"
        self.steps[self.current_step]["failure_reason"] = reason
        self._rollback_staged()
        self._save()
        log(f"[step_state] → STEP_FAILED (step {self.current_step + 1}): {reason}")

    def retry_step(self) -> None:
        assert self.state == "STEP_RUNNING"
        self.retry_count += 1
        self._rollback_staged()
        log(f"[step_state] Retry {self.retry_count} for step {self.current_step + 1}")

    def complete(self) -> None:
        self.state = "COMPLETE"
        self._save()
        log("[step_state] → COMPLETE")

    def cancel(self) -> None:
        self.state = "CANCELLED"
        self._rollback_staged()
        self._save()
        log("[step_state] → CANCELLED")

    # ── Staged write system ──────────────────────────────────────────

    def stage_file(self, filepath: str, content: str) -> None:
        """Stage a file write. Not committed until step_success()."""
        self._staged[filepath] = content
        log(f"[step_state] Staged: {filepath}")

    def _commit_staged(self) -> None:
        """Write all staged files to their real paths."""
        for filepath, content in self._staged.items():
            try:
                os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                log(f"[step_state] Committed: {filepath}")
            except Exception as e:
                log(f"[step_state] Commit failed for {filepath}: {e}")
        self._staged.clear()

    def _rollback_staged(self) -> None:
        """Discard all staged files without writing them."""
        if self._staged:
            log(f"[step_state] Rolling back {len(self._staged)} staged file(s)")
            self._staged.clear()

    def get_staged_content(self, filepath: str) -> str | None:
        """Returns staged content for a file if it exists."""
        return self._staged.get(filepath)

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def current_step_obj(self) -> dict | None:
        if 0 <= self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None

    @property
    def next_pending_index(self) -> int | None:
        for i, step in enumerate(self.steps):
            if step["status"] == "pending":
                return i
        return None

    @property
    def is_done(self) -> bool:
        return self.state in ("COMPLETE", "CANCELLED")

    @property
    def completed_count(self) -> int:
        return sum(1 for s in self.steps if s["status"] == "complete")

    @property
    def failed_count(self) -> int:
        return sum(1 for s in self.steps if s["status"] == "failed")

    # ── Persistence ──────────────────────────────────────────────────

    def _save(self) -> None:
        if not self.project_dir:
            return
        path = os.path.join(self.project_dir, STATE_FILENAME)
        try:
            data = {
                "task":         self.task,
                "state":        self.state,
                "current_step": self.current_step,
                "retry_count":  self.retry_count,
                "started_at":   self.started_at,
                "steps":        self.steps,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log(f"[step_state] Failed to save state: {e}")

    @classmethod
    def load(cls, project_dir: str) -> "StepState | None":
        """Load a saved state (for future resume support)."""
        path = os.path.join(project_dir, STATE_FILENAME)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            obj = cls.__new__(cls)
            obj.project_dir  = project_dir
            obj.task         = data["task"]
            obj.state        = data["state"]
            obj.current_step = data["current_step"]
            obj.retry_count  = data["retry_count"]
            obj.started_at   = data["started_at"]
            obj.steps        = data["steps"]
            obj._staged      = {}
            log(f"[step_state] Loaded state: {obj.state}, step {obj.current_step + 1}")
            return obj
        except Exception as e:
            log(f"[step_state] Failed to load state: {e}")
            return None
