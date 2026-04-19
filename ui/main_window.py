"""
ui/main_window.py
Top-level QMainWindow. Owns the layout, all app state, and all signal
wiring between components. Also defines the two QThread workers:
  OllamaStreamWorker — for Chat Mode and Code Chat (streaming)
  PipelineWorker     — for Code Mode task requests (3-stage pipeline)

Threading rules (spec §threading):
  - All Ollama calls happen inside workers, NEVER from the UI thread
  - start_ai_block() called BEFORE thread.start() — never inside worker
  - end_ai_block() called ONLY in _on_stream_finished — never inside worker
  - _cleanup_thread() called from every finish/error handler
  - Only one worker/thread active at a time (_worker / _thread)
"""

import json
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout,
    QApplication, QStatusBar,
)
from PySide6.QtCore import Qt, QThread, QObject, Signal
from PySide6.QtGui  import QFont

from core.config  import PALETTE, MODEL_CODER, MODEL_FALLBACK, VRAM_ESTIMATES
from engine.logger import log


# ════════════════════════════════════════════════════════════════════
# Workers
# ════════════════════════════════════════════════════════════════════

class OllamaStreamWorker(QObject):
    """
    Streams a single Ollama generation.
    Emits chunk_ready for every decoded token, finished always (even on
    cancel/error), error_occurred only for genuine failures.

    [v3.2] cancel() closes self._response to terminate the HTTP stream
    at the network layer — not just break the iteration loop.
    """
    chunk_ready    = Signal(str)
    finished       = Signal()
    error_occurred = Signal(str)

    def __init__(self, model: str, prompt: str, system: str = "", num_ctx: int = 16384):
        super().__init__()
        self.model      = model
        self.prompt     = prompt
        self.system     = system
        self.num_ctx    = num_ctx
        self._cancelled = False
        self._response  = None  # live HTTP response — closed on cancel

    def cancel(self) -> None:
        """[v3.2] True cancellation: closes the HTTP response, not just a flag."""
        self._cancelled = True
        if self._response is not None:
            try:
                self._response.close()
            except Exception:
                pass

    def run(self) -> None:
        try:
            from engine.ollama_client import safe_generate, OLLAMA_URL
            from core.config import OLLAMA_NUM_THREAD
            payload = {
                "model":   self.model,
                "prompt":  self.prompt,
                "system":  self.system,
                "stream":  True,
                "options": {
                    "num_ctx": self.num_ctx,
                    "temperature": 0.6,
                    "num_thread": OLLAMA_NUM_THREAD,  # [perf] cap CPU threads
                },
            }
            self._response = safe_generate(
                f"{OLLAMA_URL}/api/generate", payload, stream=True
            )
            self._response.raise_for_status()
            for line in self._response.iter_lines():
                if self._cancelled:
                    break
                if line:
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    if "response" in data:
                        self.chunk_ready.emit(data["response"])
                    if data.get("done"):
                        break
        except Exception as e:
            if not self._cancelled:   # suppress errors from intentional close()
                self.error_occurred.emit(str(e))
        finally:
            self.finished.emit()      # always emitted — even on cancel/error


class PipelineWorker(QObject):
    """
    Runs the Planner → Coder → Reviewer pipeline in a worker thread.
    Emits progress(stage, text) at each stage transition.
    [v3.2] Passes cancel_check into run_pipeline for stage-boundary cancellation.
    """
    progress       = Signal(str, str)   # (stage, text)
    finished       = Signal()
    error_occurred = Signal(str)

    def __init__(self, task: str, file_context: str):
        super().__init__()
        self.task         = task
        self.file_context = file_context
        self._cancelled   = False
        self.result       = {}

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            from engine.workflow import run_pipeline

            def callback(stage: str, text: str) -> None:
                if not self._cancelled:
                    self.progress.emit(stage, text)

            self.result = run_pipeline(
                self.task,
                self.file_context,
                progress_callback=callback,
                cancel_check=lambda: self._cancelled,
            )
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.finished.emit()      # always emitted


# ════════════════════════════════════════════════════════════════════
# MainWindow
# ════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VibeStudio")
        self.resize(1200, 800)

        # ── app state ────────────────────────────────────────────────
        self._messages:      list[dict] = []
        self._context_files: list[str]  = []
        self._project_dir:   str        = ""
        self._current_model: str        = MODEL_CODER
        self._worker = None
        self._thread = None

        # ── build UI ─────────────────────────────────────────────────
        self._build_ui()
        self._apply_stylesheet()
        self._wire_signals()

        # ── startup Ollama check ─────────────────────────────────────
        self._check_ollama_status()

        # [perf] Idle VRAM release — unloads model after 60s of inactivity.
        # Keeps the system responsive when VibeStudio is open but not in use.
        from PySide6.QtCore import QTimer
        from engine.ollama_client import unload_model
        self._idle_timer = QTimer(self)
        self._idle_timer.setInterval(60_000)   # 60 seconds
        self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(unload_model)

        log("[main_window] Initialised")

    # ----------------------------------------------------------------
    # UI construction
    # ----------------------------------------------------------------

    def _build_ui(self) -> None:
        from ui.chat_panel  import ChatPanel
        from ui.input_panel import InputPanel
        from ui.sidebar     import Sidebar

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Horizontal splitter: sidebar (fixed 260px) | chat area
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        self.sidebar    = Sidebar()
        self.chat_panel = ChatPanel()
        self.input_panel = InputPanel()

        # Right pane: chat + input stacked vertically
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self.chat_panel, stretch=1)
        right_layout.addWidget(self.input_panel, stretch=0)

        splitter.addWidget(self.sidebar)
        splitter.addWidget(right)
        splitter.setSizes([260, 940])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root_layout.addWidget(splitter)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

    def _wire_signals(self) -> None:
        # Sidebar → MainWindow
        self.sidebar.project_changed.connect(self._on_project_changed)
        self.sidebar.files_changed.connect(self._on_files_changed)
        self.sidebar.model_changed.connect(self._on_model_changed)
        self.sidebar.session_load_requested.connect(self._on_session_load)
        self.sidebar.session_save_requested.connect(self._on_session_save)
        self.sidebar.clear_chat_requested.connect(self.chat_panel.clear_chat)
        self.sidebar.index_project_requested.connect(self._on_index_project)

        # InputPanel → MainWindow
        self.input_panel.send_requested.connect(self._handle_send)
        self.input_panel.stop_requested.connect(self._handle_stop)

    # ----------------------------------------------------------------
    # Send / Stop
    # ----------------------------------------------------------------

    def _handle_send(self, text: str) -> None:
        if not text.strip():
            return

        # [perf] Reset idle unload timer — user is active, keep model warm
        self._idle_timer.start()

        self._messages.append({"role": "user", "content": text})
        self.chat_panel.add_user_message(text)
        self.input_panel.set_sending(True)

        from core.context import (
            is_task_prompt, filter_relevant_files,
            build_file_context, build_chat_prompt,
        )

        if self._context_files:
            relevant_files = filter_relevant_files(self._context_files, text)
            file_context   = build_file_context(relevant_files)
        else:
            file_context = ""

        is_task = is_task_prompt(text)
        # Also force pipeline when writes are enabled — the user wants a file created
        force_pipeline = self.sidebar.allow_writes_enabled() and is_task

        if self._context_files or force_pipeline:
            # ── CODE MODE — pipeline or code chat ────────────────────
            if is_task or force_pipeline:
                self._start_pipeline(text, file_context)
            else:
                prompt = build_chat_prompt(text, file_context, self._messages[:-1])
                self._start_stream(self._current_model, prompt)
        else:
            # ── CHAT MODE — direct streaming ─────────────────────────
            prompt = build_chat_prompt(text, "", self._messages[:-1])
            self._start_stream(self._current_model, prompt)

    def _handle_stop(self) -> None:
        """
        Requests cancellation on the active worker.
        For streams: closes the HTTP response (true network cancel).
        For pipeline: sets _cancelled flag checked at stage boundaries.
        _cleanup_thread() fires when the worker emits finished.
        """
        if self._worker:
            self._worker.cancel()
            log("[main_window] Stop requested")

    # ----------------------------------------------------------------
    # Thread management
    # ----------------------------------------------------------------

    def _start_pipeline(self, task: str, file_context: str) -> None:
        self._thread = QThread()
        self._worker = PipelineWorker(task, file_context)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_pipeline_progress)
        self._worker.finished.connect(self._on_pipeline_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error_occurred.connect(self._on_stream_error)
        self._thread.start()
        log("[main_window] Pipeline started")

    def _cleanup_thread(self) -> None:
        """
        [v3.1] Must be called from every finish/error handler.
        Disconnects, stops, and nullifies both worker and thread to prevent
        dangling QThreads, signal duplication, and memory leaks.
        """
        if self._worker:
            self._worker.deleteLater()
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._worker = None
        self._thread = None
        log("[main_window] _cleanup_thread complete")

    # ----------------------------------------------------------------
    # Worker signal handlers
    # ----------------------------------------------------------------

    def _on_stream_finished(self) -> None:
        """Streaming completed normally (or was cancelled)."""
        self.chat_panel.end_ai_block()
        # Capture the streamed text from the display for history
        # (we store the plain-text version — good enough for context)
        # Response text is captured via _stream_buffer in _on_chunk / _finalise_assistant_turn
        # Append last assistant response to history
        # We approximate by using whatever was streamed
        if self._messages and self._messages[-1]["role"] == "user":
            # Find the content appended since the last user message
            # Simple approach: store a marker and extract — or just use
            # the raw display. For robustness, track separately.
            pass  # response text tracked via _current_response
        self._finalise_assistant_turn()
        self._cleanup_thread()
        # [perf] Free VRAM immediately — don't leave model resident between turns
        from engine.ollama_client import unload_model
        unload_model()

    def _on_pipeline_finished(self) -> None:
        """Pipeline completed — optionally apply file writes if enabled."""
        result = self._worker.result if self._worker else {}
        self.chat_panel.end_pipeline_block()

        if result and "final_code" in result and self.sidebar.allow_writes_enabled():
            if not self._project_dir:
                self.chat_panel.add_system_message(
                    "File writes enabled but no project folder set — set one in the sidebar.",
                    level="warn",
                )
            else:
                from engine.apply_changes import extract_files, write_files
                task = self._messages[-1]["content"] if self._messages else ""
                files = extract_files(result["final_code"], task=task, context_files=self._context_files)

                if files:
                    try:
                        written = write_files(files, self._project_dir)
                        if written:
                            names = ", ".join(
                                f.replace(self._project_dir, "").lstrip("/\\")
                                for f in written
                            )
                            self.chat_panel.add_system_message(
                                f"Wrote {len(written)} file(s): {names}", level="ok",
                            )
                        else:
                            self.chat_panel.add_system_message(
                                "Files were parsed but none could be written — check folder permissions.",
                                level="warn",
                            )
                    except Exception as e:
                        self.chat_panel.add_system_message(
                            f"Write failed: {e}", level="err",
                        )
                else:
                    self.chat_panel.add_system_message(
                        "No code blocks found in output — nothing written.",
                        level="warn",
                    )
        elif result and "final_code" in result and not self.sidebar.allow_writes_enabled():
            self.chat_panel.add_system_message(
                "Preview only — enable 'Write files' in sidebar to apply changes.",
                level="info",
            )

        self.input_panel.set_sending(False)
        from core.session import autosave
        autosave(self._messages, self._context_files, self._project_dir)
        self._cleanup_thread()
        from engine.ollama_client import unload_model
        unload_model()
        log("[main_window] Pipeline finished")

    def _on_pipeline_progress(self, stage: str, text: str) -> None:
        """Routes pipeline stage output into a single selectable response block."""
        if stage == "status":
            if self.chat_panel._pipeline is None:
                self.chat_panel.start_pipeline_block()
            self.chat_panel.append_pipeline_status(text)
        elif stage in ("scan_done", "plan_done", "code_done", "review_done", "retry_done"):
            if self.chat_panel._pipeline is None:
                self.chat_panel.start_pipeline_block()
            stage_label = stage.replace("_done", "").upper()
            self.chat_panel.append_pipeline_stage(stage_label, text)

    def _on_stream_error(self, error_msg: str) -> None:
        if "stalled" in error_msg.lower() or "timeout" in error_msg.lower():
            self.chat_panel.add_system_message(
                "Model stalled — retrying...", level="warn"
            )
        else:
            self.chat_panel.add_system_message(
                f"Model failed — check Ollama: {error_msg}", level="err"
            )
        self.input_panel.set_sending(False)
        self._cleanup_thread()
        log(f"[main_window] Stream error: {error_msg}")

    # ----------------------------------------------------------------
    # Response tracking
    # ----------------------------------------------------------------

    # Accumulate streamed chunks for history
    _stream_buffer: list[str] = []

    def _start_stream(self, model: str, prompt: str, system: str = "") -> None:
        """Override to also reset the response accumulator."""
        self._stream_buffer = []
        self._thread = QThread()
        self._worker = OllamaStreamWorker(model, prompt, system)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.chunk_ready.connect(self._on_chunk)
        self._worker.finished.connect(self._on_stream_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error_occurred.connect(self._on_stream_error)
        self.chat_panel.start_ai_block()
        self._thread.start()
        log(f"[main_window] Stream started model={model}")

    def _on_chunk(self, chunk: str) -> None:
        """Receives each streaming chunk — forwards to chat panel and accumulates."""
        self._stream_buffer.append(chunk)
        self.chat_panel.append_ai_chunk(chunk)

    def _finalise_assistant_turn(self) -> None:
        """
        Appends the accumulated streamed response to _messages and autosaves.
        Called from _on_stream_finished.
        """
        response_text = "".join(self._stream_buffer)
        if response_text:
            self._messages.append({"role": "assistant", "content": response_text})
        self.input_panel.set_sending(False)
        from core.session import autosave
        autosave(self._messages, self._context_files, self._project_dir)
        log(f"[main_window] Assistant turn saved ({len(response_text)} chars)")

    # ----------------------------------------------------------------
    # Sidebar signal handlers
    # ----------------------------------------------------------------

    def _on_project_changed(self, path: str) -> None:
        self._project_dir = path
        short = path[-40:] if len(path) > 40 else path
        self._status_bar.showMessage(f"Project: {short}")
        log(f"[main_window] Project: {path}")

    def _on_files_changed(self, files: list) -> None:
        self._context_files = files
        log(f"[main_window] Context files: {len(files)}")

    def _on_model_changed(self, model: str) -> None:
        self._current_model = model
        vram = VRAM_ESTIMATES.get(model, "")
        self._status_bar.showMessage(f"Model: {model}  {vram}")
        log(f"[main_window] Model: {model}")

    def _on_session_load(self, name: str) -> None:
        try:
            from core.session import load_session
            data = load_session(name)
            self._messages       = data.get("messages", [])
            self._context_files  = data.get("files", [])
            self._project_dir    = data.get("project_dir", "")
            self.chat_panel.clear_chat()
            # Replay messages into the chat display
            for msg in self._messages:
                if msg["role"] == "user":
                    self.chat_panel.add_user_message(msg["content"])
                else:
                    self.chat_panel.add_system_message(msg["content"], level="ok")
            self.sidebar.refresh_sessions()
            log(f"[main_window] Session loaded: {name}")
        except Exception as e:
            self.chat_panel.add_system_message(f"Failed to load session: {e}", level="err")

    def _on_session_save(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save Session", "Session name:")
        if ok and name.strip():
            try:
                from core.session import save_session
                save_session(name.strip(), self._messages, self._context_files, self._project_dir)
                self.sidebar.refresh_sessions()
                self.chat_panel.add_system_message(f"Session saved: {name}", level="ok")
                log(f"[main_window] Session saved: {name}")
            except Exception as e:
                self.chat_panel.add_system_message(f"Save failed: {e}", level="err")

    def _on_index_project(self) -> None:
        from engine.rag import is_available
        if not is_available():
            self.chat_panel.add_system_message(
                "RAG not available — install chromadb: pip install chromadb", level="warn"
            )
            return
        if not self._project_dir:
            self.chat_panel.add_system_message(
                "Set a project directory first.", level="warn"
            )
            return
        self.chat_panel.add_system_message("Indexing project… this may take a while.", level="info")
        try:
            from engine.rag import index_project
            n = index_project(
                self._project_dir,
                progress_callback=lambda s, _: self.chat_panel.add_system_message(s),
            )
            self.chat_panel.add_system_message(f"Index complete — {n} chunks indexed.", level="ok")
            log(f"[main_window] RAG index: {n} chunks")
        except Exception as e:
            self.chat_panel.add_system_message(f"Index failed: {e}", level="err")

    # ----------------------------------------------------------------
    # Startup check
    # ----------------------------------------------------------------

    def _check_ollama_status(self) -> None:
        from engine.ollama_client import is_ollama_running
        if is_ollama_running():
            self._status_bar.showMessage("Ollama running  ✓")
        else:
            self._status_bar.showMessage("Ollama not detected — start Ollama then restart")
            self.chat_panel.add_system_message(
                "Ollama is not running. Start Ollama and relaunch VibeStudio.", level="warn"
            )

    # ----------------------------------------------------------------
    # Stylesheet
    # ----------------------------------------------------------------

    def _apply_stylesheet(self) -> None:
        p = PALETTE
        QApplication.instance().setStyleSheet(f"""
            QWidget {{
                background-color: {p['bg']};
                color: {p['text']};
                font-family: "Consolas", "Cascadia Code", "Courier New", monospace;
                font-size: 13px;
            }}

            /* Chat display */
            QTextEdit#chatDisplay {{
                background-color: {p['bg']};
                border: none;
                padding: 4px;
            }}
            QTextEdit#chatDisplay QScrollBar:vertical {{
                width: 6px;
                background: {p['bg2']};
                border: none;
            }}
            QTextEdit#chatDisplay QScrollBar::handle:vertical {{
                background: {p['border']};
                border-radius: 3px;
                min-height: 20px;
            }}

            /* Input box */
            QTextEdit#inputBox {{
                background-color: {p['bg2']};
                border: 1px solid {p['border']};
                border-radius: 4px;
                padding: 6px 8px;
                color: {p['text']};
            }}
            QTextEdit#inputBox:focus {{
                border-color: {p['accent']};
            }}

            /* All QPushButton defaults */
            QPushButton {{
                background-color: {p['bg3']};
                color: {p['text']};
                border: 1px solid {p['border']};
                border-radius: 4px;
                padding: 4px 12px;
            }}
            QPushButton:hover {{
                border-color: {p['accent']};
            }}
            QPushButton:disabled {{
                color: {p['text_dim']};
                border-color: {p['bg3']};
            }}

            /* Send button */
            QPushButton#sendButton {{
                background-color: {p['accent']};
                color: #000000;
                font-weight: bold;
                border: none;
            }}
            QPushButton#sendButton:hover {{
                background-color: #6ee7a0;
            }}
            QPushButton#sendButton:disabled {{
                background-color: #2d3748;
                color: {p['text_dim']};
            }}

            /* Stop button */
            QPushButton#stopButton {{
                background-color: {p['warn']};
                color: #000000;
                font-weight: bold;
                border: none;
            }}
            QPushButton#stopButton:hover {{
                background-color: #fdba74;
            }}

            /* Quick command buttons */
            QPushButton#cmdButton {{
                background-color: {p['bg2']};
                color: {p['text_dim']};
                border: 1px solid {p['bg3']};
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 11px;
            }}
            QPushButton#cmdButton:hover {{
                color: {p['text']};
                border-color: {p['border']};
            }}

            /* QListWidget */
            QListWidget {{
                background-color: {p['bg2']};
                border: 1px solid {p['border']};
                border-radius: 4px;
            }}
            QListWidget::item {{
                padding: 5px 8px;
            }}
            QListWidget::item:hover {{
                background-color: {p['bg3']};
            }}
            QListWidget::item:selected {{
                background-color: {p['bg3']};
                color: {p['accent']};
            }}

            /* QComboBox */
            QComboBox {{
                background-color: {p['bg2']};
                border: 1px solid {p['border']};
                border-radius: 4px;
                padding: 4px 8px;
                color: {p['text']};
            }}
            QComboBox:hover {{
                border-color: {p['accent']};
            }}
            QComboBox QAbstractItemView {{
                background-color: {p['bg2']};
                border: 1px solid {p['border']};
                selection-background-color: {p['bg3']};
            }}

            /* QSplitter */
            QSplitter::handle {{
                background-color: {p['border']};
                width: 1px;
            }}

            /* QStatusBar */
            QStatusBar {{
                background-color: {p['bg2']};
                color: {p['text_dim']};
                font-size: 11px;
            }}

            /* QScrollArea / sidebar */
            QScrollArea {{
                border: none;
                background-color: {p['bg2']};
            }}

            /* QLabel general */
            QLabel {{
                color: {p['text_dim']};
                font-size: 11px;
                letter-spacing: 0.5px;
            }}

            /* QCheckBox */
            QCheckBox {{
                color: {p['text']};
                font-size: 12px;
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
                border: 1px solid {p['border']};
                border-radius: 3px;
                background: {p['bg2']};
            }}
            QCheckBox::indicator:checked {{
                background: {p['accent']};
                border-color: {p['accent']};
            }}

            /* QInputDialog */
            QInputDialog {{
                background-color: {p['bg2']};
            }}
        """)
