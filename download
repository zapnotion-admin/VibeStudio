"""
ui/sidebar.py
Left panel with all app controls. Wrapped in a QScrollArea so it
handles long file lists and small screens gracefully.

Sections (top -> bottom):
  1. Mode indicator  -- "Chat Mode" / "Code Mode" / "Agent Mode"
  2. Project picker  -- directory selector
  3. Context files   -- file list with add/remove buttons
  4. Sessions        -- save/load/delete named sessions
  5. Model selector  -- QComboBox + VRAM estimate label
  6. Execution mode  -- Stable / Fast toggle
  7. Agent section   -- Write files to disk checkbox
  8. RAG controls    -- only shown when chromadb is available
  9. Project Brief   -- Edit Brief button
  10. Clear Chat     -- bottom utility button

All user actions emit signals only -- no direct state changes.
MainWindow owns all app state and handles signal responses.

v2 fixes:
  - Execution mode tooltip clarified: stable = more retries, not "stricter verification"
  - Stable mode checkbox default reads from config correctly
  - _on_stable_mode_changed emits correct bool
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QComboBox, QCheckBox,
    QScrollArea, QFileDialog, QSizePolicy, QFrame,
)
from PySide6.QtCore import Qt, Signal

from core.config import MODEL_CODER, MODEL_REASONER, MODEL_FALLBACK, VRAM_ESTIMATES, PALETTE, STABLE_MODE

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java",
    ".cpp", ".c", ".h", ".cs", ".go", ".rs",
    ".sql", ".md", ".yaml", ".yml", ".toml", ".json",
}


def _section_label(text):
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        "color: {}; font-size: 10px; letter-spacing: 1.5px; padding: 8px 0 2px 0;".format(
            PALETTE["text_dim"]
        )
    )
    return lbl


def _separator():
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: {}; margin: 4px 0;".format(PALETTE["border"]))
    return line


class Sidebar(QWidget):
    project_changed         = Signal(str)
    files_changed           = Signal(list)
    model_changed           = Signal(str)
    session_load_requested  = Signal(str)
    session_save_requested  = Signal()
    clear_chat_requested    = Signal()
    index_project_requested = Signal()
    edit_brief_requested    = Signal()
    coder_changed           = Signal(str)
    reasoner_changed        = Signal(str)
    stable_mode_changed     = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(220)
        self.setMaximumWidth(320)

        self._context_files = []

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        self._layout = QVBoxLayout(content)
        self._layout.setContentsMargins(10, 8, 10, 8)
        self._layout.setSpacing(2)

        self._build_mode_indicator()
        self._layout.addWidget(_separator())
        self._build_project_section()
        self._layout.addWidget(_separator())
        self._build_files_section()
        self._layout.addWidget(_separator())
        self._build_sessions_section()
        self._layout.addWidget(_separator())
        self._build_model_section()
        self._build_execution_mode_section()
        self._build_agent_section()
        self._build_rag_section()
        self._build_brief_section()
        self._layout.addStretch()

        scroll.setWidget(content)
        outer_layout.addWidget(scroll, stretch=1)

        clear_btn = QPushButton("Clear Chat")
        clear_btn.setObjectName("clearButton")
        clear_btn.clicked.connect(self.clear_chat_requested)
        clear_btn.setStyleSheet(
            "margin: 6px 10px; padding: 6px; color: {}; "
            "border: 1px solid {}; border-radius: 4px; background: transparent;".format(
                PALETTE["text_dim"], PALETTE["border"]
            )
        )
        outer_layout.addWidget(clear_btn)

    # ----------------------------------------------------------------
    # Section builders
    # ----------------------------------------------------------------

    def _build_mode_indicator(self):
        self._mode_label = QLabel("\U0001f4ac  Chat Mode")
        self._mode_label.setStyleSheet(
            "color: {}; font-size: 13px; font-weight: bold; padding: 4px 0 6px 0;".format(
                PALETTE["accent2"]
            )
        )
        self._layout.addWidget(self._mode_label)

    def _build_project_section(self):
        self._layout.addWidget(_section_label("Project"))
        row = QHBoxLayout()
        row.setSpacing(4)

        self._project_label = QLabel("No project set")
        self._project_label.setStyleSheet(
            "color: {}; font-size: 11px;".format(PALETTE["text"])
        )
        self._project_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        row.addWidget(self._project_label)

        browse_btn = QPushButton("...")
        browse_btn.setFixedSize(28, 24)
        browse_btn.setToolTip("Choose project folder")
        browse_btn.clicked.connect(self._browse_project)
        row.addWidget(browse_btn)
        self._layout.addLayout(row)

    def _build_files_section(self):
        self._layout.addWidget(_section_label("Context Files"))

        self._files_list = QListWidget()
        self._files_list.setMaximumHeight(140)
        self._files_list.setToolTip("Files injected into every prompt")
        self._layout.addWidget(self._files_list)

        row = QHBoxLayout()
        row.setSpacing(4)

        add_file_btn = QPushButton("+ File")
        add_file_btn.setToolTip("Add individual files")
        add_file_btn.clicked.connect(self._add_files)
        row.addWidget(add_file_btn)

        add_folder_btn = QPushButton("+ Folder")
        add_folder_btn.setToolTip("Add all code files from a folder")
        add_folder_btn.clicked.connect(self._add_folder_files)
        row.addWidget(add_folder_btn)

        remove_btn = QPushButton("- Remove")
        remove_btn.setToolTip("Remove selected file")
        remove_btn.clicked.connect(self._remove_selected_file)
        row.addWidget(remove_btn)
        self._layout.addLayout(row)

    def _build_sessions_section(self):
        self._layout.addWidget(_section_label("Sessions"))

        self._sessions_list = QListWidget()
        self._sessions_list.setMaximumHeight(120)
        self._sessions_list.itemDoubleClicked.connect(self._on_session_double_click)
        self._layout.addWidget(self._sessions_list)
        self._load_sessions()

        row = QHBoxLayout()
        row.setSpacing(4)

        save_btn = QPushButton("Save")
        save_btn.setToolTip("Save current session")
        save_btn.clicked.connect(self.session_save_requested)
        row.addWidget(save_btn)

        delete_btn = QPushButton("Delete")
        delete_btn.setToolTip("Delete selected session")
        delete_btn.clicked.connect(self._delete_session)
        row.addWidget(delete_btn)
        self._layout.addLayout(row)

    def _build_model_section(self):
        self._layout.addWidget(_section_label("Models"))

        available = self._get_available_models()

        # Coder model
        coder_label = QLabel("🔧 Coder  (Scan / Code / Review)")
        coder_label.setStyleSheet(
            "color: {}; font-size: 10px; padding: 2px 0 0 0;".format(PALETTE["accent"])
        )
        self._layout.addWidget(coder_label)
        self._coder_combo = QComboBox()
        self._coder_combo.addItems(available if available else [MODEL_CODER, MODEL_FALLBACK])
        if MODEL_CODER in available:
            self._coder_combo.setCurrentText(MODEL_CODER)
        self._coder_combo.currentTextChanged.connect(self._on_coder_changed)
        self._layout.addWidget(self._coder_combo)

        vram_c = QLabel(VRAM_ESTIMATES.get(MODEL_CODER, ""))
        vram_c.setStyleSheet(
            "color: {}; font-size: 10px; padding: 0 0 4px 2px;".format(PALETTE["text_dim"])
        )
        self._layout.addWidget(vram_c)
        self._coder_vram_label = vram_c

        # Reasoner model
        reason_label = QLabel("🧠 Reasoner  (Plan stage)")
        reason_label.setStyleSheet(
            "color: {}; font-size: 10px; padding: 2px 0 0 0;".format(PALETTE["warn"])
        )
        self._layout.addWidget(reason_label)
        self._reasoner_combo = QComboBox()
        self._reasoner_combo.addItems(available if available else [MODEL_REASONER, MODEL_FALLBACK])
        if MODEL_REASONER in available:
            self._reasoner_combo.setCurrentText(MODEL_REASONER)
        self._reasoner_combo.currentTextChanged.connect(self._on_reasoner_changed)
        self._layout.addWidget(self._reasoner_combo)

        vram_r = QLabel(VRAM_ESTIMATES.get(MODEL_REASONER, ""))
        vram_r.setStyleSheet(
            "color: {}; font-size: 10px; padding: 0 0 4px 2px;".format(PALETTE["text_dim"])
        )
        self._layout.addWidget(vram_r)
        self._reasoner_vram_label = vram_r

        note = QLabel("One model in VRAM at a time")
        note.setStyleSheet(
            "color: {}; font-size: 10px; font-style: italic; padding: 0 0 2px 0;".format(
                PALETTE["text_dim"]
            )
        )
        self._layout.addWidget(note)

    def _build_execution_mode_section(self):
        self._layout.addWidget(_separator())
        self._layout.addWidget(_section_label("Execution Mode"))

        self._stable_radio = QCheckBox("⚡ Stable Mode")
        self._stable_radio.setChecked(STABLE_MODE)
        self._stable_radio.setToolTip(
            "Stable: 2 retries per step — more robust for complex tasks.\n\n"
            "Fast: 1 retry per step — quicker iterations for simple changes.\n\n"
            "Both modes use the same verification logic."
        )
        self._stable_radio.stateChanged.connect(self._on_stable_mode_changed)
        self._layout.addWidget(self._stable_radio)

        self._mode_desc = QLabel(
            "2 retries/step — robust" if STABLE_MODE else "1 retry/step — fast"
        )
        self._mode_desc.setStyleSheet(
            "color: {}; font-size: 10px; padding: 0 0 4px 2px;".format(PALETTE["text_dim"])
        )
        self._layout.addWidget(self._mode_desc)

    def _build_agent_section(self):
        self._layout.addWidget(_separator())
        self._layout.addWidget(_section_label("Agent"))

        self._writes_checkbox = QCheckBox("Write files to disk")
        self._writes_checkbox.setToolTip(
            "When enabled, the pipeline will automatically write generated\n"
            "files into your project folder after each run.\n\n"
            "Only files inside the project folder can be written."
        )
        self._writes_checkbox.stateChanged.connect(lambda _: self._update_mode())
        self._layout.addWidget(self._writes_checkbox)

        warn = QLabel("⚠ Writes are permanent — use a project folder")
        warn.setStyleSheet(
            "color: {}; font-size: 10px; padding: 0 0 4px 2px;".format(PALETTE["warn"])
        )
        self._layout.addWidget(warn)

    def _build_rag_section(self):
        from engine.rag import is_available
        if not is_available():
            return

        self._layout.addWidget(_separator())
        self._layout.addWidget(_section_label("Project Index (RAG)"))

        self._rag_checkbox = QCheckBox("Use Project Index")
        self._rag_checkbox.setToolTip(
            "Inject semantically relevant code chunks into each prompt"
        )
        self._layout.addWidget(self._rag_checkbox)

        index_btn = QPushButton("Index Project")
        index_btn.setToolTip("Build or rebuild the project index")
        index_btn.clicked.connect(self.index_project_requested)
        self._layout.addWidget(index_btn)

    def _build_brief_section(self):
        self._layout.addWidget(_separator())
        self._layout.addWidget(_section_label("Project Brief"))

        brief_btn = QPushButton("Edit Brief")
        brief_btn.setToolTip(
            "Edit ZAPCODEFORGE_BRIEF.md — gives the AI persistent context\n"
            "about your project across all pipeline runs."
        )
        brief_btn.clicked.connect(self.edit_brief_requested)
        self._layout.addWidget(brief_btn)

        info = QLabel("Keeps AI aligned across runs")
        info.setStyleSheet(
            "color: {}; font-size: 10px; padding: 0 0 4px 2px;".format(PALETTE["text_dim"])
        )
        self._layout.addWidget(info)

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def refresh_sessions(self):
        self._load_sessions()

    def rag_enabled(self):
        return hasattr(self, "_rag_checkbox") and self._rag_checkbox.isChecked()

    def allow_writes_enabled(self):
        return hasattr(self, "_writes_checkbox") and self._writes_checkbox.isChecked()

    def stable_mode_enabled(self):
        return hasattr(self, "_stable_radio") and self._stable_radio.isChecked()

    def get_coder_model(self) -> str:
        if hasattr(self, "_coder_combo"):
            return self._coder_combo.currentText()
        return MODEL_CODER

    def get_reasoner_model(self) -> str:
        if hasattr(self, "_reasoner_combo"):
            return self._reasoner_combo.currentText()
        return MODEL_REASONER

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _get_available_models(self) -> list:
        try:
            from engine.ollama_client import list_local_models
            models = list_local_models()
            return models if models else [MODEL_CODER, MODEL_REASONER, MODEL_FALLBACK]
        except Exception:
            return [MODEL_CODER, MODEL_REASONER, MODEL_FALLBACK]

    def _on_stable_mode_changed(self, state: int) -> None:
        enabled = bool(state)
        if hasattr(self, "_mode_desc"):
            self._mode_desc.setText(
                "2 retries/step — robust" if enabled else "1 retry/step — fast"
            )
        self.stable_mode_changed.emit(enabled)

    def _on_coder_changed(self, model: str) -> None:
        if hasattr(self, "_coder_vram_label"):
            self._coder_vram_label.setText(VRAM_ESTIMATES.get(model, ""))
        self.coder_changed.emit(model)

    def _on_reasoner_changed(self, model: str) -> None:
        if hasattr(self, "_reasoner_vram_label"):
            self._reasoner_vram_label.setText(VRAM_ESTIMATES.get(model, ""))
        self.reasoner_changed.emit(model)

    def _load_sessions(self):
        from core.session import list_sessions
        self._sessions_list.clear()
        for name in list_sessions():
            self._sessions_list.addItem(name)

    def _update_mode(self):
        writes   = hasattr(self, "_writes_checkbox") and self._writes_checkbox.isChecked()
        has_files = bool(self._context_files)
        if writes and has_files:
            self._mode_label.setText("🤖  Agent Mode")
            self._mode_label.setStyleSheet(
                "color: {}; font-size: 13px; font-weight: bold; padding: 4px 0 6px 0;".format(
                    PALETTE["accent"]
                )
            )
        elif writes:
            self._mode_label.setText("🤖  Agent Mode  (no context files)")
            self._mode_label.setStyleSheet(
                "color: {}; font-size: 13px; font-weight: bold; padding: 4px 0 6px 0;".format(
                    PALETTE["accent"]
                )
            )
        elif has_files:
            self._mode_label.setText("⚡  Code Mode")
            self._mode_label.setStyleSheet(
                "color: {}; font-size: 13px; font-weight: bold; padding: 4px 0 6px 0;".format(
                    PALETTE["warn"]
                )
            )
        else:
            self._mode_label.setText("💬  Chat Mode")
            self._mode_label.setStyleSheet(
                "color: {}; font-size: 13px; font-weight: bold; padding: 4px 0 6px 0;".format(
                    PALETTE["accent2"]
                )
            )

    def _emit_files_changed(self):
        self._update_mode()
        self.files_changed.emit(list(self._context_files))

    def _refresh_files_list(self):
        self._files_list.clear()
        for fp in self._context_files:
            item = QListWidgetItem(Path(fp).name)
            item.setToolTip(fp)
            self._files_list.addItem(item)

    # ----------------------------------------------------------------
    # Slots
    # ----------------------------------------------------------------

    def _browse_project(self):
        path = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if path:
            short = path[-35:] if len(path) > 35 else path
            display = ("..." + short) if len(path) > 35 else path
            self._project_label.setText(display)
            self._project_label.setToolTip(path)
            self.project_changed.emit(path)

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add Files", "", "All Files (*)")
        if paths:
            for p in paths:
                if p not in self._context_files:
                    self._context_files.append(p)
            self._refresh_files_list()
            self._emit_files_changed()

    def _add_folder_files(self):
        folder = QFileDialog.getExistingDirectory(self, "Add Folder Files")
        if not folder:
            return
        added = 0
        for p in Path(folder).rglob("*"):
            if (
                p.is_file()
                and p.suffix.lower() in CODE_EXTENSIONS
                and str(p) not in self._context_files
            ):
                self._context_files.append(str(p))
                added += 1
        if added:
            self._refresh_files_list()
            self._emit_files_changed()

    def _remove_selected_file(self):
        selected = self._files_list.selectedItems()
        if not selected:
            return
        for item in selected:
            full_path = item.toolTip()
            if full_path in self._context_files:
                self._context_files.remove(full_path)
        self._refresh_files_list()
        self._emit_files_changed()

    def _on_session_double_click(self, item):
        self.session_load_requested.emit(item.text())

    def _delete_session(self):
        selected = self._sessions_list.selectedItems()
        if not selected:
            return
        from core.session import delete_session
        for item in selected:
            delete_session(item.text())
        self._load_sessions()
