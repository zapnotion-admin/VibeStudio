"""
ui/chat_panel.py  — v9 widget-based message renderer

Each message is a self-contained QFrame in a QVBoxLayout inside a QScrollArea.
Content areas use read-only QTextEdit so text selection and Ctrl+C work
natively — QLabel blocks mouse events in scroll areas and can't be selected.

Public API (unchanged from v8 — main_window.py needs no edits):
    add_user_message(text)
    start_ai_block(label)
    append_ai_chunk(chunk)
    end_ai_block()
    add_system_message(text, level)
    add_stage_header(stage)
    clear_chat()
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QFrame,
    QLabel, QSizePolicy, QPushButton, QHBoxLayout, QTextEdit,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui  import QGuiApplication, QTextOption
from core.config    import PALETTE

# ── colour maps ─────────────────────────────────────────────────────────────

_STAGE_COLOURS = {
    "SCAN":   "#a78bfa",   # purple — analysis
    "PLAN":   "#22d3ee",   # cyan   — reasoning output
    "CODE":   "#4ade80",   # green  — code
    "REVIEW": "#fb923c",   # orange — review
    "RETRY":  "#f87171",   # red    — retry
}

_LEVEL_COLOURS = {
    "info": PALETTE["text_dim"],
    "ok":   PALETTE["accent"],
    "warn": PALETTE["warn"],
    "err":  PALETTE["err"],
}

# ── shared helpers ───────────────────────────────────────────────────────────

def _make_content_edit(bg: str) -> QTextEdit:
    """
    Read-only QTextEdit used for all message content.
    - Supports mouse selection, Ctrl+C, and right-click copy natively.
    - No scrollbar (height expands to content via document height).
    - No border, transparent background so the parent QFrame shows through.
    """
    te = QTextEdit()
    te.setReadOnly(True)
    te.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    te.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    te.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    te.setFrameShape(QFrame.Shape.NoFrame)
    te.setStyleSheet(
        f"QTextEdit {{"
        f"  background: {bg};"
        f"  color: {PALETTE['text']};"
        f"  border: none;"
        f"  font-size: 13px;"
        f"  line-height: 1.5;"
        f"  padding: 0px;"
        f"}}"
    )
    te.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    # Auto-resize height to content
    te.document().contentsChanged.connect(lambda: _fit_height(te))
    return te


def _fit_height(te: QTextEdit) -> None:
    """Resize QTextEdit to its document height so no internal scrollbar appears.
    Uses the widget's actual viewport width so word-wrap is accounted for."""
    doc = te.document()
    vw = te.viewport().width() if te.viewport().width() > 20 else te.width()
    if vw > 20:
        doc.setTextWidth(vw)
    doc_h = int(doc.size().height())
    te.setFixedHeight(max(doc_h + 8, 24))


def _copy_btn(get_text) -> QPushButton:
    """Returns a small styled copy button that calls get_text() on click."""
    btn = QPushButton("copy")
    btn.setFixedHeight(20)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setStyleSheet(f"""
        QPushButton {{
            color: {PALETTE['text_dim']};
            background: transparent;
            border: 1px solid {PALETTE['border']};
            border-radius: 3px;
            font-size: 10px;
            padding: 0 6px;
        }}
        QPushButton:hover {{
            color: {PALETTE['text']};
            border-color: {PALETTE['accent']};
        }}
    """)

    def _do_copy():
        QGuiApplication.clipboard().setText(get_text())
        btn.setText("✓")
        QTimer.singleShot(1500, lambda: btn.setText("copy"))

    btn.clicked.connect(_do_copy)
    return btn


def _label_widget(text: str, colour: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{colour}; font-weight:bold; font-size:11px; "
        f"letter-spacing:1px; background:transparent; border:none;"
    )
    return lbl


# ── message widgets ──────────────────────────────────────────────────────────

class _MessageWidget(QFrame):
    """Completed (non-streaming) message bubble — user or finalised AI."""

    def __init__(self, label: str, label_colour: str,
                 border_colour: str, bg_colour: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._bg = bg_colour
        self.setStyleSheet(
            f"QFrame {{ background:{bg_colour}; border-left:3px solid {border_colour};"
            f" border-radius:6px; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        # label row
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(_label_widget(label, label_colour))
        row.addStretch()
        self._content_edit = _make_content_edit(bg_colour)
        row.addWidget(_copy_btn(lambda: self._content_edit.toPlainText()))
        layout.addLayout(row)

        layout.addWidget(self._content_edit)

    def set_text(self, text: str) -> None:
        self._content_edit.setPlainText(text)


class _StreamingWidget(QFrame):
    """
    Live AI response widget. Chunks stream into a read-only QTextEdit.
    The 40ms timer batches screen updates to avoid per-token repaints.
    finish() is called by end_ai_block() to flush and stop the timer.
    """

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        colour = PALETTE["accent"]
        bg     = PALETTE["ai_msg"]
        self.setStyleSheet(
            f"QFrame {{ background:{bg}; border-left:3px solid {colour};"
            f" border-radius:6px; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        # label row
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(_label_widget(label, colour))
        row.addStretch()
        self._edit = _make_content_edit(bg)
        row.addWidget(_copy_btn(lambda: self._edit.toPlainText()))
        layout.addLayout(row)

        layout.addWidget(self._edit)

        # streaming state
        self._buffer: list[str] = []
        self._full   = ""
        self._timer  = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._flush)

    def push_chunk(self, chunk: str) -> None:
        self._buffer.append(chunk)
        if not self._timer.isActive():
            self._timer.start()

    def _flush(self) -> None:
        if not self._buffer:
            self._timer.stop()
            return
        self._full += "".join(self._buffer)
        self._buffer.clear()
        self._edit.setPlainText(self._full)

    def finish(self) -> str:
        self._timer.stop()
        if self._buffer:
            self._full += "".join(self._buffer)
            self._buffer.clear()
            self._edit.setPlainText(self._full)
        return self._full


class _SystemWidget(QFrame):
    """Status / pipeline output — selectable QTextEdit so text can be copied."""

    def __init__(self, text: str, colour: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("background:transparent;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 2, 14, 2)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText(text)
        te.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        te.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        te.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        te.setFrameShape(QFrame.Shape.NoFrame)
        te.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        te.setStyleSheet(
            f"QTextEdit {{ color:{colour}; font-style:italic; font-size:11px; "
            f"background:transparent; border:none; padding:0px; }}"
        )
        te.document().contentsChanged.connect(lambda: _fit_height(te))
        _fit_height(te)
        layout.addWidget(te)


class _StageWidget(QFrame):
    """Bold pipeline stage separator."""

    def __init__(self, stage: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        colour = _STAGE_COLOURS.get(stage, PALETTE["accent"])
        self.setStyleSheet(
            f"background:transparent; border-left:3px solid {colour}; border-radius:2px;"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        lbl = QLabel(f"── {stage} ──")
        lbl.setStyleSheet(
            f"color:{colour}; font-weight:bold; font-size:11px; "
            f"letter-spacing:2px; background:transparent; border:none;"
        )
        layout.addWidget(lbl)


class _PipelineResponseWidget(QFrame):
    """
    Single selectable block for the entire pipeline response (PLAN/CODE/REVIEW).
    All stages are appended into one QTextEdit so the user can select across them.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        bg     = PALETTE["ai_msg"]
        colour = PALETTE["accent"]
        self.setStyleSheet(
            f"QFrame {{ background:{bg}; border-left:3px solid {colour};"
            f" border-radius:6px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        lbl = QLabel("PIPELINE")
        lbl.setStyleSheet(
            f"color:{colour}; font-weight:bold; font-size:11px; "
            f"letter-spacing:1px; background:transparent; border:none;"
        )
        row.addWidget(lbl)
        row.addStretch()
        self._edit = _make_content_edit(bg)
        row.addWidget(_copy_btn(lambda: self._edit.toPlainText()))
        layout.addLayout(row)
        layout.addWidget(self._edit)

    def append_stage(self, stage: str, text: str) -> None:
        colour = _STAGE_COLOURS.get(stage, PALETTE["accent"])
        cursor = self._edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        prefix = "\n\n" if self._edit.toPlainText() else ""
        cursor.insertHtml(
            f"{prefix}<span style=\"color:{colour}; font-weight:bold; "
            f"font-size:11px; letter-spacing:2px;\">── {stage} ──</span><br>"
        )
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self._edit.setTextCursor(cursor)

    def append_status(self, text: str) -> None:
        cursor = self._edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        if self._edit.toPlainText():
            cursor.insertText("\n")
        cursor.insertHtml(
            f"<span style=\"color:{PALETTE['text_dim']}; font-style:italic;\">"
            f"{text}</span><br>"
        )
        self._edit.setTextCursor(cursor)


class _DividerWidget(QFrame):
    """Thin horizontal rule between turns."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFixedHeight(1)
        self.setStyleSheet(f"background:{PALETTE['border']}; border:none;")


# ── main ChatPanel ───────────────────────────────────────────────────────────

class ChatPanel(QWidget):
    """Scrollable widget-based chat display."""

    def __init__(self, parent=None):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── toolbar ──────────────────────────────────────────────────
        toolbar = QWidget()
        toolbar.setStyleSheet(
            f"background:{PALETTE['bg2']}; border-bottom:1px solid {PALETTE['border']};"
        )
        tl = QHBoxLayout(toolbar)
        tl.setContentsMargins(8, 4, 8, 4)
        tl.setSpacing(6)
        tl.addStretch()
        self._copy_all_btn = QPushButton("Copy conversation")
        self._copy_all_btn.setFixedHeight(24)
        self._copy_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_all_btn.setStyleSheet(f"""
            QPushButton {{
                color: {PALETTE['text_dim']};
                background: transparent;
                border: 1px solid {PALETTE['border']};
                border-radius: 3px;
                font-size: 11px;
                padding: 0 10px;
            }}
            QPushButton:hover {{
                color: {PALETTE['text']};
                border-color: {PALETTE['accent']};
            }}
        """)
        self._copy_all_btn.clicked.connect(self._copy_all)
        tl.addWidget(self._copy_all_btn)
        outer.addWidget(toolbar)
        # ─────────────────────────────────────────────────────────────

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(
            f"QScrollArea {{ border:none; background:{PALETTE['bg']}; }}"
        )
        outer.addWidget(self._scroll)

        self._container = QWidget()
        self._container.setStyleSheet(f"background:{PALETTE['bg']};")
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(8)
        self._layout.addStretch()

        self._scroll.setWidget(self._container)
        self._live: _StreamingWidget | None = None
        self._pipeline: _PipelineResponseWidget | None = None

    # ── public API ───────────────────────────────────────────────────

    def start_pipeline_block(self) -> None:
        """Begin a new pipeline response block (replaces separate stage widgets)."""
        self._add_divider()
        self._pipeline = _PipelineResponseWidget()
        self._add_widget(self._pipeline)

    def append_pipeline_stage(self, stage: str, text: str) -> None:
        """Add a stage (PLAN/CODE/REVIEW/RETRY) into the current pipeline block."""
        if self._pipeline:
            self._pipeline.append_stage(stage, text)
            self._scroll_to_bottom()

    def append_pipeline_status(self, text: str) -> None:
        """Add a status line into the current pipeline block."""
        if self._pipeline:
            self._pipeline.append_status(text)
            self._scroll_to_bottom()

    def end_pipeline_block(self) -> None:
        self._pipeline = None
        self._scroll_to_bottom()

    def add_user_message(self, text: str) -> None:
        self._add_divider()
        w = _MessageWidget(
            label="YOU",
            label_colour=PALETTE["accent2"],
            border_colour=PALETTE["accent2"],
            bg_colour=PALETTE["user_msg"],
        )
        w.set_text(text)
        self._add_widget(w)

    def start_ai_block(self, label: str = "QWEN3") -> None:
        self._live = _StreamingWidget(label)
        self._add_widget(self._live)

    def append_ai_chunk(self, chunk: str) -> None:
        if self._live:
            self._live.push_chunk(chunk)

    def end_ai_block(self) -> None:
        if self._live:
            self._live.finish()
            self._live = None
        self._scroll_to_bottom()

    def add_system_message(self, text: str, level: str = "info") -> None:
        colour = _LEVEL_COLOURS.get(level, PALETTE["text_dim"])
        self._add_widget(_SystemWidget(text, colour))

    def add_stage_header(self, stage: str) -> None:
        self._add_widget(_StageWidget(stage))

    def clear_chat(self) -> None:
        if self._live:
            self._live.finish()
            self._live = None
        self._pipeline = None
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    # ── internal ─────────────────────────────────────────────────────

    def get_full_transcript(self) -> str:
        """
        Walks all message widgets in order and assembles a plain-text
        transcript of the entire conversation.
        Each widget contributes its QTextEdit content with a label prefix.
        """
        lines = []
        layout = self._layout
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if not item:
                continue
            w = item.widget()
            if w is None:
                continue
            # Each message widget has one or more QTextEdit children
            edits = w.findChildren(QTextEdit)
            if not edits:
                continue
            # Determine label from widget type
            if isinstance(w, _MessageWidget):
                label = "YOU"
            elif isinstance(w, _StreamingWidget):
                label = "AI"
            elif isinstance(w, _PipelineResponseWidget):
                label = "PIPELINE"
            elif isinstance(w, _SystemWidget):
                label = None  # status lines — include without header
            else:
                label = None

            text = "\n".join(te.toPlainText() for te in edits).strip()
            if not text:
                continue

            if label:
                lines.append(f"[{label}]\n{text}")
            else:
                lines.append(text)

        return "\n\n".join(lines)

    def _copy_all(self) -> None:
        transcript = self.get_full_transcript()
        if transcript:
            QGuiApplication.clipboard().setText(transcript)
            self._copy_all_btn.setText("Copied!")
            QTimer.singleShot(2000, lambda: self._copy_all_btn.setText("Copy conversation"))
        else:
            self._copy_all_btn.setText("Nothing to copy")
            QTimer.singleShot(2000, lambda: self._copy_all_btn.setText("Copy conversation"))

    def _add_widget(self, w: QWidget) -> None:
        self._layout.insertWidget(self._layout.count() - 1, w)
        self._scroll_to_bottom()

    def _add_divider(self) -> None:
        if self._layout.count() > 1:
            self._layout.insertWidget(self._layout.count() - 1, _DividerWidget())

    def _scroll_to_bottom(self) -> None:
        sb = self._scroll.verticalScrollBar()
        QTimer.singleShot(20, lambda: sb.setValue(sb.maximum()))
