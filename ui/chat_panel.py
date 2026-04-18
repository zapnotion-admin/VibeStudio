"""
ui/chat_panel.py
Chat display widget. A styled, read-only QTextEdit that renders HTML
message blocks and streams AI output via a 40ms QTimer buffer.

Streaming lifecycle rules (spec §chat_panel, v3.1/v3.2):
  - start_ai_block() MUST be called BEFORE the worker thread starts
    (called from MainWindow._start_stream, never from inside a worker)
  - end_ai_block() MUST be called ONLY from _on_stream_finished
    (never from inside a worker)
  - append_ai_chunk() buffers chunks; _flush_buffer() fires every 40ms
  - end_ai_block() flushes remaining buffer before closing the block
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PySide6.QtCore    import Qt, QTimer
from PySide6.QtGui     import QTextCursor
from core.config       import PALETTE

# Stage header colours (pipeline display contract)
_STAGE_COLOURS = {
    "PLAN":   "#22d3ee",   # cyan
    "CODE":   "#4ade80",   # green
    "REVIEW": "#fb923c",   # orange
    "RETRY":  "#f87171",   # red
}

# System message level colours
_LEVEL_COLOURS = {
    "info": PALETTE["text_dim"],
    "ok":   PALETTE["accent"],
    "warn": PALETTE["warn"],
    "err":  PALETTE["err"],
}


class ChatPanel(QWidget):
    """
    Left-to-right, top-to-bottom chat display.
    All text is appended to a single QTextEdit as raw HTML.
    Plain-text streaming chunks are inserted at the cursor to avoid
    full document re-renders on every token.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── layout ──────────────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── chat display ────────────────────────────────────────────
        self._display = QTextEdit()
        self._display.setReadOnly(True)
        self._display.setObjectName("chatDisplay")
        # Ensure line-wrap doesn't create horizontal scrollbar noise
        self._display.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        layout.addWidget(self._display)

        # ── streaming state ─────────────────────────────────────────
        self._streaming        = False
        self._chunk_buffer: list[str] = []

        # 40ms QTimer flush — prevents QTextEdit bottleneck on fast streams
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(40)
        self._flush_timer.timeout.connect(self._flush_buffer)

    # ----------------------------------------------------------------
    # Public API — user & AI messages
    # ----------------------------------------------------------------

    def add_user_message(self, text: str) -> None:
        """Appends a styled user message block."""
        escaped = self._escape(text)
        html = (
            f'<div style="'
            f'padding: 10px 14px;'
            f'max-width: 900px;'
            f'line-height: 1.5;'
            f'background: {PALETTE["user_msg"]};'
            f'border-left: 3px solid {PALETTE["accent2"]};'
            f'border-radius: 6px;'
            f'">'
            f'<b style="color:{PALETTE["accent2"]};font-size:11px;letter-spacing:1px;">YOU</b>'
            f'<br/><br/>'
            f'<span style="color:{PALETTE["text"]};white-space:pre-wrap;">{escaped}</span>'
            f'</div>'
        )
        self._append_html(html)

    def start_ai_block(self, label: str = "QWEN3") -> None:
        """
        Opens a streaming AI response block.
        MUST be called BEFORE the worker thread starts — from _start_stream()
        in MainWindow, never from inside the worker.
        """
        self._streaming = True
        self._chunk_buffer.clear()
        colour = PALETTE["accent"]
        html = (
            f'<div style="'
            f'padding: 10px 14px;'
            f'max-width: 900px;'
            f'line-height: 1.5;'
            f'background: {PALETTE["ai_msg"]};'
            f'border-left: 3px solid {colour};'
            f'border-radius: 6px;'
            f'">'
            f'<b style="color:{colour};font-size:11px;letter-spacing:1px;">{label}</b>'
            f'<br/><br/>'
        )
        self._append_html(html)

    def append_ai_chunk(self, chunk: str) -> None:
        """
        Buffers an incoming streaming chunk.
        The 40ms QTimer drains the buffer — never calls insertText per-chunk.
        Safe to call from a signal connected to a worker in another thread.
        """
        self._chunk_buffer.append(chunk)
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def end_ai_block(self) -> None:
        """
        Closes the open AI streaming block.
        MUST be called ONLY from _on_stream_finished — never from inside a worker.
        Flushes any remaining buffered chunks before closing.
        """
        # Stop timer and flush remaining buffer
        self._flush_timer.stop()
        self._flush_buffer()
        self._streaming = False
        # Close the div opened in start_ai_block
        self._append_html('</div>')
        # Scroll to bottom
        self._display.verticalScrollBar().setValue(
            self._display.verticalScrollBar().maximum()
        )

    def add_system_message(self, text: str, level: str = "info") -> None:
        """
        Appends a system/status message in the appropriate colour.
        Levels: info (dim), ok (green), warn (orange), err (red).
        Used for pipeline stage status lines and error messages.
        """
        colour  = _LEVEL_COLOURS.get(level, PALETTE["text_dim"])
        escaped = self._escape(text)
        html = (
            f'<div style="'
            f'margin: 2px 4px;'
            f'padding: 6px 12px;'
            f'color:{colour};'
            f'font-style:italic;'
            f'white-space:pre-wrap;'
            f'">{escaped}</div>'
        )
        self._append_html(html)

    def add_stage_header(self, stage: str) -> None:
        """
        Inserts a bold pipeline stage separator (PLAN / CODE / REVIEW / RETRY).
        Called from MainWindow._on_pipeline_progress before add_system_message.
        """
        colour = _STAGE_COLOURS.get(stage, PALETTE["accent"])
        html = (
            f'<div style="'
            f'margin: 10px 4px 2px 4px;'
            f'padding: 4px 12px;'
            f'border-left: 3px solid {colour};'
            f'border-radius: 2px;'
            f'">'
            f'<span style="color:{colour};font-weight:bold;'
            f'font-size:11px;letter-spacing:2px;">── {stage} ──</span>'
            f'</div>'
        )
        self._append_html(html)

    def clear_chat(self) -> None:
        """Clears all chat content. Connected to sidebar clear_chat_requested."""
        self._flush_timer.stop()
        self._chunk_buffer.clear()
        self._streaming = False
        self._display.clear()

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _flush_buffer(self) -> None:
        """
        Drains the chunk buffer into the QTextEdit in one insertText call.
        Fires every 40ms while streaming. Stops the timer when buffer is empty.
        """
        if not self._chunk_buffer:
            self._flush_timer.stop()
            return
        text = "".join(self._chunk_buffer)
        self._chunk_buffer.clear()
        cursor = self._display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._display.setTextCursor(cursor)
        cursor.insertText(text)
        self._display.ensureCursorVisible()

    def _append_html(self, html: str) -> None:
        """
        Appends raw HTML to the display, always starting on a new block.
        insertBlock() is used before insertHtml() because QTextEdit's
        insertHtml() inserts at the cursor inline — it never opens a new
        paragraph on its own. Without insertBlock() first, all messages
        run together on one line regardless of <p> or <div> tags.
        """
        cursor = self._display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        # Only insert a block if the document already has content —
        # avoids a blank leading line at the top of a fresh chat.
        if not self._display.document().isEmpty():
            cursor.insertBlock()
        cursor.insertHtml(html)
        self._display.setTextCursor(cursor)
        self._display.verticalScrollBar().setValue(
            self._display.verticalScrollBar().maximum()
        )

    @staticmethod
    def _escape(text: str) -> str:
        """Minimal HTML escaping for user/system text injected into HTML blocks."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
