"""
ui/input_panel.py
Bottom input area: multi-line text input, Send button, Stop button,
and optional quick-command shortcuts.

Signals:
  send_requested(str)  — emitted when user presses Enter or clicks Send
  stop_requested()     — emitted when user clicks [Stop] during generation

set_sending(True)  → disables input + send, shows [Stop]
set_sending(False) → re-enables input + send, hides [Stop]
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit
from PySide6.QtCore    import Qt, Signal
from PySide6.QtGui     import QKeyEvent
from core.config       import PALETTE


class _InputBox(QTextEdit):
    """
    QTextEdit that intercepts Enter/Shift+Enter.
    Enter alone  → emits enter_pressed signal (send)
    Shift+Enter  → inserts a newline (multi-line input)
    """
    enter_pressed = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if (
            event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        ):
            self.enter_pressed.emit()
        else:
            super().keyPressEvent(event)


class InputPanel(QWidget):
    send_requested = Signal(str)   # text content of the input box
    stop_requested = Signal()      # [v3.1] user hit [Stop] during generation

    # Line height used for min/max height calculations (approximate at 13pt)
    _LINE_H = 20

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(4)

        # ── quick command row ────────────────────────────────────────
        cmd_row = QHBoxLayout()
        cmd_row.setSpacing(4)
        cmd_row.setContentsMargins(0, 0, 0, 0)

        for label, cmd in [("/clear", "/clear"), ("/help", "/help")]:
            btn = QPushButton(label)
            btn.setObjectName("cmdButton")
            btn.setFixedHeight(22)
            btn.clicked.connect(lambda checked, c=cmd: self._insert_command(c))
            cmd_row.addWidget(btn)

        cmd_row.addStretch()
        layout.addLayout(cmd_row)

        # ── input box ────────────────────────────────────────────────
        self._input = _InputBox()
        self._input.setObjectName("inputBox")
        self._input.setPlaceholderText("Ask anything, or load files for Code Mode…")
        self._input.setMinimumHeight(self._LINE_H * 3 + 16)
        self._input.setMaximumHeight(self._LINE_H * 8 + 16)
        self._input.enter_pressed.connect(self._on_enter)
        layout.addWidget(self._input)

        # ── button row ───────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.addStretch()

        # [Stop] — hidden by default, shown during generation
        self._stop_btn = QPushButton("[Stop]")
        self._stop_btn.setObjectName("stopButton")
        self._stop_btn.setFixedHeight(32)
        self._stop_btn.setMinimumWidth(72)
        self._stop_btn.clicked.connect(self.stop_requested)
        self._stop_btn.hide()
        btn_row.addWidget(self._stop_btn)

        # Send
        self._send_btn = QPushButton("Send  ↵")
        self._send_btn.setObjectName("sendButton")
        self._send_btn.setFixedHeight(32)
        self._send_btn.setMinimumWidth(88)
        self._send_btn.clicked.connect(self._on_enter)
        btn_row.addWidget(self._send_btn)

        layout.addLayout(btn_row)

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def set_sending(self, is_sending: bool) -> None:
        """
        True  → lock input for generation: disable input + send, show [Stop].
        False → unlock: re-enable input + send, hide [Stop].
        """
        self._input.setEnabled(not is_sending)
        self._send_btn.setEnabled(not is_sending)
        if is_sending:
            self._stop_btn.show()
        else:
            self._stop_btn.hide()
            # Return focus to input box when generation ends
            self._input.setFocus()

    # ----------------------------------------------------------------
    # Internal
    # ----------------------------------------------------------------

    def _on_enter(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._input.clear()
        self.send_requested.emit(text)

    def _insert_command(self, cmd: str) -> None:
        """Appends a quick command string to the input box."""
        self._input.setPlainText(cmd)
        self._input.setFocus()
        # Move cursor to end
        cursor = self._input.textCursor()
        cursor.moveToEnd = getattr(cursor, 'movePosition', None)
        from PySide6.QtGui import QTextCursor
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._input.setTextCursor(cursor)
