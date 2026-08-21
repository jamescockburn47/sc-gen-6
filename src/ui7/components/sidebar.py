"""Minimal icon sidebar — 52px rail with text-label buttons."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPainter, QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.ui7.theme import C, S


class SidebarButton(QPushButton):
    """Sidebar button with single-letter icon and underline indicator."""

    def __init__(self, letter: str, tooltip: str, parent=None) -> None:
        super().__init__(parent)
        self._letter = letter
        self.setToolTip(tooltip)
        self.setCheckable(True)
        self.setFixedSize(44, 44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                border-radius: 10px;
                color: {C.TEXT_TERTIARY};
                font-family: "Inter", "Segoe UI", sans-serif;
                font-size: 15px;
                font-weight: 600;
                letter-spacing: 0.5px;
                padding: 0;
            }}
            QPushButton:hover {{
                background: {C.BG_RAISED};
                color: {C.TEXT_SECONDARY};
            }}
            QPushButton:checked {{
                background: {C.ACCENT_BG};
                color: {C.ACCENT};
                border: 1px solid {C.ACCENT_MUTED};
            }}
        """)

    def paintEvent(self, event) -> None:
        """Custom paint with centred letter."""
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont("Inter", 14)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)

        if self.isChecked():
            painter.setPen(QColor(C.ACCENT))
        elif self.underMouse():
            painter.setPen(QColor(C.TEXT_SECONDARY))
        else:
            painter.setPen(QColor(C.TEXT_TERTIARY))

        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._letter)
        painter.end()


class Sidebar(QWidget):
    """Vertical icon sidebar for workspace navigation."""

    workspace_changed = Signal(int)
    settings_clicked = Signal()

    ITEMS = [
        ("Q", "Query (Ctrl+1)"),
        ("D", "Documents (Ctrl+2)"),
        ("A", "Analysis (Ctrl+3)"),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(S.SIDEBAR_WIDTH)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {C.BG_SURFACE};
                border-right: 1px solid {C.BORDER};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 12, 4, 12)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Workspace buttons
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: list[SidebarButton] = []

        for i, (letter, tip) in enumerate(self.ITEMS):
            btn = SidebarButton(letter, tip)
            self._group.addButton(btn, i)
            self._buttons.append(btn)
            layout.addWidget(btn, 0, Qt.AlignmentFlag.AlignHCenter)

        layout.addStretch(1)

        # Settings button (bottom)
        self._settings_btn = SidebarButton("\u2731", "Settings (Ctrl+,)")  # ✱
        self._settings_btn.setCheckable(False)
        self._settings_btn.clicked.connect(self.settings_clicked.emit)
        layout.addWidget(self._settings_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        # Connect group
        self._group.idClicked.connect(self._on_clicked)

    def _on_clicked(self, idx: int) -> None:
        self.workspace_changed.emit(idx)

    def select(self, index: int) -> None:
        """Programmatically select a workspace."""
        if 0 <= index < len(self._buttons):
            self._buttons[index].setChecked(True)
            self.workspace_changed.emit(index)
