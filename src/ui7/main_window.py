"""SCGen7 main window — 3-workspace layout with icon sidebar.

Layout:
  ┌──────────────────────────────────────────────────────┐
  │  Status bar (LLM model · GPU · connection)     ⌘K   │
  ├────┬─────────────────────────────────────────────────┤
  │    │                                                 │
  │ 💬 │  Active workspace content                       │
  │ 📁 │                                                 │
  │ 🔗 │  Query / Documents / Analysis                   │
  │    │                                                 │
  │ ⚙  │                                                 │
  ├────┴─────────────────────────────────────────────────┤
  │  Footer: chunks · index status · matter              │
  └──────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QSize, Signal, QTimer
from PySide6.QtGui import QAction, QFont, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.ui7.theme import C, S, T, STYLESHEET
from src.ui7.components.status_bar import StatusBar
from src.ui7.components.footer_bar import FooterBar
from src.ui7.components.sidebar import Sidebar
from src.ui7.workspaces.query_workspace import QueryWorkspace
from src.ui7.workspaces.documents_workspace import DocumentsWorkspace
from src.ui7.workspaces.analysis_workspace import AnalysisWorkspace
from src.ui7.panels.settings_panel import SettingsPanel


class SCGen7Window(QMainWindow):
    """Main application window for SCGen7."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SCGen7")
        self.setMinimumSize(1100, 700)
        self.resize(1400, 900)
        self.setStyleSheet(STYLESHEET)

        self._build_ui()
        self._connect_signals()
        self._setup_shortcuts()

        # Start on Query workspace
        self.sidebar.select(0)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Assemble the main window layout."""
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Top status bar ---
        self.status_bar = StatusBar()
        root.addWidget(self.status_bar)

        # --- Middle: sidebar + workspace stack ---
        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)

        # Sidebar (48px icon rail)
        self.sidebar = Sidebar()
        middle.addWidget(self.sidebar)

        # Separator line
        sep = QWidget()
        sep.setFixedWidth(1)
        sep.setStyleSheet(f"background-color: {C.BORDER};")
        middle.addWidget(sep)

        # Workspace stack
        self.workspace_stack = QStackedWidget()
        self._init_workspaces()
        middle.addWidget(self.workspace_stack, 1)

        root.addLayout(middle, 1)

        # --- Settings panel (slide-over, hidden by default) ---
        self.settings_panel = SettingsPanel(self)
        self.settings_panel.hide()

        # --- Bottom footer bar ---
        self.footer_bar = FooterBar()
        root.addWidget(self.footer_bar)

    def _init_workspaces(self) -> None:
        """Create and add the three workspaces to the stack."""
        self.query_ws = QueryWorkspace()
        self.documents_ws = DocumentsWorkspace()
        self.analysis_ws = AnalysisWorkspace()

        self.workspace_stack.addWidget(self.query_ws)      # index 0
        self.workspace_stack.addWidget(self.documents_ws)   # index 1
        self.workspace_stack.addWidget(self.analysis_ws)    # index 2

    # ------------------------------------------------------------------
    # Signals & shortcuts
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        """Wire up cross-component signals."""
        self.sidebar.workspace_changed.connect(self._on_workspace_changed)
        self.sidebar.settings_clicked.connect(self._toggle_settings)

        # Documents workspace → footer update
        self.documents_ws.index_updated.connect(self.footer_bar.update_chunk_count)

    def _setup_shortcuts(self) -> None:
        """Register global keyboard shortcuts."""
        # Ctrl+K: command palette (placeholder for now)
        QShortcut(QKeySequence("Ctrl+K"), self, self._open_command_palette)

        # Ctrl+1/2/3: switch workspaces
        QShortcut(QKeySequence("Ctrl+1"), self, lambda: self.sidebar.select(0))
        QShortcut(QKeySequence("Ctrl+2"), self, lambda: self.sidebar.select(1))
        QShortcut(QKeySequence("Ctrl+3"), self, lambda: self.sidebar.select(2))

        # Ctrl+,: settings
        QShortcut(QKeySequence("Ctrl+,"), self, self._toggle_settings)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_workspace_changed(self, index: int) -> None:
        """Switch active workspace."""
        if 0 <= index < self.workspace_stack.count():
            self.workspace_stack.setCurrentIndex(index)

    def _toggle_settings(self) -> None:
        """Toggle settings slide-over panel."""
        if self.settings_panel.isVisible():
            self.settings_panel.hide()
        else:
            self.settings_panel.show()
            self.settings_panel.raise_()

    def _open_command_palette(self) -> None:
        """Open command palette (Ctrl+K)."""
        # TODO: implement command palette widget
        pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Clean up on close."""
        # Let workspaces clean up
        self.query_ws.cleanup()
        self.documents_ws.cleanup()
        self.analysis_ws.cleanup()
        super().closeEvent(event)
