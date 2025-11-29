"""Main window for SC Gen 6 desktop application."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QMenuBar,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.config_loader import get_settings
from src.ui.document_manager import DocumentManagerWidget
from src.ui.query_panel import QueryPanelWidget
from src.ui.results_panel import ResultsPanelWidget
from src.ui.claude_code_panel import ClaudeCodePanel
from src.ui.settings_dialog import SettingsDialog
from src.ui.technical_specs_panel import TechnicalSpecsPanel


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        """Initialize main window."""
        super().__init__()
        self.settings = get_settings()
        self.setWindowTitle("SC Gen 6 - Litigation Support RAG")
        self.setGeometry(100, 100, 1400, 900)

        # Create menu bar
        self._create_menu_bar()

        # Create central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Create splitter for left/right panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel: Document Manager
        self.document_manager = DocumentManagerWidget(self)
        splitter.addWidget(self.document_manager)

        # Right panel: Tab widget for RAG Query and Claude Code
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ccc;
            }
            QTabBar::tab {
                padding: 8px 16px;
                font-size: 11pt;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #2196f3;
                color: white;
            }
        """)

        # Tab 1: RAG Query + Results (vertical split)
        rag_widget = QWidget()
        rag_layout = QVBoxLayout(rag_widget)
        rag_layout.setContentsMargins(0, 0, 0, 0)

        right_splitter = QSplitter(Qt.Vertical)

        # Query panel (top)
        self.query_panel = QueryPanelWidget(self)
        right_splitter.addWidget(self.query_panel)

        # Results panel (bottom)
        self.results_panel = ResultsPanelWidget(self)
        right_splitter.addWidget(self.results_panel)

        # Set proportions (40% query, 60% results)
        right_splitter.setSizes([360, 540])

        rag_layout.addWidget(right_splitter)
        self.tab_widget.addTab(rag_widget, "📄 RAG Query")

        # Tab 2: Claude Code Integration
        self.claude_code_panel = ClaudeCodePanel(self)
        self.tab_widget.addTab(self.claude_code_panel, "🤖 Claude Code")

        # Tab 3: Technical Specs
        self.tech_specs_panel = TechnicalSpecsPanel(self)
        self.tab_widget.addTab(self.tech_specs_panel, "⚙️ Technical Specs")

        splitter.addWidget(self.tab_widget)

        # Set splitter proportions (30% left, 70% right)
        splitter.setSizes([420, 980])

        # Connect signals
        self._connect_signals()

    def _create_menu_bar(self):
        """Create menu bar with Settings option."""
        menubar = self.menuBar()
        settings_menu = menubar.addMenu("Settings")
        settings_action = settings_menu.addAction("Edit Configuration...")
        settings_action.triggered.connect(self._show_settings_dialog)

    def _show_settings_dialog(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self)
        dialog.exec()

    def _connect_signals(self):
        """Connect signals between widgets."""
        # Query panel -> Results panel
        self.query_panel.query_requested.connect(self._handle_query)
        self.query_panel.stop_requested.connect(self._handle_stop)

        # Document manager -> Query panel (for document selection updates)
        self.document_manager.selection_changed.connect(self.query_panel.update_selected_documents)

    def _handle_query(self, query_data: dict):
        """Handle query request from query panel.

        Args:
            query_data: Dictionary with query parameters
        """
        self.results_panel.start_query(query_data)

    def _handle_stop(self):
        """Handle stop request."""
        self.results_panel.stop_query()

