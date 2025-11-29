"""Enhanced Output Panel with better visibility and formatting."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
    QFrame,
    QProgressBar,
)
from PySide6.QtGui import QFont


class StatusBadge(QLabel):
    """A colored status badge."""

    def __init__(self, text: str, color: str, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: white;
                border-radius: 4px;
                padding: 4px 12px;
                font-weight: 600;
                font-size: 9pt;
            }}
        """)


class EnhancedOutputPanel(QWidget):
    """Enhanced output panel with better visibility."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Set up UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Header with title and actions
        header_layout = QHBoxLayout()

        title = QLabel("Answer")
        title.setProperty("styleClass", "subtitle")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Status badge
        self.status_badge = StatusBadge("Ready", "#64748b")
        header_layout.addWidget(self.status_badge)

        # Copy button
        self.copy_button = QPushButton("Copy")
        self.copy_button.setProperty("styleClass", "secondary")
        self.copy_button.clicked.connect(self._copy_to_clipboard)
        header_layout.addWidget(self.copy_button)

        # Export button
        self.export_button = QPushButton("Export")
        self.export_button.setProperty("styleClass", "secondary")
        self.export_button.clicked.connect(self._export)
        header_layout.addWidget(self.export_button)

        layout.addLayout(header_layout)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)  # Indeterminate
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        # Main output area
        self.output_text = QPlainTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setPlaceholderText(
            "Your answer will appear here...\n\n"
            "💡 Tip: Select documents on the left and enter your query to get started."
        )

        # Set a nice monospace font for better readability
        font = QFont("Consolas", 11)
        if not font.exactMatch():
            font = QFont("Courier New", 11)
        font.setStyleHint(QFont.Monospace)
        self.output_text.setFont(font)

        # Custom styling for output area
        self.output_text.setStyleSheet("""
            QPlainTextEdit {
                background-color: #0f0f12;
                border: 1px solid #27272a;
                border-radius: 10px;
                padding: 16px;
                color: #f4f4f5;
                line-height: 1.6;
            }
            QPlainTextEdit:focus {
                border: 2px solid #8b7cf6;
                padding: 15px;
            }
        """)

        layout.addWidget(self.output_text)

        # Footer with analytics (collapsible)
        self.analytics_frame = QFrame()
        self.analytics_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a20;
                border: 1px solid #27272a;
                border-radius: 10px;
                padding: 8px;
            }
        """)
        analytics_layout = QVBoxLayout(self.analytics_frame)
        analytics_layout.setContentsMargins(12, 12, 12, 12)

        # Analytics header
        analytics_header = QHBoxLayout()
        self.analytics_toggle = QPushButton("▼ Analytics")
        self.analytics_toggle.setProperty("styleClass", "secondary")
        self.analytics_toggle.clicked.connect(self._toggle_analytics)
        analytics_header.addWidget(self.analytics_toggle)
        analytics_header.addStretch()
        analytics_layout.addLayout(analytics_header)

        # Analytics content
        self.analytics_content = QWidget()
        analytics_content_layout = QVBoxLayout(self.analytics_content)
        analytics_content_layout.setContentsMargins(0, 8, 0, 0)

        self.analytics_text = QLabel("No analytics yet")
        self.analytics_text.setProperty("styleClass", "muted")
        self.analytics_text.setWordWrap(True)
        analytics_content_layout.addWidget(self.analytics_text)

        analytics_layout.addWidget(self.analytics_content)
        layout.addWidget(self.analytics_frame)

        self.analytics_frame.setVisible(False)

    def _toggle_analytics(self):
        """Toggle analytics visibility."""
        visible = self.analytics_content.isVisible()
        self.analytics_content.setVisible(not visible)
        arrow = "▼" if not visible else "▶"
        self.analytics_toggle.setText(f"{arrow} Analytics")

    def _copy_to_clipboard(self):
        """Copy output to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.output_text.toPlainText())
        self.status_badge.setText("Copied!")
        self.status_badge.setStyleSheet("""
            QLabel {
                background-color: #4ade80;
                color: #0f0f12;
                border-radius: 4px;
                padding: 4px 12px;
                font-weight: 600;
                font-size: 9pt;
            }
        """)

    def _export(self):
        """Export output to file."""
        from PySide6.QtWidgets import QFileDialog
        from datetime import datetime

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Answer",
            f"answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;Markdown Files (*.md);;All Files (*)"
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.output_text.toPlainText())
                self.status_badge.setText("Exported!")
                self.status_badge.setStyleSheet("""
                    QLabel {
                        background-color: #4ade80;
                        color: #0f0f12;
                        border-radius: 4px;
                        padding: 4px 12px;
                        font-weight: 600;
                        font-size: 9pt;
                    }
                """)
            except Exception as e:
                self.status_badge.setText("Export failed")
                self.status_badge.setStyleSheet("""
                    QLabel {
                        background-color: #f87171;
                        color: #f4f4f5;
                        border-radius: 4px;
                        padding: 4px 12px;
                        font-weight: 600;
                        font-size: 9pt;
                    }
                """)

    def set_status(self, status: str, color: str = "#64748b"):
        """Set status badge."""
        self.status_badge.setText(status)
        self.status_badge.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: white;
                border-radius: 4px;
                padding: 4px 12px;
                font-weight: 600;
                font-size: 9pt;
            }}
        """)

    def show_progress(self, show: bool = True):
        """Show or hide progress bar."""
        self.progress_bar.setVisible(show)

    def clear_output(self):
        """Clear output."""
        self.output_text.clear()

    def append_text(self, text: str):
        """Append text to output."""
        from PySide6.QtGui import QTextCursor
        self.output_text.moveCursor(QTextCursor.MoveOperation.End)
        self.output_text.insertPlainText(text)
        self.output_text.ensureCursorVisible()

    def set_text(self, text: str):
        """Set output text."""
        self.output_text.setPlainText(text)

    def set_analytics(self, analytics_html: str):
        """Set analytics text."""
        self.analytics_text.setText(analytics_html)
        self.analytics_frame.setVisible(True)
