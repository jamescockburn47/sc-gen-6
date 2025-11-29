
import markdown
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTextBrowser, QLabel, QFrame
)
from PySide6.QtCore import Qt
from pathlib import Path

class TechnicalSpecsPanel(QWidget):
    """Panel for displaying technical specifications and architecture details."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._load_content()

    def _setup_ui(self):
        """Initialize the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title = QLabel("Technical Specifications")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #1f2937;")
        layout.addWidget(title)

        # Content Viewer (Markdown supported via HTML conversion)
        self.content_viewer = QTextBrowser()
        self.content_viewer.setOpenExternalLinks(True)
        self.content_viewer.setStyleSheet("""
            QTextBrowser {
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 20px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                line-height: 1.6;
            }
        """)
        layout.addWidget(self.content_viewer)

    def _load_content(self):
        """Load and render the markdown content."""
        specs_path = Path("docs/TECHNICAL_SPECS.md")
        
        if not specs_path.exists():
            self.content_viewer.setHtml("<h1>Error</h1><p>Technical specifications file not found at docs/TECHNICAL_SPECS.md</p>")
            return

        try:
            text = specs_path.read_text(encoding="utf-8")
            # Convert Markdown to HTML
            html_content = markdown.markdown(
                text,
                extensions=['extra', 'codehilite', 'tables']
            )
            
            # Add some basic CSS styling for the HTML output
            styled_html = f"""
            <style>
                h1 {{ color: #111827; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; margin-top: 20px; }}
                h2 {{ color: #374151; margin-top: 25px; margin-bottom: 10px; }}
                h3 {{ color: #4b5563; margin-top: 20px; }}
                p {{ color: #374151; margin-bottom: 10px; }}
                li {{ color: #374151; margin-bottom: 5px; }}
                code {{ background-color: #f3f4f6; padding: 2px 4px; border-radius: 4px; font-family: 'Consolas', monospace; }}
                pre {{ background-color: #1f2937; color: #f9fafb; padding: 15px; border-radius: 8px; overflow-x: auto; }}
                strong {{ color: #111827; }}
            </style>
            {html_content}
            """
            
            self.content_viewer.setHtml(styled_html)
            
        except Exception as e:
            self.content_viewer.setHtml(f"<h1>Error</h1><p>Failed to load specifications: {str(e)}</p>")

    def refresh(self):
        """Reload content (can be connected to a refresh signal if needed)."""
        self._load_content()
