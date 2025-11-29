"""Response Popup - Large modal for viewing generated responses.

Supports inline citations with click-to-navigate functionality:
- [[@Source-N]] markers in response text are clickable
- Clicking a citation scrolls to and highlights the source
"""

import re
from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QTextBrowser,
    QPushButton,
    QFrame,
    QScrollArea,
    QWidget,
    QSplitter,
    QSizePolicy,
)
from PySide6.QtGui import QFont, QTextCursor, QTextCharFormat, QColor


class ResponsePopup(QDialog):
    """Fullscreen popup dialog for viewing the generated response with sources.
    
    Supports inline citations [[@Source-N]] that are clickable and scroll to
    the corresponding source in the sources panel.
    """
    
    # Signal emitted when user clicks on a source (for external handling)
    source_clicked = Signal(str, int, int)  # file_name, page_number, char_start

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generated Response")
        self.setMinimumSize(1200, 800)
        self.setModal(False)  # Non-blocking
        
        # Store source cards for highlighting
        self._source_cards: dict[int, QFrame] = {}
        self._source_map: dict[int, dict] = {}
        
        self._setup_ui()
        self._apply_styles()
        
        # Show maximized/fullscreen for clarity
        self.showMaximized()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setObjectName("responseHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)
        
        title = QLabel("Generated Response")
        title.setStyleSheet("font-size: 14pt; font-weight: 600; color: #f4f4f5; letter-spacing: 0.5px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Metrics in header
        self.metrics_label = QLabel("")
        self.metrics_label.setStyleSheet("color: #a1a1aa; font-size: 10pt;")
        header_layout.addWidget(self.metrics_label)
        
        close_btn = QPushButton("✕")
        close_btn.setMaximumSize(32, 32)
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #a1a1aa;
                font-size: 18px;
                border: none;
            }
            QPushButton:hover {
                color: #f4f4f5;
            }
        """)
        header_layout.addWidget(close_btn)
        
        layout.addWidget(header)

        # Main content - splitter with response and sources
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #27272a;
            }
        """)
        
        # Left: Response (Main area)
        response_frame = QFrame()
        response_frame.setObjectName("responseFrame")
        response_layout = QVBoxLayout(response_frame)
        response_layout.setContentsMargins(24, 20, 24, 20)
        
        response_header = QLabel("Answer")
        response_header.setStyleSheet("color: #8b7cf6; font-size: 12pt; font-weight: 600; margin-bottom: 8px; letter-spacing: 0.5px;")
        response_layout.addWidget(response_header)
        
        # Use QTextBrowser for clickable links
        self.response_text = QTextBrowser()
        self.response_text.setReadOnly(True)
        self.response_text.setOpenLinks(False)  # Handle links ourselves
        self.response_text.setFont(QFont("Segoe UI", 12))
        self.response_text.setStyleSheet("border: none; background: transparent;")
        self.response_text.anchorClicked.connect(self._on_citation_clicked)
        response_layout.addWidget(self.response_text)
        
        splitter.addWidget(response_frame)
        
        # Right: Sources (Side column)
        sources_frame = QFrame()
        sources_frame.setObjectName("sourcesFrame")
        sources_frame.setMinimumWidth(300)
        sources_layout = QVBoxLayout(sources_frame)
        sources_layout.setContentsMargins(12, 20, 12, 20)
        
        sources_header = QLabel("Sources Used")
        sources_header.setStyleSheet("color: #4ade80; font-size: 11pt; font-weight: 600; margin-bottom: 8px; letter-spacing: 0.5px;")
        sources_layout.addWidget(sources_header)
        
        self.sources_scroll = QScrollArea()
        self.sources_scroll.setWidgetResizable(True)
        self.sources_scroll.setFrameShape(QFrame.NoFrame)
        self.sources_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.sources_container = QWidget()
        self.sources_layout = QVBoxLayout(self.sources_container)
        self.sources_layout.setContentsMargins(0, 0, 0, 0)
        self.sources_layout.setSpacing(8)
        self.sources_layout.addStretch()
        
        self.sources_scroll.setWidget(self.sources_container)
        sources_layout.addWidget(self.sources_scroll)
        
        splitter.addWidget(sources_frame)
        
        # Set initial sizes (75% response, 25% sources)
        splitter.setSizes([900, 300])
        
        # Ensure splitter expands to fill height
        splitter.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        
        layout.addWidget(splitter, stretch=1)  # Add stretch to make splitter fill space

        # Footer with actions
        footer = QFrame()
        footer.setObjectName("responseFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(16, 12, 16, 12)
        
        self.citation_status = QLabel("")
        self.citation_status.setStyleSheet("color: #a1a1aa; font-size: 10pt;")
        footer_layout.addWidget(self.citation_status)
        
        footer_layout.addStretch()
        
        copy_btn = QPushButton("Copy Response")
        copy_btn.clicked.connect(self._copy_response)
        footer_layout.addWidget(copy_btn)
        
        layout.addWidget(footer)

    def _apply_styles(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #0f0f12;
            }
            #responseHeader {
                background-color: #1f1b18;
                border-bottom: 1px solid #27272a;
            }
            #responseFooter {
                background-color: #1f1b18;
                border-top: 1px solid #27272a;
            }
            #responseFrame, #sourcesFrame {
                background-color: #0f0f12;
            }
            QTextEdit {
                background-color: #0f0f12;
                color: #f4f4f5;
                border: 1px solid #27272a;
                border-radius: 8px;
                padding: 14px;
                font-size: 10pt;
            }
            QScrollArea {
                background-color: transparent;
            }
            QPushButton {
                background-color: #8b7cf6;
                color: #0f0f12;
                border: none;
                border-radius: 6px;
                padding: 10px 18px;
                font-weight: 600;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #d4b57a;
            }
        """)

    def set_response(self, text: str, source_map: Optional[dict[int, dict]] = None):
        """Set the response text with clickable inline citations.
        
        Args:
            text: Response text (may contain [[@Source-N]] markers)
            source_map: Optional mapping of source_id to chunk metadata
        """
        if source_map:
            self._source_map = source_map
        
        # Convert inline citations to clickable links
        html_text = self._format_response_with_links(text)
        self.response_text.setHtml(html_text)
    
    def _format_response_with_links(self, text: str) -> str:
        """Convert [[@Source-N]] markers to clickable HTML links.
        
        Args:
            text: Response text with inline citation markers
            
        Returns:
            HTML formatted text with clickable links
        """
        # Escape HTML special characters first
        import html
        escaped = html.escape(text)
        
        # Pattern to match [[@Source-N]]
        pattern = r'\[\[@Source-(\d+)\]\]'
        
        def replace_citation(match):
            source_id = int(match.group(1))
            # Create a clickable link styled as a citation
            return (
                f'<a href="source://{source_id}" '
                f'style="color: #8b7cf6; text-decoration: none; '
                f'background-color: #1e1e24; padding: 2px 6px; border-radius: 4px; '
                f'font-size: 10px;">'
                f'[{source_id}]</a>'
            )
        
        html_text = re.sub(pattern, replace_citation, escaped)
        
        # Wrap in div with proper styling
        html_text = f'''
        <div style="color: #e2e8f0; font-family: 'Segoe UI', sans-serif; 
                    font-size: 14px; line-height: 1.6;">
            {html_text.replace(chr(10), '<br>')}
        </div>
        '''
        
        return html_text
    
    def _on_citation_clicked(self, url):
        """Handle click on inline citation link.
        
        Args:
            url: QUrl with format "source://N"
        """
        url_str = url.toString()
        if url_str.startswith("source://"):
            try:
                source_id = int(url_str.replace("source://", ""))
                self._highlight_source(source_id)
                
                # Emit signal for external handling (e.g., open document)
                if source_id in self._source_map:
                    source = self._source_map[source_id]
                    self.source_clicked.emit(
                        source.get("file_name", ""),
                        source.get("page_number", 1),
                        source.get("char_start", 0),
                    )
            except ValueError:
                pass
    
    def _highlight_source(self, source_id: int):
        """Scroll to and highlight a source card.
        
        Args:
            source_id: 1-indexed source ID
        """
        # Reset all card styles
        for card in self._source_cards.values():
            card.setStyleSheet("""
                QFrame {
                    background-color: #1a1a20;
                    border: 1px solid #27272a;
                    border-radius: 8px;
                    padding: 10px;
                }
            """)
        
        # Highlight the selected card
        if source_id in self._source_cards:
            card = self._source_cards[source_id]
            card.setStyleSheet("""
                QFrame {
                    background-color: #1e1e24;
                    border: 2px solid #8b7cf6;
                    border-radius: 8px;
                    padding: 10px;
                }
            """)
            # Scroll to make visible
            self.sources_scroll.ensureWidgetVisible(card)

    def set_metrics(self, chunks: int, valid_citations: int, total_citations: int, 
                    prompt_tokens: int, output_tokens: int, tok_per_sec: float):
        """Set the metrics display."""
        self.metrics_label.setText(
            f"{chunks} chunks • {prompt_tokens:,} prompt tokens • "
            f"{output_tokens} output @ {tok_per_sec:.1f} tok/s"
        )
        
        if total_citations > 0:
            ratio = valid_citations / total_citations
            color = "#4ade80" if ratio >= 0.8 else "#fbbf24" if ratio >= 0.5 else "#f87171"
            self.citation_status.setText(
                f"Citations: {valid_citations}/{total_citations} verified"
            )
            self.citation_status.setStyleSheet(f"color: {color}; font-size: 10pt;")
        else:
            self.citation_status.setText("No citations found")
            self.citation_status.setStyleSheet("color: #fbbf24; font-size: 10pt;")

    def set_sources(self, chunks: list[dict], source_map: Optional[dict[int, dict]] = None):
        """Set the source chunks.
        
        Args:
            chunks: List of chunk dicts with text, metadata, score
            source_map: Optional mapping of source_id to chunk metadata
        """
        # Clear existing
        self._source_cards.clear()
        while self.sources_layout.count() > 1:
            item = self.sources_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Store source map if provided
        if source_map:
            self._source_map = source_map
        
        for i, chunk in enumerate(chunks):
            source_id = i + 1
            source_card = self._create_source_card(source_id, chunk)
            self._source_cards[source_id] = source_card
            self.sources_layout.insertWidget(i, source_card)

    def _create_source_card(self, source_id: int, chunk: dict) -> QFrame:
        """Create a card for a source chunk.
        
        Args:
            source_id: 1-indexed source ID (matches [[@Source-N]] markers)
            chunk: Chunk dict with text, metadata, score
            
        Returns:
            QFrame styled as a source card
        """
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #1a1a20;
                border: 1px solid #27272a;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        card.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        
        # Header: source ID badge, filename, page, and score
        header = QHBoxLayout()
        
        # Source ID badge (clickable visual)
        id_badge = QLabel(f"[{source_id}]")
        id_badge.setStyleSheet(
            "color: #8b7cf6; background-color: #1e1e24; "
            "font-size: 10pt; font-weight: 600; "
            "padding: 3px 8px; border-radius: 4px;"
        )
        header.addWidget(id_badge)
        
        filename = chunk.get("metadata", {}).get("file_name", "Unknown")
        page = chunk.get("metadata", {}).get("page_number", "")
        para = chunk.get("metadata", {}).get("paragraph_number", "")
        
        name_label = QLabel(filename)
        name_label.setStyleSheet("color: #a1a1aa; font-size: 10pt;")
        header.addWidget(name_label)
        
        if page:
            location = f"p.{page}"
            if para:
                location += f" ¶{para}"
            page_label = QLabel(location)
            page_label.setStyleSheet("color: #71717a; font-size: 9pt;")
            header.addWidget(page_label)
        
        header.addStretch()
        
        score = chunk.get("score", 0)
        score_color = "#10b981" if score > 0.8 else "#f59e0b" if score > 0.5 else "#ef4444"
        score_label = QLabel(f"{score:.2f}")
        score_label.setStyleSheet(f"color: {score_color}; font-size: 10px;")
        header.addWidget(score_label)
        
        layout.addLayout(header)
        
        # Snippet
        text = chunk.get("text", "")
        # Truncate slightly less for better context
        snippet = text[:300] + "..." if len(text) > 300 else text
        snippet_label = QLabel(snippet)
        snippet_label.setWordWrap(True)
        # Smaller font for source text as requested
        snippet_label.setStyleSheet("color: #f4f4f5; font-size: 10pt; line-height: 1.5;")
        layout.addWidget(snippet_label)
        
        return card

    def _copy_response(self):
        """Copy response to clipboard."""
        from PySide6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(self.response_text.toPlainText())

