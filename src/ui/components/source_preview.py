"""Source Preview Dialog for 'Instant' results."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QScrollArea,
    QWidget,
    QFrame,
    QPushButton,
    QHBoxLayout,
    QGraphicsOpacityEffect,
    QApplication
)
from PySide6.QtGui import QColor, QPalette

class SourcePreviewDialog(QDialog):
    """Non-blocking popup showing retrieved chunks before generation."""

    def __init__(self, parent=None, chunks=None):
        super().__init__(parent)
        self.chunks = chunks or []
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        
        self._setup_ui()
        self._position_window()
        
        # Auto-close timer (optional, or close when generation starts)
        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.close)
        # self.timer.start(5000)  # Show for 5 seconds

    def _setup_ui(self):
        """Set up the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Main container with styling
        container = QFrame()
        container.setObjectName("previewContainer")
        container.setStyleSheet("""
            #previewContainer {
                background-color: rgba(30, 41, 59, 0.95);
                border: 1px solid #475569;
                border-radius: 8px;
            }
            QLabel { color: #e2e8f0; }
        """)
        
        container_layout = QVBoxLayout(container)
        
        # Header
        header = QHBoxLayout()
        title = QLabel(f"Found {len(self.chunks)} Sources")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #60a5fa;")
        header.addWidget(title)
        
        close_btn = QPushButton("×")
        close_btn.setFlat(True)
        close_btn.setMaximumWidth(20)
        close_btn.setStyleSheet("color: #94a3b8; font-weight: bold;")
        close_btn.clicked.connect(self.close)
        header.addWidget(close_btn)
        
        container_layout.addLayout(header)
        
        # Sources List (Top 3)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        list_layout = QVBoxLayout(content)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(8)
        
        for i, chunk in enumerate(self.chunks[:3]):
            self._add_chunk_item(list_layout, chunk)
            
        if len(self.chunks) > 3:
            more_label = QLabel(f"...and {len(self.chunks) - 3} more")
            more_label.setStyleSheet("color: #94a3b8; font-style: italic; font-size: 11px;")
            list_layout.addWidget(more_label)
            
        list_layout.addStretch()
        scroll.setWidget(content)
        container_layout.addWidget(scroll)
        
        # Footer
        footer = QLabel("Generating answer...")
        footer.setStyleSheet("color: #10b981; font-size: 11px; margin-top: 4px;")
        container_layout.addWidget(footer)
        
        layout.addWidget(container)
        
        # Add drop shadow if possible
        # shadow = QGraphicsDropShadowEffect(self)
        # shadow.setBlurRadius(20)
        # shadow.setColor(QColor(0, 0, 0, 100))
        # shadow.setOffset(0, 4)
        # container.setGraphicsEffect(shadow)

    def _add_chunk_item(self, layout, chunk):
        """Add a chunk item to the list."""
        item = QFrame()
        item.setStyleSheet("background-color: rgba(255, 255, 255, 0.05); border-radius: 4px; padding: 4px;")
        item_layout = QVBoxLayout(item)
        item_layout.setContentsMargins(4, 4, 4, 4)
        item_layout.setSpacing(2)
        
        # File Name & Score
        meta_layout = QHBoxLayout()
        name = chunk.get("metadata", {}).get("file_name", "Unknown")
        score = chunk.get("score", 0)
        
        name_lbl = QLabel(name)
        name_lbl.setStyleSheet("font-weight: bold; font-size: 12px;")
        meta_layout.addWidget(name_lbl)
        
        score_lbl = QLabel(f"{score:.2f}")
        score_lbl.setStyleSheet(f"color: {'#4ade80' if score > 0.6 else '#facc15'}; font-size: 11px;")
        meta_layout.addWidget(score_lbl)
        
        item_layout.addLayout(meta_layout)
        
        # Snippet
        text = chunk.get("text", "")
        snippet = (text[:75] + "...") if len(text) > 75 else text
        snip_lbl = QLabel(snippet)
        snip_lbl.setWordWrap(True)
        snip_lbl.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        item_layout.addWidget(snip_lbl)
        
        layout.addWidget(item)

    def _position_window(self):
        """Position window at bottom right or near parent."""
        if self.parent():
            parent_geo = self.parent().geometry()
            # Position: Bottom right of parent, with margin
            x = parent_geo.x() + parent_geo.width() - 320
            y = parent_geo.y() + parent_geo.height() - 400
            self.setGeometry(x, y, 300, 350)
        else:
            screen = QApplication.primaryScreen().geometry()
            self.setGeometry(screen.width() - 320, screen.height() - 400, 300, 350)






