"""Base modal popup class for consistent warm-themed popups."""

from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, Property, QSize
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QGraphicsOpacityEffect,
    QWidget,
    QSizePolicy,
)


class ModalPopup(QDialog):
    """Base class for all modal popups with consistent warm theme styling.
    
    Features:
    - Warm dark theme matching the app
    - Fade-in animation on open
    - Standard header with close button
    - Consistent padding and styling
    """
    
    def __init__(self, title: str = "", parent=None, closable: bool = True):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumSize(700, 500)
        
        # Remove window frame for custom styling
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        
        # Base styling
        self.setStyleSheet(self._get_base_stylesheet())
        
        # Main layout
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)
        
        # Container frame for border radius
        self._container = QFrame()
        self._container.setObjectName("modalContainer")
        container_layout = QVBoxLayout(self._container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        
        # Header
        self._header = self._create_header(title, closable)
        container_layout.addWidget(self._header)
        
        # Content area (to be filled by subclasses)
        self._content_widget = QWidget()
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(24, 16, 24, 24)
        self._content_layout.setSpacing(16)
        container_layout.addWidget(self._content_widget, stretch=1)
        
        self._main_layout.addWidget(self._container)
        
        # Fade-in animation
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity_effect)
        self._opacity_effect.setOpacity(1.0)
    
    def _get_base_stylesheet(self) -> str:
        """Get the base stylesheet for the popup."""
        return """
            QDialog {
                background-color: transparent;
            }
            
            QFrame#modalContainer {
                background-color: #0f0f12;
                border: 1px solid #27272a;
                border-radius: 12px;
            }
            
            QFrame#modalHeader {
                background-color: #16161a;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                border-bottom: 1px solid #27272a;
            }
            
            QLabel#modalTitle {
                color: #f4f4f5;
                font-size: 14pt;
                font-weight: 600;
            }
            
            QPushButton#closeButton {
                background-color: transparent;
                color: #71717a;
                border: none;
                font-size: 18pt;
                font-weight: 300;
                padding: 4px 12px;
                border-radius: 4px;
            }
            QPushButton#closeButton:hover {
                background-color: #1e1e24;
                color: #f87171;
            }
            
            QLabel {
                color: #f4f4f5;
            }
            
            QPushButton {
                background-color: #1e1e24;
                color: #f4f4f5;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #362f27;
                border-color: #4a423a;
            }
            QPushButton:pressed {
                background-color: #1a1a20;
            }
            
            QPushButton#primaryButton {
                background-color: #8b7cf6;
                color: #0f0f12;
                border: none;
            }
            QPushButton#primaryButton:hover {
                background-color: #d4b07a;
            }
            
            QPushButton#dangerButton {
                background-color: transparent;
                color: #f87171;
                border: 1px solid #f87171;
            }
            QPushButton#dangerButton:hover {
                background-color: rgba(196, 90, 90, 0.1);
            }
            
            QTextEdit, QPlainTextEdit {
                background-color: #1a1a20;
                color: #f4f4f5;
                border: 1px solid #27272a;
                border-radius: 8px;
                padding: 12px;
                font-size: 10pt;
            }
            
            QScrollBar:vertical {
                background-color: #0f0f12;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: #3f3f46;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #4a423a;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """
    
    def _create_header(self, title: str, closable: bool) -> QFrame:
        """Create the header with title and close button."""
        header = QFrame()
        header.setObjectName("modalHeader")
        header.setFixedHeight(56)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 0, 16, 0)
        
        title_label = QLabel(title)
        title_label.setObjectName("modalTitle")
        layout.addWidget(title_label)
        
        layout.addStretch()
        
        if closable:
            close_btn = QPushButton("×")
            close_btn.setObjectName("closeButton")
            close_btn.setFixedSize(36, 36)
            close_btn.setCursor(Qt.PointingHandCursor)
            close_btn.clicked.connect(self.close)
            layout.addWidget(close_btn)
        
        return header
    
    def add_content(self, widget: QWidget):
        """Add a widget to the content area."""
        self._content_layout.addWidget(widget)
    
    def add_stretch(self):
        """Add stretch to content area."""
        self._content_layout.addStretch()
    
    def set_footer(self, *buttons: QPushButton):
        """Set footer with action buttons."""
        footer = QFrame()
        footer.setStyleSheet("""
            QFrame {
                background-color: #16161a;
                border-top: 1px solid #27272a;
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
            }
        """)
        
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(24, 16, 24, 16)
        
        layout.addStretch()
        for btn in buttons:
            layout.addWidget(btn)
        
        # Insert before the container's last stretch
        self._container.layout().addWidget(footer)
    
    def showEvent(self, event):
        """Animate popup appearance."""
        super().showEvent(event)
        
        # Fade in animation
        self._fade_animation = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_animation.setDuration(150)
        self._fade_animation.setStartValue(0.0)
        self._fade_animation.setEndValue(1.0)
        self._fade_animation.setEasingCurve(QEasingCurve.OutCubic)
        self._fade_animation.start()
    
    def keyPressEvent(self, event):
        """Handle key presses - Escape to close."""
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


class InfoRow(QFrame):
    """A styled info row for displaying label: value pairs."""
    
    def __init__(self, label: str, value: str = "", parent=None):
        super().__init__(parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(12)
        
        self.label = QLabel(label)
        self.label.setStyleSheet("color: #71717a; font-weight: 500; min-width: 100px;")
        layout.addWidget(self.label)
        
        self.value = QLabel(value)
        self.value.setStyleSheet("color: #f4f4f5;")
        self.value.setWordWrap(True)
        layout.addWidget(self.value, stretch=1)
    
    def set_value(self, value: str):
        """Update the value text."""
        self.value.setText(value)


class SectionHeader(QLabel):
    """A styled section header."""
    
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            color: #a1a1aa;
            font-size: 10pt;
            font-weight: 600;
            letter-spacing: 0.5px;
            padding-top: 8px;
            padding-bottom: 4px;
        """)


