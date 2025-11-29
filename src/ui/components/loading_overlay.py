"""Loading overlay and progress indicators."""

from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QFrame,
    QGraphicsOpacityEffect,
)


class LoadingSpinner(QWidget):
    """Animated loading spinner using pure Qt drawing."""
    
    def __init__(self, size: int = 40, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._angle = 0
        self._color = QColor("#8b7cf6")  # Primary gold
        
        # Animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._rotate)
        self._timer.start(50)  # 20 FPS
    
    def _rotate(self):
        """Rotate the spinner."""
        self._angle = (self._angle + 10) % 360
        self.update()
    
    def paintEvent(self, event):
        """Draw the spinner."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate dimensions
        side = min(self.width(), self.height())
        painter.translate(side / 2, side / 2)
        painter.rotate(self._angle)
        
        # Draw arcs with varying opacity
        pen = QPen(self._color, 3, Qt.SolidLine, Qt.RoundCap)
        radius = side / 2 - 4
        
        for i in range(8):
            opacity = (i + 1) / 8.0
            color = QColor(self._color)
            color.setAlphaF(opacity)
            pen.setColor(color)
            painter.setPen(pen)
            
            painter.rotate(45)
            painter.drawLine(0, int(radius * 0.5), 0, int(radius))
    
    def start(self):
        """Start the animation."""
        self._timer.start()
    
    def stop(self):
        """Stop the animation."""
        self._timer.stop()
    
    def setColor(self, color: QColor):
        """Set spinner color."""
        self._color = color
        self.update()


class LoadingOverlay(QFrame):
    """Overlay that shows loading state with spinner and message."""
    
    def __init__(self, message: str = "Loading...", parent=None):
        super().__init__(parent)
        self.setObjectName("loadingOverlay")
        self.setStyleSheet("""
            QFrame#loadingOverlay {
                background-color: rgba(23, 20, 18, 0.85);
                border-radius: 12px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(16)
        
        # Spinner
        self.spinner = LoadingSpinner(48, self)
        layout.addWidget(self.spinner, alignment=Qt.AlignCenter)
        
        # Message
        self.message_label = QLabel(message)
        self.message_label.setStyleSheet("""
            color: #f4f4f5;
            font-size: 11pt;
            font-weight: 500;
        """)
        self.message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.message_label)
        
        # Opacity effect for fade animation
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity_effect)
        self._opacity_effect.setOpacity(0.0)
        self.hide()
    
    def show_loading(self, message: str = None):
        """Show the overlay with optional message."""
        if message:
            self.message_label.setText(message)
        
        self.show()
        self.raise_()
        
        # Fade in
        self._fade_animation = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_animation.setDuration(200)
        self._fade_animation.setStartValue(0.0)
        self._fade_animation.setEndValue(1.0)
        self._fade_animation.setEasingCurve(QEasingCurve.OutCubic)
        self._fade_animation.start()
        
        self.spinner.start()
    
    def hide_loading(self):
        """Hide the overlay."""
        self.spinner.stop()
        
        # Fade out
        self._fade_animation = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_animation.setDuration(150)
        self._fade_animation.setStartValue(1.0)
        self._fade_animation.setEndValue(0.0)
        self._fade_animation.setEasingCurve(QEasingCurve.InCubic)
        self._fade_animation.finished.connect(self.hide)
        self._fade_animation.start()
    
    def set_message(self, message: str):
        """Update the loading message."""
        self.message_label.setText(message)


class PulsingDot(QWidget):
    """Small pulsing dot indicator for inline loading."""
    
    def __init__(self, size: int = 8, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._color = QColor("#8b7cf6")
        self._opacity = 1.0
        
        # Pulse animation
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._pulse)
        self._direction = -1
    
    def _pulse(self):
        """Animate the pulse."""
        self._opacity += 0.1 * self._direction
        if self._opacity <= 0.3:
            self._direction = 1
        elif self._opacity >= 1.0:
            self._direction = -1
        self.update()
    
    def paintEvent(self, event):
        """Draw the dot."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        color = QColor(self._color)
        color.setAlphaF(self._opacity)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        
        size = min(self.width(), self.height())
        painter.drawEllipse(0, 0, size, size)
    
    def start(self):
        """Start pulsing."""
        self._timer.start(50)
    
    def stop(self):
        """Stop pulsing."""
        self._timer.stop()
        self._opacity = 1.0
        self.update()


