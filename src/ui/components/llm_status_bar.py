"""Compact LLM status bar for monitoring generation progress."""

from datetime import datetime
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame


class LLMStatusBar(QWidget):
    """Compact status bar showing real-time LLM generation progress."""
    
    # Signal emitted when user clicks the status bar (to show details)
    clicked = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_time: Optional[datetime] = None
        self.current_model: str = ""
        self.current_task: str = ""
        self.last_tokens: int = 0
        self.last_tokens_per_sec: float = 0.0
        self.is_active: bool = False
        self.error_message: str = ""
        
        self._setup_ui()
        
        # Update timer (1 second interval)
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_elapsed_time)
        
    def _setup_ui(self):
        """Set up the UI."""
        self.setStyleSheet("""
            QWidget {
                background-color: #18181b;
                border-bottom: 1px solid #27272a;
            }
        """)
        self.setFixedHeight(32)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(16)
        
        # Status indicator (colored dot)
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #71717a; font-size: 16px;")
        layout.addWidget(self.status_dot)
        
        # Model name
        self.model_label = QLabel("Idle")
        self.model_label.setStyleSheet("color: #a1a1aa; font-weight: 600; font-size: 11px;")
        layout.addWidget(self.model_label)
        
        # Separator
        sep1 = QLabel("|")
        sep1.setStyleSheet("color: #3f3f46;")
        layout.addWidget(sep1)
        
        # Task description
        self.task_label = QLabel("")
        self.task_label.setStyleSheet("color: #71717a; font-size: 11px;")
        layout.addWidget(self.task_label)
        
        layout.addStretch()
        
        # Time elapsed
        self.time_label = QLabel("")
        self.time_label.setStyleSheet("color: #a1a1aa; font-size: 11px; font-family: monospace;")
        layout.addWidget(self.time_label)
        
        # Separator
        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #3f3f46;")
        layout.addWidget(sep2)
        
        # Tokens/second
        self.tps_label = QLabel("")
        self.tps_label.setStyleSheet("color: #8b7cf6; font-size: 11px; font-family: monospace; font-weight: 600;")
        layout.addWidget(self.tps_label)
        
        # Separator
        sep3 = QLabel("|")
        sep3.setStyleSheet("color: #3f3f46;")
        layout.addWidget(sep3)
        
        # Tokens generated
        self.tokens_label = QLabel("")
        self.tokens_label.setStyleSheet("color: #a1a1aa; font-size: 11px; font-family: monospace;")
        layout.addWidget(self.tokens_label)
        
        # Error indicator (hidden by default)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #ef4444; font-size: 11px; font-weight: 600;")
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)
        
        # Initially hidden
        self.setVisible(False)
    
    def start_generation(self, model: str, task: str = "Generating"):
        """Start tracking a new generation.
        
        Args:
            model: Model name (e.g., "qwen2.5:72b")
            task: Task description (e.g., "Case Overview")
        """
        self.is_active = True
        self.start_time = datetime.now()
        self.current_model = model
        self.current_task = task
        self.last_tokens = 0
        self.last_tokens_per_sec = 0.0
        self.error_message = ""
        
        # Update UI
        self.model_label.setText(model)
        self.task_label.setText(task)
        self.status_dot.setStyleSheet("color: #22c55e; font-size: 16px;")  # Green
        self.time_label.setText("0s")
        self.tps_label.setText("0 t/s")
        self.tokens_label.setText("0 tokens")
        self.error_label.setVisible(False)
        
        # Show and start timer
        self.setVisible(True)
        self.update_timer.start(1000)  # Update every second
    
    def update_progress(self, tokens: int, tokens_per_sec: float):
        """Update progress stats.
        
        Args:
            tokens: Total tokens generated so far
            tokens_per_sec: Current generation speed
        """
        if not self.is_active:
            return
        
        self.last_tokens = tokens
        self.last_tokens_per_sec = tokens_per_sec
        
        # Update labels
        self.tokens_label.setText(f"{tokens:,} tokens")
        self.tps_label.setText(f"{tokens_per_sec:.1f} t/s")
    
    def set_error(self, error: str):
        """Display an error.
        
        Args:
            error: Error message
        """
        self.error_message = error
        self.status_dot.setStyleSheet("color: #ef4444; font-size: 16px;")  # Red
        self.error_label.setText(f"⚠ {error}")
        self.error_label.setVisible(True)
        self.tps_label.setText("Error")
    
    def finish(self, success: bool = True):
        """Mark generation as complete.
        
        Args:
            success: Whether generation completed successfully
        """
        self.is_active = False
        self.update_timer.stop()
        
        if success:
            self.status_dot.setStyleSheet("color: #22c55e; font-size: 16px;")  # Green
            self.task_label.setText(f"{self.current_task} - Complete")
        else:
            self.status_dot.setStyleSheet("color: #ef4444; font-size: 16px;")  # Red
        
        # Auto-hide after 3 seconds
        QTimer.singleShot(3000, self._auto_hide)
    
    def _auto_hide(self):
        """Auto-hide the status bar if not active."""
        if not self.is_active:
            self.setVisible(False)
    
    def _update_elapsed_time(self):
        """Update the elapsed time display (called every second)."""
        if not self.is_active or not self.start_time:
            return
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        # Format time
        if elapsed < 60:
            time_str = f"{int(elapsed)}s"
        elif elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_str = f"{minutes}m {seconds}s"
        else:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            time_str = f"{hours}h {minutes}m"
        
        self.time_label.setText(time_str)
    
    def mousePressEvent(self, event):
        """Handle mouse click to show details."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)
