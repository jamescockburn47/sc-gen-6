"""Model loading status widget with dual-server support and progress bars.

Shows separate status for generation and embedding servers with:
- Loading spinner/checkmark
- Model name display
- Optional progress bar during loading
"""

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, QProgressBar

from src.servers.status_monitor import ServerStatus


class SpinnerWidget(QWidget):
    """Animated loading spinner."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(18, 18)
        self._angle = 0
        self._color = QColor("#fbbf24")
        self._spinning = False
        self._show_check = False
        self._show_x = False
        
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._rotate)
        
    def start(self):
        """Start spinning."""
        self._spinning = True
        self._show_check = False
        self._show_x = False
        self._timer.start(50)
        
    def stop(self):
        """Stop spinning."""
        self._spinning = False
        self._timer.stop()
        self.update()
    
    def show_check(self):
        """Show checkmark."""
        self.stop()
        self._show_check = True
        self._show_x = False
        self.update()
    
    def show_x(self):
        """Show X (error)."""
        self.stop()
        self._show_x = True
        self._show_check = False
        self.update()
        
    def set_color(self, color: QColor):
        self._color = color
        self.update()
        
    def _rotate(self):
        self._angle = (self._angle + 15) % 360
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        cx, cy = self.width() / 2, self.height() / 2
        pen = QPen(self._color)
        pen.setWidth(2)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        
        if self._spinning:
            rect = self.rect().adjusted(2, 2, -2, -2)
            painter.drawArc(rect, self._angle * 16, 270 * 16)
        elif self._show_check:
            painter.drawLine(int(cx-4), int(cy), int(cx-1), int(cy+3))
            painter.drawLine(int(cx-1), int(cy+3), int(cx+5), int(cy-4))
        elif self._show_x:
            painter.drawLine(int(cx-4), int(cy-4), int(cx+4), int(cy+4))
            painter.drawLine(int(cx+4), int(cy-4), int(cx-4), int(cy+4))


class ServerStatusRow(QWidget):
    """Single row showing one server's status."""
    
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label = label
        self._status = ServerStatus.LOADING  # Start in LOADING state
        self._model = ""
        self._progress = 0
        
        self._setup_ui()
        # Start spinner immediately
        self.spinner.start()
        
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(8)
        
        # Spinner
        self.spinner = SpinnerWidget(self)
        layout.addWidget(self.spinner)
        
        # Label + status - start with Loading text
        self.text_label = QLabel(f"{self._label}: Loading...")
        self.text_label.setStyleSheet("color: #fbbf24; font-size: 10pt;")
        layout.addWidget(self.text_label, stretch=1)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setFixedWidth(60)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #27272a;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #fbbf24;
                border-radius: 4px;
            }
        """)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
    def set_status(self, status: str, model: str = "", progress: int = 0):
        """Update the displayed status."""
        self._status = status
        self._model = model
        self._progress = progress
        
        if status == ServerStatus.DOWN:
            self.text_label.setText(f"{self._label}: Down")
            self.text_label.setStyleSheet("color: #71717a; font-size: 10pt;")
            self.spinner.set_color(QColor("#71717a"))
            self.spinner.show_x()
            self.progress_bar.hide()
            
        elif status == ServerStatus.PROXY_ONLY:
            self.text_label.setText(f"{self._label}: Standby")
            self.text_label.setStyleSheet("color: #60a5fa; font-size: 10pt;")  # Blue
            self.spinner.set_color(QColor("#60a5fa"))
            self.spinner.stop()
            self.spinner.update()
            self.progress_bar.hide()
            
        elif status == ServerStatus.LOADING:
            if progress > 0:
                self.text_label.setText(f"{self._label}: Loading")
                self.progress_bar.setValue(progress)
                self.progress_bar.show()
            else:
                self.text_label.setText(f"{self._label}: Loading...")
                self.progress_bar.hide()
            self.text_label.setStyleSheet("color: #fbbf24; font-size: 10pt;")
            self.spinner.set_color(QColor("#fbbf24"))
            self.spinner.start()
            
        elif status == ServerStatus.READY:
            display_model = model[:20] if len(model) > 20 else model
            if display_model:
                self.text_label.setText(f"{self._label}: {display_model}")
            else:
                self.text_label.setText(f"{self._label}: Ready")
            self.text_label.setStyleSheet("color: #4ade80; font-size: 10pt;")
            self.spinner.set_color(QColor("#4ade80"))
            self.spinner.show_check()
            self.progress_bar.hide()


class DualServerStatusWidget(QWidget):
    """Widget showing status of both generation and embedding servers."""
    
    status_changed = Signal(str, str)  # (overall_status, message)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(2)
        
        # Generation server row
        self.gen_row = ServerStatusRow("Gen", self)
        layout.addWidget(self.gen_row)
        
        # Embedding server row
        self.embed_row = ServerStatusRow("Embed", self)
        layout.addWidget(self.embed_row)
        
        # Styling
        self.setStyleSheet("""
            DualServerStatusWidget {
                background-color: rgba(39, 39, 42, 0.5);
                border: 1px solid #3f3f46;
                border-radius: 8px;
            }
        """)
        
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setMinimumHeight(60)
        
    def set_gen_status(self, status: str, model: str = "", progress: int = 0):
        """Update generation server status."""
        self.gen_row.set_status(status, model, progress)
        self._update_border()
        
    def set_embed_status(self, status: str, model: str = "", progress: int = 0):
        """Update embedding server status."""
        self.embed_row.set_status(status, model, progress)
        self._update_border()
        
    def _update_border(self):
        """Update border color based on overall status."""
        gen_ready = self.gen_row._status == ServerStatus.READY
        embed_ready = self.embed_row._status == ServerStatus.READY
        
        if gen_ready and embed_ready:
            border_color = "#4ade80"  # Green
            bg_alpha = "0.1"
        elif self.gen_row._status == ServerStatus.DOWN or self.embed_row._status == ServerStatus.DOWN:
            border_color = "#71717a"  # Gray
            bg_alpha = "0.3"
        else:
            border_color = "#fbbf24"  # Amber (loading)
            bg_alpha = "0.1"
            
        self.setStyleSheet(f"""
            DualServerStatusWidget {{
                background-color: rgba({self._hex_to_rgb(border_color)}, {bg_alpha});
                border: 1px solid {border_color};
                border-radius: 8px;
            }}
        """)
        
    def _hex_to_rgb(self, hex_color: str) -> str:
        """Convert hex color to RGB string."""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"{r}, {g}, {b}"
    
    def is_ready(self) -> bool:
        """Check if both servers ready."""
        return (
            self.gen_row._status == ServerStatus.READY and
            self.embed_row._status == ServerStatus.READY
        )


# Keep old class for backward compatibility
class ModelLoadingWidget(DualServerStatusWidget):
    """Backward-compatible alias for DualServerStatusWidget."""
    
    STATUS_STARTING = "starting"
    STATUS_LOADING = "loading"
    STATUS_READY = "ready"
    STATUS_ERROR = "error"
    
    def set_status(self, status: str, message: str = ""):
        """Legacy interface - maps to generation server status."""
        if status == self.STATUS_STARTING:
            self.set_gen_status(ServerStatus.PROXY_ONLY)
        elif status == self.STATUS_LOADING:
            self.set_gen_status(ServerStatus.LOADING)
        elif status == self.STATUS_READY:
            self.set_gen_status(ServerStatus.READY, message or "Ready")
        elif status == self.STATUS_ERROR:
            self.set_gen_status(ServerStatus.DOWN, message or "Error")
