import markdown
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, 
    QScrollArea, QLabel, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QThread, Slot, QObject
from PySide6.QtGui import QTextCursor

# Backend Imports
from src.retrieval.query_engine import QueryEngine
from src.config_loader import get_settings

class QueryWorker(QObject):
    """Worker thread for running queries without freezing the UI."""
    token_received = Signal(str)
    source_received = Signal(dict)
    finished = Signal()
    error_occurred = Signal(str)

    def __init__(self, query_text: str):
        super().__init__()
        self.query_text = query_text
        self.settings = get_settings()

    @Slot()
    def run(self):
        try:
            engine = QueryEngine(self.settings)
            # We need to adapt the generator to emit signals
            response_gen = engine.query(self.query_text)
            
            for chunk in response_gen:
                if "token" in chunk:
                    self.token_received.emit(chunk["token"])
                elif "source" in chunk:
                    self.source_received.emit(chunk["source"])
                elif "error" in chunk:
                    self.error_occurred.emit(chunk["error"])
            
            self.finished.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))
            self.finished.emit()

class ChatBubble(QFrame):
    """A single chat message bubble."""
    def __init__(self, text, is_user=False, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.label.setOpenExternalLinks(True)
        
        # Render Markdown for AI, Plain text for User
        if is_user:
            self.label.setText(text)
            self.setStyleSheet("""
                QFrame {
                    background-color: #3b82f6;
                    border-radius: 12px;
                    border-bottom-right-radius: 2px;
                }
                QLabel {
                    color: white;
                    font-size: 14px;
                }
            """)
            layout.setAlignment(Qt.AlignRight)
        else:
            html = markdown.markdown(text, extensions=['fenced_code', 'tables'])
            self.label.setText(html)
            self.setStyleSheet("""
                QFrame {
                    background-color: #27272a;
                    border-radius: 12px;
                    border-bottom-left-radius: 2px;
                }
                QLabel {
                    color: #f1f5f9;
                    font-size: 14px;
                }
            """)
            layout.setAlignment(Qt.AlignLeft)
            
        layout.addWidget(self.label)

    def update_text(self, new_text):
        """Update text (for streaming)."""
        html = markdown.markdown(new_text, extensions=['fenced_code', 'tables'])
        self.label.setText(html)

class ChatWidget(QWidget):
    """Main chat interface."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.current_worker = None
        self.current_thread = None
        self.full_response = ""
        self.ai_bubble = None

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Chat Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.addStretch() # Push messages to bottom
        
        self.scroll_area.setWidget(self.chat_container)
        layout.addWidget(self.scroll_area)
        
        # Input Area
        input_container = QFrame()
        input_container.setStyleSheet("background-color: #0f0f12; border-top: 1px solid #27272a;")
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(20, 20, 20, 20)
        
        self.input_box = QTextEdit()
        self.input_box.setObjectName("input_box")
        self.input_box.setPlaceholderText("Ask a question about your documents...")
        self.input_box.setFixedHeight(60)
        input_layout.addWidget(self.input_box)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("send_btn")
        self.send_btn.setFixedSize(80, 40)
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)
        
        layout.addWidget(input_container)

    def send_message(self):
        text = self.input_box.toPlainText().strip()
        if not text:
            return
            
        # Add User Message
        self.add_message(text, is_user=True)
        self.input_box.clear()
        self.input_box.setDisabled(True)
        self.send_btn.setDisabled(True)
        
        # Prepare AI Message placeholder
        self.full_response = ""
        self.ai_bubble = self.add_message("Thinking...", is_user=False)
        
        # Start Worker
        self.current_thread = QThread()
        self.current_worker = QueryWorker(text)
        self.current_worker.moveToThread(self.current_thread)
        
        # Connect Signals
        self.current_thread.started.connect(self.current_worker.run)
        self.current_worker.token_received.connect(self.handle_token)
        self.current_worker.source_received.connect(self.handle_source)
        self.current_worker.finished.connect(self.handle_finished)
        self.current_worker.finished.connect(self.current_thread.quit)
        self.current_worker.finished.connect(self.current_worker.deleteLater)
        self.current_thread.finished.connect(self.current_thread.deleteLater)
        
        self.current_thread.start()

    def add_message(self, text, is_user=False):
        bubble = ChatBubble(text, is_user)
        
        # Wrapper for alignment
        wrapper = QWidget()
        wrapper_layout = QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 5, 0, 5)
        
        if is_user:
            wrapper_layout.addStretch()
            wrapper_layout.addWidget(bubble)
        else:
            wrapper_layout.addWidget(bubble)
            wrapper_layout.addStretch()
            
        self.chat_layout.addWidget(wrapper)
        
        # Scroll to bottom
        QThread.msleep(10) # Small delay to let layout update
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )
        
        return bubble

    def handle_token(self, token):
        if self.full_response == "":
            self.full_response = token # Clear "Thinking..."
        else:
            self.full_response += token
        
        if self.ai_bubble:
            self.ai_bubble.update_text(self.full_response)
            
        # Auto-scroll
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )

    def handle_source(self, source):
        # Append sources to the end
        source_text = f"\n\n**Source:** {source.get('file_name', 'Unknown')} (p. {source.get('page', '?')})"
        self.full_response += source_text
        if self.ai_bubble:
            self.ai_bubble.update_text(self.full_response)

    def handle_finished(self):
        self.input_box.setDisabled(False)
        self.send_btn.setDisabled(False)
        self.input_box.setFocus()
