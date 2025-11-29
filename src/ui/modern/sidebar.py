from PySide6.QtWidgets import QFrame, QVBoxLayout, QPushButton, QLabel, QWidget, QSpacerItem, QSizePolicy
from PySide6.QtCore import Signal, Qt, QSize
from PySide6.QtGui import QIcon, QAction

class Sidebar(QFrame):
    """Navigation sidebar."""
    
    page_changed = Signal(str)  # page_name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(260)
        
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 20, 12, 20)
        layout.setSpacing(8)
        
        # App Title
        title = QLabel("SC Gen 6")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #f1f5f9; padding-left: 12px; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # New Chat Button
        self.new_chat_btn = QPushButton("+ New Chat")
        self.new_chat_btn.setObjectName("new_chat_btn")
        self.new_chat_btn.setCursor(Qt.PointingHandCursor)
        self.new_chat_btn.clicked.connect(lambda: self.page_changed.emit("chat"))
        layout.addWidget(self.new_chat_btn)
        
        layout.addSpacing(20)
        
        # Navigation Buttons
        self.nav_group = []
        
        self.add_nav_btn("Chat", "chat", layout, checked=True)
        self.add_nav_btn("Graph Explorer", "graph", layout)
        self.add_nav_btn("Documents", "documents", layout)
        self.add_nav_btn("Settings", "settings", layout)
        
        layout.addStretch()
        
        # Footer
        version = QLabel("v6.0.0")
        version.setStyleSheet("color: #52525b; font-size: 12px; padding-left: 12px;")
        layout.addWidget(version)

    def add_nav_btn(self, text, page_name, layout, checked=False):
        btn = QPushButton(text)
        btn.setObjectName("sidebar_btn")
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda: self._handle_nav_click(btn, page_name))
        layout.addWidget(btn)
        self.nav_group.append(btn)
        
    def _handle_nav_click(self, sender, page_name):
        # Uncheck others
        for btn in self.nav_group:
            if btn != sender:
                btn.setChecked(False)
        
        sender.setChecked(True)
        self.page_changed.emit(page_name)
