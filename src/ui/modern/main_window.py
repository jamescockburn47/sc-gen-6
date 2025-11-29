from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QStackedWidget, QLabel
from PySide6.QtCore import Qt

from src.ui.modern.styles import DARK_THEME
from src.ui.modern.sidebar import Sidebar
from src.ui.modern.chat_widget import ChatWidget
from src.ui.modern.graph_widget import GraphWidget
from src.ui.modern.document_manager import DocumentManagerWidget
from src.ui.modern.settings_widget import SettingsWidget

class ModernMainWindow(QMainWindow):
    """Main window for the modern desktop app."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SC Gen 6")
        self.resize(1200, 800)
        
        # Apply Theme
        self.setStyleSheet(DARK_THEME)
        
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main Layout
        self.layout = QHBoxLayout(central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Sidebar
        self.sidebar = Sidebar()
        self.sidebar.page_changed.connect(self.switch_page)
        self.layout.addWidget(self.sidebar)
        
        # Content Area (Stacked)
        self.content_stack = QStackedWidget()
        self.layout.addWidget(self.content_stack)
        
        # Initialize Pages
        self._init_pages()
        
    def _init_pages(self):
        """Initialize and add pages to the stack."""
        self.pages = {}
        
        # Chat Page
        self.pages["chat"] = ChatWidget()
        self.content_stack.addWidget(self.pages["chat"])
        
        # Graph Page
        self.pages["graph"] = GraphWidget()
        self.content_stack.addWidget(self.pages["graph"])
        
        # Documents Page
        self.pages["documents"] = DocumentManagerWidget()
        self.content_stack.addWidget(self.pages["documents"])
        
        # Settings Page
        self.pages["settings"] = SettingsWidget()
        self.content_stack.addWidget(self.pages["settings"])
        
    def _create_placeholder(self, text):
        """Create a placeholder widget for development."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 24px; color: #52525b;")
        layout.addWidget(label)
        return widget
        
    def switch_page(self, page_name):
        """Switch the visible page."""
        if page_name in self.pages:
            self.content_stack.setCurrentWidget(self.pages[page_name])
