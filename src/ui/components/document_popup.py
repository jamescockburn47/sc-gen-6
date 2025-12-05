"""Document detail popup for viewing document info, summary, and chunks."""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QTextEdit,
    QTabWidget,
    QWidget,
    QScrollArea,
    QListWidget,
    QListWidgetItem,
)

from src.ui.components.modal_popup import ModalPopup, InfoRow, SectionHeader
from src.documents.catalog import DocumentCatalog, DocumentRecord
from src.retrieval.summary_store import SummaryStore
from src.retrieval.vector_store import VectorStore
from src.config_loader import get_settings


class DocumentPopup(ModalPopup):
    """Popup showing detailed document information, summary, and chunks."""
    
    summary_requested = Signal(str)  # file_path
    delete_requested = Signal(str)  # file_path
    
    def __init__(self, record: DocumentRecord, parent=None):
        super().__init__(title=record.file_name, parent=parent)
        self.record = record
        self.settings = get_settings()
        self.resize(900, 700)
        
        self._setup_content()
        self._load_data()
    
    def _setup_content(self):
        """Set up the popup content."""
        # Tab widget for different views
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                background-color: #0f0f12;
                border: 1px solid #27272a;
                border-radius: 8px;
                margin-top: -1px;
            }
            QTabBar::tab {
                background-color: #1a1a20;
                color: #a1a1aa;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: #0f0f12;
                color: #f4f4f5;
                border-bottom: 2px solid #8b7cf6;
            }
            QTabBar::tab:hover:!selected {
                background-color: #1e1e24;
            }
        """)
        
        # Overview tab
        self.overview_tab = self._create_overview_tab()
        self.tabs.addTab(self.overview_tab, "Overview")
        
        # Summary tab
        self.summary_tab = self._create_summary_tab()
        self.tabs.addTab(self.summary_tab, "Summary")
        
        # Chunks tab
        self.chunks_tab = self._create_chunks_tab()
        self.tabs.addTab(self.chunks_tab, "Chunks")
        
        self.add_content(self.tabs)
        
        # Footer buttons
        self.generate_summary_btn = QPushButton("Generate Summary")
        self.generate_summary_btn.setObjectName("primaryButton")
        self.generate_summary_btn.clicked.connect(self._on_generate_summary)
        
        self.delete_btn = QPushButton("Delete Document")
        self.delete_btn.setObjectName("dangerButton")
        self.delete_btn.clicked.connect(self._on_delete)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        
        self.set_footer(self.delete_btn, self.generate_summary_btn, close_btn)
    
    def _create_overview_tab(self) -> QWidget:
        """Create the overview tab with document metadata."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        
        # File info section
        layout.addWidget(SectionHeader("File Information"))
        
        self.file_row = InfoRow("File:", self.record.file_name)
        layout.addWidget(self.file_row)
        
        self.path_row = InfoRow("Path:", str(self.record.file_path))
        layout.addWidget(self.path_row)
        
        self.type_row = InfoRow("Type:", self.record.category or "Unknown")
        layout.addWidget(self.type_row)
        
        self.added_row = InfoRow("Added:", self.record.ingested_at or "Unknown")
        layout.addWidget(self.added_row)
        
        # Index status section
        layout.addWidget(SectionHeader("Index Status"))
        
        status = "Indexed" if self.record.indexed else "Pending"
        status_color = "#4ade80" if self.record.indexed else "#fbbf24"
        self.status_row = InfoRow("Status:")
        self.status_row.value.setText(status)
        self.status_row.value.setStyleSheet(f"color: {status_color}; font-weight: 600;")
        layout.addWidget(self.status_row)
        
        self.chunks_row = InfoRow("Chunks:", str(self.record.chunk_count))
        layout.addWidget(self.chunks_row)
        
        if self.record.error:
            layout.addWidget(SectionHeader("Error"))
            error_label = QLabel(self.record.error)
            error_label.setStyleSheet("color: #f87171; padding: 8px; background-color: rgba(196,90,90,0.1); border-radius: 4px;")
            error_label.setWordWrap(True)
            layout.addWidget(error_label)
        
        # Graph section
        layout.addWidget(SectionHeader("Case Graph"))
        
        graph_status = "Included" if self.record.include_in_graph else "Excluded"
        self.graph_row = InfoRow("Graph:", graph_status)
        layout.addWidget(self.graph_row)
        
        layout.addStretch()
        return widget
    
    def _create_summary_tab(self) -> QWidget:
        """Create the summary tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText("No summary available. Click 'Generate Summary' to create one.")
        self.summary_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a20;
                color: #f4f4f5;
                border: 1px solid #27272a;
                border-radius: 8px;
                padding: 16px;
                font-size: 11pt;
                line-height: 1.6;
            }
        """)
        layout.addWidget(self.summary_text)
        
        return widget
    
    def _create_chunks_tab(self) -> QWidget:
        """Create the chunks tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Chunk count
        self.chunk_count_label = QLabel("Loading chunks...")
        self.chunk_count_label.setStyleSheet("color: #71717a; font-size: 9pt;")
        layout.addWidget(self.chunk_count_label)
        
        # Chunk list
        self.chunk_list = QListWidget()
        self.chunk_list.setStyleSheet("""
            QListWidget {
                background-color: #1a1a20;
                border: 1px solid #27272a;
                border-radius: 8px;
                outline: none;
            }
            QListWidget::item {
                color: #a1a1aa;
                padding: 12px;
                border-bottom: 1px solid #27272a;
            }
            QListWidget::item:hover {
                background-color: #1e1e24;
            }
            QListWidget::item:selected {
                background-color: #1e1e24;
                color: #f4f4f5;
            }
        """)
        self.chunk_list.itemClicked.connect(self._on_chunk_clicked)
        layout.addWidget(self.chunk_list)
        
        # Chunk preview
        self.chunk_preview = QTextEdit()
        self.chunk_preview.setReadOnly(True)
        self.chunk_preview.setMaximumHeight(150)
        self.chunk_preview.setPlaceholderText("Select a chunk to preview its content")
        self.chunk_preview.setStyleSheet("""
            QTextEdit {
                background-color: #16161a;
                color: #a1a1aa;
                border: 1px solid #27272a;
                border-radius: 6px;
                padding: 12px;
                font-size: 9pt;
            }
        """)
        layout.addWidget(self.chunk_preview)
        
        return widget
    
    def _load_data(self):
        """Load document data from stores."""
        # Load summary
        try:
            summary_store = SummaryStore()
            summary = summary_store.get_summary(self.record.file_path)
            if summary:
                self.summary_text.setText(summary.get("summary_text", ""))
                self.generate_summary_btn.setText("Regenerate Summary")
        except Exception as e:
            print(f"Failed to load summary: {e}")
        
        # Load chunks
        try:
            vector_store = VectorStore(settings=self.settings)
            # Get chunks for this document
            results = vector_store.collection.get(
                where={"file_name": self.record.file_name},
                include=["documents", "metadatas"]
            )
            
            if results and results.get("ids"):
                chunk_count = len(results["ids"])
                self.chunk_count_label.setText(f"{chunk_count} chunks indexed")
                
                # Add chunks to list
                for i, (chunk_id, text, meta) in enumerate(zip(
                    results["ids"],
                    results.get("documents", []),
                    results.get("metadatas", [])
                )):
                    page = meta.get("page_number", "?") if meta else "?"
                    preview = (text[:80] + "...") if text and len(text) > 80 else (text or "")
                    
                    item = QListWidgetItem(f"[{i+1}] Page {page}: {preview}")
                    item.setData(Qt.UserRole, text)
                    self.chunk_list.addItem(item)
            else:
                self.chunk_count_label.setText("No chunks found")
        except Exception as e:
            print(f"Failed to load chunks: {e}")
            self.chunk_count_label.setText(f"Error loading chunks: {e}")
    
    def _on_chunk_clicked(self, item: QListWidgetItem):
        """Handle chunk selection."""
        text = item.data(Qt.UserRole)
        if text:
            self.chunk_preview.setText(text)
    
    def _on_generate_summary(self):
        """Request summary generation."""
        self.summary_requested.emit(str(self.record.file_path))
        self.close()
    
    def _on_delete(self):
        """Request document deletion."""
        from PySide6.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self,
            "Delete Document",
            f"Are you sure you want to delete '{self.record.file_name}'?\n\n"
            "This will remove the document from the index and catalog.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.delete_requested.emit(str(self.record.file_path))
            self.close()



