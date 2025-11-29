import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, 
    QListWidgetItem, QFileDialog, QLabel, QProgressBar, QMessageBox, QFrame,
    QStyle
)
from PySide6.QtCore import Qt, QThread, Signal, Slot

from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.config_loader import get_settings

class IngestionWorker(QThread):
    """Worker thread for document ingestion."""
    progress = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, file_paths):
        super().__init__()
        self.file_paths = file_paths

    def run(self):
        try:
            # Simple sequential processing - fast and stable
            # GPU is fully utilized via batch processing (128 embeddings at once)
            pipeline = IngestionPipeline()
            for path in self.file_paths:
                self.progress.emit(f"Processing {os.path.basename(path)}...")
                # Skip summarization during ingestion (too slow with 14B model)
                # Summaries can be generated on-demand later
                success = pipeline.process_document(path, generate_summary=False)
                if not success:
                    self.error.emit(f"Failed to process {os.path.basename(path)}")
            
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))



class DocumentManagerWidget(QWidget):
    """Widget for managing and ingesting documents."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = get_settings()
        self._setup_ui()
        self.refresh_list()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QLabel("Document Management")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #f1f5f9; margin-bottom: 20px;")
        layout.addWidget(header)

        # Toolbar
        toolbar = QHBoxLayout()
        
        self.add_btn = QPushButton("+ Add Documents")
        self.add_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        self.add_btn.clicked.connect(self.add_documents)
        toolbar.addWidget(self.add_btn)

        self.add_folder_btn = QPushButton("+ Add Folder")
        self.add_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        self.add_folder_btn.clicked.connect(self.add_folder)
        toolbar.addWidget(self.add_folder_btn)

        self.refresh_btn = QPushButton("Refresh List")
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #27272a;
                color: #f1f5f9;
                border: 1px solid #3f3f46;
                padding: 10px 20px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #3f3f46;
            }
        """)
        self.refresh_btn.clicked.connect(self.refresh_list)
        toolbar.addWidget(self.refresh_btn)
        
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Document List
        self.doc_list = QListWidget()
        self.doc_list.setStyleSheet("""
            QListWidget {
                background-color: #18181b;
                border: 1px solid #27272a;
                border-radius: 8px;
                color: #f1f5f9;
                font-size: 14px;
                padding: 10px;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #27272a;
            }
            QListWidget::item:selected {
                background-color: #27272a;
            }
        """)
        layout.addWidget(self.doc_list)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #27272a;
                border-radius: 4px;
                height: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #a1a1aa; margin-top: 5px;")
        layout.addWidget(self.status_label)

    def refresh_list(self):
        """Refresh the document list from the catalog."""
        self.doc_list.clear()
        
        try:
            from src.documents.catalog import DocumentCatalog
            catalog = DocumentCatalog()
            records = catalog.list_records()
            
            if not records:
                self.status_label.setText("No documents ingested yet.")
                return
            
            # Add ingested documents to the list
            for record in records:
                # Show file name with status indicator
                status_icon = "✓" if record.indexed else "✗"
                display_text = f"{status_icon} {record.file_name}"
                if record.chunk_count > 0:
                    display_text += f" ({record.chunk_count} chunks)"
                
                item = QListWidgetItem(display_text)
                item.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
                
                # Store file path in item data for potential future use
                item.setData(Qt.UserRole, record.file_path)
                
                self.doc_list.addItem(item)
            
            self.status_label.setText(f"{len(records)} document(s) ingested")
            
        except Exception as e:
            self.status_label.setText(f"Error loading documents: {str(e)}")

    def add_documents(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select Documents", 
            "", 
            "Documents (*.pdf *.docx *.txt);;All Files (*.*)"
        )
        if not files:
            return

        self._start_ingestion(files)

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if not folder:
            return

        # Recursively scan folder for supported document types
        supported_extensions = ('.pdf', '.docx', '.txt', '.doc', '.xlsx', '.xls')
        files = []
        for root, dirs, filenames in os.walk(folder):
            for filename in filenames:
                if filename.lower().endswith(supported_extensions):
                    files.append(os.path.join(root, filename))
        
        if not files:
            QMessageBox.information(self, "No Documents", "No supported documents found in the selected folder.")
            return

        self._start_ingestion(files)

    def _start_ingestion(self, files):
        """Start ingestion worker with the given file list."""
        self.add_btn.setDisabled(True)
        self.add_folder_btn.setDisabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate
        
        self.worker = IngestionWorker(files)
        self.worker.progress.connect(self.update_status)
        self.worker.finished.connect(self.ingestion_finished)
        self.worker.error.connect(self.ingestion_error)
        self.worker.start()

    def update_status(self, msg):
        self.status_label.setText(msg)

    def ingestion_finished(self):
        self.add_btn.setDisabled(False)
        self.add_folder_btn.setDisabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ingestion complete.")
        self.refresh_list()
        QMessageBox.information(self, "Success", "Documents processed successfully.")

    def ingestion_error(self, err):
        self.add_btn.setDisabled(False)
        self.add_folder_btn.setDisabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Error occurred.")
        QMessageBox.critical(self, "Error", f"Ingestion failed: {err}")
