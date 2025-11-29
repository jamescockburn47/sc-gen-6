"""Document Manager widget for left panel."""
from pathlib import Path
from typing import Optional
import hashlib

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.config_loader import get_settings
from src.documents.catalog import DocumentCatalog, DocumentRecord
from src.ingestion.chunkers.adaptive_chunker import AdaptiveChunker
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.retrieval.fts5_index import FTS5IndexCompat
from src.retrieval import get_embedding_service
from src.retrieval.vector_store import VectorStore
from src.schema import ParsedDocument
from src.graph.extraction import GraphExtractionService
from src.ui.graph_review_dialog import GraphReviewDialog
from src.ui.document_metadata_dialog import DocumentMetadataDialog
from src.ui.components.document_popup import DocumentPopup
from src.ingestion.system_check import (
    IngestionEnvironment,
    verify_local_ingestion,
)


class IngestionWorker(QThread):
    """Worker thread for document ingestion."""

    progress = Signal(str)  # Progress message
    document_parsed = Signal(object)  # ParsedDocument
    finished = Signal(list)  # List of ParsedDocument
    error = Signal(str)  # Error message

    def __init__(
        self,
        file_paths: list[Path],
        catalog: DocumentCatalog,
        auto_index: bool = True,
        hardware_env: Optional[IngestionEnvironment] = None,
        parent=None,
    ):
        """Initialize worker.

        Args:
            file_paths: List of file paths to ingest
            auto_index: Whether to automatically chunk and index documents
            parent: Parent widget
        """
        super().__init__(parent)
        self.file_paths = file_paths
        self.auto_index = auto_index
        self.pipeline = IngestionPipeline()
        self.graph_extractor = GraphExtractionService()
        self.catalog = catalog
        self.hardware_env = hardware_env

    def run(self):
        """Run ingestion in background thread."""
        try:
            parsed_docs = []
            if self.hardware_env:
                if self.hardware_env.has_gpu:
                    self.progress.emit(
                        f"Using GPU acceleration ({self.hardware_env.gpu_name or 'Unknown GPU'})"
                    )
                else:
                    self.progress.emit("GPU unavailable. Using CPU-only ingestion.")

            # Use parallel ingestion for speed
            import os
            # Limit workers to avoid UI freeze, leave some cores for system
            max_workers = max(1, os.cpu_count() - 1)
            
            self.progress.emit(f"Parsing {len(self.file_paths)} files (Parallel - {max_workers} workers)...")
            
            # Batch parse
            parsed_results = self.pipeline.ingest_files(
                self.file_paths, 
                max_workers=max_workers
            )
            
            # Process results
            for parsed in parsed_results:
                if parsed:
                    record = self.catalog.ensure_record(parsed)
                    parsed.document_type = record.category
                    parsed_docs.append(parsed)
                    self.document_parsed.emit(parsed)
                    
                    if record.include_in_graph:
                        try:
                            self.graph_extractor.queue_update(parsed)
                        except Exception as exc:
                            print(f"Graph extraction skipped for {parsed.file_name}: {exc}")
                else:
                    self.error.emit(f"Failed to parse a file")

            # Auto-index if requested
            if self.auto_index and parsed_docs:
                self.progress.emit("Chunking and indexing documents...")
                from src.config_loader import get_settings
                from datetime import datetime

                settings = get_settings()
                chunker = AdaptiveChunker(settings=settings)
                
                # Use factory function to get best embedding service (ONNX GPU or CPU)
                embedding_service = get_embedding_service(settings=settings)
                
                vector_store = VectorStore(settings=settings)
                
                # Always use FTS5 keyword index
                from src.retrieval.fts5_index import FTS5IndexCompat
                keyword_index = FTS5IndexCompat(settings=settings)

                all_chunks = []
                doc_chunk_counts: dict[str, int] = {}  # Track chunks per doc
                
                for doc in parsed_docs:
                    chunks = chunker.chunk_document(doc)
                    doc_chunk_counts[doc.file_path] = len(chunks)
                    all_chunks.extend(chunks)

                if all_chunks:
                    gpu_status = "GPU" if embedding_service.is_gpu_available() else "CPU"
                    self.progress.emit(f"Generating embeddings for {len(all_chunks)} chunks ({gpu_status})...")
                    # Generate embeddings for all chunks
                    chunk_texts = [chunk.text for chunk in all_chunks]
                    embeddings = embedding_service.embed_batch(chunk_texts)

                    self.progress.emit(f"Adding {len(all_chunks)} chunks to indexes...")
                    vector_store.add_chunks(all_chunks, embeddings)
                    
                    # Keyword index: Add chunks incrementally (handles merging internally)
                    keyword_index.add_chunks(all_chunks)
                    keyword_index.save()  # No-op for FTS5, saves pickle for BM25
                    
                    # Update catalog records with indexing status
                    timestamp = datetime.now().isoformat()
                    for doc in parsed_docs:
                        record = self.catalog.get_record(doc.file_path)
                        if record:
                            record.indexed = True
                            record.chunk_count = doc_chunk_counts.get(doc.file_path, 0)
                            record.ingested_at = timestamp
                            record.error = None
                            self.catalog.update_record(record)
                    
                    self.progress.emit("Indexing complete!")

            self.finished.emit(parsed_docs)
        except Exception as e:
            self.error.emit(f"Ingestion error: {str(e)}")


class DocumentManagerWidget(QWidget):
    """Document Manager widget for managing ingested documents."""

    documents_changed = Signal()  # Emitted when document list changes
    selection_changed = Signal(list)  # Emitted when document selection changes (list of file names)

    def __init__(self, parent=None):
        """Initialize document manager."""
        super().__init__(parent)
        self.settings = get_settings()
        self.catalog = DocumentCatalog()
        self.documents: dict[str, ParsedDocument] = {}  # file_path -> ParsedDocument
        self.ingestion_worker: Optional[IngestionWorker] = None
        self.hardware_env: Optional[IngestionEnvironment] = None
        
        # Initialize stores for management
        self.vector_store = VectorStore(settings=self.settings)
        self.keyword_index = FTS5IndexCompat(settings=self.settings)
        try:
            self.keyword_index.load()
        except Exception:
            pass  # Index will be built during ingestion

        self._setup_ui()
        self._load_documents_from_catalog()

    def _setup_ui(self):
        """Set up UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title = QLabel("Documents")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Buttons
        button_layout = QHBoxLayout()
        ingest_files_btn = QPushButton("Add Files...")
        ingest_files_btn.clicked.connect(self._ingest_files)
        button_layout.addWidget(ingest_files_btn)

        ingest_folder_btn = QPushButton("Add Folder...")
        ingest_folder_btn.clicked.connect(self._ingest_folder)
        button_layout.addWidget(ingest_folder_btn)

        review_graph_btn = QPushButton("Review Case Graph")
        review_graph_btn.clicked.connect(self._review_graph)
        button_layout.addWidget(review_graph_btn)

        layout.addLayout(button_layout)

        # Auto-index checkbox
        self.auto_index_checkbox = QCheckBox("Auto-index after ingestion")
        self.auto_index_checkbox.setChecked(True)  # Default to enabled
        layout.addWidget(self.auto_index_checkbox)

        # Document selection for queries
        select_layout = QHBoxLayout()
        select_label = QLabel("Select for query:")
        select_layout.addWidget(select_label)

        select_all_btn = QPushButton("All")
        select_all_btn.setMaximumWidth(50)
        select_all_btn.clicked.connect(self._select_all_for_query)
        select_layout.addWidget(select_all_btn)

        select_none_btn = QPushButton("None")
        select_none_btn.setMaximumWidth(50)
        select_none_btn.clicked.connect(self._select_none_for_query)
        select_layout.addWidget(select_none_btn)

        select_layout.addStretch()
        layout.addLayout(select_layout)

        # Document list with card-like styling
        self.document_list = QListWidget()
        self.document_list.setStyleSheet("""
            QListWidget {
                background-color: #0f0f12;
                border: none;
                outline: none;
                padding: 4px;
            }
            QListWidget::item {
                background-color: #1a1a20;
                color: #f4f4f5;
                border: 1px solid #27272a;
                border-radius: 8px;
                margin: 4px 2px;
                padding: 12px 16px;
                min-height: 32px;
            }
            QListWidget::item:hover {
                background-color: #1e1e24;
                border-color: #3f3f46;
            }
            QListWidget::item:selected {
                background-color: #1e1e24;
                border-color: #8b7cf6;
                border-width: 2px;
            }
            QListWidget::item:checked {
                background-color: rgba(201, 166, 107, 0.1);
            }
        """)
        self.document_list.setSpacing(2)
        self.document_list.itemChanged.connect(self._on_selection_changed)
        self.document_list.itemClicked.connect(self._on_document_clicked)
        self.document_list.currentItemChanged.connect(self._on_document_selected)
        self.document_list.itemDoubleClicked.connect(self._on_document_double_clicked)
        layout.addWidget(self.document_list, stretch=2)
        
        # Document preview panel (shows summary when document selected)
        self.preview_frame = QFrame()
        self.preview_frame.setMinimumHeight(250)  # Ensure visible
        self.preview_frame.setFrameStyle(QFrame.StyledPanel)
        self.preview_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a20;
                border: 1px solid #27272a;
                border-radius: 10px;
            }
        """)
        preview_layout = QVBoxLayout(self.preview_frame)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(4)
        
        # Preview header
        self.preview_title = QLabel("Select a document to view details")
        self.preview_title.setStyleSheet("font-weight: 600; color: #f4f4f5; font-size: 11pt;")
        self.preview_title.setWordWrap(True)
        preview_layout.addWidget(self.preview_title)
        
        # Document metadata
        self.preview_meta = QLabel("")
        self.preview_meta.setStyleSheet("color: #a1a1aa; font-size: 10pt;")
        self.preview_meta.setWordWrap(True)
        preview_layout.addWidget(self.preview_meta)
        
        # Summary section
        self.preview_summary_label = QLabel("Summary:")
        self.preview_summary_label.setStyleSheet("font-weight: 600; color: #f4f4f5; margin-top: 8px;")
        self.preview_summary_label.setVisible(False)
        preview_layout.addWidget(self.preview_summary_label)
        
        self.preview_summary = QTextEdit()
        self.preview_summary.setReadOnly(True)
        self.preview_summary.setMaximumHeight(150)
        self.preview_summary.setStyleSheet("""
            QTextEdit {
                background-color: #0f0f12;
                border: 1px solid #27272a;
                border-radius: 8px;
                color: #f4f4f5;
                padding: 8px;
                font-size: 10pt;
            }
        """)
        self.preview_summary.setVisible(False)
        preview_layout.addWidget(self.preview_summary)
        
        # Generate summary button
        self.generate_summary_btn = QPushButton("Generate Summary")
        self.generate_summary_btn.setVisible(False)
        self.generate_summary_btn.clicked.connect(self._generate_summary_for_selected)
        preview_layout.addWidget(self.generate_summary_btn)
        
        layout.addWidget(self.preview_frame)

        # Action buttons
        action_layout = QHBoxLayout()
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._delete_selected)
        action_layout.addWidget(delete_btn)

        metadata_btn = QPushButton("Edit Metadata")
        metadata_btn.clicked.connect(self._edit_metadata)
        action_layout.addWidget(metadata_btn)

        update_btn = QPushButton("Update")
        update_btn.clicked.connect(self._update_selected)
        action_layout.addWidget(update_btn)
        
        # Rebuild Indexes Button
        rebuild_btn = QPushButton("Rebuild")
        rebuild_btn.setToolTip("Re-index all documents (clears indexes, keeps catalog)")
        rebuild_btn.clicked.connect(self._rebuild_indexes)
        action_layout.addWidget(rebuild_btn)
        
        # Reset All Button
        reset_btn = QPushButton("Reset All")
        reset_btn.setStyleSheet("color: red;")
        reset_btn.setToolTip("Delete everything and start fresh")
        reset_btn.clicked.connect(self._reset_all_data)
        action_layout.addWidget(reset_btn)

        layout.addLayout(action_layout)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

    def _ingest_files(self):
        """Open file dialog and ingest selected files."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Documents",
            str(Path.home()),
            "All Supported (*.pdf *.docx *.doc *.xlsx *.xls *.eml *.msg);;PDF (*.pdf);;Word (*.docx *.doc);;Excel (*.xlsx *.xls);;Email (*.eml *.msg)",
        )

        if file_paths:
            self._start_ingestion([Path(f) for f in file_paths])

    def _review_graph(self):
        """Open graph review dialog."""
        dialog = GraphReviewDialog(self)
        dialog.exec()
        self.status_label.setText("Graph review updated.")

    def _ingest_folder(self):
        """Open folder dialog and ingest all files in folder."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder", str(Path.home())
        )

        if folder_path:
            self._start_ingestion([Path(folder_path)])

    def _start_ingestion(self, paths: list[Path]):
        """Start ingestion worker thread.

        Args:
            paths: List of file or folder paths
        """
        if self.ingestion_worker and self.ingestion_worker.isRunning():
            self.status_label.setText("Ingestion already in progress...")
            return

        # Collect all file paths
        file_paths = []
        for path in paths:
            if path.is_file():
                file_paths.append(path)
            elif path.is_dir():
                # Add all files in directory
                file_paths.extend(
                    [
                        f
                        for f in path.rglob("*")
                        if f.is_file()
                        and f.suffix.lower()
                        in [".pdf", ".docx", ".doc", ".xlsx", ".xls", ".eml", ".msg"]
                    ]
                )

        if not file_paths:
            self.status_label.setText("No files found to ingest.")
            return

        env = verify_local_ingestion(self.settings)
        if not env.local_only:
            QMessageBox.critical(
                self,
                "Local Compute Required",
                "Proxy or remote configuration detected. Disable external routing to run ingestion locally.",
            )
            return

        if env.warnings:
            QMessageBox.warning(self, "Ingestion Warnings", "\n".join(env.warnings))

        self.hardware_env = env
        gpu_msg = (
            f"GPU: {env.gpu_name or 'Detected'} / {env.total_vram_gb or '?.?'} GB"
            if env.has_gpu
            else "GPU not detected (CPU fallback)"
        )
        self.status_label.setText(
            f"Ingesting {len(file_paths)} file(s)... {gpu_msg}"
        )

        # Start worker thread with auto-index setting
        auto_index = self.auto_index_checkbox.isChecked()
        self.ingestion_worker = IngestionWorker(
            file_paths,
            catalog=self.catalog,
            auto_index=auto_index,
            hardware_env=self.hardware_env,
        )
        self.ingestion_worker.progress.connect(self._on_ingestion_progress)
        self.ingestion_worker.document_parsed.connect(self._on_document_parsed)
        self.ingestion_worker.finished.connect(self._on_ingestion_finished)
        self.ingestion_worker.error.connect(self._on_ingestion_error)
        self.ingestion_worker.start()

    def _on_ingestion_progress(self, message: str):
        """Handle ingestion progress update."""
        self.status_label.setText(message)

    def _on_document_parsed(self, document: ParsedDocument):
        """Handle parsed document."""
        self.documents[document.file_path] = document
        self._add_document_to_list(document)

    def _on_ingestion_finished(self, documents: list[ParsedDocument]):
        """Handle ingestion completion."""
        total = len(list(self.catalog.all_records()))
        self.status_label.setText(f"Ready - {total} document(s) in catalog")
        self.documents_changed.emit()

    def _on_ingestion_error(self, error_message: str):
        """Handle ingestion error."""
        self.status_label.setText(f"Error: {error_message}")

    def _add_document_to_list(self, document: ParsedDocument):
        """Add or update a document entry when newly parsed."""
        record = self.catalog.get(document.file_path)
        self._upsert_list_item(document.file_path, record, document)

    def _get_document_id(self, file_path: str) -> str:
        """Generate document ID from file path (matching AdaptiveChunker logic)."""
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        return f"doc_{file_hash}"

    def _delete_selected(self):
        """Delete selected document."""
        current_item = self.document_list.currentItem()
        if not current_item:
            return
            
        file_path = current_item.data(Qt.UserRole)
        if not file_path:
            return
            
        # Confirm deletion
        reply = QMessageBox.question(
            self, 
            "Confirm Delete", 
            f"Are you sure you want to delete {Path(file_path).name}?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        if reply == QMessageBox.No:
            return

        try:
            # Generate ID and delete from backend stores
            doc_id = self._get_document_id(file_path)
            
            # Delete from Vector Store
            try:
                self.vector_store.delete_document(doc_id)
            except Exception as e:
                print(f"Warning: Failed to delete from vector store: {e}")
                
            # Delete from BM25 Index
            try:
                self.keyword_index.delete_document(doc_id)
                self.keyword_index.save()
            except Exception as e:
                print(f"Warning: Failed to delete from BM25 index: {e}")

            # Delete from memory and catalog
            if file_path in self.documents:
                del self.documents[file_path]
            self.catalog.delete_record(file_path)
            
            # Remove from UI
            self.document_list.takeItem(self.document_list.row(current_item))
            self.status_label.setText(f"Removed {Path(file_path).name} from catalog and indexes")
            self.documents_changed.emit()
            self._emit_selection()
            
        except Exception as e:
            QMessageBox.critical(self, "Delete Error", f"Failed to delete document: {str(e)}")

    def _rebuild_indexes(self):
        """Clear indexes and re-index all documents from catalog."""
        records = list(self.catalog.all_records())
        if not records:
            QMessageBox.information(self, "Nothing to Rebuild", "No documents in catalog.")
            return
            
        reply = QMessageBox.question(
            self,
            "Confirm Rebuild",
            f"This will clear indexes and re-index {len(records)} document(s).\n\n"
            "Document catalog and metadata will be preserved.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            try:
                self.status_label.setText("Clearing indexes...")
                
                # Clear Vector Store
                self.vector_store.reset()
                
                # Clear FTS5 Index
                self.keyword_index.reset()
                self.keyword_index = FTS5IndexCompat(settings=self.settings)
                
                # Reset indexed status in catalog
                for record in records:
                    record.indexed = False
                    record.chunk_count = 0
                    record.ingested_at = None
                    record.error = None
                    self.catalog.update_record(record)
                
                self.status_label.setText("Re-ingesting documents...")
                
                # Collect all file paths that still exist
                file_paths = [
                    Path(r.file_path) for r in records 
                    if Path(r.file_path).exists()
                ]
                
                if file_paths:
                    self._start_ingestion(file_paths)
                else:
                    self.status_label.setText("No valid files found to rebuild.")
                    
            except Exception as e:
                self.status_label.setText(f"Error rebuilding: {str(e)}")
                QMessageBox.critical(self, "Rebuild Error", f"Failed to rebuild indexes: {str(e)}")

    def _reset_all_data(self):
        """Clear all documents and reset indexes."""
        reply = QMessageBox.warning(
            self,
            "Confirm Reset",
            "This will DELETE ALL DOCUMENTS and clear the database.\n\nAre you sure?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                self.status_label.setText("Resetting database...")
                
                # Clear Vector Store
                self.vector_store.reset()
                
                # Clear FTS5 Index
                self.keyword_index.reset()
                self.keyword_index = FTS5IndexCompat(settings=self.settings)
                
                # Clear Catalog - delete file
                catalog_path = self.catalog.path
                if catalog_path.exists():
                    catalog_path.unlink()
                # Re-init empty catalog
                self.catalog = DocumentCatalog()
                
                # Clear memory and UI
                self.documents.clear()
                self.document_list.clear()
                
                self.status_label.setText("Database cleared. Ready to ingest.")
                self.documents_changed.emit()
                self._emit_selection()
                
                QMessageBox.information(self, "Reset Complete", "All documents and indexes have been cleared.")
                
            except Exception as e:
                self.status_label.setText(f"Error resetting: {str(e)}")
                QMessageBox.critical(self, "Reset Error", f"Failed to reset database: {str(e)}")

    def _update_selected(self):
        """Update/reparse selected document."""
        current_item = self.document_list.currentItem()
        if current_item:
            file_path = current_item.data(Qt.UserRole)
            self._start_ingestion([Path(file_path)])

    def _edit_metadata(self):
        """Open metadata dialog for selected document."""
        current_item = self.document_list.currentItem()
        if not current_item:
            return
        file_path = current_item.data(Qt.UserRole)
        record = self.catalog.get(file_path)
        if not record:
            self.status_label.setText("No metadata found. Re-ingest document first.")
            return

        dialog = DocumentMetadataDialog(self.catalog, record, self)
        if dialog.exec():
            updated = self.catalog.update_record(file_path, **dialog.updated_values())
            self._upsert_list_item(
                file_path, updated, self.documents.get(file_path)
            )
            self.status_label.setText("Metadata updated.")
            if dialog.should_reprocess():
                self._start_ingestion([Path(file_path)])

    def _load_documents_from_catalog(self):
        """Load documents from the metadata catalog."""
        self.document_list.clear()
        records = sorted(
            self.catalog.all_records(), key=lambda r: r.label.lower()
        )
        if not records:
            self.status_label.setText("Ready - no documents ingested yet")
            return
        for record in records:
            self._upsert_list_item(record.file_path, record, None)
        self.status_label.setText(f"Ready - {len(records)} document(s) in catalog")
        self._emit_selection()

    # ------------------------------------------------------------------#
    # Helpers for list rendering
    # ------------------------------------------------------------------#
    def _upsert_list_item(
        self,
        file_path: str,
        record: Optional[DocumentRecord],
        document: Optional[ParsedDocument],
    ):
        item = self._find_item(file_path)
        if item is None:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, file_path)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item.setCheckState(Qt.Checked)
            self.document_list.addItem(item)
        self._update_item_text(item, record, document)

    def _update_item_text(
        self,
        item: QListWidgetItem,
        record: Optional[DocumentRecord],
        document: Optional[ParsedDocument],
    ):
        file_path = item.data(Qt.UserRole)
        file_name = Path(file_path).name
        
        # Try to get suggested name from document renamer
        display_name = file_name
        try:
            from src.generation.document_renamer import DocumentRenamer
            renamer = DocumentRenamer()
            doc_id = self._get_document_id(file_path)
            suggested_display = renamer.get_display_name(doc_id, file_name)
            if suggested_display and suggested_display != file_name:
                display_name = suggested_display
        except Exception:
            # Fallback to original name if renamer fails
            pass
        
        # Use record label if available, otherwise use display name
        label = record.label if record else display_name
        
        category = record.category if record else (
            document.document_type if document else "unknown"
        )
        include_flag = (
            record.include_in_graph
            if record
            else self.catalog.default_include_flag(category)
        )
        graph_state = "Graph" if include_flag else ""
        indexed = record.indexed if record else False
        index_state = "Indexed" if indexed else "Pending"
        
        category_text = category.replace("_", " ").title()
        
        # Clean two-line format: Name on top, metadata below
        meta_parts = [category_text]
        if graph_state:
            meta_parts.append(graph_state)
        meta_parts.append(index_state)
        
        item.setText(f"{label}\n{' · '.join(meta_parts)}")
        
        # Set tooltip to show original filename if using suggested name
        if display_name != file_name:
            item.setToolTip(f"Original: {file_name}\nSuggested: {display_name}")


    def _find_item(self, file_path: str) -> Optional[QListWidgetItem]:
        for i in range(self.document_list.count()):
            item = self.document_list.item(i)
            if item.data(Qt.UserRole) == file_path:
                return item
        return None

    def _select_all_for_query(self):
        """Select all documents for querying."""
        for i in range(self.document_list.count()):
            item = self.document_list.item(i)
            if item.flags() & Qt.ItemIsUserCheckable:
                item.setCheckState(Qt.Checked)

    def _select_none_for_query(self):
        """Deselect all documents from querying."""
        for i in range(self.document_list.count()):
            item = self.document_list.item(i)
            if item.flags() & Qt.ItemIsUserCheckable:
                item.setCheckState(Qt.Unchecked)

    def _on_document_clicked(self, item):
        """Handle document click - toggle checkbox and select item."""
        # Ensure item is selected so preview updates
        self.document_list.setCurrentItem(item)
        
        if item.flags() & Qt.ItemIsUserCheckable:
            new_state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
            item.setCheckState(new_state)
    
    def _on_document_selected(self, current, previous):
        """Handle document selection change - show preview with summary."""
        if not current:
            self.preview_title.setText("Select a document to view details")
            self.preview_meta.setText("")
            self.preview_summary_label.setVisible(False)
            self.preview_summary.setVisible(False)
            self.generate_summary_btn.setVisible(False)
            return
        
        # Get file path from item data
        file_path = current.data(Qt.UserRole)
        if not file_path:
            return
        
        # Get document record from catalog
        record = self.catalog.get_record(file_path)
        if not record:
            return
        
        # Update preview title
        self.preview_title.setText(record.file_name)
        
        # Update metadata
        meta_parts = []
        if record.category:
            meta_parts.append(f"Type: {record.category}")
        if record.chunk_count:
            meta_parts.append(f"Chunks: {record.chunk_count}")
        if record.indexed:
            meta_parts.append("Indexed")
        else:
            meta_parts.append("⚠ Not indexed")
        if record.ingested_at:
            meta_parts.append(f"Added: {record.ingested_at[:10]}")
        
        self.preview_meta.setText(" • ".join(meta_parts))
        
        # #region agent log
        open(r'c:\Users\James\Desktop\SC Gen 6\.cursor\debug.log','a').write('{"location":"document_manager.py:_on_document_selected","message":"about to call _load_document_summary","data":{"file_name":"'+str(record.file_name)[:30].replace('"','\\"')+'"},"timestamp":'+str(int(__import__('time').time()*1000))+',"sessionId":"debug-session","hypothesisId":"H1-E"}\n')
        # #endregion
        # Try to load summary
        self._load_document_summary(file_path, record)
    
    def _load_document_summary(self, file_path: str, record: DocumentRecord):
        """Load and display document summary if available."""
        # #region agent log
        open(r'c:\Users\James\Desktop\SC Gen 6\.cursor\debug.log','a').write('{"location":"document_manager.py:_load_document_summary","message":"ENTRY","data":{"file_path_tail":"'+str(file_path)[-30:].replace('\\','\\\\').replace('"','\\"')+'"},"timestamp":'+str(int(__import__('time').time()*1000))+',"sessionId":"debug-session","hypothesisId":"H1-F"}\n')
        # #endregion
        try:
            from src.retrieval.summary_store import SummaryStore
            
            summary_store = SummaryStore(settings=self.settings)
            
            # Get document ID (same logic as chunker)
            doc_id = self._get_document_id(file_path)
            # #region agent log
            open(r'c:\Users\James\Desktop\SC Gen 6\.cursor\debug.log','a').write('{"location":"document_manager.py:_load_document_summary","message":"looking up summary","data":{"doc_id":"'+str(doc_id)[:50]+'","file_path":"'+str(file_path)[-40:].replace('\\','\\\\').replace('"','\\"')+'"},"timestamp":'+str(int(__import__('time').time()*1000))+',"sessionId":"debug-session","hypothesisId":"H1-C"}\n')
            # #endregion
            
            # Get summaries for this document
            summaries = summary_store.get_document_summaries(
                document_id=doc_id,
                summary_level="document",
            )
            # #region agent log
            open(r'c:\Users\James\Desktop\SC Gen 6\.cursor\debug.log','a').write('{"location":"document_manager.py:_load_document_summary","message":"summaries result","data":{"count":'+str(len(summaries) if summaries else 0)+',"doc_id":"'+str(doc_id)[:50]+'"},"timestamp":'+str(int(__import__('time').time()*1000))+',"sessionId":"debug-session","hypothesisId":"H1-D"}\n')
            # #endregion
            
            if summaries:
                # Show the first summary (usually "overview")
                summary = summaries[0]
                self.preview_summary_label.setVisible(True)
                self.preview_summary_label.setText(f"Summary ({summary.summary_type}):")
                self.preview_summary.setVisible(True)
                self.preview_summary.setText(summary.content)
                self.generate_summary_btn.setVisible(False)
            else:
                # No summary - show generate button
                self.preview_summary_label.setText("No summary available")
                self.preview_summary_label.setVisible(True)
                self.preview_summary.setVisible(False)
                self.generate_summary_btn.setVisible(True)
                self.generate_summary_btn.setProperty("file_path", file_path)
                
        except Exception as e:
            # Error loading summary
            self.preview_summary_label.setVisible(True)
            self.preview_summary_label.setText("Summary:")
            self.preview_summary.setVisible(True)
            self.preview_summary.setText(f"Error loading summary: {e}")
            self.generate_summary_btn.setVisible(True)
    
    def _on_document_double_clicked(self, item: QListWidgetItem):
        """Open document detail popup on double-click."""
        if not item:
            return
        
        file_path = item.data(Qt.UserRole)
        if not file_path:
            return
        
        record = self.catalog.get_record(file_path)
        if not record:
            return
        
        # Open the document popup
        self._document_popup = DocumentPopup(record, parent=self)
        self._document_popup.summary_requested.connect(self._on_popup_summary_requested)
        self._document_popup.delete_requested.connect(self._on_popup_delete_requested)
        self._document_popup.show()
    
    def _on_popup_summary_requested(self, file_path: str):
        """Handle summary request from popup."""
        # Find and select the item
        for i in range(self.document_list.count()):
            item = self.document_list.item(i)
            if item and item.data(Qt.UserRole) == file_path:
                self.document_list.setCurrentItem(item)
                break
        
        # Trigger summary generation
        self._generate_summary_for_selected()
    
    def _on_popup_delete_requested(self, file_path: str):
        """Handle delete request from popup."""
        try:
            # Delete from indexes
            vector_store = VectorStore(settings=self.settings)
            vector_store.delete_document(Path(file_path).name)
            
            # Delete from catalog
            self.catalog.delete_record(file_path)
            
            # Refresh list
            self._refresh_document_list()
            self.status_label.setText("Document deleted")
        except Exception as e:
            QMessageBox.warning(self, "Delete Error", f"Failed to delete document: {e}")
    
    def _generate_summary_for_selected(self):
        """Generate summary for the currently selected document."""
        current = self.document_list.currentItem()
        if not current:
            return
        
        file_path = current.data(Qt.UserRole)
        if not file_path:
            return
        
        record = self.catalog.get_record(file_path)
        if not record:
            return
        
        # Get document text from chunks
        try:
            from src.retrieval.fts5_index import FTS5Index
            from src.generation.summarizer import SummarizerService
            
            self.status_label.setText(f"Generating summary for {record.file_name}...")
            self.generate_summary_btn.setEnabled(False)
            
            # Get document ID
            doc_id = self._get_document_id(file_path)
            
            # Get text from FTS5 index
            fts5 = FTS5Index(settings=self.settings)
            conn = fts5._get_conn()
            cursor = conn.execute(
                "SELECT chunk_text FROM chunks WHERE document_id = ? ORDER BY id",
                (doc_id,)
            )
            text_parts = [r[0] for r in cursor.fetchall()]
            
            if not text_parts:
                self.status_label.setText("No indexed text found for this document")
                self.generate_summary_btn.setEnabled(True)
                return
            
            full_text = "\n\n".join(text_parts)
            
            # Generate summary
            summarizer = SummarizerService(settings=self.settings)
            summary = summarizer.generate_summary(
                document_id=doc_id,
                document_text=full_text,
                file_name=record.file_name,
                doc_type=record.category or "unknown",
                summary_type="overview",
            )
            
            if summary:
                # Save and display
                from src.retrieval.summary_store import SummaryStore
                summary_store = SummaryStore(settings=self.settings)
                summary_store.add_summary(summary)
                
                self.preview_summary_label.setVisible(True)
                self.preview_summary_label.setText(f"Summary ({summary.summary_type}):")
                self.preview_summary.setVisible(True)
                self.preview_summary.setText(summary.content)
                self.generate_summary_btn.setVisible(False)
                
                self.status_label.setText(f"Summary generated for {record.file_name}")
            else:
                self.status_label.setText("Failed to generate summary")
                
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
        finally:
            self.generate_summary_btn.setEnabled(True)

    def _on_selection_changed(self, item=None):
        """Handle checkbox state change."""
        self._emit_selection()

    def _emit_selection(self):
        """Emit the list of selected document file names."""
        selected_docs = []
        for i in range(self.document_list.count()):
            item = self.document_list.item(i)
            if item.flags() & Qt.ItemIsUserCheckable and item.checkState() == Qt.Checked:
                file_path = item.data(Qt.UserRole)
                if file_path:
                    selected_docs.append(Path(file_path).name)

        self.selection_changed.emit(selected_docs)

    def get_selected_documents(self):
        """Get list of currently selected document file names.

        Returns:
            List of file names (not paths)
        """
        selected_docs = []
        for i in range(self.document_list.count()):
            item = self.document_list.item(i)
            if item.flags() & Qt.ItemIsUserCheckable and item.checkState() == Qt.Checked:
                file_path = item.data(Qt.UserRole)
                if file_path:
                    selected_docs.append(Path(file_path).name)

        return selected_docs

