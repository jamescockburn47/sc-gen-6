"""Summary Generation Panel for UI.

Provides controls for generating document summaries:
- Model selection
- Progress tracking
- Start/cancel buttons
"""

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QProgressBar,
    QTextEdit,
    QGroupBox,
    QSpinBox,
    QCheckBox,
    QFrame,
)

from src.config_loader import get_settings
from src.generation.summarizer import (
    SummarizerService,
    SummarizerProgressSignals,
    estimate_summarization_time,
)
from src.retrieval.summary_store import SummaryStore
from src.retrieval.fts5_index import FTS5Index


class SummaryWorker(QThread):
    """Background worker for summary generation."""
    
    progress = Signal(int, int, str)
    document_completed = Signal(str, str, float)
    error = Signal(str, str)
    finished = Signal(int, float)
    model_status = Signal(str, str)
    log = Signal(str)
    
    def __init__(
        self,
        documents: list,
        model: str,
        summary_types: list,
        parallel: int,
    ):
        super().__init__()
        self.documents = documents
        self.model = model
        self.summary_types = summary_types
        self.parallel = parallel
        self.summarizer = None
    
    def run(self):
        """Run summary generation."""
        settings = get_settings()
        
        # Create summarizer with our signals
        signals = SummarizerProgressSignals()
        signals.progress.connect(self.progress.emit)
        signals.document_completed.connect(self.document_completed.emit)
        signals.error.connect(self.error.emit)
        signals.finished.connect(self.finished.emit)
        signals.model_status.connect(self.model_status.emit)
        
        self.summarizer = SummarizerService(
            settings=settings,
            progress_signals=signals,
        )
        
        self.log.emit(f"Starting summarization with {self.model}...")
        
        # Ensure model available
        success, message = self.summarizer.ensure_model_available(self.model)
        if not success:
            self.error.emit("", message)
            return
        
        self.log.emit(f"Model ready: {message}")
        
        # Generate summaries
        if self.parallel > 1:
            self.summarizer.summarize_documents_parallel(
                documents=self.documents,
                summary_types=self.summary_types,
                model=self.model,
                max_workers=self.parallel,
            )
        else:
            self.summarizer.summarize_documents(
                documents=self.documents,
                summary_types=self.summary_types,
                model=self.model,
            )
    
    def cancel(self):
        """Cancel the summarization."""
        if self.summarizer:
            self.summarizer.cancel()


class SummaryPanel(QWidget):
    """Panel for generating document summaries."""
    
    summaries_generated = Signal()  # Emitted when summaries are generated
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = get_settings()
        self.worker = None
        self._setup_ui()
        self._refresh_stats()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Header
        header = QLabel("Document Summarization")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #f1f5f9;")
        layout.addWidget(header)
        
        desc = QLabel(
            "Generate AI summaries for your documents to improve search quality. "
            "Uses a dedicated smaller model for fast bulk processing."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #94a3b8; font-size: 12px;")
        layout.addWidget(desc)
        
        # Stats group
        stats_group = QGroupBox("Current Status")
        stats_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #334155;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                color: #60a5fa;
            }
        """)
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("Loading...")
        self.stats_label.setStyleSheet("color: #cbd5e1;")
        stats_layout.addWidget(self.stats_label)
        
        layout.addWidget(stats_group)
        
        # Settings group
        settings_group = QGroupBox("Generation Settings")
        settings_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #334155;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                color: #60a5fa;
            }
        """)
        settings_layout = QVBoxLayout(settings_group)
        
        # Model selector
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.settings.summary.summarization_model_alternatives)
        self.model_combo.setCurrentText(self.settings.summary.summarization_model)
        self.model_combo.currentTextChanged.connect(self._update_estimate)
        model_row.addWidget(self.model_combo, 1)
        settings_layout.addLayout(model_row)
        
        # Parallel workers
        parallel_row = QHBoxLayout()
        parallel_row.addWidget(QLabel("Parallel workers:"))
        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(1, 4)
        self.parallel_spin.setValue(self.settings.summary.parallel_summaries)
        parallel_row.addWidget(self.parallel_spin)
        parallel_row.addStretch()
        settings_layout.addLayout(parallel_row)
        
        # Summary types
        types_row = QHBoxLayout()
        types_row.addWidget(QLabel("Summary types:"))
        self.overview_check = QCheckBox("Overview")
        self.overview_check.setChecked(True)
        self.keypoints_check = QCheckBox("Key Points")
        types_row.addWidget(self.overview_check)
        types_row.addWidget(self.keypoints_check)
        types_row.addStretch()
        settings_layout.addLayout(types_row)
        
        # Regenerate option
        self.regenerate_check = QCheckBox("Regenerate existing summaries")
        self.regenerate_check.setStyleSheet("color: #fbbf24;")
        settings_layout.addWidget(self.regenerate_check)
        
        layout.addWidget(settings_group)
        
        # Time estimate
        self.estimate_label = QLabel("")
        self.estimate_label.setStyleSheet("color: #4ade80; font-size: 10pt;")
        layout.addWidget(self.estimate_label)
        
        # Progress section
        self.progress_frame = QFrame()
        self.progress_frame.setVisible(False)
        progress_layout = QVBoxLayout(self.progress_frame)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #334155;
                border-radius: 4px;
                text-align: center;
                color: #f1f5f9;
            }
            QProgressBar::chunk {
                background-color: #8b7cf6;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        progress_layout.addWidget(self.progress_label)
        
        layout.addWidget(self.progress_frame)
        
        # Log output
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #0f172a;
                color: #94a3b8;
                border: 1px solid #334155;
                border-radius: 4px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.log_text)
        
        # Buttons
        btn_row = QHBoxLayout()
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_stats)
        btn_row.addWidget(self.refresh_btn)
        
        btn_row.addStretch()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setStyleSheet("background-color: #f87171;")
        btn_row.addWidget(self.cancel_btn)
        
        self.generate_btn = QPushButton("▶ Generate Summaries")
        self.generate_btn.clicked.connect(self._start_generation)
        self.generate_btn.setStyleSheet("background-color: #4ade80; color: #0f0f12;")
        btn_row.addWidget(self.generate_btn)
        
        layout.addLayout(btn_row)
        
        layout.addStretch()
        
        # Initial estimate
        self._update_estimate()
    
    def _refresh_stats(self):
        """Refresh summary statistics."""
        try:
            summary_store = SummaryStore(settings=self.settings)
            fts5_index = FTS5Index(settings=self.settings)
            
            summary_stats = summary_store.stats()
            fts5_stats = fts5_index.stats()
            
            total_docs = fts5_stats.get("file_count", 0)
            docs_with_summaries = summary_stats.get("documents_with_summaries", 0)
            total_summaries = summary_stats.get("total_summaries", 0)
            
            pending = total_docs - docs_with_summaries
            
            self.stats_label.setText(
                f"Documents indexed: {total_docs}\n"
                f"Documents with summaries: {docs_with_summaries}\n"
                f"Total summaries: {total_summaries}\n"
                f"Pending: {pending}"
            )
            
            self._pending_count = pending
            self._total_docs = total_docs
            self._update_estimate()
            
        except Exception as e:
            self.stats_label.setText(f"Error loading stats: {e}")
    
    def _update_estimate(self):
        """Update time estimate based on current settings."""
        try:
            count = self._pending_count if hasattr(self, '_pending_count') else 0
            
            if self.regenerate_check.isChecked():
                count = self._total_docs if hasattr(self, '_total_docs') else 0
            
            if count == 0:
                self.estimate_label.setText("All documents have summaries")
                return
            
            summary_types = []
            if self.overview_check.isChecked():
                summary_types.append("overview")
            if self.keypoints_check.isChecked():
                summary_types.append("key_points")
            
            estimate = estimate_summarization_time(
                num_documents=count,
                model=self.model_combo.currentText(),
                summary_types=len(summary_types) or 1,
            )
            
            self.estimate_label.setText(
                f"⏱ Estimated time: {estimate['formatted_time']} "
                f"({count} documents × {len(summary_types) or 1} summary types)"
            )
            
        except Exception:
            self.estimate_label.setText("")
    
    def _get_documents_to_summarize(self) -> list:
        """Get list of documents that need summaries."""
        fts5_index = FTS5Index(settings=self.settings)
        summary_store = SummaryStore(settings=self.settings)
        
        conn = fts5_index._get_conn()
        cursor = conn.execute("""
            SELECT DISTINCT document_id, file_name, document_type
            FROM chunks
        """)
        
        documents = []
        for row in cursor.fetchall():
            doc_id = row[0]
            file_name = row[1]
            doc_type = row[2] or "unknown"
            
            # Check if summary exists (unless regenerating)
            if not self.regenerate_check.isChecked():
                existing = summary_store.get_document_summaries(
                    document_id=doc_id,
                    summary_level="document",
                )
                if existing:
                    continue
            
            # Get document text
            chunk_cursor = conn.execute("""
                SELECT chunk_text FROM chunks 
                WHERE document_id = ? 
                ORDER BY id
            """, (doc_id,))
            
            text_parts = [r[0] for r in chunk_cursor.fetchall()]
            full_text = "\n\n".join(text_parts)
            
            documents.append({
                "document_id": doc_id,
                "file_name": file_name,
                "doc_type": doc_type,
                "text": full_text,
            })
        
        return documents
    
    def _start_generation(self):
        """Start summary generation."""
        documents = self._get_documents_to_summarize()
        
        if not documents:
            self._log("No documents need summaries.")
            return
        
        summary_types = []
        if self.overview_check.isChecked():
            summary_types.append("overview")
        if self.keypoints_check.isChecked():
            summary_types.append("key_points")
        
        if not summary_types:
            summary_types = ["overview"]
        
        # Update UI
        self.generate_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.progress_frame.setVisible(True)
        self.progress_bar.setMaximum(len(documents) * len(summary_types))
        self.progress_bar.setValue(0)
        
        self._log(f"Starting generation for {len(documents)} documents...")
        
        # Create and start worker
        self.worker = SummaryWorker(
            documents=documents,
            model=self.model_combo.currentText(),
            summary_types=summary_types,
            parallel=self.parallel_spin.value(),
        )
        
        self.worker.progress.connect(self._on_progress)
        self.worker.document_completed.connect(self._on_doc_complete)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)
        self.worker.model_status.connect(self._on_model_status)
        self.worker.log.connect(self._log)
        
        self.worker.start()
    
    def _cancel(self):
        """Cancel generation."""
        if self.worker:
            self._log("Cancelling...")
            self.worker.cancel()
    
    def _on_progress(self, completed: int, total: int, current_file: str):
        """Handle progress update."""
        self.progress_bar.setValue(completed)
        self.progress_label.setText(f"Processing: {current_file}")
    
    def _on_doc_complete(self, file_name: str, status: str, time_sec: float):
        """Handle document completion."""
        self._log(f"{file_name} ({time_sec:.1f}s)")
    
    def _on_error(self, file_name: str, error: str):
        """Handle error."""
        if file_name:
            self._log(f"Error: {file_name}: {error}")
        else:
            self._log(f"Error: {error}")
    
    def _on_model_status(self, model: str, status: str):
        """Handle model status update."""
        self._log(f"Model {model}: {status}")
    
    def _on_finished(self, total: int, time_sec: float):
        """Handle completion."""
        self._log(f"\nCompleted: {total} summaries in {int(time_sec // 60)}m {int(time_sec % 60)}s")
        
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.progress_frame.setVisible(False)
        
        self._refresh_stats()
        self.summaries_generated.emit()
    
    def _log(self, message: str):
        """Add message to log."""
        self.log_text.append(message)
        # Scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

