"""Results Panel widget for right bottom section."""

import re
import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.config_loader import get_settings
from src.generation.citation import CitationVerifier, verify_citations
from src.generation.llm_service import LLMService, GenerationProgressSignals
from src.retrieval.hybrid_retriever import HybridRetriever, RetrieverProgressSignals
from src.ui.progress_panel import ProgressPanelWidget


class QueryWorker(QThread):
    """Worker thread for running queries."""

    token_received = Signal(str)  # Streaming token
    finished = Signal(str, dict, list)  # Complete response, verification result, chunks
    error = Signal(str)  # Error message

    def __init__(
        self,
        query_data: dict,
        retriever: HybridRetriever,
        llm_service: LLMService,
        parent=None,
    ):
        """Initialize worker.

        Args:
            query_data: Query parameters dict
            retriever: HybridRetriever instance
            llm_service: LLMService instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.query_data = query_data
        self.retriever = retriever
        self.llm_service = llm_service
        self.stop_event = threading.Event()
        self.retrieved_chunks = []

    def run(self):
        """Run query in background thread."""
        try:
            # Check if we should refuse before generation
            chunks = self.retriever.retrieve(
                query=self.query_data["query"],
                semantic_top_n=self.query_data.get("semantic_top_n", 50),
                keyword_top_n=self.query_data.get("keyword_top_n", 50),
                rerank_top_k=self.query_data.get("rerank_top_k", 20),
                context_to_llm=self.query_data.get("context_to_llm", 5),
                confidence_threshold=self.query_data.get("confidence_threshold", 0.70),
                doc_type_filter=self.query_data.get("doc_type_filter"),
                selected_documents=self.query_data.get("selected_documents"),
                cancel_event=self.stop_event,
                skip_reranking=self.query_data.get("skip_reranking"),
            )

            if not chunks:
                self.error.emit("No relevant chunks found. Try rephrasing your query.")
                return

            # Store chunks for display
            self.retrieved_chunks = chunks

            # Check confidence threshold
            verifier = CitationVerifier(chunks, self.query_data.get("confidence_threshold", 0.70))
            should_refuse, reason = verifier.should_refuse_before_generation()
            if should_refuse:
                self.error.emit(f"Cannot answer: {reason}")
                return

            # Generate with streaming
            response_parts = []

            def token_callback(token: str):
                if not self.stop_event.is_set():
                    response_parts.append(token)
                    self.token_received.emit(token)

            response = self.llm_service.generate_with_context(
                query=self.query_data["query"],
                chunks=chunks,
                model=self.query_data.get("model"),
                stream=True,
                callback=token_callback,
                cancel_event=self.stop_event,
            )

            if self.stop_event.is_set():
                self.error.emit("Query stopped by user")
                return

            # Verify citations
            verification_result = verify_citations(
                response,
                chunks,
                confidence_threshold=self.query_data.get("confidence_threshold", 0.70),
            )

            # Convert dataclass to dict
            from dataclasses import asdict
            self.finished.emit(response, asdict(verification_result), chunks)

        except Exception as e:
            if self.stop_event.is_set():
                self.error.emit("Query stopped by user")
            else:
                self.error.emit(f"Query error: {str(e)}")

    def stop(self):
        """Stop the query."""
        self.stop_event.set()


class ResultsPanelWidget(QWidget):
    """Results panel for displaying query results."""

    def __init__(self, parent=None):
        """Initialize results panel."""
        super().__init__(parent)
        self.settings = get_settings()

        # Create progress signal objects
        self.retriever_signals = RetrieverProgressSignals()
        self.generation_signals = GenerationProgressSignals()

        self.retriever: Optional[HybridRetriever] = None
        self.llm_service: Optional[LLMService] = None
        self.query_worker: Optional[QueryWorker] = None
        self.current_chunks: list[dict] = []

        self._setup_ui()
        self._initialize_services()

    def _setup_ui(self):
        """Set up UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title_layout = QHBoxLayout()
        title = QLabel("Results")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        title_layout.addWidget(title)

        # Toggles and actions
        self.show_confidence_checkbox = QCheckBox("Show Confidence")
        self.show_confidence_checkbox.setChecked(self.settings.ui.show_confidence_scores)
        title_layout.addWidget(self.show_confidence_checkbox)

        self.show_chunks_checkbox = QCheckBox("Show Retrieved Chunks")
        title_layout.addWidget(self.show_chunks_checkbox)

        self.show_progress_checkbox = QCheckBox("Show Progress")
        self.show_progress_checkbox.setChecked(True)  # Default to visible
        title_layout.addWidget(self.show_progress_checkbox)

        # Export button
        export_btn = QPushButton("Export...")
        export_btn.clicked.connect(self._export_results)
        title_layout.addWidget(export_btn)

        # Copy button
        copy_btn = QPushButton("Copy Answer")
        copy_btn.clicked.connect(self._copy_answer)
        title_layout.addWidget(copy_btn)

        title_layout.addStretch()
        layout.addLayout(title_layout)

        # Progress Panel (collapsible)
        self.progress_panel = ProgressPanelWidget()
        self.progress_panel.setVisible(True)
        layout.addWidget(self.progress_panel)

        # Connect progress toggle
        self.show_progress_checkbox.toggled.connect(self.progress_panel.setVisible)

        # Connect retriever signals to progress panel
        self.retriever_signals.stage_started.connect(self.progress_panel.on_stage_started)
        self.retriever_signals.stage_completed.connect(self.progress_panel.on_stage_completed)
        self.retriever_signals.semantic_progress.connect(self.progress_panel.on_semantic_progress)
        self.retriever_signals.keyword_progress.connect(self.progress_panel.on_keyword_progress)
        self.retriever_signals.fusion_complete.connect(self.progress_panel.on_fusion_complete)
        self.retriever_signals.rerank_progress.connect(self.progress_panel.on_rerank_progress)
        self.retriever_signals.filtering_complete.connect(self.progress_panel.on_filtering_complete)
        self.retriever_signals.retrieval_complete.connect(self.progress_panel.on_retrieval_complete)

        # Connect generation signals to progress panel
        self.generation_signals.generation_started.connect(self.progress_panel.on_generation_started)
        self.generation_signals.generation_progress.connect(self.progress_panel.on_generation_progress)
        self.generation_signals.generation_completed.connect(self.progress_panel.on_generation_completed)
        self.generation_signals.stage_changed.connect(self.progress_panel.on_generation_stage_changed)
        self.generation_signals.batch_progress.connect(self.progress_panel.on_generation_batch_progress)

        # Answer display
        answer_label = QLabel("Answer:")
        layout.addWidget(answer_label)

        self.answer_text = QPlainTextEdit()
        self.answer_text.setReadOnly(True)
        self.answer_text.setPlaceholderText("Query results will appear here...")
        layout.addWidget(self.answer_text, stretch=1)

        # Retrieved chunks display (collapsible)
        chunks_label = QLabel("Retrieved Chunks:")
        layout.addWidget(chunks_label)

        self.chunks_text = QPlainTextEdit()
        self.chunks_text.setReadOnly(True)
        self.chunks_text.setMaximumHeight(150)
        self.chunks_text.setVisible(False)
        layout.addWidget(self.chunks_text)

        # Connect toggles
        self.show_chunks_checkbox.toggled.connect(self.chunks_text.setVisible)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

    def _initialize_services(self):
        """Initialize retrieval and LLM services."""
        try:
            from src.retrieval.hybrid_retriever import HybridRetriever
            from src.generation.llm_service import LLMService

            self.retriever = HybridRetriever(
                settings=self.settings,
                progress_signals=self.retriever_signals
            )
            self.llm_service = LLMService(
                settings=self.settings,
                progress_signals=self.generation_signals
            )
        except Exception as e:
            self.status_label.setText(f"Error initializing services: {str(e)}")

    def start_query(self, query_data: dict):
        """Start a query.

        Args:
            query_data: Query parameters dictionary
        """
        if not self.retriever or not self.llm_service:
            self.status_label.setText("Services not initialized")
            return

        self.status_label.setText("Retrieving relevant chunks...")

        # Clear previous results
        self.answer_text.clear()
        self.chunks_text.clear()
        self.current_chunks = []

        # Notify progress panel that query is starting
        self.progress_panel.start_query()

        # Start worker thread
        self.query_worker = QueryWorker(query_data, self.retriever, self.llm_service)
        self.query_worker.token_received.connect(self._on_token_received)
        self.query_worker.finished.connect(self._on_query_finished)
        self.query_worker.error.connect(self._on_query_error)
        self.query_worker.start()

    def stop_query(self):
        """Stop current query."""
        if self.query_worker and self.query_worker.isRunning():
            self.query_worker.stop()
            self.status_label.setText("Stopping query...")

    def _on_token_received(self, token: str):
        """Handle streaming token."""
        self.answer_text.insertPlainText(token)
        # Auto-scroll to bottom
        scrollbar = self.answer_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_query_finished(self, response: str, verification_result: dict, chunks: list[dict]):
        """Handle query completion.

        Args:
            response: Complete response text
            verification_result: Citation verification result dict
            chunks: Retrieved chunks
        """
        self.current_chunks = chunks
        self.status_label.setText(
            f"Query complete. Verified: {verification_result.get('verified', False)}"
        )

        # Highlight citations if needed
        if self.show_confidence_checkbox.isChecked():
            self._highlight_citations(response, verification_result)

        # Show retrieved chunks if enabled
        if self.show_chunks_checkbox.isChecked():
            self._display_retrieved_chunks()

        # Notify query panel
        if self.parent():
            query_panel = getattr(self.parent(), "query_panel", None)
            if query_panel:
                query_panel.on_query_finished()

    def _on_query_error(self, error_message: str):
        """Handle query error."""
        self.status_label.setText(f"Error: {error_message}")
        self.answer_text.setPlainText(f"Error: {error_message}")

        if self.parent():
            query_panel = getattr(self.parent(), "query_panel", None)
            if query_panel:
                query_panel.on_query_finished()

    def _highlight_citations(self, response: str, verification_result: dict):
        """Highlight citations in response (placeholder for future enhancement)."""
        # This could be enhanced with rich text formatting
        pass

    def _display_retrieved_chunks(self):
        """Display retrieved chunks."""
        if not self.current_chunks:
            self.chunks_text.setPlainText("No chunks retrieved.")
            return

        chunks_text_parts = []
        for i, chunk in enumerate(self.current_chunks, 1):
            metadata = chunk.get("metadata", {})
            score = chunk.get("score", 0.0)
            chunks_text_parts.append(
                f"{i}. [{chunk.get('chunk_id', 'unknown')}] "
                f"Score: {score:.3f} | "
                f"{metadata.get('file_name', 'unknown')} | "
                f"Page {metadata.get('page_number', 'N/A')}"
            )
            chunks_text_parts.append(f"   {chunk.get('text', '')[:200]}...")
            chunks_text_parts.append("")

        self.chunks_text.setPlainText("\n".join(chunks_text_parts))

    def _export_results(self):
        """Export results to file."""
        if not self.answer_text.toPlainText().strip():
            QMessageBox.information(self, "No Results", "No answer to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "Text Files (*.txt);;All Files (*)",
        )

        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("=" * 60 + "\n")
                    f.write("SC Gen 6 - Query Results\n")
                    f.write("=" * 60 + "\n\n")
                    f.write("Answer:\n")
                    f.write("-" * 60 + "\n")
                    f.write(self.answer_text.toPlainText())
                    f.write("\n\n")

                    if self.current_chunks:
                        f.write("Retrieved Chunks:\n")
                        f.write("-" * 60 + "\n")
                        f.write(self.chunks_text.toPlainText())

                QMessageBox.information(
                    self, "Export Successful", f"Results exported to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Failed", f"Failed to export results:\n{str(e)}"
                )

    def _copy_answer(self):
        """Copy answer to clipboard."""
        answer_text = self.answer_text.toPlainText()
        if answer_text:
            clipboard = QApplication.clipboard()
            clipboard.setText(answer_text)
            self.status_label.setText("Answer copied to clipboard")
        else:
            QMessageBox.information(self, "No Answer", "No answer to copy.")

