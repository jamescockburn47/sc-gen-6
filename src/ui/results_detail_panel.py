"""Results detail panel with chunk list, logs, diagnostics, and RAG graph placeholder."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPlainTextEdit, QTabWidget, QVBoxLayout, QWidget, QScrollArea

from src.ui.components.diagnostics_panel import DiagnosticsPanel


class ResultsDetailPanel(QWidget):
    """Tabbed panel that surfaces retrieved chunks, run logs, diagnostics, and graph placeholder."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Diagnostics tab (first - most important during queries)
        self.diagnostics_panel = DiagnosticsPanel()
        diagnostics_scroll = QScrollArea()
        diagnostics_scroll.setWidget(self.diagnostics_panel)
        diagnostics_scroll.setWidgetResizable(True)
        diagnostics_scroll.setFrameShape(QScrollArea.NoFrame)
        self.tabs.addTab(diagnostics_scroll, "⚡ Pipeline")

        # Chunks tab
        self.chunks_text = QPlainTextEdit()
        self.chunks_text.setReadOnly(True)
        self.chunks_text.setPlaceholderText("Retrieved chunks will appear here after a query runs.")
        self.tabs.addTab(self.chunks_text, "Chunks")

        # Run log tab
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("Status updates and debug notes will stream here.")
        self.tabs.addTab(self.log_text, "Log")

        # Graph placeholder tab (for upcoming RAG graph integration)
        self.graph_placeholder = QLabel(
            "Graph view coming soon.\n\n"
            "This tab will visualize entity/event relationships across documents "
            "to help trace fraud chains and competition issues."
        )
        self.graph_placeholder.setAlignment(Qt.AlignCenter)
        self.graph_placeholder.setWordWrap(True)
        self.tabs.addTab(self.graph_placeholder, "🔗 Graph")

    # ------------------------------------------------------------------#
    # Public API
    # ------------------------------------------------------------------#
    def clear_all(self) -> None:
        """Reset all tabs."""
        self.chunks_text.clear()
        self.log_text.clear()

    def set_chunks(self, chunks: Sequence[dict[str, Any]]) -> None:
        """Render retrieved chunks into the chunks tab."""
        if not chunks:
            self.chunks_text.setPlainText("No chunks retrieved.")
            return

        lines: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})
            score = chunk.get("score", 0.0)
            lines.append(
                f"{idx}. {metadata.get('file_name', 'unknown')} | "
                f"Page {metadata.get('page_number', 'N/A')} | "
                f"Score {score:.3f}"
            )
            snippet = chunk.get("text", "").strip()
            if snippet:
                snippet = snippet.replace("\n", " ")
                if len(snippet) > 280:
                    snippet = snippet[:277] + "..."
                lines.append(f"    {snippet}")
            lines.append("")

        self.chunks_text.setPlainText("\n".join(lines))

    def append_log(self, message: str) -> None:
        """Append a timestamped line to the run log tab."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{timestamp}] {message}")

    def set_graph_message(self, message: str) -> None:
        """Update the placeholder graph message."""
        self.graph_placeholder.setText(message)

    def progress_widget(self) -> "ResultsDetailPanel":
        """Return self as progress widget (stub for compatibility)."""
        return self

    def start_progress_monitor(self) -> None:
        """Start progress monitoring (stub)."""
        pass

    def set_llm_console(self, text: str) -> None:
        """Set LLM console output (appends to log)."""
        # For now, we don't display raw console output
        pass

    # Progress signal handlers (stubs for compatibility)
    def on_stage_started(self, stage_name: str) -> None:
        """Handle stage started signal."""
        pass

    def on_stage_completed(self, stage_name: str, stats: dict) -> None:
        """Handle stage completed signal."""
        pass

    def on_semantic_progress(self, count: int) -> None:
        """Handle semantic search progress."""
        pass

    def on_keyword_progress(self, count: int) -> None:
        """Handle BM25 search progress."""
        pass

    def on_fusion_complete(self, count: int) -> None:
        """Handle fusion complete signal."""
        pass

    def on_rerank_progress(self, completed: int, total: int) -> None:
        """Handle rerank progress signal."""
        pass

    def on_filtering_complete(self, count: int) -> None:
        """Handle filtering complete signal."""
        pass

    def on_retrieval_complete(self, count: int) -> None:
        """Handle retrieval complete signal."""
        pass

    def on_generation_started(self, model: str) -> None:
        """Handle generation started signal."""
        pass

    def on_generation_progress(self, tokens: int, time_ms: float, tok_per_sec: float) -> None:
        """Handle generation progress signal."""
        pass

    def on_generation_completed(self, total_tokens: int, total_time_ms: float, avg_tok_per_sec: float) -> None:
        """Handle generation completed signal."""
        pass

    def on_generation_stage_changed(self, stage: str) -> None:
        """Handle generation stage change signal."""
        pass

    def on_generation_batch_progress(self, completed: int, total: int) -> None:
        """Handle generation batch progress signal."""
        pass

