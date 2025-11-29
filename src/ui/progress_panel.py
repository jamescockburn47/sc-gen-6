"""Progress Panel widget for displaying retrieval and generation progress."""

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class ProgressPanelWidget(QWidget):
    """Widget for displaying real-time query progress."""

    def __init__(self, parent=None):
        """Initialize progress panel."""
        super().__init__(parent)
        self._setup_ui()
        self._reset()

    def _setup_ui(self):
        """Set up UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Retrieval Progress Section
        retrieval_group = QGroupBox("Retrieval Pipeline")
        retrieval_layout = QVBoxLayout()

        # Stage indicators
        self.stage_labels = {}
        stages = [
            ("semantic_search", "Semantic Search"),
            ("bm25_search", "BM25 Search"),
            ("fusion", "Fusion"),
            ("reranking", "Reranking"),
            ("filtering", "Filtering"),
        ]

        for stage_key, stage_name in stages:
            stage_layout = QHBoxLayout()

            # Stage name label
            name_label = QLabel(f"{stage_name}:")
            name_label.setMinimumWidth(120)
            stage_layout.addWidget(name_label)

            # Status label
            status_label = QLabel("⏸")
            status_label.setMinimumWidth(20)
            stage_layout.addWidget(status_label)

            # Stats label
            stats_label = QLabel("—")
            stats_label.setStyleSheet("color: gray;")
            stage_layout.addWidget(stats_label, stretch=1)

            retrieval_layout.addLayout(stage_layout)
            self.stage_labels[stage_key] = (status_label, stats_label)

        # Overall retrieval stats
        retrieval_layout.addSpacing(5)
        self.retrieval_summary = QLabel("Ready")
        self.retrieval_summary.setStyleSheet("font-weight: bold; color: #2196F3;")
        retrieval_layout.addWidget(self.retrieval_summary)

        retrieval_group.setLayout(retrieval_layout)
        layout.addWidget(retrieval_group)

        # Generation Progress Section
        generation_group = QGroupBox("Generation")
        generation_layout = QVBoxLayout()

        # Model label
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_label = QLabel("—")
        self.model_label.setStyleSheet("color: gray;")
        model_layout.addWidget(self.model_label, stretch=1)
        generation_layout.addLayout(model_layout)

        self.stage_label = QLabel("Stage: —")
        self.stage_label.setStyleSheet("color: gray;")
        generation_layout.addWidget(self.stage_label)

        self.batch_label = QLabel("Batches: —")
        self.batch_label.setStyleSheet("color: gray;")
        generation_layout.addWidget(self.batch_label)

        # Progress bar
        self.generation_progress = QProgressBar()
        self.generation_progress.setMinimum(0)
        self.generation_progress.setMaximum(100)
        self.generation_progress.setValue(0)
        self.generation_progress.setTextVisible(False)
        generation_layout.addWidget(self.generation_progress)

        # Stats labels
        stats_layout = QHBoxLayout()

        self.tokens_label = QLabel("Tokens: 0")
        stats_layout.addWidget(self.tokens_label)

        self.speed_label = QLabel("Speed: —")
        stats_layout.addWidget(self.speed_label)

        self.time_label = QLabel("Time: 0.0s")
        stats_layout.addWidget(self.time_label, stretch=1)

        generation_layout.addLayout(stats_layout)

        generation_group.setLayout(generation_layout)
        layout.addWidget(generation_group)

        # Documents Used Section
        docs_group = QGroupBox("Context")
        docs_layout = QVBoxLayout()

        self.docs_label = QLabel("No documents selected")
        self.docs_label.setStyleSheet("color: gray;")
        self.docs_label.setWordWrap(True)
        docs_layout.addWidget(self.docs_label)

        docs_group.setLayout(docs_layout)
        layout.addWidget(docs_group)

        layout.addStretch()

    def _reset(self):
        """Reset all progress indicators."""
        # Reset retrieval stages
        for status_label, stats_label in self.stage_labels.values():
            status_label.setText("⏸")
            status_label.setStyleSheet("")
            stats_label.setText("—")
            stats_label.setStyleSheet("color: gray;")

        self.retrieval_summary.setText("Ready")
        self.retrieval_summary.setStyleSheet("font-weight: bold; color: #2196F3;")

        # Reset generation
        self.model_label.setText("—")
        self.model_label.setStyleSheet("color: gray;")
        self.stage_label.setText("Stage: —")
        self.stage_label.setStyleSheet("color: gray;")
        self.batch_label.setText("Batches: —")
        self.batch_label.setStyleSheet("color: gray;")
        self.generation_progress.setValue(0)
        self.tokens_label.setText("Tokens: 0")
        self.speed_label.setText("Speed: —")
        self.time_label.setText("Time: 0.0s")

        # Reset documents
        self.docs_label.setText("No documents selected")
        self.docs_label.setStyleSheet("color: gray;")

    @Slot(str)
    def on_stage_started(self, stage_name: str):
        """Handle stage started signal."""
        if stage_name in self.stage_labels:
            status_label, stats_label = self.stage_labels[stage_name]
            status_label.setText("▶")
            status_label.setStyleSheet("color: #2196F3;")
            stats_label.setText("In progress...")
            stats_label.setStyleSheet("color: #2196F3;")

    @Slot(str, dict)
    def on_stage_completed(self, stage_name: str, stats: dict):
        """Handle stage completed signal."""
        if stage_name in self.stage_labels:
            status_label, stats_label = self.stage_labels[stage_name]
            status_label.setText("✓")
            status_label.setStyleSheet("color: #4CAF50;")

            # Format stats based on stage
            time_ms = stats.get("time_ms", 0)
            count = stats.get("count", 0)

            if stage_name in ["semantic_search", "bm25_search"]:
                top_score = stats.get("top_score", 0)
                stats_text = f"{count} candidates, top={top_score:.3f}, {time_ms:.0f}ms"
            elif stage_name == "fusion":
                stats_text = f"{count} merged, {time_ms:.0f}ms"
            elif stage_name == "reranking":
                stats_text = f"{count} reranked, {time_ms:.0f}ms"
            elif stage_name == "filtering":
                passed = stats.get("passed_count", 0)
                avg_conf = stats.get("avg_confidence", 0)
                stats_text = f"{passed} passed (avg conf: {avg_conf:.3f}), {time_ms:.0f}ms"
            else:
                stats_text = f"{time_ms:.0f}ms"

            stats_label.setText(stats_text)
            stats_label.setStyleSheet("color: #4CAF50;")

    @Slot(int, int, float, float)
    def on_semantic_progress(self, current: int, total: int, top_score: float, time_ms: float):
        """Handle semantic search progress."""
        # Update handled by stage_completed
        pass

    @Slot(int, int, float, float)
    def on_keyword_progress(self, current: int, total: int, top_score: float, time_ms: float):
        """Handle BM25 search progress."""
        # Update handled by stage_completed
        pass

    @Slot(int, float)
    def on_fusion_complete(self, merged_count: int, time_ms: float):
        """Handle fusion complete."""
        # Update handled by stage_completed
        pass

    @Slot(int, int, float)
    def on_rerank_progress(self, current: int, total: int, time_ms: float):
        """Handle reranking progress."""
        # Update handled by stage_completed
        pass

    @Slot(int, float, float, float, float)
    def on_filtering_complete(self, passed: int, avg: float, min_val: float, max_val: float, time_ms: float):
        """Handle filtering complete."""
        # Update handled by stage_completed
        pass

    @Slot(int, float, dict)
    def on_retrieval_complete(self, results_count: int, total_time_ms: float, stats: dict):
        """Handle overall retrieval complete."""
        unique_docs = stats.get("unique_documents", 0)
        avg_conf = stats.get("avg_confidence", 0)

        summary = f"✓ Retrieved {results_count} chunks from {unique_docs} documents in {total_time_ms/1000:.1f}s (avg conf: {avg_conf:.3f})"
        self.retrieval_summary.setText(summary)
        self.retrieval_summary.setStyleSheet("font-weight: bold; color: #4CAF50;")

        # Update documents used (will be refined when we have actual doc names)
        if unique_docs > 0:
            self.docs_label.setText(f"{results_count} chunks from {unique_docs} unique documents")
            self.docs_label.setStyleSheet("color: #4CAF50;")

    @Slot(str)
    def on_generation_started(self, model_name: str):
        """Handle generation started."""
        self.model_label.setText(model_name)
        self.model_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        self.stage_label.setText("Stage: Initializing")
        self.stage_label.setStyleSheet("color: #2196F3;")
        self.batch_label.setText("Batches: —")
        self.batch_label.setStyleSheet("color: gray;")
        self.generation_progress.setValue(0)
        self.tokens_label.setText("Tokens: 0")
        self.speed_label.setText("Speed: —")
        self.time_label.setText("Time: 0.0s")

    @Slot(int, float, float)
    def on_generation_progress(self, tokens: int, time_ms: float, tok_per_sec: float):
        """Handle generation progress."""
        self.tokens_label.setText(f"Tokens: {tokens}")
        self.speed_label.setText(f"Speed: {tok_per_sec:.1f} tok/s")
        self.time_label.setText(f"Time: {time_ms/1000:.1f}s")

        # Animate progress bar (indeterminate since we don't know total)
        current_value = self.generation_progress.value()
        self.generation_progress.setValue((current_value + 5) % 100)

    @Slot(int, float, float)
    def on_generation_completed(self, total_tokens: int, total_time_ms: float, avg_tok_per_sec: float):
        """Handle generation completed."""
        self.tokens_label.setText(f"Tokens: {total_tokens}")
        self.speed_label.setText(f"Speed: {avg_tok_per_sec:.1f} tok/s (avg)")
        self.speed_label.setStyleSheet("color: #4CAF50;")
        self.time_label.setText(f"Time: {total_time_ms/1000:.1f}s")
        self.generation_progress.setValue(100)
        self.model_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

    @Slot(str)
    def on_generation_stage_changed(self, stage_name: str):
        """Update stage label."""
        display = stage_name or "—"
        color = "#2196F3" if stage_name else "gray"
        self.stage_label.setText(f"Stage: {display}")
        self.stage_label.setStyleSheet(f"color: {color}; font-weight: bold;" if stage_name else "color: gray;")

    @Slot(int, int)
    def on_generation_batch_progress(self, completed: int, total: int):
        """Update batch progress indicator."""
        if total <= 0:
            self.batch_label.setText("Batches: —")
            self.batch_label.setStyleSheet("color: gray;")
            return

        self.batch_label.setText(f"Batches: {completed}/{total}")
        self.batch_label.setStyleSheet("color: #2196F3;")
        pct = max(0, min(100, int((completed / total) * 100)))
        self.generation_progress.setValue(pct)

    def start_query(self):
        """Called when a new query starts."""
        self._reset()
        self.retrieval_summary.setText("Starting query...")
        self.retrieval_summary.setStyleSheet("font-weight: bold; color: #2196F3;")
