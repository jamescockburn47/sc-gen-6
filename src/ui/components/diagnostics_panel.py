"""Diagnostics Panel - Shows RAG pipeline status in a user-friendly way."""

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QProgressBar,
    QGridLayout,
)


class MetricCard(QFrame):
    """A small card showing a single metric."""

    def __init__(self, label: str, value: str = "-", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            MetricCard {
                background-color: rgba(36, 32, 25, 0.6);
                border: 1px solid #27272a;
                border-radius: 8px;
                padding: 4px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)
        
        self.label = QLabel(label)
        self.label.setStyleSheet("color: #a1a1aa; font-size: 10px;")
        layout.addWidget(self.label)
        
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("color: #f4f4f5; font-size: 13px; font-weight: bold;")
        layout.addWidget(self.value_label)
        
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #71717a; font-size: 9px;")
        self.status_label.hide()
        layout.addWidget(self.status_label)

    def set_value(self, value: str, status: str = "", status_color: str = "#71717a"):
        """Update the metric value and optional status."""
        self.value_label.setText(value)
        if status:
            self.status_label.setText(status)
            self.status_label.setStyleSheet(f"color: {status_color}; font-size: 9px;")
            self.status_label.show()
        else:
            self.status_label.hide()


class PipelineStage(QFrame):
    """A horizontal bar showing pipeline stage progress."""

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)
        
        self.name_label = QLabel(name)
        self.name_label.setMinimumWidth(100)
        self.name_label.setStyleSheet("color: #a1a1aa; font-size: 11px;")
        layout.addWidget(self.name_label)
        
        self.progress = QProgressBar()
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(8)
        self.progress.setStyleSheet("""
            QProgressBar {
                background-color: #0f0f12;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #8b7cf6;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress, stretch=1)
        
        self.status_label = QLabel("-")
        self.status_label.setMinimumWidth(80)
        self.status_label.setAlignment(Qt.AlignRight)
        self.status_label.setStyleSheet("color: #71717a; font-size: 10px;")
        layout.addWidget(self.status_label)

    def set_running(self):
        """Mark stage as running."""
        self.progress.setMaximum(0)  # Indeterminate
        self.status_label.setText("Running...")
        self.status_label.setStyleSheet("color: #8b7cf6; font-size: 10px;")

    def set_complete(self, time_ms: float, count: int = 0, extra: str = ""):
        """Mark stage as complete."""
        self.progress.setMaximum(100)
        self.progress.setValue(100)
        
        if time_ms < 1000:
            time_str = f"{time_ms:.0f}ms"
        else:
            time_str = f"{time_ms/1000:.1f}s"
        
        status = time_str
        if count > 0:
            status = f"{count} items • {time_str}"
        if extra:
            status = f"{status} • {extra}"
            
        self.status_label.setText(status)
        self.status_label.setStyleSheet("color: #4ade80; font-size: 10px;")
        self.progress.setStyleSheet("""
            QProgressBar {
                background-color: #0f0f12;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #4ade80;
                border-radius: 4px;
            }
        """)

    def set_warning(self, message: str):
        """Mark stage with warning."""
        self.progress.setMaximum(100)
        self.progress.setValue(100)
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #fbbf24; font-size: 10px;")
        self.progress.setStyleSheet("""
            QProgressBar {
                background-color: #0f0f12;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #fbbf24;
                border-radius: 4px;
            }
        """)

    def reset(self):
        """Reset to initial state."""
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.status_label.setText("-")
        self.status_label.setStyleSheet("color: #71717a; font-size: 10px;")
        self.progress.setStyleSheet("""
            QProgressBar {
                background-color: #0f0f12;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #8b7cf6;
                border-radius: 4px;
            }
        """)


class DiagnosticsPanel(QWidget):
    """Panel showing RAG pipeline diagnostics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Title
        title = QLabel("Pipeline Diagnostics")
        title.setStyleSheet("color: #f4f4f5; font-size: 12px; font-weight: bold;")
        layout.addWidget(title)

        # Metrics grid
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(8)
        
        self.model_card = MetricCard("Model")
        self.gpu_card = MetricCard("GPU Memory")
        self.context_card = MetricCard("Context")
        self.speed_card = MetricCard("Speed")
        
        metrics_grid.addWidget(self.model_card, 0, 0)
        metrics_grid.addWidget(self.gpu_card, 0, 1)
        metrics_grid.addWidget(self.context_card, 1, 0)
        metrics_grid.addWidget(self.speed_card, 1, 1)
        
        layout.addLayout(metrics_grid)

        # Pipeline stages
        stages_label = QLabel("Pipeline Stages")
        stages_label.setStyleSheet("color: #a1a1aa; font-size: 11px; margin-top: 8px;")
        layout.addWidget(stages_label)

        self.semantic_stage = PipelineStage("Semantic Search")
        self.keyword_stage = PipelineStage("Keyword Search")
        self.fusion_stage = PipelineStage("Fusion (RRF)")
        self.rerank_stage = PipelineStage("Reranking")
        self.generation_stage = PipelineStage("Generation")

        layout.addWidget(self.semantic_stage)
        layout.addWidget(self.keyword_stage)
        layout.addWidget(self.fusion_stage)
        layout.addWidget(self.rerank_stage)
        layout.addWidget(self.generation_stage)

        # Warnings area
        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("""
            color: #fbbf24;
            background-color: rgba(212, 165, 74, 0.1);
            border: 1px solid #fbbf24;
            border-radius: 6px;
            padding: 8px;
            font-size: 11px;
        """)
        self.warning_label.hide()
        layout.addWidget(self.warning_label)

        layout.addStretch()

    def reset(self):
        """Reset all stages to initial state."""
        self.semantic_stage.reset()
        self.keyword_stage.reset()
        self.fusion_stage.reset()
        self.rerank_stage.reset()
        self.generation_stage.reset()
        self.warning_label.hide()

    def set_model_info(self, name: str, quant: str = "", size_gb: float = 0):
        """Set model information."""
        self.model_card.set_value(name, quant if quant else "")
        
    def set_gpu_info(self, used_gb: float, free_gb: float):
        """Set GPU memory info."""
        self.gpu_card.set_value(
            f"{used_gb:.1f} GB",
            f"{free_gb:.0f} GB free"
        )

    def set_context_info(self, used: int, limit: int, truncated: bool = False):
        """Set context window info."""
        if truncated:
            self.context_card.set_value(
                f"{limit:,} / {limit:,}",
                f"Truncated from {used:,}",
                "#fbbf24"
            )
            self.show_warning(f"Prompt truncated: {used:,} tokens → {limit:,} tokens. Some context was lost.")
        else:
            self.context_card.set_value(f"{used:,} / {limit:,}")

    def set_speed_info(self, tokens: int, tok_per_sec: float):
        """Set generation speed info."""
        self.speed_card.set_value(
            f"{tok_per_sec:.1f} tok/s",
            f"{tokens} tokens"
        )

    def show_warning(self, message: str):
        """Show a warning message."""
        self.warning_label.setText(message)
        self.warning_label.show()

    # Slot methods for pipeline signals
    @Slot(str)
    def on_stage_started(self, stage: str):
        """Handle stage started signal."""
        stage_map = {
            "semantic_search": self.semantic_stage,
            "keyword_search": self.keyword_stage,
            "fusion": self.fusion_stage,
            "reranking": self.rerank_stage,
            "filtering": self.rerank_stage,  # Part of reranking
            "generation": self.generation_stage,
        }
        if stage in stage_map:
            stage_map[stage].set_running()

    @Slot(str, dict)
    def on_stage_completed(self, stage: str, stats: dict):
        """Handle stage completed signal."""
        stage_map = {
            "semantic_search": self.semantic_stage,
            "keyword_search": self.keyword_stage,
            "fusion": self.fusion_stage,
            "reranking": self.rerank_stage,
            "filtering": None,  # Merged into reranking display
            "generation": self.generation_stage,
        }
        
        widget = stage_map.get(stage)
        if widget:
            time_ms = stats.get("time_ms", 0)
            count = stats.get("count", 0)
            widget.set_complete(time_ms, count)
