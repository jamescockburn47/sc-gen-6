"""Background task progress dialog.

Shows progress for background generation tasks (case graph, timeline, overview, etc.)
with real-time updates and cancellation support.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
)

from src.config_loader import get_settings
from src.generation.background_task_manager import BackgroundTaskManager

logger = logging.getLogger(__name__)


class BackgroundTaskWorker(QThread):
    """Worker thread for running background tasks."""
    
    progress = Signal(str, int, int)  # message, current, total
    finished = Signal(dict)  # results
    error = Signal(str)  # error message
    
    def __init__(
        self,
        task_manager: BackgroundTaskManager,
        incremental: bool = False,
        single_task: Optional[str] = None,
    ):
        super().__init__()
        self.task_manager = task_manager
        self.incremental = incremental
        self.single_task = single_task
    
    def run(self):
        """Execute background tasks."""
        try:
            def progress_callback(message: str, current: int, total: int):
                self.progress.emit(message, current, total)
            
            if self.single_task:
                # Run single task
                result = self._run_single_task(progress_callback)
            else:
                # Run all tasks
                result = self.task_manager.run_all_tasks(
                    progress_callback=progress_callback,
                    incremental=self.incremental
                )
            
            self.finished.emit(result)
            
        except Exception as e:
            logger.error(f"Error in background task worker: {e}", exc_info=True)
            self.error.emit(str(e))
    
    def _run_single_task(self, progress_callback):
        """Run a single task."""
        task_type = self.single_task
        
        try:
            if task_type == "case_graph_generation":
                from src.graph.graph_generator import CaseGraphGenerator
                generator = CaseGraphGenerator(settings=self.task_manager.settings)
                return self.task_manager.run_task(
                    task_type,
                    generator.generate_full_graph,
                    progress_callback,
                    self.incremental
                )
            
            elif task_type == "timeline_generation":
                from src.graph.timeline_generator import TimelineGenerator
                generator = TimelineGenerator(settings=self.task_manager.settings)
                return self.task_manager.run_task(
                    task_type,
                    generator.generate_full_timeline,
                    progress_callback,
                    self.incremental
                )
            
            elif task_type == "case_overview_generation":
                from src.generation.case_overview_generator import CaseOverviewGenerator
                generator = CaseOverviewGenerator(settings=self.task_manager.settings)
                return self.task_manager.run_task(
                    task_type,
                    generator.generate_overview,
                    progress_callback,
                    self.incremental
                )
            
            elif task_type == "document_renaming":
                from src.generation.document_renamer import DocumentRenamer
                generator = DocumentRenamer(settings=self.task_manager.settings)
                return self.task_manager.run_task(
                    task_type,
                    generator.rename_all_documents,
                    progress_callback,
                    self.incremental
                )
            
        except Exception as e:
            return {"error": str(e)}


class BackgroundTaskDialog(QDialog):
    """Dialog showing progress of background generation tasks."""
    
    def __init__(
        self,
        incremental: bool = False,
        single_task: Optional[str] = None,
        parent=None
    ):
        super().__init__(parent)
        self.incremental = incremental
        self.single_task = single_task
        self.task_manager = BackgroundTaskManager(settings=get_settings())
        self.worker: Optional[BackgroundTaskWorker] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("Background Tasks")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        mode = "Incremental Update" if self.incremental else "Full Generation"
        if self.single_task:
            title_text = f"{self.single_task.replace('_', ' ').title()}"
        else:
            title_text = f"Running All Background Tasks ({mode})"
        
        title = QLabel(title_text)
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #f1f5f9;")
        layout.addWidget(title)
        
        # Model info
        optimal_model = self.task_manager.get_optimal_model()
        model_label = QLabel(f"Using model: {optimal_model}")
        model_label.setStyleSheet("color: #a1a1aa; font-size: 13px;")
        layout.addWidget(model_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3f3f46;
                border-radius: 4px;
                background-color: #27272a;
                text-align: center;
                color: #f1f5f9;
            }
            QProgressBar::chunk {
                background-color: #8b5cf6;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("color: #e4e4e7; font-size: 13px;")
        layout.addWidget(self.status_label)
        
        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("""
            QTextEdit {
                background-color: #18181b;
                color: #a1a1aa;
                border: 1px solid #27272a;
                border-radius: 4px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.log_output)
        
        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc2626;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #b91c1c;
            }
            QPushButton:disabled {
                background-color: #3f3f46;
                color: #71717a;
            }
        """)
        self.cancel_btn.clicked.connect(self._cancel_tasks)
        layout.addWidget(self.cancel_btn)
    
    def start_tasks(self):
        """Start running background tasks."""
        self.log_output.append("Starting background tasks...")
        
        # Create worker thread
        self.worker = BackgroundTaskWorker(
            self.task_manager,
            self.incremental,
            self.single_task
        )
        
        # Connect signals
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        
        # Start worker
        self.worker.start()
    
    def _on_progress(self, message: str, current: int, total: int):
        """Handle progress updates."""
        self.status_label.setText(message)
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
        self.log_output.append(f"[{current}/{total}] {message}")
    
    def _on_finished(self, results: dict):
        """Handle task completion."""
        self.status_label.setText("✅ All tasks completed!")
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.cancel_btn.setText("Close")
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.clicked.disconnect()
        self.cancel_btn.clicked.connect(self.accept)
        
        # Log results
        self.log_output.append("\n=== Results ===")
        for task_type, result in results.items():
            if isinstance(result, dict) and "error" in result:
                self.log_output.append(f"❌ {task_type}: {result['error']}")
            else:
                self.log_output.append(f"✅ {task_type}: Success")
    
    def _on_error(self, error_message: str):
        """Handle errors."""
        self.status_label.setText(f"❌ Error: {error_message}")
        self.log_output.append(f"\n❌ ERROR: {error_message}")
        self.cancel_btn.setText("Close")
        self.cancel_btn.setEnabled(True)
    
    def _cancel_tasks(self):
        """Cancel running tasks."""
        if self.worker and self.worker.isRunning():
            self.log_output.append("\nCancelling tasks...")
            self.task_manager.cancel()
            self.worker.wait(5000)  # Wait up to 5 seconds
            self.status_label.setText("Cancelled")
            self.cancel_btn.setEnabled(False)
