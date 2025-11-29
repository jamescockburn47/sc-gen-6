from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFrame, QMessageBox
)
from PySide6.QtCore import Qt

class BackgroundTasksTab(QWidget):
    """Tab for managing background tasks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Description
        desc_label = QLabel(
            "Generate case insights from document summaries using the largest available model.\n"
            "Tasks can be run incrementally (only new documents) or fully regenerated."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #555; font-size: 13px; margin-bottom: 10px;")
        layout.addWidget(desc_label)
        
        # Generate All Button
        generate_all_btn = QPushButton("🚀 Generate All (Full)")
        generate_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b5cf6;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #7c3aed;
            }
        """)
        generate_all_btn.setToolTip("Run all background tasks: case graph, timeline, overview, and document renaming")
        generate_all_btn.clicked.connect(lambda: self._run_background_tasks(incremental=False))
        layout.addWidget(generate_all_btn)
        
        # Update All Button (Incremental)
        update_all_btn = QPushButton("⚡ Update All (Incremental)")
        update_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #06b6d4;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0891b2;
            }
        """)
        update_all_btn.setToolTip("Update with only new documents added since last run")
        update_all_btn.clicked.connect(lambda: self._run_background_tasks(incremental=True))
        layout.addWidget(update_all_btn)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # Individual task buttons
        individual_label = QLabel("Individual Tasks:")
        individual_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 5px;")
        layout.addWidget(individual_label)
        
        # Case Graph button
        graph_btn = QPushButton("📊 Generate Case Graph")
        graph_btn.setStyleSheet(self._task_button_style())
        graph_btn.clicked.connect(lambda: self._run_single_task("case_graph_generation"))
        layout.addWidget(graph_btn)
        
        # Timeline button
        timeline_btn = QPushButton("📅 Generate Timeline")
        timeline_btn.setStyleSheet(self._task_button_style())
        timeline_btn.clicked.connect(lambda: self._run_single_task("timeline_generation"))
        layout.addWidget(timeline_btn)
        
        # Case Overview button
        overview_btn = QPushButton("📋 Generate Case Overview")
        overview_btn.setStyleSheet(self._task_button_style())
        overview_btn.clicked.connect(lambda: self._run_single_task("case_overview_generation"))
        layout.addWidget(overview_btn)
        
        # Document Renaming button
        rename_btn = QPushButton("✏️ Rename Documents")
        rename_btn.setStyleSheet(self._task_button_style())
        rename_btn.clicked.connect(lambda: self._run_single_task("document_renaming"))
        layout.addWidget(rename_btn)
        
        layout.addStretch()

    def _task_button_style(self):
        return """
            QPushButton {
                background-color: #f8f9fa;
                color: #333;
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 4px;
                font-size: 13px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #e9ecef;
                border-color: #ccc;
            }
        """
    
    def _run_background_tasks(self, incremental: bool = False):
        """Run all background tasks."""
        try:
            from src.ui.background_task_dialog import BackgroundTaskDialog
            
            # Create and show progress dialog
            # Find the parent dialog or window to attach to
            parent = self.window()
            dialog = BackgroundTaskDialog(incremental=incremental, parent=parent)
            dialog.show()
            
            # Run tasks in background thread
            dialog.start_tasks()
        except ImportError:
            QMessageBox.critical(self, "Error", "Could not load background task dialog.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start tasks: {e}")
    
    def _run_single_task(self, task_type: str):
        """Run a single background task."""
        try:
            from src.ui.background_task_dialog import BackgroundTaskDialog
            
            parent = self.window()
            dialog = BackgroundTaskDialog(
                incremental=False,
                single_task=task_type,
                parent=parent
            )
            dialog.show()
            dialog.start_tasks()
        except ImportError:
            QMessageBox.critical(self, "Error", "Could not load background task dialog.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start task: {e}")
