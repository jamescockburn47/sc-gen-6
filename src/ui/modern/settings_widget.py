from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QComboBox, QLineEdit, 
    QLabel, QPushButton, QFrame, QScrollArea, QMessageBox
)
from src.config_loader import get_settings

class SettingsWidget(QWidget):
    """Configuration settings widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = get_settings()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent; border: none;")
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(40, 40, 40, 40)
        content_layout.setSpacing(20)

        # Header
        header = QLabel("Settings")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #f1f5f9; margin-bottom: 10px;")
        content_layout.addWidget(header)

        # LLM Section
        llm_group = self._create_group("LLM Configuration")
        llm_form = QFormLayout(llm_group)
        llm_form.setSpacing(15)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.settings.models.llm.available)
        self.model_combo.setCurrentText(self.settings.models.llm.default)
        self.model_combo.setStyleSheet(self._input_style())
        llm_form.addRow(self._label("Default Model:"), self.model_combo)

        self.temp_input = QLineEdit(str(self.settings.models.llm.temperature))
        self.temp_input.setStyleSheet(self._input_style())
        llm_form.addRow(self._label("Temperature:"), self.temp_input)

        content_layout.addWidget(llm_group)

        # Retrieval Section
        retrieval_group = self._create_group("Retrieval Configuration")
        ret_form = QFormLayout(retrieval_group)
        ret_form.setSpacing(15)

        self.top_k_input = QLineEdit(str(self.settings.retrieval.semantic_top_n))
        self.top_k_input.setStyleSheet(self._input_style())
        ret_form.addRow(self._label("Semantic Top N:"), self.top_k_input)

        self.rerank_k_input = QLineEdit(str(self.settings.retrieval.rerank_top_k))
        self.rerank_k_input.setStyleSheet(self._input_style())
        ret_form.addRow(self._label("Rerank Top K:"), self.rerank_k_input)

        content_layout.addWidget(retrieval_group)

        # Background Tasks Section
        bg_tasks_group = self._create_group("Background Tasks")
        bg_tasks_layout = QVBoxLayout()
        
        # Description
        desc_label = QLabel(
            "Generate case insights from document summaries using the largest available model.\n"
            "Tasks can be run incrementally (only new documents) or fully regenerated."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #a1a1aa; font-size: 12px; margin-bottom: 10px;")
        bg_tasks_layout.addWidget(desc_label)
        
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
        bg_tasks_layout.addWidget(generate_all_btn)
        
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
        bg_tasks_layout.addWidget(update_all_btn)
        
        # Individual task buttons
        individual_label = QLabel("Individual Tasks:")
        individual_label.setStyleSheet("color: #e4e4e7; font-size: 13px; font-weight: bold; margin-top: 15px;")
        bg_tasks_layout.addWidget(individual_label)
        
        # Case Graph button
        graph_btn = QPushButton("📊 Generate Case Graph")
        graph_btn.setStyleSheet(self._task_button_style())
        graph_btn.clicked.connect(lambda: self._run_single_task("case_graph_generation"))
        bg_tasks_layout.addWidget(graph_btn)
        
        # Timeline button
        timeline_btn = QPushButton("📅 Generate Timeline")
        timeline_btn.setStyleSheet(self._task_button_style())
        timeline_btn.clicked.connect(lambda: self._run_single_task("timeline_generation"))
        bg_tasks_layout.addWidget(timeline_btn)
        
        # Case Overview button
        overview_btn = QPushButton("📋 Generate Case Overview")
        overview_btn.setStyleSheet(self._task_button_style())
        overview_btn.clicked.connect(lambda: self._run_single_task("case_overview_generation"))
        bg_tasks_layout.addWidget(overview_btn)
        
        # Document Renaming button
        rename_btn = QPushButton("✏️ Rename Documents")
        rename_btn.setStyleSheet(self._task_button_style())
        rename_btn.clicked.connect(lambda: self._run_single_task("document_renaming"))
        bg_tasks_layout.addWidget(rename_btn)
        
        bg_tasks_group.layout().addLayout(bg_tasks_layout)
        content_layout.addWidget(bg_tasks_group)

        content_layout.addStretch()

        # Save Button
        save_btn = QPushButton("Save Changes")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        save_btn.clicked.connect(self.save_settings)
        content_layout.addWidget(save_btn)

        scroll.setWidget(content)
        layout.addWidget(scroll)
    
    def _task_button_style(self):
        return """
            QPushButton {
                background-color: #27272a;
                color: #f1f5f9;
                border: 1px solid #3f3f46;
                padding: 10px;
                border-radius: 4px;
                font-size: 13px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #3f3f46;
                border-color: #52525b;
            }
        """
    
    def _run_background_tasks(self, incremental: bool = False):
        """Run all background tasks."""
        from src.generation.background_task_manager import BackgroundTaskManager
        from PySide6.QtCore import QThread
        from src.ui.background_task_dialog import BackgroundTaskDialog
        
        # Create and show progress dialog
        dialog = BackgroundTaskDialog(incremental=incremental, parent=self)
        dialog.show()
        
        # Run tasks in background thread
        dialog.start_tasks()
    
    def _run_single_task(self, task_type: str):
        """Run a single background task."""
        from src.generation.background_task_manager import BackgroundTaskManager
        from src.ui.background_task_dialog import BackgroundTaskDialog
        
        # Create and show progress dialog for single task
        dialog = BackgroundTaskDialog(
            incremental=False,
            single_task=task_type,
            parent=self
        )
        dialog.show()
        dialog.start_tasks()

    def _create_group(self, title):
        group = QFrame()
        group.setStyleSheet("""
            QFrame {
                background-color: #18181b;
                border: 1px solid #27272a;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        layout = QVBoxLayout(group)
        label = QLabel(title)
        label.setStyleSheet("font-size: 16px; font-weight: bold; color: #e4e4e7; margin-bottom: 10px;")
        layout.addWidget(label)
        return group

    def _label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #a1a1aa; font-size: 14px;")
        return lbl

    def _input_style(self):
        return """
            background-color: #27272a;
            color: #f1f5f9;
            border: 1px solid #3f3f46;
            padding: 8px;
            border-radius: 4px;
        """

    def save_settings(self):
        # In a real app, we would write back to config.yaml
        # For now, we just update the runtime object
        try:
            self.settings.models.llm.default = self.model_combo.currentText()
            self.settings.models.llm.temperature = float(self.temp_input.text())
            self.settings.retrieval.semantic_top_n = int(self.top_k_input.text())
            self.settings.retrieval.rerank_top_k = int(self.rerank_k_input.text())
            
            QMessageBox.information(self, "Success", "Settings saved successfully (Runtime only).")
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid input values.")
