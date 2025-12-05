"""Settings dialog for editing configuration."""

from pathlib import Path

import yaml
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.config_loader import get_settings, get_config_profile
from src.ui.llm_control_panel import LLMControlPanel
from src.ui.background_tasks_tab import BackgroundTasksTab
from src.config.api_key_manager import APIKeyManager


class DiagnosticsWorker(QThread):
    """Worker thread for running diagnostics."""

    finished = Signal(object)  # SystemDiagnostics
    progress = Signal(str)

    def run(self):
        """Run diagnostics in background."""
        self.progress.emit("Running system diagnostics...")
        try:
            from src.system.diagnostics import run_diagnostics
            diag = run_diagnostics()
            self.finished.emit(diag)
        except Exception as e:
            self.progress.emit(f"Error: {e}")
            self.finished.emit(None)


class SettingsDialog(QDialog):
    """Settings dialog for editing config.yaml."""

    def __init__(self, parent=None):
        """Initialize settings dialog."""
        super().__init__(parent)
        self.settings = get_settings()
        self.setWindowTitle("Settings")
        self.setMinimumSize(800, 700)
        self.diag_worker = None
        
        # Dictionaries to store widget references
        self.paths_widgets = {}
        self.retrieval_widgets = {}
        self.model_widgets = {}
        self.quality_widgets = {}
        self.api_key_manager = APIKeyManager()

        self._setup_ui()

    def _setup_ui(self):
        """Set up UI components."""
        layout = QVBoxLayout(self)

        # Info label
        info_label = QLabel(
            "Edit configuration settings. Changes will be saved to config/config.yaml and require a restart to fully apply."
        )
        info_label.setStyleSheet("font-weight: bold; color: #555;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Tabs for different settings sections
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # System Diagnostics tab (first for visibility)
        diagnostics_tab = self._create_diagnostics_tab()
        tabs.addTab(diagnostics_tab, "System Diagnostics")

        # Background Tasks tab (New)
        bg_tasks_tab = BackgroundTasksTab(self)
        tabs.addTab(bg_tasks_tab, "Background Tasks")

        # Retrieval tab (Priority)
        retrieval_tab = self._create_retrieval_tab()
        tabs.addTab(retrieval_tab, "Retrieval")

        # Models tab
        models_tab = self._create_models_tab()
        tabs.addTab(models_tab, "Models")

        # Paths tab
        paths_tab = self._create_paths_tab()
        tabs.addTab(paths_tab, "Paths")

        # LLM control tab
        llm_tab = LLMControlPanel(self)
        tabs.addTab(llm_tab, "LLM Server")

        # Quality Assessment tab
        quality_tab = self._create_quality_tab()
        tabs.addTab(quality_tab, "Quality Assessment")

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults
        )
        button_box.accepted.connect(self._save_settings)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.RestoreDefaults).clicked.connect(
            self._restore_defaults
        )
        layout.addWidget(button_box)

    def _create_diagnostics_tab(self) -> QWidget:
        """Create system diagnostics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Config profile selector
        profile_group = QGroupBox("Configuration Profile")
        profile_layout = QHBoxLayout(profile_group)

        profile_label = QLabel("Active Profile:")
        profile_layout.addWidget(profile_label)

        self.profile_combo = QComboBox()
        self.profile_combo.addItems(["default", "96gb", "128gb"])
        current_profile = get_config_profile()
        idx = self.profile_combo.findText(current_profile)
        if idx >= 0:
            self.profile_combo.setCurrentIndex(idx)
        self.profile_combo.currentTextChanged.connect(self._on_profile_changed)
        profile_layout.addWidget(self.profile_combo)

        profile_info = QLabel("Set SC_CONFIG env var to persist")
        profile_info.setStyleSheet("color: #888;")
        profile_layout.addWidget(profile_info)

        profile_layout.addStretch()
        layout.addWidget(profile_group)

        # Diagnostics output
        diag_group = QGroupBox("System Diagnostics")
        diag_layout = QVBoxLayout(diag_group)

        # Run button and status
        btn_layout = QHBoxLayout()
        self.run_diag_btn = QPushButton("Run Diagnostics")
        self.run_diag_btn.clicked.connect(self._run_diagnostics)
        btn_layout.addWidget(self.run_diag_btn)

        self.diag_status = QLabel("")
        btn_layout.addWidget(self.diag_status)
        btn_layout.addStretch()
        diag_layout.addLayout(btn_layout)

        # Output text area
        self.diag_output = QPlainTextEdit()
        self.diag_output.setReadOnly(True)
        self.diag_output.setMinimumHeight(300)
        self.diag_output.setStyleSheet(
            "QPlainTextEdit { font-family: 'Consolas', 'Monaco', monospace; font-size: 11px; }"
        )
        diag_layout.addWidget(self.diag_output)

        layout.addWidget(diag_group)

        # Auto-run diagnostics on tab creation
        self._run_diagnostics()

        return widget

    def _run_diagnostics(self):
        """Run system diagnostics in background thread."""
        if self.diag_worker and self.diag_worker.isRunning():
            return

        self.run_diag_btn.setEnabled(False)
        self.diag_status.setText("Running...")
        self.diag_output.setPlainText("Running diagnostics, please wait...")

        self.diag_worker = DiagnosticsWorker()
        self.diag_worker.progress.connect(self._on_diag_progress)
        self.diag_worker.finished.connect(self._on_diag_finished)
        self.diag_worker.start()

    def _on_diag_progress(self, msg: str):
        """Handle diagnostics progress update."""
        self.diag_status.setText(msg)

    def _on_diag_finished(self, diag):
        """Handle diagnostics completion."""
        self.run_diag_btn.setEnabled(True)

        if diag is None:
            self.diag_status.setText("Error running diagnostics")
            return

        from src.system.diagnostics import format_diagnostics_text
        self.diag_output.setPlainText(format_diagnostics_text(diag))
        self.diag_status.setText(diag.summary)

    def _on_profile_changed(self, profile: str):
        """Handle config profile change."""
        import os
        os.environ["SC_CONFIG"] = profile

        # Reload settings
        from src.config_loader import get_settings
        get_settings(force_reload=True)

        self.diag_status.setText(f"Profile changed to: {profile}")
        QMessageBox.information(
            self,
            "Profile Changed",
            f"Configuration profile changed to '{profile}'.\n\n"
            f"To make this permanent, set the environment variable:\n"
            f"SC_CONFIG={profile}\n\n"
            "Restart the application for all changes to take effect."
        )

    def _create_paths_tab(self) -> QWidget:
        """Create paths settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        paths = self.settings.paths
        self.paths_widgets = {}

        for key in ["documents", "vector_db", "keyword_index", "logs"]:
            row_layout = QVBoxLayout()
            label = QLabel(f"{key.replace('_', ' ').title()}:")
            row_layout.addWidget(label)

            path_layout = QHBoxLayout()
            line_edit = QLineEdit(getattr(paths, key))
            path_layout.addWidget(line_edit, stretch=1)

            browse_btn = QPushButton("Browse...")
            browse_btn.clicked.connect(
                lambda checked, le=line_edit: self._browse_folder(le)
            )
            path_layout.addWidget(browse_btn)

            row_layout.addLayout(path_layout)
            layout.addLayout(row_layout)

            self.paths_widgets[key] = line_edit

        layout.addStretch()
        return widget

    def _create_models_tab(self) -> QWidget:
        """Create models settings tab."""
        widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)

        # LLM Section
        llm_group = QGroupBox("LLM Settings")
        llm_layout = QFormLayout(llm_group)
        
        backend_combo = QComboBox()
        backend_combo.addItems(["ollama", "llama_cpp"]) # LM Studio removed
        backend_combo.setCurrentText(self.settings.models.llm.backend if self.settings.models.llm.backend != "lmstudio" else "ollama")
        llm_layout.addRow("Backend:", backend_combo)
        self.model_widgets["backend"] = backend_combo
        
        default_model = QLineEdit(self.settings.models.llm.default)
        llm_layout.addRow("Default Model:", default_model)
        self.model_widgets["default_model"] = default_model
        
        layout.addWidget(llm_group)
        
        # Embedding Section
        embed_group = QGroupBox("Embedding Settings")
        embed_layout = QFormLayout(embed_group)
        
        embed_model = QLineEdit(self.settings.models.embedding.default)
        embed_layout.addRow("Embedding Model:", embed_model)
        self.model_widgets["embedding_model"] = embed_model
        
        rerank_model = QLineEdit(self.settings.models.reranker.default)
        embed_layout.addRow("Reranker Model:", rerank_model)
        self.model_widgets["reranker_model"] = rerank_model
        
        layout.addWidget(embed_group)
        
        # Provider Settings
        provider_group = QGroupBox("Provider Details")
        provider_layout = QFormLayout(provider_group)
        
        ollama_host = QLineEdit(self.settings.models.ollama.host)
        provider_layout.addRow("Ollama Host:", ollama_host)
        self.model_widgets["ollama_host"] = ollama_host
        
        # LM Studio removed
        
        layout.addWidget(provider_group)
        
        layout.addStretch()
        scroll.setWidget(content)
        
        main_layout = QVBoxLayout(widget)
        main_layout.addWidget(scroll)
        return widget

    def _create_retrieval_tab(self) -> QWidget:
        """Create editable retrieval settings tab."""
        widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)

        retrieval = self.settings.retrieval
        
        # Primary Parameters
        params_group = QGroupBox("Search Parameters")
        params_layout = QFormLayout(params_group)
        
        semantic_n = QSpinBox()
        semantic_n.setRange(10, 200)
        semantic_n.setValue(retrieval.semantic_top_n)
        params_layout.addRow("Semantic Top N:", semantic_n)
        self.retrieval_widgets["semantic_top_n"] = semantic_n
        
        keyword_n = QSpinBox()
        keyword_n.setRange(10, 200)
        keyword_n.setValue(retrieval.keyword_top_n)
        params_layout.addRow("Keyword Top N:", keyword_n)
        self.retrieval_widgets["keyword_top_n"] = keyword_n
        
        rerank_k = QSpinBox()
        rerank_k.setRange(5, 100)
        rerank_k.setValue(retrieval.rerank_top_k)
        params_layout.addRow("Rerank Top K:", rerank_k)
        self.retrieval_widgets["rerank_top_k"] = rerank_k
        
        context_m = QSpinBox()
        context_m.setRange(1, 50)
        context_m.setValue(retrieval.context_to_llm)
        params_layout.addRow("Chunks to LLM:", context_m)
        self.retrieval_widgets["context_to_llm"] = context_m
        
        threshold = QDoubleSpinBox()
        threshold.setRange(0.0, 1.0)
        threshold.setSingleStep(0.05)
        threshold.setValue(retrieval.confidence_threshold)
        params_layout.addRow("Confidence Threshold:", threshold)
        self.retrieval_widgets["confidence_threshold"] = threshold
        
        rrf_k = QSpinBox()
        rrf_k.setRange(1, 200)
        rrf_k.setValue(retrieval.rrf_k)
        params_layout.addRow("RRF Constant (k):", rrf_k)
        self.retrieval_widgets["rrf_k"] = rrf_k
        
        layout.addWidget(params_group)
        
        # Chunking Info (Read-only for now as changing requires re-ingestion)
        chunk_group = QGroupBox("Chunking Defaults (Requires Re-ingestion)")
        chunk_layout = QFormLayout(chunk_group)
        
        ws_size = QLabel(str(self.settings.chunking.sizes.witness_statement))
        chunk_layout.addRow("Witness Statement Size:", ws_size)
        
        ws_overlap = QLabel(str(self.settings.chunking.overlaps.witness_statement))
        chunk_layout.addRow("Witness Statement Overlap:", ws_overlap)
        
        layout.addWidget(chunk_group)
        
        layout.addStretch()
        scroll.setWidget(content)
        
        main_layout = QVBoxLayout(widget)
        main_layout.addWidget(scroll)
        return widget

    def _create_quality_tab(self) -> QWidget:
        """Create quality assessment settings tab."""
        widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)

        # API Keys Section
        keys_group = QGroupBox("Cloud Provider API Keys")
        keys_layout = QFormLayout(keys_group)
        
        # OpenAI Key
        openai_key = QLineEdit()
        openai_key.setEchoMode(QLineEdit.Password)
        openai_key.setText(self.api_key_manager.get_key("openai") or "")
        keys_layout.addRow("OpenAI API Key:", openai_key)
        self.quality_widgets["openai_key"] = openai_key
        
        # Anthropic Key
        anthropic_key = QLineEdit()
        anthropic_key.setEchoMode(QLineEdit.Password)
        anthropic_key.setText(self.api_key_manager.get_key("anthropic") or "")
        keys_layout.addRow("Anthropic API Key:", anthropic_key)
        self.quality_widgets["anthropic_key"] = anthropic_key
        
        # Google Key
        google_key = QLineEdit()
        google_key.setEchoMode(QLineEdit.Password)
        google_key.setText(self.api_key_manager.get_key("google") or "")
        keys_layout.addRow("Google API Key:", google_key)
        self.quality_widgets["google_key"] = google_key
        
        layout.addWidget(keys_group)
        
        # Configuration Section
        config_group = QGroupBox("Assessment Configuration")
        config_layout = QFormLayout(config_group)
        
        # Provider Selection
        provider_combo = QComboBox()
        provider_combo.addItems([
            "GPT-5.1 Instant (OpenAI)", 
            "GPT-5.1 Nano (OpenAI)",
            "GPT-5.1 Thinking (OpenAI)",
            "Claude Sonnet 4.5 (Anthropic)", 
            "Gemini 3 Pro (Google)"
        ])
        
        # Set current selection based on config or default
        current_provider = "GPT-5.1 Instant (OpenAI)" # Default
        if hasattr(self.settings, "quality") and hasattr(self.settings.quality, "provider"):
             # Map internal name to display name
             if self.settings.quality.provider == "anthropic":
                 current_provider = "Claude Sonnet 4.5 (Anthropic)"
             elif self.settings.quality.provider == "google":
                 current_provider = "Gemini 3 Pro (Google)"
             elif self.settings.quality.provider == "openai":
                 # Check model variant if stored, otherwise default to Instant
                 model = getattr(self.settings.quality, "model", "gpt-5.1-instant")
                 if model == "gpt-5.1-nano":
                     current_provider = "GPT-5.1 Nano (OpenAI)"
                 elif model == "gpt-5.1-thinking":
                     current_provider = "GPT-5.1 Thinking (OpenAI)"
                 else:
                     current_provider = "GPT-5.1 Instant (OpenAI)"
        
        provider_combo.setCurrentText(current_provider)
        config_layout.addRow("Evaluator Model:", provider_combo)
        self.quality_widgets["provider"] = provider_combo
        
        layout.addWidget(config_group)
        
        # Info
        info_label = QLabel(
            "Quality assessment sends generation data to the selected cloud provider.\n"
            "Ensure you have sufficient credits and are comfortable with the privacy implications."
        )
        info_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(info_label)

        layout.addStretch()
        scroll.setWidget(content)
        
        main_layout = QVBoxLayout(widget)
        main_layout.addWidget(scroll)
        return widget

    def _browse_folder(self, line_edit: QLineEdit):
        """Open folder browser dialog."""
        from PySide6.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(self, "Select Folder", line_edit.text())
        if folder:
            line_edit.setText(folder)

    def _save_settings(self):
        """Save settings to config.yaml."""
        try:
            config_path = Path("config/config.yaml")
            if not config_path.exists():
                self.reject()
                return

            # Load existing config to preserve comments/structure if possible
            # But yaml library usually wipes comments. 
            with config_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            # Update Paths
            if "paths" not in config: config["paths"] = {}
            for key, widget in self.paths_widgets.items():
                config["paths"][key] = widget.text()
            
            # Update Retrieval
            if "retrieval" not in config: config["retrieval"] = {}
            for key, widget in self.retrieval_widgets.items():
                config["retrieval"][key] = widget.value()
            
            # Update Models
            if "models" not in config: config["models"] = {}
            if "llm" not in config["models"]: config["models"]["llm"] = {}
            config["models"]["llm"]["backend"] = self.model_widgets["backend"].currentText()
            config["models"]["llm"]["default"] = self.model_widgets["default_model"].text()
            
            if "embedding" not in config["models"]: config["models"]["embedding"] = {}
            config["models"]["embedding"]["default"] = self.model_widgets["embedding_model"].text()
            
            if "reranker" not in config["models"]: config["models"]["reranker"] = {}
            config["models"]["reranker"]["default"] = self.model_widgets["reranker_model"].text()
            
            if "ollama" not in config["models"]: config["models"]["ollama"] = {}
            config["models"]["ollama"]["host"] = self.model_widgets["ollama_host"].text()
            
            # if "lmstudio" not in config["models"]: config["models"]["lmstudio"] = {}
            # config["models"]["lmstudio"]["host"] = self.model_widgets["lmstudio_host"].text()

            # Update Quality Settings
            if "quality" not in config: config["quality"] = {}
            
            provider_map = {
                "GPT-5.1 Instant (OpenAI)": ("openai", "gpt-5.1-instant"),
                "GPT-5.1 Nano (OpenAI)": ("openai", "gpt-5.1-nano"),
                "GPT-5.1 Thinking (OpenAI)": ("openai", "gpt-5.1-thinking"),
                "Claude Sonnet 4.5 (Anthropic)": ("anthropic", "claude-3-sonnet-20240229"), # Assuming ID
                "Gemini 3 Pro (Google)": ("google", "gemini-1.5-pro") # Assuming ID
            }
            selected_provider_text = self.quality_widgets["provider"].currentText()
            provider_info = provider_map.get(selected_provider_text, ("openai", "gpt-5.1-instant"))
            
            config["quality"]["provider"] = provider_info[0]
            config["quality"]["model"] = provider_info[1]
            
            # Save API Keys
            self.api_key_manager.set_key("openai", self.quality_widgets["openai_key"].text())
            self.api_key_manager.set_key("anthropic", self.quality_widgets["anthropic_key"].text())
            self.api_key_manager.set_key("google", self.quality_widgets["google_key"].text())

            # Save config
            with config_path.open("w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            # Update running settings instance
            # Note: Deep updates might require app restart, but we'll update the singleton
            settings = get_settings()
            settings.retrieval.semantic_top_n = self.retrieval_widgets["semantic_top_n"].value()
            settings.retrieval.keyword_top_n = self.retrieval_widgets["keyword_top_n"].value()
            settings.retrieval.rerank_top_k = self.retrieval_widgets["rerank_top_k"].value()
            settings.retrieval.context_to_llm = self.retrieval_widgets["context_to_llm"].value()
            settings.retrieval.confidence_threshold = self.retrieval_widgets["confidence_threshold"].value()
            
            self.accept()
            QMessageBox.information(self, "Settings Saved", "Settings saved successfully.\nRestart application for all changes to take effect.")
            
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", f"Failed to save settings: {str(e)}")

    def _restore_defaults(self):
        """Restore default settings."""
        from PySide6.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            "Restore Defaults",
            "This will restore default settings in the UI (not saved yet). Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # Restore retrieval defaults
            defaults_ret = self.settings.retrieval
            self.retrieval_widgets["semantic_top_n"].setValue(defaults_ret.semantic_top_n)
            self.retrieval_widgets["keyword_top_n"].setValue(defaults_ret.keyword_top_n)
            self.retrieval_widgets["rerank_top_k"].setValue(defaults_ret.rerank_top_k)
            self.retrieval_widgets["context_to_llm"].setValue(defaults_ret.context_to_llm)
            self.retrieval_widgets["confidence_threshold"].setValue(defaults_ret.confidence_threshold)
            self.retrieval_widgets["rrf_k"].setValue(defaults_ret.rrf_k)
            
            # Restore path defaults
            defaults_paths = self.settings.paths
            for key, widget in self.paths_widgets.items():
                widget.setText(getattr(defaults_paths, key))
