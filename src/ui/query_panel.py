"""Query Panel widget for right top section."""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal, QDate
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QListWidget,
    QListWidgetItem,
    QCheckBox,
    QGroupBox,
)

from src.config.llm_config import load_llm_config, save_llm_config
from src.config_loader import get_settings
from src.llm.model_presets import (
    ModelPreset,
    apply_model_preset,
    get_model_presets,
)
from src.schema import DocumentType
from src.retrieval.vector_store import VectorStore


from src.generation.prompts import SYSTEM_LIT_RAG, SYSTEM_FACT_LOOKUP, SYSTEM_DEEP_ANALYSIS

class QueryPanelWidget(QWidget):
    # ... (existing code) ...

    def _run_query(self):
        # ... (existing code) ...
        
        # Determine system prompt
        # If customized, use the text box. Otherwise, derive from mode (though text box should be synced)
        system_prompt = self.system_prompt_edit.toPlainText().strip()
        if not system_prompt:
             # Fallback if empty
            mode = self.mode_combo.currentData()
            if mode == "fact":
                system_prompt = SYSTEM_FACT_LOOKUP
            elif mode == "analysis":
                system_prompt = SYSTEM_DEEP_ANALYSIS
            else:
                system_prompt = SYSTEM_LIT_RAG

        # Collect parameters
        query_data = {
            "query": query_text,
            "model": self._current_model_identifier(),
            "system_prompt": system_prompt,  # Pass the selected prompt
            "doc_type_filter": self.doc_type_combo.currentData(),
            # ... (rest of dict)
        }
        # ...
        """Initialize query panel."""
        super().__init__(parent)
        self.settings = get_settings()
        self.llm_env_config = load_llm_config()
        self.is_querying = False
        self.sliders = {}  # Initialize sliders dictionary
        self.selected_documents = []  # List of selected document file names
        self.model_presets: list[ModelPreset] = []
        self.model_lookup: dict[str, ModelPreset] = {}
        self._presets_enabled = False
        self._model_combo_refreshing = False

        self._setup_ui()

    def _setup_ui(self):
        """Set up UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title = QLabel("Query")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Query text box
        query_label = QLabel("Enter your question:")
        layout.addWidget(query_label)

        self.query_text = QTextEdit()
        self.query_text.setPlaceholderText("What are the key facts in the witness statements?")
        self.query_text.setMaximumHeight(100)
        layout.addWidget(self.query_text)

        # Mode Selection
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Search Mode:")
        mode_layout.addWidget(mode_label)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Standard", "standard")
        self.mode_combo.addItem("⚡ Fact Lookup (High Precision)", "fact")
        self.mode_combo.addItem("🧠 Deep Analysis (High Recall)", "analysis")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo, stretch=1)

        layout.addLayout(mode_layout)

        # System Prompt Editor (Collapsible)
        self.sys_prompt_check = QCheckBox("Customize System Prompt")
        self.sys_prompt_check.toggled.connect(self._toggle_system_prompt)
        layout.addWidget(self.sys_prompt_check)

        self.system_prompt_edit = QTextEdit()
        self.system_prompt_edit.setPlaceholderText("Enter system prompt...")
        self.system_prompt_edit.setMaximumHeight(150)
        self.system_prompt_edit.setVisible(False)  # Hidden by default
        self.system_prompt_edit.setAcceptRichText(False)
        layout.addWidget(self.system_prompt_edit)

        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        model_layout.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.addItems(self.settings.models.llm.available)
        env_model = self.llm_env_config.model_name
        if env_model and env_model not in self.settings.models.llm.available:
            self.model_combo.addItem(env_model)
        self.model_combo.setCurrentText(env_model or self.settings.ui.default_model)
        self.model_combo.currentIndexChanged.connect(self._handle_model_changed)
        model_layout.addWidget(self.model_combo, stretch=1)

        # Refresh models button
        refresh_models_btn = QPushButton("Refresh")
        refresh_models_btn.setMaximumWidth(30)
        refresh_models_btn.setToolTip("Refresh available models from the active provider")
        refresh_models_btn.clicked.connect(self._refresh_available_models)
        model_layout.addWidget(refresh_models_btn)

        layout.addLayout(model_layout)
        layout.addLayout(model_layout)
        self._load_model_presets()

        # Initialize system prompt with default
        self.system_prompt_edit.setPlainText(SYSTEM_LIT_RAG)

        # Filters
        filter_layout = QHBoxLayout()
        doc_type_label = QLabel("Doc Type:")
        filter_layout.addWidget(doc_type_label)

        self.doc_type_combo = QComboBox()
        self.doc_type_combo.addItem("All Types", None)
        
        # Document types organized by category
        doc_type_categories = {
            "👤 Witness Evidence": [
                "witness_statement",
            ],
            "Pleadings": [
                "pleading",
                "skeleton_argument",
                "court_filing",
            ],
            "Expert/Technical": [
                "expert_report",
                "schedule_of_loss",
                "medical_report",
            ],
            "📚 Legal Sources": [
                "statute",
                "case_law",
                "contract",
            ],
            "📧 Correspondence": [
                "email",
                "letter",
            ],
            "Disclosure": [
                "disclosure",
                "disclosure_list",
            ],
            "Court Forms": [
                "court_form",
                "case_management",
                "chronology",
            ],
            "🏛️ Specialist": [
                "tribunal_document",
                "regulatory_document",
            ],
        }
        
        for category, types in doc_type_categories.items():
            # Add category separator (disabled item)
            self.doc_type_combo.addItem(f"── {category} ──", "__separator__")
            # Make separator non-selectable
            idx = self.doc_type_combo.count() - 1
            self.doc_type_combo.model().item(idx).setEnabled(False)
            
            for doc_type in types:
                display_name = doc_type.replace("_", " ").title()
                self.doc_type_combo.addItem(f"    {display_name}", doc_type)
        
        filter_layout.addWidget(self.doc_type_combo)
        layout.addLayout(filter_layout)

        # Date range filter
        date_filter_layout = QHBoxLayout()
        date_label = QLabel("Date Range:")
        date_filter_layout.addWidget(date_label)
        
        self.date_filter_enabled = QCheckBox()
        self.date_filter_enabled.setToolTip("Enable date filtering")
        date_filter_layout.addWidget(self.date_filter_enabled)
        
        self.date_from = QDateEdit()
        self.date_from.setCalendarPopup(True)
        self.date_from.setDate(QDate.currentDate().addYears(-5))  # Default: 5 years ago
        self.date_from.setDisplayFormat("dd/MM/yyyy")
        self.date_from.setEnabled(False)
        date_filter_layout.addWidget(self.date_from)
        
        date_to_label = QLabel("to")
        date_filter_layout.addWidget(date_to_label)
        
        self.date_to = QDateEdit()
        self.date_to.setCalendarPopup(True)
        self.date_to.setDate(QDate.currentDate())  # Default: today
        self.date_to.setDisplayFormat("dd/MM/yyyy")
        self.date_to.setEnabled(False)
        date_filter_layout.addWidget(self.date_to)
        
        # Connect checkbox to enable/disable date inputs
        self.date_filter_enabled.toggled.connect(self.date_from.setEnabled)
        self.date_filter_enabled.toggled.connect(self.date_to.setEnabled)
        
        layout.addLayout(date_filter_layout)

        # Selected documents display
        doc_select_group = QGroupBox("Documents Selected for Query")
        doc_select_layout = QVBoxLayout()

        self.selected_docs_label = QLabel("No documents selected (select from left panel)")
        self.selected_docs_label.setWordWrap(True)
        self.selected_docs_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                padding: 8px;
                font-size: 9pt;
            }
        """)
        doc_select_layout.addWidget(self.selected_docs_label)

        doc_select_group.setLayout(doc_select_layout)
        layout.addWidget(doc_select_group)

        # Retrieval parameters (sliders)
        self._add_slider(
            layout,
            "Semantic Top N:",
            "semantic_top_n",
            self.settings.retrieval.semantic_top_n,
            10,
            200,
        )
        self._add_slider(
            layout,
            "Keyword Top N:",
            "keyword_top_n",
            self.settings.retrieval.keyword_top_n,
            10,
            200,
        )
        self._add_slider(
            layout,
            "Rerank Top K:",
            "rerank_top_k",
            self.settings.retrieval.rerank_top_k,
            5,
            100,
        )
        self._add_slider(
            layout,
            "Context to LLM:",
            "context_to_llm",
            self.settings.retrieval.context_to_llm,
            1,
            100,  # Increased max to 100 chunks for larger context models
        )
        self._add_slider(
            layout,
            "Confidence Threshold:",
            "confidence_threshold",
            int(self.settings.retrieval.confidence_threshold * 100),
            0,
            100,
            is_percent=True,
        )

        answer_layout = QHBoxLayout()
        answer_label = QLabel("Answer Tokens:")
        answer_layout.addWidget(answer_label)
        self.answer_slider = QSlider(Qt.Horizontal)
        self.answer_slider.setMinimum(200)
        self.answer_slider.setMaximum(2048)
        self.answer_slider.setSingleStep(50)
        default_answer = getattr(self.settings.generation, "synthesis_max_tokens", 800)
        self.answer_slider.setValue(default_answer)
        self.answer_slider.valueChanged.connect(self._on_answer_slider_changed)
        answer_layout.addWidget(self.answer_slider, stretch=1)
        self.answer_value_label = QLabel(f"{default_answer} tok")
        self.answer_value_label.setMinimumWidth(70)
        answer_layout.addWidget(self.answer_value_label)
        layout.addLayout(answer_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Query")
        self.run_button.clicked.connect(self._run_query)
        self.run_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        button_layout.addWidget(self.run_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_query)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)

    def _add_slider(
        self,
        layout: QVBoxLayout,
        label_text: str,
        key: str,
        default_value: int,
        min_value: int,
        max_value: int,
        is_percent: bool = False,
    ):
        """Add a slider control.

        Args:
            layout: Layout to add to
            label_text: Label text
            key: Key for storing slider reference
            default_value: Default slider value
            min_value: Minimum value
            max_value: Maximum value
            is_percent: Whether to display as percentage
        """
        slider_layout = QHBoxLayout()

        label = QLabel(label_text)
        slider_layout.addWidget(label)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(default_value)
        slider_layout.addWidget(slider, stretch=1)

        value_label = QLabel(str(default_value))
        value_label.setMinimumWidth(50)
        slider_layout.addWidget(value_label)

        # Update label when slider changes
        def update_label(value):
            if is_percent:
                value_label.setText(f"{value / 100:.2f}")
            else:
                value_label.setText(str(value))

        slider.valueChanged.connect(update_label)
        layout.addLayout(slider_layout)

        self.sliders[key] = (slider, value_label, is_percent)

    def _on_mode_changed(self, index: int):
        """Handle search mode change."""
        mode = self.mode_combo.currentData()
        
        if mode == "fact":
            # High precision, strict threshold
            self._set_slider("semantic_top_n", 50)
            self._set_slider("rerank_top_k", 15)
            self._set_slider("context_to_llm", 5)
            self._set_slider("confidence_threshold", 65) # 0.65
            self.query_text.setPlaceholderText("State the specific dates and amounts...")
            
        elif mode == "analysis":
            # High recall, broader search
            self._set_slider("semantic_top_n", 100)
            self._set_slider("rerank_top_k", 40)
            self._set_slider("context_to_llm", 20)
            self._set_slider("confidence_threshold", 40) # 0.40
            self.query_text.setPlaceholderText("Analyze the relationship between...")
            
        else: # Standard
            defaults = self.settings.retrieval
            self._set_slider("semantic_top_n", defaults.semantic_top_n)
            self._set_slider("rerank_top_k", defaults.rerank_top_k)
            self._set_slider("context_to_llm", defaults.context_to_llm)
            self._set_slider("confidence_threshold", int(defaults.confidence_threshold * 100))
            self.query_text.setPlaceholderText("What are the key facts...")

        # Update system prompt text area
        if mode == "fact":
            self.system_prompt_edit.setPlainText(SYSTEM_FACT_LOOKUP)
        elif mode == "analysis":
            self.system_prompt_edit.setPlainText(SYSTEM_DEEP_ANALYSIS)
        else:
            self.system_prompt_edit.setPlainText(SYSTEM_LIT_RAG)

    def _toggle_system_prompt(self, checked: bool):
        """Toggle visibility of system prompt editor."""
        self.system_prompt_edit.setVisible(checked)

    def _set_slider(self, key: str, value: int):
        """Helper to set slider value safely."""
        if key in self.sliders:
            slider, _, _ = self.sliders[key]
            slider.setValue(value)

    def _run_query(self):
        """Run query with current parameters."""
        query_text = self.query_text.toPlainText().strip()
        if not query_text:
            return

        # Build date range filter if enabled
        date_range = None
        if self.date_filter_enabled.isChecked():
            from_date = self.date_from.date().toPython()
            to_date = self.date_to.date().toPython()
            date_range = (from_date, to_date)
        
        # Collect parameters
        query_data = {
            "query": query_text,
            "model": self._current_model_identifier(),
            "doc_type_filter": self.doc_type_combo.currentData(),
            "date_range": date_range,  # New: date range filter
            "selected_documents": self.selected_documents if self.selected_documents else None,  # None = all documents
            "semantic_top_n": self.sliders["semantic_top_n"][0].value(),
            "keyword_top_n": self.sliders["keyword_top_n"][0].value(),
            "rerank_top_k": self.sliders["rerank_top_k"][0].value(),
            "context_to_llm": self.sliders["context_to_llm"][0].value(),
            "confidence_threshold": self.sliders["confidence_threshold"][0].value() / 100.0,
            "skip_reranking": self.settings.retrieval.skip_reranking,
            "answer_tokens": self.answer_slider.value(),
            "search_mode": self.mode_combo.currentData(),
        }

        self.is_querying = True
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.query_requested.emit(query_data)

    def _stop_query(self):
        """Stop current query."""
        self.stop_requested.emit()
        self.is_querying = False
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def on_query_finished(self):
        """Called when query finishes."""
        self.is_querying = False
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _on_answer_slider_changed(self, value: int):
        self.answer_value_label.setText(f"{value} tok")

    def update_selected_documents(self, selected_docs: list):
        """Update the display of selected documents.

        Args:
            selected_docs: List of selected document file names
        """
        self.selected_documents = selected_docs

        if not selected_docs:
            self.selected_docs_label.setText("⚠ No documents selected (select from left panel)")
            self.selected_docs_label.setStyleSheet("""
                QLabel {
                    background-color: #fff3cd;
                    border: 1px solid #ffc107;
                    padding: 8px;
                    font-size: 9pt;
                    color: #856404;
                }
            """)
        else:
            count = len(selected_docs)
            # Show first few documents, then "and X more"
            if count <= 5:
                doc_list = "\n• ".join(selected_docs)
                text = f"{count} document(s) selected:\n{doc_list}"
            else:
                first_docs = selected_docs[:5]
                doc_list = "\n• ".join(first_docs)
                text = f"{count} document(s) selected:\n{doc_list}\n...and {count - 5} more"

            self.selected_docs_label.setText(text)
            self.selected_docs_label.setStyleSheet("""
                QLabel {
                    background-color: #d4edda;
                    border: 1px solid #28a745;
                    padding: 8px;
                    font-size: 9pt;
                    color: #155724;
                }
            """)

    def _refresh_available_models(self):
        """Query the current provider for available models and update dropdown."""
        self.llm_env_config = load_llm_config()
        
        # If using llama.cpp, refresh presets
        if self.llm_env_config.provider == "llama_cpp":
            self._load_model_presets(force_refresh=True)
            return

        # If using generic provider (Ollama/LM Studio), fetch models
        try:
            from src.generation.llm_service import LLMService

            llm_service = LLMService()
            models = llm_service.get_available_models()

            if models:
                self._model_combo_refreshing = True
                current_model = self.model_combo.currentText()
                self.model_combo.clear()
                self.model_combo.addItems(models)

                if current_model in models:
                    self.model_combo.setCurrentText(current_model)
                self._model_combo_refreshing = False
                
                QMessageBox.information(self, "Models Refreshed", f"Found {len(models)} models.")

        except Exception as e:
            QMessageBox.warning(self, "Refresh Failed", f"Failed to refresh models: {e}")

    def _load_model_presets(self, force_refresh: bool = False) -> None:
        """Populate the model dropdown with available options."""

        self.llm_env_config = load_llm_config()
        
        # For llama_cpp, use the preset system
        if self.llm_env_config.provider == "llama_cpp":
            presets = get_model_presets(force_refresh=force_refresh)
            if not presets:
                self._presets_enabled = False
                self.model_lookup.clear()
                return

            self._presets_enabled = True
            self.model_presets = presets
            self.model_lookup = {preset.label: preset for preset in presets}

            target_label = self._find_label_for_model(self.llm_env_config.model_name) or presets[0].label

            self._model_combo_refreshing = True
            self.model_combo.blockSignals(True)
            self.model_combo.clear()
            for preset in presets:
                self.model_combo.addItem(preset.label)
            self.model_combo.setCurrentText(target_label)
            self.model_combo.blockSignals(False)
            self._model_combo_refreshing = False
        
        # For Ollama/LM Studio, stick with raw model names
        else:
            self._presets_enabled = False
            self.model_lookup.clear()
            # List is already populated in __init__ or refreshed via button

    def _handle_model_changed(self) -> None:
        """Handle model change."""

        if self._model_combo_refreshing:
            return

        # 1. Handle llama.cpp Preset Switch (Requires Server Restart)
        if self._presets_enabled:
            label = self.model_combo.currentText()
            preset = self.model_lookup.get(label)
            if not preset:
                return

            try:
                apply_model_preset(preset, restart_server=True)
                self.llm_env_config = load_llm_config()
                QMessageBox.information(self, "Model Switched", f"Switched to {label} (Server Restarted)")
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Model Switch Failed",
                    f"Could not apply preset '{label}':\n{exc}",
                )
            return

        # 2. Handle Hot Swap (Ollama/LM Studio)
        new_model = self.model_combo.currentText()
        if not new_model:
            return
            
        try:
            # Just update the active config without restarting anything
            self.llm_env_config.model_name = new_model
            save_llm_config(self.llm_env_config)
            # No message box needed for instant switch, maybe just log status
            print(f"Active model updated to: {new_model}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to update model config: {e}")

    def _find_label_for_model(self, model_name: str | None) -> str | None:
        if not model_name:
            return None
        for preset in self.model_presets:
            if preset.model_name == model_name:
                return preset.label
        return None

    def _current_model_identifier(self) -> str:
        if self._presets_enabled:
            preset = self.model_lookup.get(self.model_combo.currentText())
            return preset.model_name if preset else self.model_combo.currentText()
        return self.model_combo.currentText()
