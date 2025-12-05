"""Compact Query Panel with fixed top inputs and scrollable settings."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QCheckBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QFrame,
    QSlider,
    QScrollArea,
    QSizePolicy
)

from src.config.llm_config import load_llm_config
from src.config_loader import get_settings
from src.llm.model_presets import (
    ModelPreset,
    apply_model_preset,
    get_model_presets,
)


class CollapsibleSection(QWidget):
    """A collapsible section widget."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.is_collapsed = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(0)

        # Header button
        self.toggle_button = QPushButton(f"▶ {title}")
        self.toggle_button.setProperty("styleClass", "secondary")
        self.toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 8px 12px;
                font-weight: 600;
                font-size: 10pt;
            }
        """)
        self.toggle_button.clicked.connect(self.toggle)
        layout.addWidget(self.toggle_button)

        # Content container
        self.content_area = QFrame()
        self.content_area.setStyleSheet("""
            QFrame {
                background-color: transparent;
                border: none;
            }
        """)
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(12, 8, 12, 8)
        self.content_area.setMaximumHeight(0)
        layout.addWidget(self.content_area)

    def toggle(self):
        """Toggle collapsed state."""
        self.is_collapsed = not self.is_collapsed
        arrow = "▼" if not self.is_collapsed else "▶"
        self.toggle_button.setText(f"{arrow} {self.toggle_button.text()[2:]}")

        if self.is_collapsed:
            self.content_area.setMaximumHeight(0)
        else:
            self.content_area.setMaximumHeight(16777215)

    def add_widget(self, widget: QWidget):
        """Add widget to content area."""
        self.content_layout.addWidget(widget)


class CompactQueryPanel(QWidget):
    """Compact query panel with fixed inputs and scrollable advanced settings."""

    query_requested = Signal(dict)
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = get_settings()
        self.llm_env_config = load_llm_config()
        self.is_querying = False
        self.selected_documents = []
        self.param_widgets: dict[str, dict] = {}
        self.model_presets: list[ModelPreset] = []
        self.model_lookup: dict[str, ModelPreset] = {}
        self._presets_enabled = False
        self._model_combo_refreshing = False

        self._setup_ui()

    def _setup_ui(self):
        """Set up UI components."""
        # Main layout for the panel
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # --- Fixed Top Section ---
        # Title
        title = QLabel("Query")
        title.setProperty("styleClass", "subtitle")
        main_layout.addWidget(title)

        # Query input card
        query_section = QFrame()
        query_section.setStyleSheet("""
            QFrame {
                background-color: #1a1a20;
                border: 1px solid #27272a;
                border-radius: 12px;
            }
        """)
        query_layout = QVBoxLayout(query_section)
        query_layout.setContentsMargins(16, 16, 16, 16)
        query_layout.setSpacing(12)

        self.query_text = QTextEdit()
        self.query_text.setPlaceholderText("Ask about your documents...")
        self.query_text.setMinimumHeight(80)
        self.query_text.setMaximumHeight(120)
        self.query_text.setStyleSheet("""
            QTextEdit {
                background-color: #0f0f12;
                border: 1px solid #27272a;
                border-radius: 8px;
                padding: 12px;
                font-size: 11pt;
                color: #f4f4f5;
            }
            QTextEdit:focus {
                border-color: #8b7cf6;
            }
        """)
        query_layout.addWidget(self.query_text)

        main_layout.addWidget(query_section)

        # System Prompt Section (Prominent position)
        prompt_section = QFrame()
        prompt_section.setStyleSheet("""
            QFrame {
                background-color: #1a1a20;
                border: 1px solid #27272a;
                border-radius: 12px;
            }
        """)
        prompt_layout = QVBoxLayout(prompt_section)
        prompt_layout.setContentsMargins(16, 12, 16, 12)
        prompt_layout.setSpacing(8)

        # Header with preset dropdown
        prompt_header = QHBoxLayout()
        prompt_label = QLabel("💬 System Prompt:")
        prompt_label.setProperty("styleClass", "muted")
        prompt_label.setStyleSheet("font-size: 11pt; font-weight: 600;")
        prompt_header.addWidget(prompt_label)
        prompt_header.addStretch()

        # Preset dropdown
        preset_label = QLabel("Preset:")
        preset_label.setProperty("styleClass", "muted")
        prompt_header.addWidget(preset_label)
        
        self.prompt_preset_combo = QComboBox()
        self.prompt_preset_combo.setMaximumWidth(200)
        self.prompt_preset_combo.addItem("Mode Default", None)
        self.prompt_preset_combo.addItem("Document Synthesis", "litigation")
        self.prompt_preset_combo.addItem("Factual Extraction", "factual")
        self.prompt_preset_combo.addItem("Timeline Builder", "timeline")
        self.prompt_preset_combo.addItem("Witness Comparison", "witness")
        self.prompt_preset_combo.addItem("Gap Analysis", "gaps")
        self.prompt_preset_combo.addItem("Contradiction Finder", "contradictions")
        self.prompt_preset_combo.addItem("Custom", "custom")
        self.prompt_preset_combo.currentIndexChanged.connect(self._on_prompt_preset_changed)
        prompt_header.addWidget(self.prompt_preset_combo)
        
        prompt_layout.addLayout(prompt_header)

        # Prompt text editor
        self.system_prompt_text = QTextEdit()
        self.system_prompt_text.setPlaceholderText(
            "System prompt will auto-populate based on mode or preset..."
        )
        self.system_prompt_text.setMinimumHeight(80)
        self.system_prompt_text.setMaximumHeight(120)
        self.system_prompt_text.setStyleSheet("""
            QTextEdit {
                background-color: #0f0f12;
                border: 1px solid #27272a;
                border-radius: 8px;
                padding: 10px;
                font-size: 10pt;
                color: #d4d4d8;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                line-height: 1.4;
            }
            QTextEdit:focus {
                border-color: #8b7cf6;
            }
        """)
        prompt_layout.addWidget(self.system_prompt_text)
        
        main_layout.addWidget(prompt_section)

        # Quick model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        model_label.setProperty("styleClass", "muted")
        model_layout.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.addItems(self.settings.models.llm.available)
        self.model_combo.setCurrentText(self.settings.ui.default_model)
        self.model_combo.currentIndexChanged.connect(self._handle_model_changed)
        model_layout.addWidget(self.model_combo, stretch=1)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setProperty("styleClass", "icon")
        refresh_btn.setToolTip("Refresh models")
        refresh_btn.setMaximumWidth(32)
        refresh_btn.clicked.connect(self._refresh_models)
        model_layout.addWidget(refresh_btn)

        main_layout.addLayout(model_layout)
        self._load_model_presets()

        # Search mode selection
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Mode:")
        mode_label.setProperty("styleClass", "muted")
        mode_layout.addWidget(mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Standard", "Fact Lookup", "Deep Analysis"])
        self.mode_combo.setToolTip(
            "Standard: Balanced retrieval and generation\n"
            "Fact Lookup: Quick, narrow search with concise answers\n"
            "Deep Analysis: Wider search, more thorough analysis"
        )
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.mode_combo.currentTextChanged.connect(self._update_system_prompt_from_mode)
        mode_layout.addWidget(self.mode_combo, stretch=1)

        main_layout.addLayout(mode_layout)
        
        # Initialize system prompt with current mode default
        self._update_system_prompt_from_mode()

        # Action buttons
        button_layout = QHBoxLayout()

        self.submit_button = QPushButton("🚀 Generate Answer")
        self.submit_button.clicked.connect(self._on_submit)
        button_layout.addWidget(self.submit_button)

        # Quality Assessment Toggle
        self.assess_quality_checkbox = QCheckBox("Assess Quality")
        self.assess_quality_checkbox.setToolTip("Send result to cloud AI for quality assessment")
        self.assess_quality_checkbox.setProperty("styleClass", "muted")
        button_layout.addWidget(self.assess_quality_checkbox)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self._on_stop)
        button_layout.addWidget(self.stop_button)

        main_layout.addLayout(button_layout)

        # --- Scrollable Bottom Section (Collapsibles) ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Allow scroll area to shrink
        scroll_area.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)

        # Collapsible: Selected Documents
        self.doc_section = CollapsibleSection("Selected Documents")
        self.selected_docs_label = QLabel("All documents")
        self.selected_docs_label.setProperty("styleClass", "muted")
        self.selected_docs_label.setWordWrap(True)
        self.doc_section.add_widget(self.selected_docs_label)
        content_layout.addWidget(self.doc_section)

        # Collapsible: Advanced Settings
        self.advanced_section = CollapsibleSection("Advanced Settings")

        # Doc type filter
        doc_type_layout = QHBoxLayout()
        doc_type_label = QLabel("Doc Type:")
        doc_type_label.setProperty("styleClass", "muted")
        doc_type_layout.addWidget(doc_type_label)

        self.doc_type_combo = QComboBox()
        self.doc_type_combo.addItem("All Types", None)
        
        # Full list of document types
        all_doc_types = [
            # Witness evidence
            "witness_statement",
            # Pleadings
            "pleading", "skeleton_argument", "court_filing",
            # Expert/Technical
            "expert_report", "schedule_of_loss", "medical_report",
            # Legal sources
            "statute", "case_law", "contract",
            # Correspondence
            "email", "letter",
            # Disclosure
            "disclosure", "disclosure_list",
            # Court forms
            "court_form", "case_management", "chronology",
            # Specialist
            "tribunal_document", "regulatory_document",
        ]
        
        for doc_type in all_doc_types:
            self.doc_type_combo.addItem(doc_type.replace("_", " ").title(), doc_type)
        
        doc_type_layout.addWidget(self.doc_type_combo)

        self.advanced_section.content_layout.addLayout(doc_type_layout)

        # Reranker toggle
        self.skip_rerank_checkbox = QCheckBox("Skip reranking (faster, lower quality)")
        self.skip_rerank_checkbox.setChecked(self.settings.retrieval.skip_reranking)
        self.skip_rerank_checkbox.setProperty("styleClass", "muted")
        self.skip_rerank_checkbox.toggled.connect(self._on_skip_rerank_toggled)
        self.advanced_section.content_layout.addWidget(self.skip_rerank_checkbox)

        # Graph-enhanced search section (optional features, OFF by default)
        graph_label = QLabel("🔗 Graph Features (Optional):")
        graph_label.setProperty("styleClass", "muted")
        self.advanced_section.content_layout.addWidget(graph_label)

        self.query_expansion_checkbox = QCheckBox("Expand query with entity aliases")
        self.query_expansion_checkbox.setToolTip(
            "Automatically include known aliases when searching.\n"
            "E.g., 'John Smith' also finds 'Mr Smith', 'JS', 'the Claimant'"
        )
        self.query_expansion_checkbox.setChecked(self.settings.retrieval.use_query_expansion)
        self.query_expansion_checkbox.setProperty("styleClass", "muted")
        self.advanced_section.content_layout.addWidget(self.query_expansion_checkbox)

        self.graph_context_checkbox = QCheckBox("Include entity context in LLM")
        self.graph_context_checkbox.setToolTip(
            "Add brief entity/timeline context to help the LLM.\n"
            "Uses ~500 extra tokens of context budget."
        )
        self.graph_context_checkbox.setChecked(self.settings.retrieval.use_graph_context)
        self.graph_context_checkbox.setProperty("styleClass", "muted")
        self.advanced_section.content_layout.addWidget(self.graph_context_checkbox)

        # Date filter mode
        date_mode_layout = QHBoxLayout()
        date_mode_label = QLabel("Date Filter:")
        date_mode_label.setProperty("styleClass", "muted")
        date_mode_layout.addWidget(date_mode_label)
        
        self.date_mode_combo = QComboBox()
        self.date_mode_combo.addItem("Document Date", "document")
        self.date_mode_combo.addItem("Dates Mentioned", "mentioned")
        self.date_mode_combo.addItem("Both", "both")
        self.date_mode_combo.setToolTip(
            "Document Date: When the document was created\n"
            "Dates Mentioned: Dates appearing in the text\n"
            "Both: Match either"
        )
        # Set current value from settings
        mode_index = self.date_mode_combo.findData(self.settings.retrieval.date_filter_mode)
        if mode_index >= 0:
            self.date_mode_combo.setCurrentIndex(mode_index)
        date_mode_layout.addWidget(self.date_mode_combo)
        self.advanced_section.content_layout.addLayout(date_mode_layout)

        # Retrieval parameters in a compact grid
        params_header = QHBoxLayout()
        params_label = QLabel("Retrieval Parameters:")
        params_label.setProperty("styleClass", "muted")
        params_header.addWidget(params_label)
        params_header.addStretch()
        reset_btn = QPushButton("Reset")
        reset_btn.setProperty("styleClass", "text")
        reset_btn.setToolTip("Reset retrieval parameters to defaults")
        reset_btn.clicked.connect(self._reset_parameters)
        params_header.addWidget(reset_btn)
        self.advanced_section.content_layout.addLayout(params_header)

        for cfg in self._parameter_configs():
            self._build_param_row(cfg)

        # Answer length slider
        answer_layout = QHBoxLayout()
        answer_label = QLabel("Answer Tokens:")
        answer_label.setProperty("styleClass", "muted")
        answer_label.setMinimumWidth(150)
        answer_layout.addWidget(answer_label)

        self.answer_slider = QSlider(Qt.Horizontal)
        self.answer_slider.setRange(200, 1600)
        self.answer_slider.setSingleStep(50)
        default_answer = getattr(self.settings.generation, "synthesis_max_tokens", 800)
        self.answer_slider.setValue(default_answer)
        self.answer_slider.valueChanged.connect(self._on_answer_slider_changed)
        answer_layout.addWidget(self.answer_slider, stretch=1)

        self.answer_value_label = QLabel(f"{default_answer} tok")
        self.answer_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.answer_value_label.setMinimumWidth(70)
        answer_layout.addWidget(self.answer_value_label)

        self.advanced_section.content_layout.addLayout(answer_layout)

        content_layout.addWidget(self.advanced_section)
        content_layout.addStretch()

        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

    def _refresh_models(self):
        """Refresh available models from LM Studio or local presets."""
        self.llm_env_config = load_llm_config()
        if self.llm_env_config.provider == "llama_cpp":
            self._load_model_presets(force_refresh=True)
            return

        try:
            from src.generation.llm_service import LLMService

            llm_service = LLMService()
            models = llm_service.get_available_models()

            if models:
                current_model = self.model_combo.currentText()
                self.model_combo.clear()
                self.model_combo.addItems(models)

                if current_model in models:
                    self.model_combo.setCurrentText(current_model)

        except Exception as e:
            print(f"Failed to refresh models: {e}")

    def _on_mode_changed(self, mode: str):
        """Handle search mode change - adjust parameters accordingly."""
        if mode == "Fact Lookup":
            # Quick, narrow search with concise answers
            self._set_param_value("semantic_top_n", 20)
            self._set_param_value("keyword_top_n", 20)
            self._set_param_value("rerank_top_k", 10)
            self._set_param_value("context_to_llm", 3)
            self._set_param_value("confidence_threshold", 0.75)
            self.answer_slider.setValue(400)
            self.query_text.setPlaceholderText("Quick fact question (e.g., 'What date was the contract signed?')")
        elif mode == "Deep Analysis":
            # Wider search, more thorough analysis
            self._set_param_value("semantic_top_n", 80)
            self._set_param_value("keyword_top_n", 80)
            self._set_param_value("rerank_top_k", 30)
            self._set_param_value("context_to_llm", 12)
            self._set_param_value("confidence_threshold", 0.50)
            self.answer_slider.setValue(1200)
            self.query_text.setPlaceholderText("Complex analysis question (e.g., 'What evidence supports the fraud claim?')")
        else:  # Standard
            # Balanced defaults from config
            self._reset_parameters()
            self.query_text.setPlaceholderText("What are the key facts?")

    def _set_param_value(self, param_id: str, value: int | float):
        """Set a parameter slider value."""
        if param_id in self.param_widgets:
            widget_info = self.param_widgets[param_id]
            slider = widget_info.get("slider")
            label = widget_info.get("value_label")
            if slider:
                slider.setValue(int(value * 100) if isinstance(value, float) and value <= 1 else int(value))
            if label:
                if isinstance(value, float) and value <= 1:
                    label.setText(f"{value:.2f}")
                else:
                    label.setText(str(int(value)))

    def _get_system_prompt_for_mode(self) -> str | None:
        """Get the appropriate system prompt based on selected mode."""
        mode = self.mode_combo.currentText()
        if mode == "Fact Lookup":
            return (
                "You are a litigation support assistant retrieving specific facts from documents. "
                "Provide ONLY the requested information from retrieved documents. "
                "No preamble, no conclusion. Just the facts with [Doc Name, Page X] citations. "
                "Use complete, grammatical sentences. Answer in 1-3 sentences if possible. "
                "If not in documents, state: 'Not found in documents.'"
            )
        elif mode == "Deep Analysis":
            return (
                "You are a litigation support assistant conducting comprehensive document analysis. "
                "Analyze ONLY retrieved sources. Synthesize patterns and connections. "
                "Cite EVERY factual claim with [Doc Name, Page X]. Use complete sentences. "
                "Structure with ## headings and bullet points. Present conclusions clearly."
            )
        return None  # Standard mode uses default prompt

    def _get_prompt_presets(self) -> dict[str, str]:
        """RAG-specific litigation query presets."""
        return {
            "litigation": (
                "Answer ONLY using the retrieved documents. Synthesize information from multiple sources. "
                "Cite every fact with [Doc Name, Page X]. If information is not in the retrieved documents, "
                "state 'Not found in retrieved documents.' Do NOT use external knowledge.\n\n"
                "OUTPUT STRUCTURE:\n"
                "- Use clear section headings (## Section Name)\n"
                "- Write in complete, grammatically correct sentences\n"
                "- Format lists with bullet points or numbers\n"
                "- Use Markdown tables for comparisons:\n"
                "  | Document | Page | Finding |\n"
                "  |----------|------|---------|\n"
                "  | Doc A    | 5    | Fact 1  |"
            ),
            "factual": (
                "Extract ONLY the specific facts requested from the retrieved documents. "
                "Provide direct quotes where relevant. Cite each fact with [Doc Name, Page X]. "
                "Write in complete sentences. Be extremely concise—answer in 1-3 sentences if possible. "
                "If the fact is not in the documents, say 'Not found in documents.'"
            ),
            "timeline": (
                "Build a chronological timeline using ONLY events mentioned in the retrieved documents.\n\n"
                "FORMAT AS MARKDOWN TABLE:\n"
                "| Date | Event | Source |\n"
                "|------|-------|--------|\n"
                "| YYYY-MM-DD | Description | [Doc, Page X] |\n\n"
                "Flag any date conflicts between documents. Note timeline gaps. "
                "Do NOT infer dates not explicitly stated in the documents. Use complete sentences in Event column."
            ),
            "witness": (
                "Compare witness accounts using ONLY the retrieved statements/testimony.\n\n"
                "FORMAT AS TABLE:\n"
                "| Witness | Statement | Source | Type |\n"
                "|---------|-----------|--------|------|\n"
                "| Name    | Quote     | [Doc, Page X, Line Y] | Corroboration/Contradiction |\n\n"
                "Identify: (1) Corroborations, (2) Contradictions, (3) Unique claims. "
                "Present findings objectively without speculation. Use complete sentences."
            ),
            "gaps": (
                "Identify information gaps in the retrieved documents regarding the query. "
                "List: (1) What the documents DO say (with citations), (2) What is MISSING or unclear. "
                "Suggest specific documents or discovery items needed. "
                "Do NOT speculate about missing information."
            ),
            "contradictions": (
                "Find contradictions or inconsistencies across the retrieved documents. "
                "For each contradiction: cite both sources with [Doc A, Page X] vs [Doc B, Page Y]. "
                "Quote conflicting statements directly. Categorize as factual conflicts, timeline issues, or claim disputes. "
                "Do NOT interpret—only report what the documents state."
            ),
        }

    def _update_system_prompt_from_mode(self, mode: str = None):
        """Update system prompt text when mode changes."""
        # Only update if user hasn't selected a preset manually
        if self.prompt_preset_combo.currentData() is not None:
            return
        
        prompt = self._get_system_prompt_for_mode()
        if prompt:
            self.system_prompt_text.setPlainText(prompt)
        else:
            self.system_prompt_text.setPlainText("")
    
    def _on_prompt_preset_changed(self):
        """Handle prompt preset selection."""
        preset_key = self.prompt_preset_combo.currentData()
        
        if preset_key is None:
            # Mode default - update from current mode
            self._update_system_prompt_from_mode()
        elif preset_key == "custom":
            # Custom - clear prompt for user to type
            if not self.system_prompt_text.toPlainText().strip():
                self.system_prompt_text.setPlainText("")
                self.system_prompt_text.setFocus()
        else:
            # Legal preset
            presets = self._get_prompt_presets()
            if preset_key in presets:
                self.system_prompt_text.setPlainText(presets[preset_key])

    def _on_submit(self):
        """Handle submit button click."""
        query = self.query_text.toPlainText().strip()
        if not query:
            return

        self.is_querying = True
        self.submit_button.setVisible(False)
        self.stop_button.setVisible(True)

        query_data = {
            "query": query,
            "model": self._current_model_identifier(),
            "mode": self.mode_combo.currentText(),
            "system_prompt": self.system_prompt_text.toPlainText().strip() or self._get_system_prompt_for_mode(),
            "doc_type_filter": self.doc_type_combo.currentData(),
            "selected_documents": self.selected_documents,
            "skip_reranking": self.skip_rerank_checkbox.isChecked(),
            "answer_tokens": self.answer_slider.value(),
            # Graph-enhanced search (user-optional)
            "use_query_expansion": self.query_expansion_checkbox.isChecked(),
            "use_graph_context": self.graph_context_checkbox.isChecked(),
            "date_filter_mode": self.date_mode_combo.currentData(),
            "assess_quality": self.assess_quality_checkbox.isChecked(),
            **self._collect_param_values(),
        }

        self.query_requested.emit(query_data)

    def _on_stop(self):
        """Handle stop button click."""
        self.stop_requested.emit()
        self.query_finished()

    def query_finished(self):
        """Called when query finishes."""
        self.is_querying = False
        self.submit_button.setVisible(True)
        self.stop_button.setVisible(False)

    def update_selected_documents(self, selected_documents: list[str]):
        """Update the list of selected documents.

        Args:
            selected_documents: List of selected document file names
        """
        self.selected_documents = selected_documents

        if not selected_documents:
            self.selected_docs_label.setText("All documents")
        elif len(selected_documents) == 1:
            self.selected_docs_label.setText(f"{selected_documents[0]}")
        else:
            docs_text = ", ".join(selected_documents[:3])
            if len(selected_documents) > 3:
                docs_text += f" (+{len(selected_documents) - 3} more)"
            self.selected_docs_label.setText(f"{docs_text}")

    # ------------------------------------------------------------------#
    # Retrieval parameter helpers
    # ------------------------------------------------------------------#
    def _parameter_configs(self) -> list[dict]:
        cfg = self.settings.retrieval
        return [
            {
                "id": "semantic_top_n",
                "label": "Semantic Top N",
                "min": 10,
                "max": 200,
                "default": cfg.semantic_top_n,
                "step": 5,
            },
            {
                "id": "keyword_top_n",
                "label": "Keyword Top N",
                "min": 10,
                "max": 200,
                "default": cfg.keyword_top_n,
                "step": 5,
            },
            {
                "id": "rerank_top_k",
                "label": "Rerank Top K",
                "min": 5,
                "max": 120,
                "default": cfg.rerank_top_k,
                "step": 5,
            },
            {
                "id": "context_to_llm",
                "label": "Context to LLM",
                "min": 1,
                "max": 150,
                "default": cfg.context_to_llm,
                "step": 1,
            },
            {
                "id": "confidence_threshold",
                "label": "Confidence Threshold",
                "min": 0,
                "max": 100,
                "default": int(cfg.confidence_threshold * 100),
                "step": 1,
                "scale": 0.01,
                "formatter": lambda v: f"{v / 100:.2f}",
            },
        ]

    def _build_param_row(self, cfg: dict) -> None:
        param_layout = QHBoxLayout()

        title = QLabel(f"{cfg['label']}:")
        title.setProperty("styleClass", "muted")
        title.setMinimumWidth(150)
        param_layout.addWidget(title)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(cfg["min"], cfg["max"])
        slider.setSingleStep(cfg.get("step", 1))
        slider.setTracking(True)
        default_value = max(cfg["min"], min(cfg["max"], cfg["default"]))
        slider.setValue(default_value)
        slider.valueChanged.connect(
            lambda value, pid=cfg["id"]: self._on_param_changed(pid, value)
        )
        param_layout.addWidget(slider, stretch=1)

        value_label = QLabel(self._format_param_display(cfg["id"], default_value, cfg))
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value_label.setMinimumWidth(60)
        param_layout.addWidget(value_label)

        self.param_widgets[cfg["id"]] = {
            "slider": slider,
            "label": value_label,
            "config": cfg,
        }

        self.advanced_section.content_layout.addLayout(param_layout)

    def _on_param_changed(self, param_id: str, slider_value: int) -> None:
        widget = self.param_widgets.get(param_id)
        if not widget:
            return
        widget["label"].setText(self._format_param_display(param_id, slider_value))

    def _format_param_display(
        self, param_id: str, slider_value: int, cfg: dict | None = None
    ) -> str:
        config = cfg
        if config is None:
            widget = self.param_widgets.get(param_id)
            config = widget["config"] if widget else None
        if config and "formatter" in config:
            return config["formatter"](slider_value)
        return str(slider_value)

    def _collect_param_values(self) -> dict:
        values: dict[str, float] = {}
        for param_id, widget in self.param_widgets.items():
            cfg = widget["config"]
            slider_val = widget["slider"].value()
            scale = cfg.get("scale", 1.0)
            value = slider_val * scale
            if scale == 1.0:
                values[param_id] = int(value)
            else:
                values[param_id] = round(value, 4)
        return values

    def _reset_parameters(self) -> None:
        for widget in self.param_widgets.values():
            cfg = widget["config"]
            widget["slider"].setValue(cfg["default"])

        self.skip_rerank_checkbox.setChecked(self.settings.retrieval.skip_reranking)
        default_answer = getattr(self.settings.generation, "synthesis_max_tokens", 800)
        self.answer_slider.setValue(default_answer)

    def _on_answer_slider_changed(self, value: int) -> None:
        self.answer_value_label.setText(f"{value} tok")

    def _on_skip_rerank_toggled(self, checked: bool) -> None:
        """Enable/disable rerank controls when skipping."""
        rerank_widget = self.param_widgets.get("rerank_top_k")
        if rerank_widget:
            rerank_widget["slider"].setEnabled(not checked)
            rerank_widget["label"].setEnabled(not checked)

    def _load_model_presets(self, force_refresh: bool = False) -> None:
        """Refresh the model dropdown with detected presets."""

        self.llm_env_config = load_llm_config()
        if self.llm_env_config.provider != "llama_cpp":
            self._presets_enabled = False
            self.model_lookup.clear()
            return

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

    def _handle_model_changed(self) -> None:
        if self._model_combo_refreshing or not self._presets_enabled:
            return

        label = self.model_combo.currentText()
        preset = self.model_lookup.get(label)
        if not preset:
            return

        try:
            apply_model_preset(preset, restart_server=True)
            self.llm_env_config = load_llm_config()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Model Switch Failed",
                f"Could not apply preset '{label}':\n{exc}",
            )

    def _find_label_for_model(self, model_name: str | None) -> str | None:
        if not model_name:
            return None
        for preset in self.model_presets:
            if preset.model_name == model_name:
                return preset.label
        return None

    def _current_model_identifier(self) -> str:
        if not self._presets_enabled:
            return self.model_combo.currentText()
        preset = self.model_lookup.get(self.model_combo.currentText())
        return preset.model_name if preset else self.model_combo.currentText()
