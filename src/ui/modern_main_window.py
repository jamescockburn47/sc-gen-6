"""Modern main window with redesigned dashboard layout."""

from __future__ import annotations

import shlex
import threading
from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QAction, QIcon, QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QMainWindow,
    QSplitter,
    QWidget,
    QFrame,
    QPushButton,
    QLabel,
    QStackedWidget,
    QSizePolicy,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
)

from src.config_loader import get_settings
from src.ui.document_manager import DocumentManagerWidget
from src.ui.compact_query_panel import CompactQueryPanel
from src.ui.enhanced_output_panel import EnhancedOutputPanel
from src.ui.results_detail_panel import ResultsDetailPanel
from src.ui.styles import get_modern_stylesheet
from src.ui.settings_dialog import SettingsDialog
from src.ui.technical_specs_panel import TechnicalSpecsPanel
from src.ui.components.source_preview import SourcePreviewDialog
from src.ui.components.diagnostics_panel import DiagnosticsPanel
from src.ui.components.response_popup import ResponsePopup
from src.ui.document_stats_widget import DocumentStatsWidget
from src.ui.entity_panel import EntityPanel
from src.ui.timeline_panel import TimelinePanel
from src.ui.graph_visualization import GraphVisualizationWidget
from src.ui.summary_panel import SummaryPanel
from src.ui.case_overview_widget import CaseOverviewWidget
from src.generation.llm_service import LLMService, GenerationProgressSignals
from src.generation.chunk_batcher import ChunkBatchGenerator
from src.retrieval.hybrid_retriever import HybridRetriever, RetrieverProgressSignals
from src.generation.citation import verify_citations
from src.guardrails.service import GuardrailService
from src.guardrails.service import GuardrailService
from src.ui.components.llm_status_bar import LLMStatusBar
from src.ui.components.suggestions_viewer import SuggestionsViewer
from src.assessment.assessment_collector import AssessmentCollector
from src.assessment.assessment_worker import AssessmentWorker


import time
from urllib.parse import urlparse

from src.config.llm_config import load_llm_config
from src.config.runtime_store import load_runtime_state
from src.llm.constants import LLAMA_SERVER_LOG_PATH
from src.llm.server_manager import manager as llama_manager
from src.system.gpu_monitor import get_gpu_stats, get_ollama_gpu_info
from src.chat.history import ChatHistory, ChatEntry


import requests
from time import sleep

class ServerReadinessWorker(QThread):
    """Worker to wait for LLM server to be ready."""
    ready = Signal()
    failed = Signal(str)
    
    def __init__(self, url: str, timeout: int = 60):
        super().__init__()
        self.url = url
        self.timeout = timeout
        self._stop_event = threading.Event()

    def run(self):
        start_time = time.time()
        # Try /health first, then /v1/models
        endpoints = ["/health", "/v1/models"]
        
        while not self._stop_event.is_set():
            if time.time() - start_time > self.timeout:
                self.failed.emit(f"Server startup timed out after {self.timeout}s")
                return

            for endpoint in endpoints:
                try:
                    target = f"{self.url.rstrip('/')}{endpoint}"
                    response = requests.get(target, timeout=2)
                    if response.status_code == 200:
                        self.ready.emit()
                        return
                except requests.RequestException:
                    pass
            
            sleep(1)

    def stop(self):
        self._stop_event.set()



class QueryWorker(QThread):
    """Worker thread for running queries."""

    token_received = Signal(str)
    finished = Signal(str, dict, list, dict)
    error = Signal(str)
    status_update = Signal(str, str)  # status text, color
    chunks_ready = Signal(list)  # New signal for instant results

    def __init__(
        self,
        query_data: dict,
        retriever: HybridRetriever,
        llm_service: LLMService,
        chunk_batcher: ChunkBatchGenerator | None = None,
    ):
        super().__init__()
        self.query_data = query_data
        self.retriever = retriever
        self.llm_service = llm_service
        self.chunk_batcher = chunk_batcher
        self.guardrails = GuardrailService()
        self.stop_event = threading.Event()
        self.retrieved_chunks = []

    def run(self):
        """Run query in background."""
        try:
            # 1. Input Guardrail
            input_check = self.guardrails.validate_input(self.query_data["query"])
            if not input_check.passed and input_check.severity == "block":
                self.error.emit(f"Input Blocked: {input_check.reason}")
                return
            if not input_check.passed and input_check.severity == "warning":
                self.token_received.emit(f"Warning: {input_check.reason}\n\n")

            # Retrieval phase
            self.status_update.emit("Retrieving...", "#8b7cf6")

            # Safety check: limit retrieval parameters to prevent OOM
            MAX_SAFE_TOP_N = 100  # Reasonable max for safety
            semantic_top_n = min(self.query_data.get("semantic_top_n", 30), MAX_SAFE_TOP_N)
            keyword_top_n = min(self.query_data.get("keyword_top_n", 30), MAX_SAFE_TOP_N)
            rerank_top_k = min(self.query_data.get("rerank_top_k", 40), MAX_SAFE_TOP_N)

            chunks = self.retriever.retrieve(
                query=self.query_data["query"],
                semantic_top_n=semantic_top_n,
                keyword_top_n=keyword_top_n,
                rerank_top_k=rerank_top_k,
                context_to_llm=self.query_data.get("context_to_llm", 10),
                confidence_threshold=self.query_data.get("confidence_threshold", 0.30),
                doc_type_filter=self.query_data.get("doc_type_filter"),
                selected_documents=self.query_data.get("selected_documents"),
                cancel_event=self.stop_event,
                skip_reranking=self.query_data.get("skip_reranking"),
            )

            if not chunks:
                self.error.emit("No relevant chunks found. Try rephrasing your query.")
                return

            self.retrieved_chunks = chunks
            
            # Emit chunks immediately for "Instant Results"
            self.chunks_ready.emit(chunks)

            # 2. Retrieval Guardrail (Confidence Check)
            # Only apply if we strictly enforce it (optional config, currently implicit)
            # For now, just warn if low confidence but don't block unless 0 valid
            
            # Generation phase
            plan = self.chunk_batcher.describe_plan(len(chunks)) if self.chunk_batcher else {"enabled": False}
            use_batching = bool(plan.get("enabled"))
            answer_tokens = self.query_data.get("answer_tokens")
            if use_batching:
                batch_count = plan.get("batch_count")
                batch_size = plan.get("batch_size")
                last_batch = plan.get("last_batch_size")
                covered = plan.get("covered_chunks")
                plan_msg = (
                    f"Parallel generation: {batch_count} batches (size≈{batch_size}, "
                    f"final batch {last_batch}, covering {covered}/{len(chunks)} chunks)"
                )
                self.status_update.emit(
                    plan_msg,
                    "#4ade80",
                )
            else:
                self.status_update.emit("Generating...", "#4ade80")

            response_parts = []
            generation_start = time.time()
            stream_token_count = [0]

            def token_callback(token: str):
                if not self.stop_event.is_set():
                    response_parts.append(token)
                    self.token_received.emit(token)
                    stream_token_count[0] += 1

            if use_batching and self.chunk_batcher:
                response, batch_stats = self.chunk_batcher.generate(
                    query=self.query_data["query"],
                    chunks=chunks,
                    model=self.query_data.get("model"),
                    token_callback=token_callback,
                    cancel_event=self.stop_event,
                    max_output_tokens=answer_tokens,
                    system_prompt=self.query_data.get("system_prompt"),
                )
                generation_stats = batch_stats.copy()
                if generation_stats.get("token_count"):
                    stream_token_count[0] = generation_stats["token_count"]  # approximate
            else:
                response = self.llm_service.generate_with_context(
                    query=self.query_data["query"],
                    chunks=chunks,
                    system_prompt=self.query_data.get("system_prompt"),
                    model=self.query_data.get("model"),
                    stream=True,
                    callback=token_callback,
                    cancel_event=self.stop_event,
                    max_tokens=answer_tokens,
                )
                generation_stats = (
                    self.llm_service.last_generation_stats.copy()
                    if getattr(self.llm_service, "last_generation_stats", None)
                    else {}
                )

            if self.stop_event.is_set():
                self.error.emit("Query stopped by user")
                return

            # Verify citations
            verification_result = verify_citations(response, chunks)

            verification_dict = asdict(verification_result)
            # Add computed fields for UI
            verification_dict['valid_citations'] = len(verification_result.citations)
            verification_dict['total_citations'] = len(verification_result.citations) + len(verification_result.missing)

            if not generation_stats:
                generation_stats = (
                    self.llm_service.last_generation_stats.copy()
                    if getattr(self.llm_service, "last_generation_stats", None)
                    else {}
                )
            total_duration_ms = generation_stats.get("duration_ms")
            if not total_duration_ms:
                total_duration_ms = (time.time() - generation_start) * 1000
            tokens_per_sec = generation_stats.get("tokens_per_sec")
            if tokens_per_sec is None and total_duration_ms > 0:
                tokens_per_sec = stream_token_count[0] / (total_duration_ms / 1000)

            metrics = {
                "chunk_count": generation_stats.get("chunk_count", len(chunks)),
                "token_count": generation_stats.get("token_count", stream_token_count[0]),
                "duration_ms": total_duration_ms,
                "tokens_per_sec": tokens_per_sec,
                "prompt_tokens": generation_stats.get("prompt_tokens"),
                "prompt_chars": generation_stats.get("prompt_chars"),
                "prompt_build_ms": generation_stats.get("prompt_build_ms"),
                "batch_count": generation_stats.get("batch_count"),
                "synthesis_reasoning": generation_stats.get("synthesis_reasoning"),
                "synthesis_full": generation_stats.get("synthesis_full"),
            }

            self.status_update.emit("Complete", "#4ade80")
            self.finished.emit(response, verification_dict, chunks, metrics)

        except MemoryError:
            self.error.emit(
                "Out of memory! Try:\n"
                "• Reduce retrieval parameters (N, K values)\n"
                "• Select fewer documents\n"
                "• Close other applications\n"
                "• Enable 'Skip Reranking' for faster queries"
            )
        except Exception as e:
            if self.stop_event.is_set():
                self.error.emit("Query stopped by user")
            else:
                error_msg = str(e)
                # Provide more helpful error messages
                if "CUDA" in error_msg or "memory" in error_msg.lower():
                    self.error.emit(
                        f"Memory error: {error_msg}\n\n"
                        "Try reducing retrieval parameters or selecting fewer documents."
                    )
                elif "timeout" in error_msg.lower():
                    self.error.emit(
                        f"Timeout error: {error_msg}\n\n"
                        "The LLM server may be overloaded. Try again or restart the server."
                    )
                else:
                    self.error.emit(f"Error: {error_msg}")

    def stop(self):
        """Stop the worker."""
        self.stop_event.set()


class ModernMainWindow(QMainWindow):
    """Modern main window with Dashboard layout."""

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.worker = None
        self.readiness_worker = None
        self.auto_started_llama = False
        self.server_ready = False  # Track server readiness
        self._last_logged_batch_progress = (-1, -1)
        self.preview_dialog = None
        self.chat_history = ChatHistory()

        # Progress signals
        self.retriever_signals = RetrieverProgressSignals()
        self.generation_signals = GenerationProgressSignals()

        # Initialize services with progress hooks
        self.retriever = HybridRetriever(progress_signals=self.retriever_signals)
        self.llm_service = LLMService(progress_signals=self.generation_signals)
        self.retriever = HybridRetriever(progress_signals=self.retriever_signals)
        self.llm_service = LLMService(progress_signals=self.generation_signals)
        self.chunk_batcher = ChunkBatchGenerator(self.llm_service)
        self.assessment_collector = AssessmentCollector()
        self._last_query_data = {}

        self._setup_window()
        self._setup_ui()
        self._connect_signals()
        self._wire_progress_signals()
        self._setup_llama_console_monitor()
        self._setup_gpu_monitor()
        self._auto_start_llama_if_needed()

    def _setup_window(self):
        """Set up window properties."""
        # Windows-only deployment (Vulkan optimized)
        self.setWindowTitle("SC Gen 6 - Litigation Support RAG")
        self.resize(1600, 900)
        self.setMinimumSize(1200, 800)

        # Apply modern stylesheet
        self.setStyleSheet(get_modern_stylesheet())

    def _setup_ui(self):
        """Set up Dashboard UI."""
        # Central container
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)  # Changed to VBoxLayout to add status bar
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # LLM Status Bar at top
        self.llm_status_bar = LLMStatusBar(self)
        main_layout.addWidget(self.llm_status_bar)
        
        # Main content area (horizontal: sidebar + content)
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # === 1. Sidebar (Left) ===
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(220)
        self.sidebar.setStyleSheet("""
            QFrame#sidebar {
                background-color: #0f0f12;
                border-right: 1px solid #27272a;
            }
        """)
        
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(16, 24, 16, 24)
        sidebar_layout.setSpacing(12)

        # Title / Logo
        title_label = QLabel("SC GEN 6")
        title_label.setProperty("styleClass", "title")
        title_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(title_label)
        
        subtitle_label = QLabel("LITIGATION RAG")
        subtitle_label.setProperty("styleClass", "section")
        subtitle_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(subtitle_label)
        
        sidebar_layout.addSpacing(20)

        # Navigation Buttons - Clean text-based
        self.btn_chat = QPushButton("Query")
        self.btn_chat.setCheckable(True)
        self.btn_chat.setChecked(True)
        self.btn_chat.setProperty("styleClass", "nav")
        self.btn_chat.clicked.connect(lambda: self._switch_view(0))
        
        self.btn_docs = QPushButton("Documents")
        self.btn_docs.setCheckable(True)
        self.btn_docs.setProperty("styleClass", "nav")
        self.btn_docs.clicked.connect(lambda: self._switch_view(1))
        
        self.btn_graph = QPushButton("Case Graph")
        self.btn_graph.setCheckable(True)
        self.btn_graph.setProperty("styleClass", "nav")
        self.btn_graph.clicked.connect(lambda: self._switch_view(2))
        
        self.btn_timeline = QPushButton("Timeline")
        self.btn_timeline.setCheckable(True)
        self.btn_timeline.setProperty("styleClass", "nav")
        self.btn_timeline.clicked.connect(lambda: self._switch_view(3))
        
        self.btn_summaries = QPushButton("Summaries")
        self.btn_summaries.setCheckable(True)
        self.btn_summaries.setProperty("styleClass", "nav")
        self.btn_summaries.clicked.connect(lambda: self._switch_view(4))
        
        self.btn_overview = QPushButton("Case Overview")
        self.btn_overview.setCheckable(True)
        self.btn_overview.setProperty("styleClass", "nav")
        self.btn_overview.clicked.connect(lambda: self._switch_view(5))
        
        self.btn_specs = QPushButton("System")
        self.btn_specs.setCheckable(True)
        self.btn_specs.setProperty("styleClass", "nav")
        self.btn_specs.clicked.connect(lambda: self._switch_view(6))
        
        self.btn_performance = QPushButton("Performance")
        self.btn_performance.setCheckable(True)
        self.btn_performance.setProperty("styleClass", "nav")
        self.btn_performance.clicked.connect(lambda: self._switch_view(7))
        
        self.btn_quality = QPushButton("Quality")
        self.btn_quality.setCheckable(True)
        self.btn_quality.setProperty("styleClass", "nav")
        self.btn_quality.clicked.connect(lambda: self._switch_view(8))
        
        self.btn_settings = QPushButton("Settings")
        self.btn_settings.setProperty("styleClass", "nav")
        self.btn_settings.clicked.connect(self._show_settings)

        sidebar_layout.addWidget(self.btn_chat)
        sidebar_layout.addWidget(self.btn_docs)
        sidebar_layout.addWidget(self.btn_graph)
        sidebar_layout.addWidget(self.btn_timeline)
        sidebar_layout.addWidget(self.btn_summaries)
        sidebar_layout.addWidget(self.btn_overview)
        sidebar_layout.addWidget(self.btn_specs)
        sidebar_layout.addWidget(self.btn_performance)
        sidebar_layout.addWidget(self.btn_quality)
        sidebar_layout.addWidget(self.btn_settings)
        
        sidebar_layout.addSpacing(16)
        
        # Chat History Section
        history_header = QLabel("Recent Queries")
        history_header.setStyleSheet("color: #71717a; font-size: 9pt; font-weight: 600; letter-spacing: 0.5px;")
        sidebar_layout.addWidget(history_header)
        
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(200)
        self.history_list.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                border: none;
                outline: none;
            }
            QListWidget::item {
                background-color: transparent;
                color: #a1a1aa;
                padding: 8px 4px;
                border-radius: 4px;
                font-size: 9pt;
            }
            QListWidget::item:hover {
                background-color: #1a1a20;
                color: #f4f4f5;
            }
            QListWidget::item:selected {
                background-color: #1e1e24;
                color: #8b7cf6;
            }
        """)
        self.history_list.itemClicked.connect(self._on_history_item_clicked)
        sidebar_layout.addWidget(self.history_list)
        
        # Load existing history
        self._refresh_history_list()
        
        sidebar_layout.addStretch()
        
        # System Stats (GPU/RAM)
        self.sys_stats_label = QLabel("System: Initializing...")
        self.sys_stats_label.setProperty("styleClass", "muted")
        self.sys_stats_label.setWordWrap(True)
        self.sys_stats_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(self.sys_stats_label)

        # === 2. Main Content Area (Center) ===
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(1)
        
        # Stacked Widget for Views (Chat vs Docs)
        self.view_stack = QStackedWidget()
        
        # -- View 0: Chat --
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(0)
        
        # Chat Output (Top)
        self.output_panel = EnhancedOutputPanel(self)
        
        # Query Input (Bottom)
        self.query_panel = CompactQueryPanel(self)
        
        chat_layout.addWidget(self.output_panel, 1)  # Stretch
        chat_layout.addWidget(self.query_panel, 0)   # Fixed
        
        self.view_stack.addWidget(chat_widget)
        
        # -- View 1: Documents --
        docs_container = QWidget()
        docs_layout = QHBoxLayout(docs_container)
        docs_layout.setContentsMargins(0, 0, 0, 0)
        docs_layout.setSpacing(0)
        
        # Document manager on left
        self.document_manager = DocumentManagerWidget(self)
        docs_layout.addWidget(self.document_manager, stretch=2)
        
        # Document stats on right
        self.document_stats = DocumentStatsWidget(self)
        self.document_stats.setMaximumWidth(350)
        docs_layout.addWidget(self.document_stats, stretch=1)
        
        self.view_stack.addWidget(docs_container)

        # -- View 2: Case Graph --
        graph_container = QWidget()
        graph_layout = QHBoxLayout(graph_container)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(0)
        
        # Graph visualization (main area)
        self.graph_viz = GraphVisualizationWidget(self)
        graph_layout.addWidget(self.graph_viz, stretch=2)
        
        # Entity panel (sidebar)
        self.entity_panel = EntityPanel(self)
        self.entity_panel.setMaximumWidth(400)
        graph_layout.addWidget(self.entity_panel, stretch=1)
        
        self.view_stack.addWidget(graph_container)

        # -- View 3: Timeline --
        self.timeline_panel = TimelinePanel(self)
        self.view_stack.addWidget(self.timeline_panel)

        # -- View 4: Summaries --
        self.summary_panel = SummaryPanel(self)
        self.summary_panel.summaries_generated.connect(self._on_summaries_generated)
        self.view_stack.addWidget(self.summary_panel)

        # -- View 5: Case Overview --
        self.case_overview_widget = CaseOverviewWidget(self)
        self.view_stack.addWidget(self.case_overview_widget)

        # -- View 6: Technical Specs --
        self.specs_panel = TechnicalSpecsPanel(self)
        self.view_stack.addWidget(self.specs_panel)

        # -- View 7: Performance Analytics --
        from src.ui.performance_dashboard import PerformanceDashboard
        self.performance_dashboard = PerformanceDashboard(self)
        self.performance_dashboard = PerformanceDashboard(self)
        self.view_stack.addWidget(self.performance_dashboard)

        # -- View 8: Quality Suggestions --
        self.suggestions_viewer = SuggestionsViewer(self)
        self.view_stack.addWidget(self.suggestions_viewer)

        # === 3. Details Panel (Right) ===
        self.detail_panel = ResultsDetailPanel(self)
        self.detail_panel.setMinimumWidth(300)
        
        # Assemble Splitter
        self.main_splitter.addWidget(self.view_stack)
        self.main_splitter.addWidget(self.detail_panel)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setCollapsible(1, True)

        # Final Assembly - add sidebar and splitter to content layout
        content_layout.addWidget(self.sidebar)
        content_layout.addWidget(self.main_splitter)
        
        # Add content widget to main layout (below status bar)
        main_layout.addWidget(content_widget)
        
        # Link document manager selection to query panel
        self.document_manager.selection_changed.connect(
            self.query_panel.update_selected_documents
        )
        
        # Link entity panel to graph visualization
        self.entity_panel.entity_selected.connect(self.graph_viz.focus_on_node)
        
        # Link timeline events to graph
        self.timeline_panel.event_selected.connect(self.graph_viz.focus_on_node)

    def _switch_view(self, index: int):
        """Switch between Chat, Documents, Graph, Timeline, and Specs views."""
        self.view_stack.setCurrentIndex(index)
        
        # Update button states
        nav_buttons = [
            (self.btn_chat, 0),
            (self.btn_docs, 1),
            (self.btn_graph, 2),
            (self.btn_timeline, 3),
            (self.btn_summaries, 4),
            (self.btn_overview, 5),
            (self.btn_specs, 6),
            (self.btn_specs, 6),
            (self.btn_performance, 7),
            (self.btn_quality, 8),
        ]
        
        for btn, btn_index in nav_buttons:
            btn.setChecked(index == btn_index)
            btn.setProperty("styleClass", "" if index == btn_index else "secondary")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        
        # Refresh data when switching to certain views
        if index == 1:  # Documents
            self.document_stats.refresh_stats()
        elif index == 2:  # Graph
            self.graph_viz.refresh_graph()
            self.entity_panel.refresh_entities()
        elif index == 3:  # Timeline
            self.timeline_panel.refresh_timeline()
        elif index == 4:  # Summaries
            self.summary_panel._refresh_stats()
        elif index == 5:  # Case Overview
            self.case_overview_widget._load_overview()
        elif index == 7:  # Performance
            self.performance_dashboard.refresh_insights()
        elif index == 8:  # Quality
            self.suggestions_viewer.refresh_data()

    def _on_summaries_generated(self):
        """Handle completion of summary generation."""
        # Refresh document stats to show updated summary counts
        self.document_stats.refresh_stats()
        self.output_panel.append_text("\nDocument summaries generated successfully.\n")

    def _connect_signals(self):
        """Connect signals between widgets."""
        # Query actions
        self.query_panel.query_requested.connect(self._handle_query)
        self.query_panel.stop_requested.connect(self._handle_stop)

    def _wire_progress_signals(self):
        """Bridge retriever/LLM progress into the UI."""
        progress_widget = self.detail_panel.progress_widget()
        diag = self.detail_panel.diagnostics_panel

        if self.retriever_signals:
            self.retriever_signals.stage_started.connect(progress_widget.on_stage_started)
            self.retriever_signals.stage_completed.connect(progress_widget.on_stage_completed)
            self.retriever_signals.semantic_progress.connect(progress_widget.on_semantic_progress)
            self.retriever_signals.keyword_progress.connect(progress_widget.on_keyword_progress)
            self.retriever_signals.fusion_complete.connect(progress_widget.on_fusion_complete)
            self.retriever_signals.rerank_progress.connect(progress_widget.on_rerank_progress)
            self.retriever_signals.filtering_complete.connect(progress_widget.on_filtering_complete)
            self.retriever_signals.retrieval_complete.connect(progress_widget.on_retrieval_complete)

            self.retriever_signals.stage_started.connect(self._log_stage_started)
            self.retriever_signals.stage_completed.connect(self._log_stage_completed)
            
            # Wire to diagnostics panel
            self.retriever_signals.stage_started.connect(diag.on_stage_started)
            self.retriever_signals.stage_completed.connect(diag.on_stage_completed)

        if self.generation_signals:
            self.generation_signals.generation_started.connect(progress_widget.on_generation_started)
            self.generation_signals.generation_progress.connect(progress_widget.on_generation_progress)
            self.generation_signals.generation_completed.connect(progress_widget.on_generation_completed)
            self.generation_signals.stage_changed.connect(progress_widget.on_generation_stage_changed)
            self.generation_signals.batch_progress.connect(progress_widget.on_generation_batch_progress)
            self.generation_signals.stage_changed.connect(self._log_generation_stage)
            self.generation_signals.batch_progress.connect(self._log_generation_batch_progress)
            
            # Wire generation to diagnostics
            self.generation_signals.generation_started.connect(
                lambda model: diag.set_model_info(model.split(":")[0] if ":" in model else model)
            )
            self.generation_signals.stage_changed.connect(
                lambda stage: diag.generation_stage.set_running() if "generation" in stage.lower() else None
            )

    def _show_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self)
        dialog.exec()

    def _handle_query(self, query_data: dict):
        """Handle query request."""
        self._switch_view(0) # Ensure we are in chat view
        
        # Check server readiness if we auto-started it
        if self.auto_started_llama and not self.server_ready:
            self.output_panel.clear_output()
            self.output_panel.show_progress(True)
            self.output_panel.set_status("Waiting for Server...", "#fbbf24")
            self.detail_panel.append_log("Query queued: waiting for llama.cpp server to be ready...")
            
            # Queue the query to run when ready
            self._queued_query = query_data
            return

        self._last_logged_batch_progress = (-1, -1)
        # Store query for history and assessment
        self._current_query = query_data.get("query", "")
        self._last_query_data = query_data
        
        # Clear previous output
        
        # Clear previous output
        self.output_panel.clear_output()
        self.output_panel.show_progress(True)
        self.output_panel.set_status("Starting...", "#8b7cf6")
        self.detail_panel.clear_all()
        self.detail_panel.append_log("Starting query...")
        self.detail_panel.start_progress_monitor()

        # Create and start worker
        self.worker = QueryWorker(query_data, self.retriever, self.llm_service, self.chunk_batcher)
        self.worker.token_received.connect(self.output_panel.append_text)
        self.worker.status_update.connect(self._handle_worker_status)
        self.worker.finished.connect(self._handle_query_finished)
        self.worker.error.connect(self._handle_query_error)
        self.worker.chunks_ready.connect(self._show_source_preview)
        self.worker.start()
        
        # Show LLM status bar
        model = query_data.get("model") or self.settings.models.llm.default
        self.llm_status_bar.start_generation(model, "Query Generation")

    def _show_source_preview(self, chunks: list):
        """Show instant preview of found sources."""
        if not chunks:
            return
            
        # Close existing preview if any
        if self.preview_dialog:
            try:
                self.preview_dialog.close()
            except Exception:
                pass
        
        self.preview_dialog = SourcePreviewDialog(self, chunks)
        self.preview_dialog.show()
        
        # Update diagnostics panel
        diag = self.detail_panel.diagnostics_panel
        diag.rerank_stage.set_complete(0, len(chunks), "sources found")

    def _show_response_popup(self, response: str, chunks: list[dict], verification: dict, metrics: dict):
        """Show the large response popup for clarity."""
        # Store as instance variable to prevent garbage collection (non-modal dialogs need persistent reference)
        self._response_popup = ResponsePopup(self)
        self._response_popup.set_response(response)
        self._response_popup.set_sources(chunks)
        self._response_popup.set_metrics(
            chunks=len(chunks),
            valid_citations=verification.get("valid_citations", 0),
            total_citations=verification.get("total_citations", 0),
            prompt_tokens=int(metrics.get("prompt_tokens", 0) or 0),
            output_tokens=int(metrics.get("token_count", 0) or 0),
            tok_per_sec=float(metrics.get("tokens_per_sec", 0) or 0)
        )
        self._response_popup.show()

    def _handle_stop(self):
        """Handle stop request."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.output_panel.show_progress(False)
            self.output_panel.set_status("Stopped", "#fbbf24")
            self.detail_panel.append_log("Stop requested by user.")
            self.query_panel.query_finished()
            if self.preview_dialog:
                self.preview_dialog.close()

    def _handle_query_finished(self, response: str, verification: dict, chunks: list[dict], metrics: dict):
        """Handle query completion."""
        self.output_panel.show_progress(False)
        self.output_panel.set_status("Complete", "#4ade80")
        
        # Update status bar with final metrics
        tokens = metrics.get("token_count", 0) or 0
        tokens_per_sec = metrics.get("tokens_per_sec", 0) or 0
        self.llm_status_bar.update_progress(int(tokens), float(tokens_per_sec))
        self.llm_status_bar.finish(success=True)
        
        if self.preview_dialog:
            self.preview_dialog.close()

        # Trigger Quality Assessment if requested
        if self._last_query_data.get("assess_quality"):
            self._start_quality_assessment(response, chunks, metrics, verification)

    def _start_quality_assessment(self, response: str, chunks: list, metrics: dict, verification: dict):
        """Start the background quality assessment."""
        self.llm_status_bar.set_status("Running Quality Assessment...", "#8b7cf6")
        
        # Collect payload
        payload = self.assessment_collector.collect(
            query=self._last_query_data.get("query", ""),
            retrieved_chunks=chunks,
            generated_answer=response,
            model_used=self._last_query_data.get("model", "unknown"),
            system_prompt=self._last_query_data.get("system_prompt", ""),
            generation_config=self._last_query_data,
            diagnostics={**metrics, **verification}
        )
        
        # Start worker
        self.assessment_worker = AssessmentWorker(payload)
        self.assessment_worker.finished.connect(self._handle_assessment_finished)
        self.assessment_worker.error.connect(self._handle_assessment_error)
        self.assessment_worker.start()
        
    def _handle_assessment_finished(self, result):
        """Handle successful assessment."""
        self.llm_status_bar.set_status(f"Assessment Complete: {result.overall_rating}/10", "#4ade80")
        self.detail_panel.append_log(f"Quality Assessment: {result.overall_rating}/10")
        # TODO: Show a toast or notification, or update a specific UI element
        
    def _handle_assessment_error(self, error_msg):
        """Handle assessment error."""
        self.llm_status_bar.set_status("Assessment Failed", "#ef4444")
        self.detail_panel.append_log(f"Assessment Error: {error_msg}")
        
        # Ensure the final response is displayed (overwriting any streaming artifacts if needed)
        # This fixes the "no output" issue if streaming didn't catch the final synthesis
        self.output_panel.set_text(response)
        
        self.query_panel.query_finished()
        self.detail_panel.set_chunks(chunks)
        
        # Update diagnostics panel with final metrics
        diag = self.detail_panel.diagnostics_panel
        diag.set_speed_info(
            tokens=int(metrics.get("token_count", 0) or 0),
            tok_per_sec=float(metrics.get("tokens_per_sec", 0) or 0)
        )
        
        # Check for context truncation (prompt > 4096 but we sent more)
        prompt_tokens = metrics.get("prompt_tokens", 0) or 0
        if prompt_tokens > 4096:
            # We increased context to 16K, so warn if it's still very large
            if prompt_tokens > 16000:
                diag.set_context_info(prompt_tokens, 16384, truncated=True)
            else:
                diag.set_context_info(prompt_tokens, 16384, truncated=False)
        
        # Store response data for popup
        self._last_response = response
        self._last_chunks = chunks
        self._last_verification = verification
        self._last_metrics = metrics
        
        # Show response popup for clarity
        self._show_response_popup(response, chunks, verification, metrics)

        # Save to chat history
        self._save_to_history(response, chunks, metrics)

        # Build analytics text
        analytics = []

        if verification.get("total_citations", 0) > 0:
            valid = verification.get("valid_citations", 0)
            total = verification.get("total_citations", 0)
            analytics.append(f"Citations: {valid}/{total} valid")

        if verification.get("missing_citations"):
            missing = ", ".join(verification["missing_citations"])
            analytics.append(f"Missing references: {missing}")

        if verification.get("all_below_threshold"):
            analytics.append("Low confidence - answer may not be reliable")

        if analytics:
            analytics_text = "\n".join(analytics)
        else:
            analytics_text = ""

        metrics_lines = []
        if metrics.get("chunk_count") is not None:
            metrics_lines.append(f"Chunks: {metrics.get('chunk_count')}")
        if metrics.get("prompt_tokens") is not None:
            build_ms = metrics.get("prompt_build_ms")
            build_text = f" ({build_ms/1000:.2f}s)" if build_ms else ""
            metrics_lines.append(f"Prompt: {int(metrics['prompt_tokens']):,} tokens{build_text}")
        if metrics.get("token_count") is not None:
            metrics_lines.append(f"Output: {int(metrics.get('token_count'))} tokens")
        if metrics.get("tokens_per_sec") is not None:
            metrics_lines.append(f"Speed: {metrics.get('tokens_per_sec'):.1f} tok/s")
        if metrics.get("batch_count"):
            metrics_lines.append(f"Batches: {int(metrics.get('batch_count'))}")

        combined = "\n".join(filter(None, [analytics_text, "\n".join(metrics_lines)]))
        if combined:
            self.output_panel.set_analytics(combined)

        # Show synthesis reasoning if available (for debugging)
        synthesis_reasoning = metrics.get("synthesis_reasoning")
        if synthesis_reasoning:
            self.output_panel.set_reasoning(synthesis_reasoning)

        # Store full synthesis output and show view button
        # NOTE: Disabled - EnhancedOutputPanel doesn't have set_full_synthesis method
        # synthesis_full = metrics.get("synthesis_full")
        # if synthesis_full:
        #     self.output_panel.set_full_synthesis(synthesis_full)

        tokens = metrics.get("token_count", 0) or 0
        speed = metrics.get("tokens_per_sec", 0) or 0
        prompt_tokens = metrics.get("prompt_tokens")
        prompt_segment = f" • {int(prompt_tokens):,} prompt tok" if prompt_tokens else ""
        batch_segment = f" • {int(metrics.get('batch_count'))} batches" if metrics.get("batch_count") else ""
        self.detail_panel.append_log(
            f"Query complete • {len(chunks)} chunks • "
            f"{verification.get('valid_citations', 0)}/{verification.get('total_citations', 0)} citations valid"
            f"{prompt_segment}{batch_segment} • {tokens} tokens @ {speed:.1f} tok/s"
        )

    def _handle_query_error(self, error_message: str):
        """Handle query error."""
        self.output_panel.show_progress(False)
        self.output_panel.set_status("Error", "#ef4444")
        self.output_panel.append_text(f"\n\n❌ {error_message}")
        self.query_panel.query_finished()
        
        # Update status bar with error
        self.llm_status_bar.set_error(error_message[:50])  # Truncate long errors
        self.llm_status_bar.finish(success=False)
        self.detail_panel.append_log(f"Error: {error_message}")
        if self.preview_dialog:
            self.preview_dialog.close()

    def _handle_worker_status(self, status: str, color: str):
        """Mirror worker status to output panel and log."""
        self.output_panel.set_status(status, color)
        self.detail_panel.append_log(status)

    def _log_stage_started(self, stage_name: str):
        """Write retriever stage transitions into the run log."""
        readable = stage_name.replace("_", " ").title()
        self.detail_panel.append_log(f"[Retriever] {readable}…")

    def _log_stage_completed(self, stage_name: str, stats: dict):
        """Write retriever stage completions with timing."""
        readable = stage_name.replace("_", " ").title()
        time_ms = stats.get("time_ms")
        count = stats.get("count") or stats.get("results") or stats.get("merged_count")
        extra = []
        if count is not None:
            extra.append(f"{int(count)} items")
        if time_ms is not None:
            extra.append(f"{time_ms:.0f} ms")
        if stats.get("top_score") is not None:
            extra.append(f"top={stats['top_score']:.3f}")
        summary = ", ".join(extra) if extra else ""
        self.detail_panel.append_log(f"[Retriever] {readable} - {summary}")

    def _log_generation_stage(self, stage_name: str):
        """Log generation stage transitions."""
        if not stage_name:
            return
        readable = stage_name.replace("_", " ").title()
        self.detail_panel.append_log(f"[LLM] Stage → {readable}")

    def _log_generation_batch_progress(self, completed: int, total: int):
        """Log batch progress without spamming identical entries."""
        if total <= 0:
            return
        if self._last_logged_batch_progress == (completed, total):
            return
        self._last_logged_batch_progress = (completed, total)
        self.detail_panel.append_log(f"[LLM] Batches {completed}/{total}")

    def _auto_start_llama_if_needed(self):
        """Start llama.cpp server automatically when configured."""
        try:
            # CRITICAL: Fix AMD Vulkan allocation limit on Windows
            import os
            os.environ['GGML_VK_FORCE_MAX_ALLOCATION_SIZE'] = '4294967295'
            
            cfg = load_llm_config()
            if cfg.provider != "llama_cpp":
                self.detail_panel.append_log("LLM provider is not llama_cpp; auto-start skipped.")
                return

            runtime_state = load_runtime_state()
            llama_cfg = runtime_state.get("llama_server", {})
            executable = llama_cfg.get("executable")
            model_path = llama_cfg.get("model_path")
            if not executable or not model_path:
                self.detail_panel.append_log("llama.cpp auto-start skipped: missing executable or model path.")
                return

            if llama_manager.is_running():
                self.detail_panel.append_log("llama.cpp server already running; auto-start skipped.")
                return

            host, port = _parse_host_port(cfg.base_url)
            extra_args: list[str] = []
            if llama_cfg.get("flash_attn"):
                extra_args.extend(["-fa", "on"])
            if llama_cfg.get("extra_args"):
                extra_args.extend(shlex.split(llama_cfg.get("extra_args", "")))

            llama_manager.start(
                executable=executable,
                model_path=model_path,
                host=host,
                port=port,
                api_key=cfg.api_key,
                context=int(llama_cfg.get("context", 65536)),
                gpu_layers=int(llama_cfg.get("gpu_layers", 999)),
                parallel=int(llama_cfg.get("parallel", 2)),
                batch=int(llama_cfg.get("batch", 1024)),
                timeout=int(llama_cfg.get("timeout", 1800)),
                detached=True,
                log_path=self.llama_log_path,
                extra_args=extra_args,
            )
            self.auto_started_llama = True
            self.detail_panel.append_log("llama.cpp server process started. Waiting for readiness...")
            
            # Start readiness checker
            base_url = f"http://{host}:{port}"
            self.readiness_worker = ServerReadinessWorker(base_url, timeout=120) # 2 min timeout for 80B model
            self.readiness_worker.ready.connect(self._on_server_ready)
            self.readiness_worker.failed.connect(self._on_server_failed)
            self.readiness_worker.start()
            
            self.llm_status_bar.set_status("Loading Model...", True)

        except Exception as exc:
            self.detail_panel.append_log(f"llama.cpp auto-start failed: {exc}")

    def _on_server_ready(self):
        """Handle server ready state."""
        self.detail_panel.append_log("llama.cpp server is ready and listening.")
        self.llm_status_bar.finish(True)
        self.server_ready = True
        
        # Process queued query if any
        if hasattr(self, "_queued_query") and self._queued_query:
            self.detail_panel.append_log("Processing queued query...")
            query = self._queued_query
            del self._queued_query
            self._handle_query(query)

    def _on_server_failed(self, error: str):
        """Handle server startup failure."""
        self.detail_panel.append_log(f"llama.cpp startup failed: {error}")
        self.llm_status_bar.set_error("Server Timeout")

    def closeEvent(self, event):
        """Ensure auto-started server is stopped."""
        if hasattr(self, "llama_log_timer"):
            self.llama_log_timer.stop()
        if hasattr(self, "gpu_monitor_timer"):
            self.gpu_monitor_timer.stop()
        if self.auto_started_llama and llama_manager.is_running():
            llama_manager.stop()
            self.detail_panel.append_log("llama.cpp server stopped on exit.")
        if self.preview_dialog:
            self.preview_dialog.close()
        super().closeEvent(event)

    def _setup_llama_console_monitor(self) -> None:
        self.llama_log_candidates = [
            LLAMA_SERVER_LOG_PATH,
            Path("logs/llama_server_default.err.log"),
            Path("logs/llama_server_default.out.log"),
            Path("logs/llama_server_auto.log"),
        ]
        self.llama_log_path = None
        self.llama_log_tail = ""
        self.llama_log_pos = 0
        self.llama_log_timer = QTimer(self)
        # Reduced from 1500ms to 300ms for real-time progress monitoring
        self.llama_log_timer.setInterval(300)
        self.llama_log_timer.timeout.connect(self._refresh_llama_console)
        # DISABLED FOR PERFORMANCE: Timer causes UI sluggishness
        # self.llama_log_timer.start()
        self._select_llama_log_path(announce=True)
        self._refresh_llama_console(force_full=True)

    def _resolve_llama_log_path(self) -> Path | None:
        candidates = getattr(self, "llama_log_candidates", [])
        existing: list[tuple[float, Path]] = []
        for path in candidates:
            try:
                if path.exists():
                    existing.append((path.stat().st_mtime, path))
            except OSError:
                continue
        if not existing:
            return None
        existing.sort(reverse=True)
        return existing[0][1]

    def _select_llama_log_path(self, announce: bool = False) -> bool:
        path = self._resolve_llama_log_path()
        if not path:
            return False
        if path != getattr(self, "llama_log_path", None):
            self.llama_log_path = path
            self.llama_log_tail = ""
            self.llama_log_pos = 0
            if announce:
                self.detail_panel.append_log(f"LLM console: tailing {path}")
        return True

    def _refresh_llama_console(self, force_full: bool = False) -> None:
        path = getattr(self, "llama_log_path", None)
        if not path or not path.exists():
            if not self._select_llama_log_path(announce=True):
                return
            path = self.llama_log_path
        try:
            size = path.stat().st_size
            if force_full or size < self.llama_log_pos:
                self.llama_log_pos = 0
            with path.open("r", encoding="utf-8", errors="ignore") as log_file:
                log_file.seek(self.llama_log_pos)
                chunk = log_file.read()
                self.llama_log_pos = log_file.tell()
        except Exception:
            return

        if not chunk:
            return

        self.llama_log_tail += chunk.replace("\r\n", "\n")
        lines = self.llama_log_tail.splitlines()
        if len(lines) > 400:
            lines = lines[-400:]
        self.llama_log_tail = "\n".join(lines)

        # Parse and highlight progress information in the console
        annotated_console = self._parse_llama_progress(self.llama_log_tail)
        self.detail_panel.set_llm_console(annotated_console)

    def _parse_llama_progress(self, log_text: str) -> str:
        """Parse llama.cpp logs and add progress annotations."""
        import re

        lines = log_text.splitlines()
        annotated_lines = []
        progress_summary = []

        for line in lines:
            annotated_line = line

            # Highlight token generation speed
            speed_match = re.search(r'(\d+\.\d+)\s*(tokens?\s*per\s*second|t/s|tok/s)', line, re.IGNORECASE)
            if speed_match:
                speed = float(speed_match.group(1))
                progress_summary.append(f"[{speed:.1f} tok/s]")
                annotated_line = f">>> {line}"

            # Highlight prompt evaluation
            if re.search(r'prompt\s+eval', line, re.IGNORECASE):
                annotated_line = f">>> {line}"

            # Highlight generation eval time
            if re.search(r'eval\s+time.*\d+\.\d+\s*ms', line, re.IGNORECASE):
                annotated_line = f">>> {line}"

            # Highlight slot processing
            if re.search(r'slot\s+\d+.*processing', line, re.IGNORECASE):
                annotated_line = f"==> {line}"

            # Highlight completion status
            if re.search(r'completion|finished|done', line, re.IGNORECASE):
                annotated_line = f"==> {line}"

            annotated_lines.append(annotated_line)

        # Add progress summary at the top
        if progress_summary:
            summary_line = f"[LIVE PROGRESS] {' '.join(progress_summary[-3:])}"
            return f"{summary_line}\n{'='*60}\n" + "\n".join(annotated_lines)

        return "\n".join(annotated_lines)

    def _setup_gpu_monitor(self) -> None:
        self.gpu_monitor_timer = QTimer(self)
        # Update every 5 seconds (was 1s - too frequent, spams Ollama logs)
        self.gpu_monitor_timer.setInterval(5000)
        self.gpu_monitor_timer.timeout.connect(self._refresh_gpu_stats)
        # DISABLED FOR PERFORMANCE: GPU monitoring not needed with llama.cpp
        # self.gpu_monitor_timer.start()
        self._refresh_gpu_stats()
        
        # Initialize diagnostics panel with GPU info
        self._init_diagnostics_gpu_info()

    def _refresh_gpu_stats(self) -> None:
        try:
            # Try Ollama GPU info first (shows actual VRAM usage for AMD ROCm)
            ollama_info = get_ollama_gpu_info()
            
            if ollama_info and ollama_info.is_loaded:
                # Show Ollama's GPU usage (more accurate for AMD)
                used = ollama_info.vram_used_gb
                total = ollama_info.vram_total_gb
                pct = (used / total * 100) if total > 0 else 0
                
                lines = [
                    f"GPU: {used:.1f} / {total:.0f} GB",
                    f"Model: {ollama_info.model_name.split(':')[0]}",
                ]
                self.sys_stats_label.setText("\n".join(lines))
                
                # Color based on VRAM usage - using warm theme colors
                color = "#f87171" if pct > 80 else ("#fbbf24" if pct > 50 else "#4ade80")
                self.sys_stats_label.setStyleSheet(f"color: {color}; font-weight: 600;")
                return
            elif ollama_info and not ollama_info.is_loaded:
                # Ollama running but no model loaded
                self.sys_stats_label.setText(f"GPU: Ready\n{ollama_info.vram_total_gb:.0f} GB available")
                self.sys_stats_label.setStyleSheet("color: #4ade80; font-weight: 600;")
                return
            
            # Fallback to Windows performance counters
            stats = get_gpu_stats()
            if not stats:
                self.sys_stats_label.setText("System: Monitoring...")
                return
                
            lines = []
            metric = stats[0]
            
            pct = metric.utilization if metric.utilization > 0 else (metric.usage_mb / metric.limit_mb * 100 if metric.limit_mb else 0)
            
            label = "SYS" if metric.is_fallback else "GPU"
            lines.append(f"{label}: {metric.usage_mb/1024:.1f} GB")
            lines.append(f"Load: {pct:.0f}%")
            
            self.sys_stats_label.setText("\n".join(lines))
            
            # Color code based on load - using warm theme colors
            color = "#f87171" if pct > 80 else ("#fbbf24" if pct > 50 else "#4ade80")
            self.sys_stats_label.setStyleSheet(f"color: {color}; font-weight: 600;")
            
        except Exception:
            self.sys_stats_label.setText("System: N/A")

    def _init_diagnostics_gpu_info(self) -> None:
        """Initialize diagnostics panel with GPU/system info."""
        try:
            diag = self.detail_panel.diagnostics_panel
            
            # Get Ollama GPU info
            ollama_info = get_ollama_gpu_info()
            if ollama_info:
                diag.set_gpu_info(
                    used_gb=ollama_info.vram_used_gb,
                    free_gb=ollama_info.vram_total_gb - ollama_info.vram_used_gb
                )
                if ollama_info.is_loaded:
                    diag.set_model_info(
                        name=ollama_info.model_name.split(":")[0] if ":" in ollama_info.model_name else ollama_info.model_name,
                        quant="Q4_K_M",  # Common quantization
                        size_gb=ollama_info.vram_used_gb
                    )
            else:
                # Fallback: show 96GB available (your system config)
                diag.set_gpu_info(used_gb=0, free_gb=96.0)
                
        except Exception as e:
            print(f"Failed to init diagnostics GPU info: {e}")

    # === Chat History Methods ===
    
    def _refresh_history_list(self):
        """Refresh the sidebar history list."""
        self.history_list.clear()
        
        for entry in self.chat_history.get_recent(15):
            item = QListWidgetItem()
            item.setText(entry.query_preview)
            item.setToolTip(f"{entry.query}\n\n{entry.display_time}")
            item.setData(Qt.UserRole, entry.id)
            self.history_list.addItem(item)
    
    def _save_to_history(self, response: str, chunks: list[dict], metrics: dict):
        """Save the current query/response to history."""
        try:
            query = getattr(self, '_current_query', "")
            if not query:
                return
            
            # Get model name from settings or metrics
            model = self.settings.models.llm.default
            
            entry = ChatEntry.create(
                query=query,
                response=response,
                model=model,
                chunk_count=len(chunks),
                duration_ms=int(metrics.get("duration_ms", 0) or 0),
                sources=[c.get("metadata", {}).get("file_name", "") for c in chunks[:5]],
                metrics=metrics,
            )
            
            self.chat_history.add(entry)
            self._refresh_history_list()
        except Exception as e:
            print(f"Failed to save to history: {e}")
    
    def _on_history_item_clicked(self, item: QListWidgetItem):
        """Handle click on a history item."""
        entry_id = item.data(Qt.UserRole)
        entry = self.chat_history.get(entry_id)
        
        if entry and entry.full_response:
            # Show the response popup with stored data
            self._response_popup = ResponsePopup(self)
            self._response_popup.set_response(entry.full_response)
            
            # Reconstruct minimal chunk data for display
            if entry.sources:
                chunks = [{"metadata": {"file_name": s}} for s in entry.sources if s]
                self._response_popup.set_sources(chunks)
            
            self._response_popup.set_metrics(
                chunks=entry.chunk_count,
                valid_citations=0,
                total_citations=0,
                prompt_tokens=entry.metrics.get("prompt_tokens", 0) or 0,
                output_tokens=entry.metrics.get("token_count", 0) or 0,
                tok_per_sec=entry.metrics.get("tokens_per_sec", 0) or 0,
            )
            self._response_popup.show()


def _parse_host_port(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    return host, port
