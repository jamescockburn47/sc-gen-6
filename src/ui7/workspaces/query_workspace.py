"""Query workspace — persistent multi-turn chat with diagnostics.

Improvements over baseline:
  - Copy / Export buttons on every answer card
  - Loading indicator ("Thinking...") while generating
  - Stop button to cancel mid-generation
  - Source chip click → popup with full chunk text
  - Answer length toggle (Concise / Detailed)
  - Turn separator for visual clarity in long conversations
  - Conversation context injected into LLM for follow-ups
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.ui7.theme import C, S, T


# ─── Conversation model ───────────────────────────────────────────────

@dataclass
class Turn:
    """Single Q&A turn in a conversation."""
    query: str
    response: str = ""
    sources: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


# ─── Worker ────────────────────────────────────────────────────────────

class QueryWorker(QThread):
    """Run RAG query with conversation context, emitting timing data."""

    chunk_received = Signal(str)       # final answer token
    thinking_chunk = Signal(str)       # chain-of-thought token
    sources_ready = Signal(list)
    finished = Signal(str, dict)
    error = Signal(str)

    def __init__(
        self,
        query: str,
        history: list[dict[str, str]],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.query = query
        self.history = history
        self._cancelled = False

    def run(self) -> None:
        try:
            from src.config_loader import get_settings
            from src.retrieval.query_engine import QueryEngine
            from src.utils.memory_manager import flush_for_query

            # Kick off page-cache drop + conditional swap flush in background.
            # Runs async so it doesn't block the UI — by the time the LLM
            # starts generating (after embedding + retrieval, ~1-3s) the
            # memory should already be freed.
            flush_for_query(blocking=False)

            t0 = time.time()
            settings = get_settings()
            engine = QueryEngine(settings)


            sources: list[dict] = []
            full_text = ""
            token_count = 0
            retrieval_ms = 0.0
            gen_start = 0.0

            for event in engine.query(
                self.query,
                conversation_history=self.history,
                max_tokens=None,          # unlimited — context window is the cap
                thinking_callback=self.thinking_chunk.emit,
            ):
                if self._cancelled:
                    return

                if "source" in event:
                    src = event["source"]
                    sources.append({
                        "text": (src.get("text") or "")[:300],
                        "file_name": src.get("file_name") or "Unknown",
                        "page": src.get("page") or "?",
                    })
                elif "thinking" in event:
                    # Thinking tokens arrive via thinking_callback directly
                    pass
                elif "token" in event:
                    if not full_text:
                        retrieval_ms = (time.time() - t0) * 1000
                        gen_start = time.time()
                        if sources:
                            self.sources_ready.emit(sources)
                    token = event["token"]
                    full_text += token
                    token_count += 1
                    self.chunk_received.emit(token)
                elif "error" in event:
                    self.error.emit(event["error"])
                    return

            gen_ms = (time.time() - gen_start) * 1000 if gen_start else 0
            total_ms = (time.time() - t0) * 1000
            tok_sec = (token_count / (gen_ms / 1000)) if gen_ms > 0 else 0

            self.finished.emit(full_text, {
                "retrieval_ms": retrieval_ms,
                "generation_ms": gen_ms,
                "total_ms": total_ms,
                "token_count": token_count,
                "tokens_per_sec": tok_sec,
                "source_count": len(sources),
                "model": settings.models.llm.default,
            })
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self) -> None:
        self._cancelled = True


# ─── Source popup ──────────────────────────────────────────────────────

class SourcePopup(QDialog):
    """Modal popup showing full source chunk text."""

    def __init__(self, source: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{source.get('file_name', 'Source')}  —  Page {source.get('page', '?')}")
        self.setMinimumSize(500, 350)
        self.resize(600, 400)
        self.setStyleSheet(f"background: {C.BG_SURFACE};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # Header
        header = QLabel(f"{source.get('file_name', 'Unknown')}  \u00B7  Page {source.get('page', '?')}")
        header.setStyleSheet(f"""
            font-size: 14px; font-weight: 600; color: {C.TEXT_PRIMARY};
            background: transparent;
        """)
        layout.addWidget(header)

        # Full text
        text_area = QTextEdit()
        text_area.setReadOnly(True)
        text_area.setPlainText(source.get("text", "No text available."))
        text_area.setStyleSheet(f"""
            QTextEdit {{
                background: {C.BG_BASE}; border: 1px solid {C.BORDER};
                border-radius: 8px; padding: 12px; font-size: 13px;
                color: {C.TEXT_PRIMARY}; line-height: 160%;
            }}
        """)
        layout.addWidget(text_area, 1)

        # Close + Copy buttons
        btns = QHBoxLayout()
        btns.addStretch(1)

        copy_btn = QPushButton("Copy Text")
        copy_btn.clicked.connect(lambda: QGuiApplication.clipboard().setText(source.get("text", "")))
        btns.addWidget(copy_btn)

        close_btn = QPushButton("Close")
        close_btn.setProperty("class", "primary")
        close_btn.clicked.connect(self.close)
        btns.addWidget(close_btn)

        layout.addLayout(btns)


# ─── Diagnostics bar ───────────────────────────────────────────────────

class DiagnosticsBar(QWidget):
    """Compact metrics strip under an answer."""

    def __init__(self, diag: dict, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(36, 4, 0, 0)
        layout.setSpacing(14)

        mono = (
            f"color: {C.TEXT_TERTIARY}; font-size: 10px;"
            f" font-family: 'JetBrains Mono','Consolas',monospace;"
            f" background: transparent;"
        )
        for text in [
            f"Retrieval {diag.get('retrieval_ms',0):.0f}ms",
            f"{diag.get('source_count',0)} sources",
            f"Gen {diag.get('generation_ms',0)/1000:.1f}s",
            f"{diag.get('token_count',0)} tokens",
            f"{diag.get('tokens_per_sec',0):.0f} tok/s",
            f"{diag.get('model','?')}",
        ]:
            lbl = QLabel(text)
            lbl.setStyleSheet(mono)
            layout.addWidget(lbl)
        layout.addStretch(1)


# ─── Answer card ───────────────────────────────────────────────────────

class AnswerCard(QWidget):
    """Single Q&A turn with streaming thinking panel + prominent answer."""

    def __init__(self, question: str, parent=None) -> None:
        super().__init__(parent)
        self._question = question
        self._thinking_buf = ""
        self._answer_buf = ""
        self._thinking_collapsed = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 16)
        root.setSpacing(8)

        # ── Separator ──
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {C.BORDER}; margin: 0 36px;")
        root.addWidget(sep)

        # ── Question ──
        q_row = QHBoxLayout()
        q_row.setSpacing(10)
        q_badge = QLabel("Q")
        q_badge.setFixedSize(26, 26)
        q_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        q_badge.setStyleSheet(f"background:{C.ACCENT}; color:{C.TEXT_INVERSE}; font-size:12px; font-weight:700; border-radius:6px;")
        q_row.addWidget(q_badge, 0, Qt.AlignmentFlag.AlignTop)
        q_text = QLabel(question)
        q_text.setWordWrap(True)
        q_text.setStyleSheet(f"color:{C.TEXT_PRIMARY}; font-size:14px; font-weight:600; background:transparent; padding-top:3px;")
        q_row.addWidget(q_text, 1)
        root.addLayout(q_row)

        # ── Thinking section ──
        self._thinking_wrapper = QWidget()
        self._thinking_wrapper.setStyleSheet("background:transparent;")
        tw = QVBoxLayout(self._thinking_wrapper)
        tw.setContentsMargins(36, 0, 0, 0)
        tw.setSpacing(4)

        # Header row: pulsing indicator + label + collapse toggle
        think_header = QHBoxLayout()
        think_header.setSpacing(6)
        self._think_pulse = QLabel("●")
        self._think_pulse.setStyleSheet(f"color:{C.ACCENT}; font-size:10px; background:transparent;")
        think_header.addWidget(self._think_pulse)
        think_header.addWidget(QLabel(
            "Thinking…",
            styleSheet=f"color:{C.TEXT_TERTIARY}; font-size:11px; font-style:italic; font-weight:500; background:transparent;"
        ))
        think_header.addStretch(1)
        self._toggle_btn = QPushButton("▲ Collapse")
        self._toggle_btn.setFlat(True)
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.setStyleSheet(f"color:{C.TEXT_TERTIARY}; font-size:10px; background:transparent; border:none; padding:0;")
        self._toggle_btn.clicked.connect(self._toggle_thinking)
        self._toggle_btn.hide()  # shown only after answer arrives
        think_header.addWidget(self._toggle_btn)
        tw.addLayout(think_header)

        # Thinking text area (scrollable)
        self._thinking_scroll = QScrollArea()
        self._thinking_scroll.setWidgetResizable(True)
        self._thinking_scroll.setMaximumHeight(350)   # live generation view
        self._thinking_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._thinking_scroll.setStyleSheet(f"""
            QScrollArea {{ border:1px solid {C.BORDER}; border-radius:8px; background:{C.BG_BASE}; }}
            QScrollBar:vertical {{ width:4px; background:transparent; }}
            QScrollBar::handle:vertical {{ background:{C.BORDER}; border-radius:2px; }}
        """)
        self._thinking_label = QLabel("")
        self._thinking_label.setWordWrap(True)
        self._thinking_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._thinking_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._thinking_label.setStyleSheet(f"""
            QLabel {{
                color:{C.TEXT_TERTIARY}; font-size:11px; font-style:italic;
                font-family:'JetBrains Mono','Consolas',monospace;
                background:{C.BG_BASE}; padding:10px 14px; line-height:160%;
            }}
        """)
        self._thinking_scroll.setWidget(self._thinking_label)
        tw.addWidget(self._thinking_scroll)
        root.addWidget(self._thinking_wrapper)
        self._thinking_wrapper.hide()  # revealed on first thinking token

        # ── Loading indicator (before first thinking token) ──
        self._loading = QLabel("• • •  Retrieving sources…")
        self._loading.setStyleSheet(f"color:{C.TEXT_TERTIARY}; font-size:12px; font-style:italic; background:transparent; margin-left:36px; padding:8px 0;")
        root.addWidget(self._loading)

        # ── Answer section ──
        self._answer_wrapper = QWidget()
        self._answer_wrapper.setStyleSheet("background:transparent;")
        aw = QVBoxLayout(self._answer_wrapper)
        aw.setContentsMargins(36, 0, 0, 0)
        aw.setSpacing(6)

        ans_header = QLabel("Answer")
        ans_header.setStyleSheet(f"color:{C.ACCENT}; font-size:11px; font-weight:700; letter-spacing:0.5px; background:transparent; text-transform:uppercase;")
        aw.addWidget(ans_header)

        from PySide6.QtWidgets import QTextEdit
        self.answer_text = QTextEdit()
        self.answer_text.setReadOnly(True)
        self.answer_text.setFrameStyle(0)  # no frame
        self.answer_text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.answer_text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.answer_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.answer_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.answer_text.setStyleSheet(f"""
            QTextEdit {{
                color:{C.TEXT_PRIMARY}; font-size:14px;
                background:{C.BG_SURFACE}; border:1px solid {C.BORDER};
                border-radius:10px; padding:18px 22px;
                font-family:'Inter','Segoe UI',system-ui;
                line-height:175%;
            }}
            QScrollBar:vertical {{ width:4px; background:transparent; }}
            QScrollBar::handle:vertical {{ background:{C.BORDER}; border-radius:2px; }}
        """)
        aw.addWidget(self.answer_text)
        self._answer_wrapper.hide()
        root.addWidget(self._answer_wrapper)

        # ── Sources (grouped by document, wrapping rows) ──
        self._sources: list[dict] = []
        self.sources_widget = QWidget()
        self.sources_widget.setStyleSheet("background:transparent;")
        self._sources_outer = QVBoxLayout(self.sources_widget)
        self._sources_outer.setContentsMargins(36, 4, 0, 0)
        self._sources_outer.setSpacing(4)
        self.sources_widget.hide()
        root.addWidget(self.sources_widget)

        # ── Action row ──
        self._actions = QWidget()
        self._actions.setStyleSheet("background:transparent;")
        self._actions.hide()
        al = QHBoxLayout(self._actions)
        al.setContentsMargins(36, 0, 0, 0)
        al.setSpacing(8)
        ghost = f"""
            QPushButton {{ background:transparent; color:{C.TEXT_TERTIARY}; border:none; padding:3px 8px; font-size:10px; font-weight:500; }}
            QPushButton:hover {{ color:{C.TEXT_PRIMARY}; background:{C.BG_RAISED}; border-radius:4px; }}
        """
        copy_btn = QPushButton("Copy")
        copy_btn.setStyleSheet(ghost)
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.clicked.connect(self._copy_answer)
        al.addWidget(copy_btn)
        export_btn = QPushButton("Export as Markdown")
        export_btn.setStyleSheet(ghost)
        export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        export_btn.clicked.connect(self._export_markdown)
        al.addWidget(export_btn)
        al.addStretch(1)
        root.addWidget(self._actions)

    # ── Streaming ──

    def append_thinking_token(self, token: str) -> None:
        """Called for each reasoning/chain-of-thought token."""
        if not self._thinking_buf:
            self._loading.hide()
            self._thinking_wrapper.show()
        self._thinking_buf += token
        self._thinking_label.setText(self._thinking_buf)
        sb = self._thinking_scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def append_token(self, token: str) -> None:
        """Called for each final-answer token."""
        if not self._answer_buf:
            self._on_answer_started()
        self._answer_buf += token
        self.answer_text.setPlainText(self._answer_buf)

    def set_full_text(self, text: str) -> None:
        """Set complete answer (for restored conversations)."""
        self._loading.hide()
        self._thinking_wrapper.hide()
        self._answer_wrapper.show()
        self._answer_buf = text
        self.answer_text.setPlainText(text)
        self._actions.show()

    def _on_answer_started(self) -> None:
        """Transition from thinking to answering — collapse to summary height."""
        self._think_pulse.setText("✓")
        self._think_pulse.setStyleSheet(f"color:{C.ACCENT}; font-size:10px; background:transparent;")
        self._thinking_scroll.setMaximumHeight(120)   # collapsed summary
        self._toggle_btn.setText("▼ Show reasoning")
        self._toggle_btn.show()
        self._thinking_collapsed = True   # starts collapsed once answer begins
        # Update label
        for i in range(self._thinking_wrapper.layout().count()):
            item = self._thinking_wrapper.layout().itemAt(i)
            if item and item.layout():
                for j in range(item.layout().count()):
                    w = item.layout().itemAt(j)
                    if w and isinstance(w.widget(), QLabel) and "Thinking" in (w.widget().text() or ""):
                        w.widget().setText("Reasoning complete  ·  expand to read")
                        break
        self._answer_wrapper.show()

    def _toggle_thinking(self) -> None:
        if self._thinking_collapsed:
            # Expand fully — use Qt's maximum widget size (uncapped)
            self._thinking_scroll.show()
            self._thinking_scroll.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
            self._toggle_btn.setText("▲ Collapse")
            self._thinking_collapsed = False
        else:
            self._thinking_scroll.setMaximumHeight(120)
            self._toggle_btn.setText("▼ Show reasoning")
            self._thinking_collapsed = True


    def set_sources(self, sources: list[dict]) -> None:
        self._sources = sources

        # Group chunks by document name to avoid 20 chips for the same PDF
        doc_map: dict[str, list[dict]] = {}
        for src in sources:
            name = src.get("file_name", "Unknown")
            doc_map.setdefault(name, []).append(src)

        chip_style = f"""
            QPushButton {{ background:{C.BG_RAISED}; color:{C.TEXT_SECONDARY}; border:1px solid {C.BORDER}; border-radius:5px; padding:4px 10px; font-size:11px; font-weight:500; }}
            QPushButton:hover {{ background:{C.BG_OVERLAY}; color:{C.TEXT_PRIMARY}; border-color:{C.BORDER_STRONG}; }}
        """

        # Build wrapping rows — at most 6 chips per row
        chips: list[QPushButton] = []
        for doc_name, srcs in doc_map.items():
            pages = sorted({str(s.get('page', '?')) for s in srcs})
            if len(pages) == 1:
                label = f"{doc_name}  p.{pages[0]}"
            elif len(pages) <= 4:
                label = f"{doc_name}  pp. {'·'.join(pages)}"
            else:
                label = f"{doc_name}  ({len(srcs)} passages)"
            btn = QPushButton(label)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(f"Pages: {', '.join(pages)}")
            btn.setStyleSheet(chip_style)
            btn.clicked.connect(lambda checked, s=srcs[0]: self._show_source(s))
            chips.append(btn)

        # Place chips in rows of up to 6
        ROW_MAX = 6
        for row_start in range(0, len(chips), ROW_MAX):
            row_widget = QWidget()
            row_widget.setStyleSheet("background:transparent;")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            for chip in chips[row_start:row_start + ROW_MAX]:
                row_layout.addWidget(chip)
            row_layout.addStretch(1)
            self._sources_outer.addWidget(row_widget)

        self.sources_widget.show()

    def set_diagnostics(self, diag: dict) -> None:
        self._loading.hide()
        self._answer_wrapper.show()
        self._actions.show()
        self.layout().addWidget(DiagnosticsBar(diag))

    def set_error(self, message: str) -> None:
        self._loading.hide()
        self._answer_wrapper.show()
        self._answer_buf = message
        self.answer_text.setPlainText(message)
        self.answer_text.setStyleSheet(
            self.answer_text.styleSheet() + f"\nQTextEdit {{ color:{C.ERROR}; border-color:{C.ERROR}; }}"
        )


    # ── Actions ──

    def _copy_answer(self) -> None:
        QGuiApplication.clipboard().setText(self._answer_buf)

    def _export_markdown(self) -> None:
        parts = [f"**Q:** {self._question}\n", self._answer_buf]
        if self._sources:
            parts.append("\n**Sources:**")
            for src in self._sources:
                parts.append(f"- {src['file_name']} p.{src['page']}")
        QGuiApplication.clipboard().setText("\n".join(parts))

    def _show_source(self, source: dict) -> None:
        SourcePopup(source, self).exec()


# ─── History rail ──────────────────────────────────────────────────────

class HistoryRail(QWidget):
    """Left sidebar listing past conversations."""

    entry_selected = Signal(str)
    new_chat = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(200)
        self.setStyleSheet(f"""
            QWidget#histRail {{
                background:{C.BG_SURFACE}; border-right:1px solid {C.BORDER};
            }}
        """)
        self.setObjectName("histRail")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(8)

        new_btn = QPushButton("+ New Chat")
        new_btn.setStyleSheet(f"""
            QPushButton {{
                background:{C.ACCENT_BG}; color:{C.ACCENT};
                border:1px solid {C.ACCENT_MUTED}; border-radius:6px;
                padding:6px 12px; font-size:12px; font-weight:600;
            }}
            QPushButton:hover {{ background:{C.ACCENT}; color:{C.TEXT_INVERSE}; }}
        """)
        new_btn.clicked.connect(self.new_chat.emit)
        layout.addWidget(new_btn)

        self.history_list = QListWidget()
        self.history_list.setStyleSheet(f"""
            QListWidget {{ background:transparent; border:none; }}
            QListWidget::item {{
                padding:6px 8px; border-radius:4px; margin:1px 0;
                color:{C.TEXT_SECONDARY}; font-size:11px;
            }}
            QListWidget::item:selected {{
                background:{C.ACCENT_BG}; color:{C.TEXT_PRIMARY};
            }}
            QListWidget::item:hover:!selected {{ background:{C.BG_RAISED}; }}
        """)
        self.history_list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.history_list, 1)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        eid = item.data(Qt.ItemDataRole.UserRole)
        if eid:
            self.entry_selected.emit(eid)

    def load_entries(self, entries) -> None:
        self.history_list.clear()
        for entry in entries:
            item = QListWidgetItem()
            item.setText(f"{entry.query_preview}\n{entry.display_time}")
            item.setData(Qt.ItemDataRole.UserRole, entry.id)
            self.history_list.addItem(item)

    def prepend_entry(self, entry) -> None:
        item = QListWidgetItem()
        item.setText(f"{entry.query_preview}\n{entry.display_time}")
        item.setData(Qt.ItemDataRole.UserRole, entry.id)
        self.history_list.insertItem(0, item)


# ─── Workspace ─────────────────────────────────────────────────────────

class QueryWorkspace(QWidget):
    """Multi-turn chat workspace with all quality-of-life features."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._worker: Optional[QueryWorker] = None
        self._current_card: Optional[AnswerCard] = None
        self._turns: list[Turn] = []

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── History rail ──
        self.history_rail = HistoryRail()
        self.history_rail.new_chat.connect(self._new_chat)
        self.history_rail.entry_selected.connect(self._restore_entry)
        root.addWidget(self.history_rail)

        # ── Main content ──
        main = QWidget()
        ml = QVBoxLayout(main)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea {{ border:none; background:{C.BG_BASE}; }}")

        self._container = QWidget()
        self._container.setStyleSheet(f"background:{C.BG_BASE};")
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(40, 24, 40, 16)
        self._layout.setSpacing(0)
        self._layout.addStretch(1)

        self._welcome = QLabel(
            "Ask a question about your case documents.\n"
            "Answers are grounded in your indexed evidence with full citations.\n"
            "Follow-up questions carry the conversation context."
        )
        self._welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._welcome.setWordWrap(True)
        self._welcome.setStyleSheet(f"""
            color:{C.TEXT_TERTIARY}; font-size:14px; line-height:160%;
            padding:80px 40px; background:transparent;
        """)
        self._layout.insertWidget(0, self._welcome)

        scroll.setWidget(self._container)
        self._scroll = scroll
        ml.addWidget(scroll, 1)

        # ── Input bar ──
        input_wrapper = QWidget()
        input_wrapper.setStyleSheet(f"""
            QWidget#inputWrap {{ background:{C.BG_BASE}; border-top:1px solid {C.BORDER}; }}
        """)
        input_wrapper.setObjectName("inputWrap")
        iw = QVBoxLayout(input_wrapper)
        iw.setContentsMargins(40, 10, 40, 14)
        iw.setSpacing(8)

        # Options row: answer length
        opts_row = QHBoxLayout()
        opts_row.setSpacing(8)

        len_label = QLabel("Length:")
        len_label.setStyleSheet(f"color:{C.TEXT_TERTIARY}; font-size:10px; background:transparent;")
        opts_row.addWidget(len_label)

        self.length_combo = QComboBox()
        self.length_combo.addItems(["Concise", "Standard", "Detailed"])
        self.length_combo.setCurrentIndex(1)
        self.length_combo.setFixedWidth(90)
        self.length_combo.setStyleSheet(f"""
            QComboBox {{
                background:{C.BG_RAISED}; color:{C.TEXT_SECONDARY};
                border:1px solid {C.BORDER}; border-radius:4px;
                padding:2px 6px; font-size:10px;
            }}
        """)
        opts_row.addWidget(self.length_combo)

        opts_row.addStretch(1)

        # Turn count indicator
        self._turn_label = QLabel("")
        self._turn_label.setStyleSheet(f"color:{C.TEXT_TERTIARY}; font-size:10px; background:transparent;")
        opts_row.addWidget(self._turn_label)

        iw.addLayout(opts_row)

        # Input row
        input_bar = QWidget()
        input_bar.setStyleSheet(f"""
            QWidget#inputBar {{
                background:{C.BG_SURFACE}; border:1px solid {C.BORDER}; border-radius:12px;
            }}
        """)
        input_bar.setObjectName("inputBar")
        ib = QHBoxLayout(input_bar)
        ib.setContentsMargins(16, 8, 8, 8)
        ib.setSpacing(8)

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Ask about your case documents...")
        self.query_input.setStyleSheet(f"""
            QLineEdit {{
                background:transparent; border:none;
                color:{C.TEXT_PRIMARY}; font-size:14px; padding:4px 0;
            }}
        """)
        self.query_input.returnPressed.connect(self._submit_query)
        ib.addWidget(self.query_input, 1)

        # Stop button (hidden by default)
        self.stop_btn = QPushButton("\u25A0")  # ■
        self.stop_btn.setFixedSize(34, 34)
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setToolTip("Stop generating")
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background:{C.ERROR}; color:white;
                border:none; border-radius:8px; font-size:12px; font-weight:700;
            }}
            QPushButton:hover {{ background:{C.ERROR_BG}; color:{C.ERROR}; border:1px solid {C.ERROR}; }}
        """)
        self.stop_btn.clicked.connect(self._stop_generation)
        self.stop_btn.hide()
        ib.addWidget(self.stop_btn)

        # Send button
        self.send_btn = QPushButton("\u2192")
        self.send_btn.setFixedSize(34, 34)
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.setStyleSheet(f"""
            QPushButton {{
                background:{C.ACCENT}; color:{C.TEXT_INVERSE};
                border:none; border-radius:8px; font-size:16px; font-weight:700;
            }}
            QPushButton:hover {{ background:{C.ACCENT_HOVER}; }}
        """)
        self.send_btn.clicked.connect(self._submit_query)
        ib.addWidget(self.send_btn)

        iw.addWidget(input_bar)
        ml.addWidget(input_wrapper)

        root.addWidget(main, 1)
        self._load_history()

    # ── History ──

    def _load_history(self) -> None:
        try:
            from src.chat.history import ChatHistory
            entries = ChatHistory().get_recent(30)
            self.history_rail.load_entries(entries)
        except Exception:
            pass

    def _new_chat(self) -> None:
        self._turns = []
        self._clear_cards()
        self._welcome.show()
        self._update_turn_label()
        self.query_input.setFocus()

    def _restore_entry(self, entry_id: str) -> None:
        try:
            from src.chat.history import ChatHistory
            entry = ChatHistory().get(entry_id)
            if not entry:
                return

            self._turns = []
            self._clear_cards()
            self._welcome.hide()

            card = AnswerCard(entry.query)
            card.set_full_text(entry.full_response)
            if entry.sources:
                card.set_sources(entry.sources)
            if entry.metrics:
                card.set_diagnostics(entry.metrics)
            self._layout.insertWidget(self._layout.count() - 1, card)

            self._turns.append(Turn(
                query=entry.query, response=entry.full_response,
                sources=entry.sources, diagnostics=entry.metrics,
            ))
            self._update_turn_label()
        except Exception:
            pass

    def _clear_cards(self) -> None:
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            w = item.widget()
            if w and w is not self._welcome:
                w.deleteLater()
            elif w is self._welcome:
                self._layout.insertWidget(0, self._welcome)
                break

    def _update_turn_label(self) -> None:
        n = len(self._turns)
        if n == 0:
            self._turn_label.setText("")
        else:
            self._turn_label.setText(f"Turn {n} \u00B7 context active")

    # ── Query submission ──

    def _submit_query(self) -> None:
        query = self.query_input.text().strip()
        if not query:
            return

        if self._welcome.isVisible():
            self._welcome.hide()

        card = AnswerCard(query)
        self._layout.insertWidget(self._layout.count() - 1, card)
        self._current_card = card

        self.query_input.clear()
        self.query_input.setEnabled(False)
        self.send_btn.hide()
        self.stop_btn.show()

        conv_history = [{"query": t.query, "response": t.response} for t in self._turns]

        self._worker = QueryWorker(query, conv_history)
        self._worker.chunk_received.connect(card.append_token)
        self._worker.thinking_chunk.connect(card.append_thinking_token)
        self._worker.sources_ready.connect(card.set_sources)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(lambda text, diag: self._on_finished(query, text, diag))
        self._worker.start()

    def _stop_generation(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            if self._current_card:
                self._current_card._loading.hide()
                self._current_card._answer_wrapper.show()
                if not self._current_card._answer_buf:
                    self._current_card.set_error("Generation stopped by user.")
        self._reset_input()

    def _on_error(self, message: str) -> None:
        if self._current_card:
            self._current_card.set_error(message)
        self._reset_input()

    def _on_finished(self, query: str, text: str, diagnostics: dict) -> None:
        if self._current_card:
            self._current_card.set_diagnostics(diagnostics)

        turn = Turn(query=query, response=text, diagnostics=diagnostics)
        self._turns.append(turn)
        self._update_turn_label()

        try:
            from src.chat.history import ChatEntry, ChatHistory
            entry = ChatEntry.create(
                query=query, response=text,
                model=diagnostics.get("model", "?"),
                chunk_count=diagnostics.get("source_count", 0),
                duration_ms=int(diagnostics.get("total_ms", 0)),
                metrics=diagnostics,
            )
            ChatHistory().add(entry)
            self.history_rail.prepend_entry(entry)
        except Exception:
            pass

        self._reset_input()
        QTimer.singleShot(50, self._scroll_to_bottom)

    def _reset_input(self) -> None:
        self.query_input.setEnabled(True)
        self.query_input.setFocus()
        self.stop_btn.hide()
        self.send_btn.show()

    def _scroll_to_bottom(self) -> None:
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def cleanup(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
