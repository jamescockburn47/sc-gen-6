"""Standalone anonymisation review panel — human-in-the-loop review UI.

A sophisticated, colour-coded review interface that allows the user to:
  - See all detected PII entities with risk-level colour coding
  - View original text alongside anonymised text (side-by-side)
  - Approve or reject the anonymisation for each document/response
  - Manually de-anonymise specific tokens (selective reveal)
  - Manually add extra anonymisation (paint over missed PII)
  - View privilege warnings and data sovereignty notes
  - See detection confidence and layer information
  - Navigate through a queue of pending reviews

Colour scheme (risk levels):
  CRITICAL  → Red (#ef4444)      — victim/perpetrator identifiers
  HIGH      → Orange (#f97316)   — addresses, witnesses, relationships
  MEDIUM    → Yellow (#eab308)   — legal professionals, dates, case refs
  LOW       → Blue (#3b82f6)     — organisations, URLs
  PRIVILEGE → Purple (#a855f7)   — legally privileged content
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import (
    QColor,
    QFont,
    QTextCharFormat,
    QTextCursor,
    QPalette,
    QBrush,
    QPainter,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from src.anonymisation.models import (
    AnonymisedDocument,
    PIICategory,
    PIIEntity,
    ReviewStatus,
    RiskLevel,
)
from src.anonymisation.review.human_review import HumanReviewQueue, ReviewItem

# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------

RISK_COLOURS: dict[RiskLevel, tuple[str, str, str]] = {
    # (background, text, border)
    RiskLevel.CRITICAL: ("#fef2f2", "#991b1b", "#ef4444"),
    RiskLevel.HIGH: ("#fff7ed", "#9a3412", "#f97316"),
    RiskLevel.MEDIUM: ("#fefce8", "#854d0e", "#eab308"),
    RiskLevel.LOW: ("#eff6ff", "#1e40af", "#3b82f6"),
}

RISK_HIGHLIGHT_COLOURS: dict[RiskLevel, QColor] = {
    RiskLevel.CRITICAL: QColor(254, 202, 202),   # Red-100
    RiskLevel.HIGH: QColor(254, 215, 170),        # Orange-200
    RiskLevel.MEDIUM: QColor(254, 240, 138),      # Yellow-200
    RiskLevel.LOW: QColor(191, 219, 254),          # Blue-200
}

PRIVILEGE_COLOUR = QColor(233, 213, 255)  # Purple-200

STATUS_COLOURS = {
    ReviewStatus.PENDING: "#eab308",
    ReviewStatus.APPROVED: "#22c55e",
    ReviewStatus.REJECTED: "#ef4444",
    ReviewStatus.EDITED: "#3b82f6",
    ReviewStatus.NOT_REQUIRED: "#6b7280",
}


def _risk_badge_style(risk: RiskLevel) -> str:
    """CSS style for a risk level badge."""
    bg, fg, border = RISK_COLOURS.get(risk, ("#f3f4f6", "#374151", "#d1d5db"))
    return (
        f"background-color: {bg}; color: {fg}; border: 1px solid {border}; "
        f"border-radius: 4px; padding: 2px 8px; font-weight: 600; font-size: 12px;"
    )


# ---------------------------------------------------------------------------
# Review Panel
# ---------------------------------------------------------------------------

class AnonymisationReviewPanel(QMainWindow):
    """Standalone review panel for anonymised documents.

    Can be launched as a separate window from the main application.

    Args:
        review_queue: The HumanReviewQueue instance.
        parent: Optional parent widget.
    """

    # Signals
    review_completed = Signal(str, str)  # (item_id, status)

    def __init__(
        self,
        review_queue: HumanReviewQueue,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._queue = review_queue
        self._current_item: Optional[ReviewItem] = None
        self._manual_additions: list[dict[str, Any]] = []

        self.setWindowTitle("Anonymisation Review — Human-in-the-Loop")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)

        self._setup_ui()
        self._apply_styles()
        self._refresh_queue_list()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the full UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left panel: review queue list
        self._queue_panel = self._build_queue_panel()
        main_layout.addWidget(self._queue_panel)

        # Right panel: document review area
        self._review_area = self._build_review_area()
        main_layout.addWidget(self._review_area, stretch=1)

    def _build_queue_panel(self) -> QFrame:
        """Build the left-side queue list panel."""
        panel = QFrame()
        panel.setObjectName("queue_panel")
        panel.setFixedWidth(320)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header
        header = QLabel("Review Queue")
        header.setObjectName("panel_header")
        layout.addWidget(header)

        # Stats bar
        self._stats_label = QLabel("0 pending")
        self._stats_label.setObjectName("stats_label")
        layout.addWidget(self._stats_label)

        # Queue list
        self._queue_table = QTableWidget()
        self._queue_table.setColumnCount(3)
        self._queue_table.setHorizontalHeaderLabels(["Document", "Entities", "Risk"])
        self._queue_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._queue_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._queue_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._queue_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._queue_table.setSelectionMode(QTableWidget.SingleSelection)
        self._queue_table.verticalHeader().setVisible(False)
        self._queue_table.cellClicked.connect(self._on_queue_item_clicked)
        layout.addWidget(self._queue_table, stretch=1)

        # Refresh button
        refresh_btn = QPushButton("Refresh Queue")
        refresh_btn.setObjectName("secondary_btn")
        refresh_btn.clicked.connect(self._refresh_queue_list)
        layout.addWidget(refresh_btn)

        return panel

    def _build_review_area(self) -> QWidget:
        """Build the main review area with side-by-side text views."""
        area = QWidget()
        layout = QVBoxLayout(area)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Document header bar
        header_bar = QHBoxLayout()
        self._doc_title = QLabel("Select a document to review")
        self._doc_title.setObjectName("doc_title")
        header_bar.addWidget(self._doc_title)

        self._review_status_badge = QLabel("")
        self._review_status_badge.setObjectName("status_badge")
        header_bar.addWidget(self._review_status_badge)
        header_bar.addStretch()
        layout.addLayout(header_bar)

        # Warning banners area
        self._warnings_area = QVBoxLayout()
        layout.addLayout(self._warnings_area)

        # Main splitter: entity table | text views
        main_splitter = QSplitter(Qt.Horizontal)

        # Left: entity table
        entity_panel = self._build_entity_panel()
        main_splitter.addWidget(entity_panel)

        # Right: side-by-side text
        text_splitter = QSplitter(Qt.Horizontal)
        self._original_view = self._build_text_view("Original Text (LOCAL ONLY)")
        self._anonymised_view = self._build_text_view("Anonymised Text (Cloud-Safe)")
        text_splitter.addWidget(self._original_view)
        text_splitter.addWidget(self._anonymised_view)
        text_splitter.setSizes([500, 500])
        main_splitter.addWidget(text_splitter)

        main_splitter.setSizes([400, 800])
        layout.addWidget(main_splitter, stretch=1)

        # Action buttons bar
        action_bar = self._build_action_bar()
        layout.addLayout(action_bar)

        return area

    def _build_entity_panel(self) -> QWidget:
        """Build the detected entities table with colour coding."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        header = QLabel("Detected PII Entities")
        header.setObjectName("section_header")
        layout.addWidget(header)

        self._entity_table = QTableWidget()
        self._entity_table.setColumnCount(5)
        self._entity_table.setHorizontalHeaderLabels([
            "Category", "Original Text", "Confidence", "Risk", "Layer",
        ])
        self._entity_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._entity_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._entity_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._entity_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._entity_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._entity_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._entity_table.verticalHeader().setVisible(False)
        self._entity_table.cellClicked.connect(self._on_entity_clicked)
        layout.addWidget(self._entity_table, stretch=1)

        # Manual controls
        manual_layout = QHBoxLayout()

        self._manual_input = QLineEdit()
        self._manual_input.setPlaceholderText("Select text in original view to add manually...")
        manual_layout.addWidget(self._manual_input, stretch=1)

        add_anon_btn = QPushButton("+ Add Anonymisation")
        add_anon_btn.setObjectName("add_btn")
        add_anon_btn.clicked.connect(self._on_add_manual_anonymisation)
        manual_layout.addWidget(add_anon_btn)

        remove_btn = QPushButton("- Remove Selected")
        remove_btn.setObjectName("remove_btn")
        remove_btn.clicked.connect(self._on_remove_entity)
        manual_layout.addWidget(remove_btn)

        layout.addLayout(manual_layout)

        return panel

    def _build_text_view(self, title: str) -> QWidget:
        """Build a labelled text view."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        label = QLabel(title)
        label.setObjectName("section_header")
        layout.addWidget(label)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setObjectName("text_view")
        layout.addWidget(text_edit, stretch=1)

        # Store reference
        if "Original" in title:
            self._original_text = text_edit
        else:
            self._anonymised_text = text_edit

        return panel

    def _build_action_bar(self) -> QHBoxLayout:
        """Build the approve/reject/edit action buttons."""
        bar = QHBoxLayout()
        bar.setSpacing(12)

        # Left: info
        self._action_info = QLabel("")
        bar.addWidget(self._action_info, stretch=1)

        # Selective de-anonymise button
        deanon_btn = QPushButton("Selective De-anonymise")
        deanon_btn.setObjectName("secondary_btn")
        deanon_btn.setToolTip("Reveal specific tokens in the anonymised output (for fine-tuning)")
        deanon_btn.clicked.connect(self._on_selective_deanonymise)
        bar.addWidget(deanon_btn)

        # Reject
        reject_btn = QPushButton("  Reject — Block Export  ")
        reject_btn.setObjectName("reject_btn")
        reject_btn.clicked.connect(self._on_reject)
        bar.addWidget(reject_btn)

        # Approve
        approve_btn = QPushButton("  Approve for Export  ")
        approve_btn.setObjectName("approve_btn")
        approve_btn.clicked.connect(self._on_approve)
        bar.addWidget(approve_btn)

        return bar

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------

    def _apply_styles(self) -> None:
        """Apply the dark theme styling."""
        self.setStyleSheet("""
            QMainWindow { background-color: #0f0f12; color: #f1f5f9; }
            QWidget { background-color: #0f0f12; color: #f1f5f9;
                      font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 13px; }

            /* Queue panel */
            QFrame#queue_panel { background-color: #18181b; border-right: 1px solid #27272a; }
            QLabel#panel_header { font-size: 18px; font-weight: 700; color: #f1f5f9;
                                  padding: 4px 0; }
            QLabel#stats_label { font-size: 12px; color: #a1a1aa; }

            /* Document title */
            QLabel#doc_title { font-size: 16px; font-weight: 600; color: #f1f5f9; }
            QLabel#section_header { font-size: 13px; font-weight: 600; color: #a1a1aa;
                                    text-transform: uppercase; letter-spacing: 1px; }

            /* Text views */
            QTextEdit#text_view { background-color: #1a1a1e; color: #e4e4e7;
                                  border: 1px solid #27272a; border-radius: 6px;
                                  padding: 8px; font-family: 'JetBrains Mono', 'Consolas', monospace;
                                  font-size: 13px; line-height: 1.5; }

            /* Tables */
            QTableWidget { background-color: #18181b; color: #e4e4e7;
                           border: 1px solid #27272a; border-radius: 6px;
                           gridline-color: #27272a; }
            QTableWidget::item { padding: 4px 8px; }
            QTableWidget::item:selected { background-color: #27272a; }
            QHeaderView::section { background-color: #18181b; color: #a1a1aa;
                                   border: none; border-bottom: 1px solid #27272a;
                                   padding: 6px 8px; font-weight: 600; font-size: 12px; }

            /* Buttons */
            QPushButton#approve_btn { background-color: #166534; color: white;
                                      border: none; border-radius: 6px; padding: 10px 20px;
                                      font-weight: 600; font-size: 14px; }
            QPushButton#approve_btn:hover { background-color: #15803d; }

            QPushButton#reject_btn { background-color: #991b1b; color: white;
                                     border: none; border-radius: 6px; padding: 10px 20px;
                                     font-weight: 600; font-size: 14px; }
            QPushButton#reject_btn:hover { background-color: #b91c1c; }

            QPushButton#secondary_btn { background-color: #27272a; color: #e4e4e7;
                                        border: 1px solid #3f3f46; border-radius: 6px;
                                        padding: 8px 16px; font-weight: 500; }
            QPushButton#secondary_btn:hover { background-color: #3f3f46; }

            QPushButton#add_btn { background-color: #1e3a5f; color: #93c5fd;
                                  border: 1px solid #3b82f6; border-radius: 4px;
                                  padding: 4px 12px; font-size: 12px; font-weight: 600; }
            QPushButton#add_btn:hover { background-color: #1e40af; }

            QPushButton#remove_btn { background-color: #3b1818; color: #fca5a5;
                                     border: 1px solid #ef4444; border-radius: 4px;
                                     padding: 4px 12px; font-size: 12px; font-weight: 600; }
            QPushButton#remove_btn:hover { background-color: #991b1b; }

            /* Input */
            QLineEdit { background-color: #18181b; color: #e4e4e7;
                        border: 1px solid #3f3f46; border-radius: 4px;
                        padding: 6px 8px; }

            /* Warning banners */
            QFrame#warning_banner { border-radius: 6px; padding: 8px 12px; }

            /* Status badge */
            QLabel#status_badge { border-radius: 4px; padding: 2px 10px;
                                  font-weight: 600; font-size: 12px; }

            /* Scroll bars */
            QScrollBar:vertical { border: none; background: #18181b; width: 8px; }
            QScrollBar::handle:vertical { background: #3f3f46; border-radius: 4px; min-height: 20px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def _refresh_queue_list(self) -> None:
        """Refresh the queue list from the review queue."""
        items = self._queue.get_all()
        self._queue_table.setRowCount(len(items))

        pending = 0
        for row, item in enumerate(items):
            doc = item.document

            # Document name
            name_item = QTableWidgetItem(doc.source_filename or f"Doc {doc.id[:8]}")
            status_colour = STATUS_COLOURS.get(doc.review_status, "#6b7280")
            name_item.setForeground(QColor(status_colour))
            self._queue_table.setItem(row, 0, name_item)

            # Entity count
            count_item = QTableWidgetItem(str(doc.entity_count))
            count_item.setTextAlignment(Qt.AlignCenter)
            self._queue_table.setItem(row, 1, count_item)

            # Risk level indicator
            if doc.has_critical_entities:
                risk_text = "CRITICAL"
                risk_colour = "#ef4444"
            elif any(e.risk_level == RiskLevel.HIGH for e in doc.entities_detected):
                risk_text = "HIGH"
                risk_colour = "#f97316"
            else:
                risk_text = "MEDIUM"
                risk_colour = "#eab308"

            risk_item = QTableWidgetItem(risk_text)
            risk_item.setForeground(QColor(risk_colour))
            risk_item.setTextAlignment(Qt.AlignCenter)
            self._queue_table.setItem(row, 2, risk_item)

            if doc.review_status == ReviewStatus.PENDING:
                pending += 1

            # Store item reference
            name_item.setData(Qt.UserRole, item.id)

        self._stats_label.setText(
            f"{pending} pending  •  {len(items)} total"
        )

    def _on_queue_item_clicked(self, row: int, _col: int) -> None:
        """Handle queue item selection."""
        name_item = self._queue_table.item(row, 0)
        if not name_item:
            return

        item_id = name_item.data(Qt.UserRole)
        item = self._queue.get_item(item_id)
        if item:
            self._load_review_item(item)

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------

    def _load_review_item(self, item: ReviewItem) -> None:
        """Load a review item into the review area."""
        self._current_item = item
        doc = item.document

        # Update header
        self._doc_title.setText(doc.source_filename or f"Document {doc.id[:8]}")
        self._update_status_badge(doc.review_status)

        # Clear warnings area
        while self._warnings_area.count():
            child = self._warnings_area.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add warning banners
        if doc.has_critical_entities:
            self._add_warning_banner(
                "CRITICAL RISK: This document contains victim/perpetrator identifiers. "
                "Careful review is mandatory before any cloud export.",
                "#991b1b", "#fef2f2",
            )

        if not doc.validation_passed:
            self._add_warning_banner(
                "VALIDATION FAILED: Double-pass detection found potential residual PII "
                "in the anonymised output. Review carefully.",
                "#92400e", "#fffbeb",
            )

        low_conf = [e for e in doc.entities_detected if e.confidence < 0.7]
        if low_conf:
            self._add_warning_banner(
                f"LOW CONFIDENCE: {len(low_conf)} entities detected with confidence below 70%. "
                "These may be false positives or may have missed context. Check each one.",
                "#854d0e", "#fefce8",
            )

        # Load entity table
        self._load_entity_table(doc.entities_detected)

        # Load text views with highlighting
        self._load_original_text(doc)
        self._load_anonymised_text(doc)

        # Update action info
        self._action_info.setText(
            f"{doc.entity_count} entities detected  •  "
            f"Review status: {doc.review_status.value}"
        )

    def _update_status_badge(self, status: ReviewStatus) -> None:
        """Update the status badge colour and text."""
        colour = STATUS_COLOURS.get(status, "#6b7280")
        self._review_status_badge.setText(f"  {status.value.upper()}  ")
        self._review_status_badge.setStyleSheet(
            f"background-color: {colour}; color: white; "
            f"border-radius: 4px; padding: 2px 10px; font-weight: 600;"
        )

    def _add_warning_banner(self, text: str, fg_colour: str, bg_colour: str) -> None:
        """Add a coloured warning banner to the warnings area."""
        banner = QFrame()
        banner.setObjectName("warning_banner")
        banner.setStyleSheet(
            f"background-color: {bg_colour}; border: 1px solid {fg_colour}; "
            f"border-radius: 6px; padding: 8px 12px;"
        )
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(8, 6, 8, 6)

        icon = QLabel("⚠")
        icon.setStyleSheet(f"color: {fg_colour}; font-size: 16px; background: transparent;")
        layout.addWidget(icon)

        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {fg_colour}; font-size: 12px; background: transparent;")
        layout.addWidget(label, stretch=1)

        self._warnings_area.addWidget(banner)

    # ------------------------------------------------------------------
    # Entity table
    # ------------------------------------------------------------------

    def _load_entity_table(self, entities: list[PIIEntity]) -> None:
        """Populate the entity table with colour-coded rows."""
        self._entity_table.setRowCount(len(entities))

        for row, entity in enumerate(entities):
            # Category
            cat_item = QTableWidgetItem(entity.category.value.replace("_", " ").title())
            bg, fg, border = RISK_COLOURS.get(entity.risk_level, ("#f3f4f6", "#374151", "#d1d5db"))
            cat_item.setBackground(QColor(bg))
            cat_item.setForeground(QColor(fg))
            self._entity_table.setItem(row, 0, cat_item)

            # Original text
            text_item = QTableWidgetItem(entity.original_text[:60])
            text_item.setToolTip(entity.original_text)
            self._entity_table.setItem(row, 1, text_item)

            # Confidence
            conf_text = f"{entity.confidence:.0%}"
            conf_item = QTableWidgetItem(conf_text)
            conf_item.setTextAlignment(Qt.AlignCenter)
            if entity.confidence < 0.7:
                conf_item.setForeground(QColor("#ef4444"))
            elif entity.confidence < 0.85:
                conf_item.setForeground(QColor("#eab308"))
            else:
                conf_item.setForeground(QColor("#22c55e"))
            self._entity_table.setItem(row, 2, conf_item)

            # Risk level
            risk_item = QTableWidgetItem(entity.risk_level.value.upper())
            risk_item.setBackground(QColor(bg))
            risk_item.setForeground(QColor(fg))
            risk_item.setTextAlignment(Qt.AlignCenter)
            self._entity_table.setItem(row, 3, risk_item)

            # Detection layer
            layers = entity.metadata.get("detection_layers", [entity.detection_layer.value])
            layer_text = ", ".join(layers)
            layer_item = QTableWidgetItem(layer_text)
            layer_item.setForeground(QColor("#a1a1aa"))
            self._entity_table.setItem(row, 4, layer_item)

            # Store entity reference
            cat_item.setData(Qt.UserRole, entity.id)

    def _on_entity_clicked(self, row: int, _col: int) -> None:
        """Highlight the corresponding text when an entity is clicked."""
        if not self._current_item:
            return

        entities = self._current_item.document.entities_detected
        if row >= len(entities):
            return

        entity = entities[row]

        # Scroll original text to entity position
        cursor = self._original_text.textCursor()
        cursor.setPosition(min(entity.start, len(self._original_text.toPlainText())))
        self._original_text.setTextCursor(cursor)
        self._original_text.ensureCursorVisible()

    # ------------------------------------------------------------------
    # Text views with highlighting
    # ------------------------------------------------------------------

    def _load_original_text(self, doc: AnonymisedDocument) -> None:
        """Load original text with PII entities highlighted by risk level."""
        self._original_text.clear()
        text = doc.original_text

        if not text:
            self._original_text.setPlainText("(Original text not available)")
            return

        self._original_text.setPlainText(text)

        # Apply highlighting for each entity
        cursor = self._original_text.textCursor()
        for entity in sorted(doc.entities_detected, key=lambda e: e.start):
            fmt = QTextCharFormat()
            colour = RISK_HIGHLIGHT_COLOURS.get(entity.risk_level, QColor(200, 200, 200))
            fmt.setBackground(QBrush(colour))
            fmt.setForeground(QBrush(QColor("#1a1a1a")))
            fmt.setToolTip(
                f"{entity.category.value} | Risk: {entity.risk_level.value} | "
                f"Confidence: {entity.confidence:.0%}"
            )

            # Apply format to the entity span
            start = min(entity.start, len(text))
            end = min(entity.end, len(text))
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.KeepAnchor)
            cursor.mergeCharFormat(fmt)

        self._original_text.setTextCursor(QTextCursor(self._original_text.document()))

    def _load_anonymised_text(self, doc: AnonymisedDocument) -> None:
        """Load anonymised text with tokens highlighted."""
        self._anonymised_text.clear()
        text = doc.anonymised_text

        if not text:
            self._anonymised_text.setPlainText("(Anonymised text not available)")
            return

        self._anonymised_text.setPlainText(text)

        # Highlight all tokens [CATEGORY_NNN]
        import re
        token_pattern = re.compile(r"\[[A-Z_]+_\d{3}(?:,\s*[^\]]+)?\]")
        cursor = self._anonymised_text.textCursor()

        for match in token_pattern.finditer(text):
            fmt = QTextCharFormat()
            fmt.setBackground(QBrush(QColor("#3b82f6")))
            fmt.setForeground(QBrush(QColor("#ffffff")))
            fmt.setFontWeight(QFont.Bold)

            cursor.setPosition(match.start())
            cursor.setPosition(match.end(), QTextCursor.KeepAnchor)
            cursor.mergeCharFormat(fmt)

        self._anonymised_text.setTextCursor(QTextCursor(self._anonymised_text.document()))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_approve(self) -> None:
        """Approve the current document for export."""
        if not self._current_item:
            return

        reply = QMessageBox.question(
            self,
            "Approve Anonymisation",
            "Are you satisfied that all PII has been correctly identified and "
            "anonymised? Approving allows this document to be exported to "
            "cloud services.\n\n"
            "This action will be logged in the audit trail.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            doc = self._queue.approve(self._current_item.id, reviewer="user")
            if doc:
                self._update_status_badge(ReviewStatus.APPROVED)
                self.review_completed.emit(self._current_item.id, "approved")
                self._refresh_queue_list()
                self._action_info.setText("Approved — document cleared for export")

    def _on_reject(self) -> None:
        """Reject the document — blocks export."""
        if not self._current_item:
            return

        reply = QMessageBox.question(
            self,
            "Reject Anonymisation",
            "Rejecting will block this document from being exported to cloud "
            "services. The anonymisation will need to be re-done.\n\n"
            "Proceed with rejection?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            doc = self._queue.reject(
                self._current_item.id,
                reviewer="user",
                reason="Manual rejection during review",
            )
            if doc:
                self._update_status_badge(ReviewStatus.REJECTED)
                self.review_completed.emit(self._current_item.id, "rejected")
                self._refresh_queue_list()
                self._action_info.setText("Rejected — export blocked")

    def _on_add_manual_anonymisation(self) -> None:
        """Add a manual anonymisation from selected text or input field."""
        if not self._current_item:
            return

        # Try to get selected text from original view
        cursor = self._original_text.textCursor()
        selected = cursor.selectedText().strip()

        if not selected:
            # Fall back to manual input
            selected = self._manual_input.text().strip()

        if not selected:
            QMessageBox.information(
                self,
                "Add Anonymisation",
                "Select text in the original document view, or type the PII "
                "text in the input field, then click 'Add Anonymisation'.",
            )
            return

        # Create a manual entity
        start = self._current_item.document.original_text.find(selected)
        entity = PIIEntity(
            category=PIICategory.CUSTOM,
            original_text=selected,
            start=max(0, start),
            end=max(0, start) + len(selected) if start >= 0 else 0,
            confidence=1.0,
            detection_layer=PIIEntity.__dataclass_fields__["detection_layer"].default,
            risk_level=RiskLevel.HIGH,
            metadata={"source": "human_review", "reviewer": "user"},
        )
        # Override to HUMAN_REVIEW
        from src.anonymisation.models import DetectionLayer
        entity.detection_layer = DetectionLayer.HUMAN_REVIEW

        self._queue.add_entity(self._current_item.id, entity, reviewer="user")
        self._manual_input.clear()

        # Reload
        self._load_review_item(self._current_item)
        self._action_info.setText(f"Added manual anonymisation: '{selected[:40]}'")

    def _on_remove_entity(self) -> None:
        """Remove the selected entity (false positive)."""
        if not self._current_item:
            return

        row = self._entity_table.currentRow()
        if row < 0:
            QMessageBox.information(
                self,
                "Remove Entity",
                "Select an entity row in the table to remove it as a false positive.",
            )
            return

        entities = self._current_item.document.entities_detected
        if row >= len(entities):
            return

        entity = entities[row]
        reply = QMessageBox.question(
            self,
            "Remove Entity",
            f"Remove '{entity.original_text[:50]}' ({entity.category.value}) "
            "as a false positive?\n\nThis entity will not be anonymised in the export.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self._queue.remove_entity(self._current_item.id, entity.id, reviewer="user")
            self._load_review_item(self._current_item)
            self._action_info.setText(f"Removed entity: '{entity.original_text[:40]}'")

    def _on_selective_deanonymise(self) -> None:
        """Allow the user to selectively reveal specific tokens."""
        if not self._current_item:
            return

        # Get selected token from anonymised view
        cursor = self._anonymised_text.textCursor()
        selected = cursor.selectedText().strip()

        if not selected or not selected.startswith("["):
            QMessageBox.information(
                self,
                "Selective De-anonymise",
                "Select a token (e.g. [PERSON_NAME_001]) in the anonymised text view "
                "to reveal its original value. This is for fine-tuning only — "
                "the revealed value will NOT be exported.",
            )
            return

        # Show the original value
        doc = self._current_item.document
        for token in doc.tokens_applied:
            if token.anonymised_value in selected or selected in token.anonymised_value:
                QMessageBox.information(
                    self,
                    "Token Revealed (LOCAL ONLY)",
                    f"Token: {token.anonymised_value}\n"
                    f"Original: {token.original_value}\n"
                    f"Category: {token.category.value}\n\n"
                    "This information is for review purposes only and will "
                    "NOT be included in any cloud export.",
                )
                return

        QMessageBox.information(
            self,
            "Token Not Found",
            f"Could not find mapping for '{selected}'. "
            "Ensure you select the complete token including brackets.",
        )

    # ------------------------------------------------------------------
    # Public API for external use
    # ------------------------------------------------------------------

    def add_document_for_review(self, doc: AnonymisedDocument) -> None:
        """Add a document to the review queue and refresh the list.

        Args:
            doc: AnonymisedDocument to enqueue for review.
        """
        self._queue.enqueue(doc)
        self._refresh_queue_list()

    def load_next_pending(self) -> bool:
        """Load the next pending review item.

        Returns:
            True if a pending item was found and loaded.
        """
        item = self._queue.get_next()
        if item:
            self._load_review_item(item)
            return True
        return False
