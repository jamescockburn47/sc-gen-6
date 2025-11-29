"""Timeline panel for viewing chronological events from the case graph."""

from datetime import date, datetime
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.graph.storage import GraphStore
from src.graph.models import GraphNode


class TimelineEventItem(QFrame):
    """A single event in the timeline."""

    clicked = Signal(str)  # node_id

    def __init__(self, node: GraphNode, parent=None):
        super().__init__(parent)
        self.node = node
        self.setStyleSheet("""
            TimelineEventItem {
                background-color: #1a1a20;
                border-left: 3px solid #8b7cf6;
                margin-left: 20px;
                padding: 8px 12px;
            }
            TimelineEventItem:hover {
                background-color: #1e1e24;
            }
        """)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # Date and label
        header = QHBoxLayout()

        event_date = node.metadata.get("date", "Unknown date")
        date_label = QLabel(event_date)
        date_label.setStyleSheet("font-weight: 600; color: #8b7cf6;")
        header.addWidget(date_label)

        header.addStretch()

        # Source documents (new field from TimelineEvent)
        source_docs = node.metadata.get("source_documents", [])
        if not source_docs:
            # Fallback to old source_file field
            source = node.metadata.get("source_file", "")
            if source:
                source_docs = [source]
        
        if source_docs:
            sources_text = ", ".join(source_docs[:2])  # Show first 2
            if len(source_docs) > 2:
                sources_text += f" +{len(source_docs) - 2} more"
            source_label = QLabel(f"📄 {sources_text}")
            source_label.setStyleSheet("color: #71717a; font-size: 9pt;")
            source_label.setToolTip("Source documents:\n" + "\n".join(source_docs))
            header.addWidget(source_label)

        layout.addLayout(header)

        # Event description
        desc_label = QLabel(node.label)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-size: 10pt; color: #f4f4f5;")
        layout.addWidget(desc_label)

    def mousePressEvent(self, event):
        self.clicked.emit(self.node.node_id)
        super().mousePressEvent(event)


class YearMarker(QFrame):
    """Year divider in timeline."""

    def __init__(self, year: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            YearMarker {
                background-color: transparent;
                padding: 8px 0;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)

        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setStyleSheet("background-color: #27272a;")
        layout.addWidget(line1)

        year_label = QLabel(f" {year} ")
        year_label.setStyleSheet("font-weight: 600; font-size: 12pt; color: #a1a1aa;")
        layout.addWidget(year_label)

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("background-color: #27272a;")
        layout.addWidget(line2)


class TimelinePanel(QWidget):
    """Panel for viewing case timeline."""

    event_selected = Signal(str)  # node_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.graph_store = GraphStore()
        self._all_events: list[GraphNode] = []
        self._filtered_events: list[GraphNode] = []

        self._setup_ui()
        self.refresh_timeline()

    def _setup_ui(self):
        """Set up UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header
        header = QHBoxLayout()
        title = QLabel("Case Timeline")
        title.setStyleSheet("font-size: 12pt; font-weight: bold; color: #f4f4f5;")
        header.addWidget(title)
        header.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setProperty("styleClass", "ghost")
        refresh_btn.setToolTip("Refresh timeline")
        refresh_btn.clicked.connect(self.refresh_timeline)
        header.addWidget(refresh_btn)

        layout.addLayout(header)

        # Stats
        self.stats_label = QLabel("Loading...")
        self.stats_label.setStyleSheet("color: #71717a; font-size: 9pt;")
        layout.addWidget(self.stats_label)

        # Date range filter
        filter_layout = QHBoxLayout()

        filter_layout.addWidget(QLabel("From:"))
        self.date_from = QDateEdit()
        self.date_from.setCalendarPopup(True)
        self.date_from.setDate(date(2020, 1, 1))
        self.date_from.setDisplayFormat("dd/MM/yyyy")
        self.date_from.dateChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.date_from)

        filter_layout.addWidget(QLabel("To:"))
        self.date_to = QDateEdit()
        self.date_to.setCalendarPopup(True)
        self.date_to.setDate(date.today())
        self.date_to.setDisplayFormat("dd/MM/yyyy")
        self.date_to.dateChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.date_to)

        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # Timeline list (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.timeline_widget = QWidget()
        self.timeline_layout = QVBoxLayout(self.timeline_widget)
        self.timeline_layout.setContentsMargins(0, 0, 0, 0)
        self.timeline_layout.setSpacing(0)
        self.timeline_layout.addStretch()

        scroll.setWidget(self.timeline_widget)
        layout.addWidget(scroll, stretch=1)

    def refresh_timeline(self):
        """Reload events from graph store."""
        self._all_events = []

        # Load accepted graph nodes, filter for events
        for record in self.graph_store.load_graph_records():
            nodes = record.get("nodes", [])
            for node_data in nodes:
                node = GraphNode(**node_data)
                if node.node_type == "event":
                    self._all_events.append(node)

        # Sort by date
        self._all_events.sort(key=lambda n: self._parse_date(n.metadata.get("date", "")))

        # Update stats
        if self._all_events:
            dates = [self._parse_date(n.metadata.get("date", "")) for n in self._all_events]
            valid_dates = [d for d in dates if d]
            if valid_dates:
                min_date = min(valid_dates)
                max_date = max(valid_dates)
                self.stats_label.setText(
                    f"{len(self._all_events)} events | {min_date.strftime('%b %Y')} - {max_date.strftime('%b %Y')}"
                )
            else:
                self.stats_label.setText(f"{len(self._all_events)} events")
        else:
            self.stats_label.setText("No events found")

        self._apply_filter()

    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse date from various formats."""
        if not date_str:
            return None

        # Try ISO format first
        for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d %B %Y", "%d %b %Y"]:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        return None

    def _apply_filter(self):
        """Apply date range filter."""
        from_date = self.date_from.date().toPython()
        to_date = self.date_to.date().toPython()

        self._filtered_events = []
        for event in self._all_events:
            event_date = self._parse_date(event.metadata.get("date", ""))
            if event_date:
                if from_date <= event_date <= to_date:
                    self._filtered_events.append(event)
            else:
                # Include events without parseable dates
                self._filtered_events.append(event)

        self._rebuild_timeline()

    def _rebuild_timeline(self):
        """Rebuild the timeline widget."""
        # Clear existing items
        while self.timeline_layout.count() > 1:
            item = self.timeline_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._filtered_events:
            no_events = QLabel("No events in selected date range")
            no_events.setStyleSheet("color: #71717a; padding: 20px;")
            no_events.setAlignment(Qt.AlignCenter)
            self.timeline_layout.insertWidget(0, no_events)
            return

        # Group by year
        current_year = None
        for event in reversed(self._filtered_events):  # Most recent first
            event_date = self._parse_date(event.metadata.get("date", ""))
            year = event_date.year if event_date else "Unknown"

            if year != current_year:
                current_year = year
                year_marker = YearMarker(str(year))
                self.timeline_layout.insertWidget(
                    self.timeline_layout.count() - 1, year_marker
                )

            item = TimelineEventItem(event)
            item.clicked.connect(self._on_event_clicked)
            self.timeline_layout.insertWidget(
                self.timeline_layout.count() - 1, item
            )

    def _on_event_clicked(self, node_id: str):
        """Handle event click."""
        self.event_selected.emit(node_id)
