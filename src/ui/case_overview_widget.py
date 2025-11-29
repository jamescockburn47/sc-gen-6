"""Case overview widget.

Displays high-level case overview including key parties, dates, issues,
and document statistics.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QFrame,
    QGroupBox,
)

from src.generation.case_overview_generator import CaseOverview, CaseOverviewGenerator

logger = logging.getLogger(__name__)


class CaseOverviewWidget(QWidget):
    """Widget to display case overview."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.overview: CaseOverview | None = None
        self._setup_ui()
        self._load_overview()
    
    def _setup_ui(self):
        """Set up the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header with refresh button
        header_layout = QHBoxLayout()
        
        title = QLabel("Case Overview")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #f1f5f9;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        self.refresh_btn = QPushButton("🔄 Regenerate")
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b5cf6;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7c3aed;
            }
        """)
        self.refresh_btn.clicked.connect(self._regenerate_overview)
        header_layout.addWidget(self.refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent; border: none;")
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setSpacing(20)
        
        scroll.setWidget(self.content_widget)
        layout.addWidget(scroll)
    
    def _load_overview(self):
        """Load existing overview from file."""
        try:
            generator = CaseOverviewGenerator()
            self.overview = generator.load_overview()
            
            if self.overview:
                self._display_overview()
            else:
                self._show_empty_state()
                
        except Exception as e:
            logger.error(f"Error loading case overview: {e}")
            self._show_empty_state()
    
    def _show_empty_state(self):
        """Show empty state when no overview exists."""
        # Clear existing content
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        empty_label = QLabel(
            "No case overview available.\n\n"
            "Click 'Regenerate' or go to Settings → Background Tasks to generate one."
        )
        empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_label.setStyleSheet("color: #a1a1aa; font-size: 14px; padding: 40px;")
        self.content_layout.addWidget(empty_label)
    
    def _display_overview(self):
        """Display the case overview."""
        if not self.overview:
            return
        
        # Clear existing content
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Case Title
        title = QLabel(self.overview.case_title)
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #f1f5f9; margin-bottom: 10px;")
        title.setWordWrap(True)
        self.content_layout.addWidget(title)
        
        # Case Summary
        summary_group = self._create_group("Summary")
        summary_label = QLabel(self.overview.case_summary)
        summary_label.setWordWrap(True)
        summary_label.setStyleSheet("color: #e4e4e7; line-height: 1.6;")
        summary_group.layout().addWidget(summary_label)
        self.content_layout.addWidget(summary_group)
        
        # Key Parties
        if self.overview.key_parties:
            parties_group = self._create_group("Key Parties")
            for party in self.overview.key_parties:
                party_widget = self._create_party_widget(party)
                parties_group.layout().addWidget(party_widget)
            self.content_layout.addWidget(parties_group)
        
        # Key Dates
        if self.overview.key_dates:
            dates_group = self._create_group("Key Dates")
            for date_info in self.overview.key_dates:
                date_widget = self._create_date_widget(date_info)
                dates_group.layout().addWidget(date_widget)
            self.content_layout.addWidget(dates_group)
        
        # Key Issues
        if self.overview.key_issues:
            issues_group = self._create_group("Key Issues")
            for idx, issue in enumerate(self.overview.key_issues, 1):
                issue_label = QLabel(f"{idx}. {issue}")
                issue_label.setWordWrap(True)
                issue_label.setStyleSheet("color: #e4e4e7; padding: 5px 0;")
                issues_group.layout().addWidget(issue_label)
            self.content_layout.addWidget(issues_group)
        
        # Document Statistics
        stats_group = self._create_group("Document Statistics")
        stats_text = f"Total Documents: {self.overview.document_count}\n\n"
        stats_text += "Document Types:\n"
        for doc_type, count in self.overview.document_types.items():
            stats_text += f"  • {doc_type}: {count}\n"
        
        stats_label = QLabel(stats_text)
        stats_label.setStyleSheet("color: #e4e4e7; font-family: monospace;")
        stats_group.layout().addWidget(stats_label)
        self.content_layout.addWidget(stats_group)
        
        # Metadata
        meta_text = f"Generated: {self.overview.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        meta_text += f"Model: {self.overview.model_used}\n"
        meta_text += f"Generation Time: {self.overview.generation_time_seconds:.2f}s"
        
        meta_label = QLabel(meta_text)
        meta_label.setStyleSheet("color: #71717a; font-size: 11px; margin-top: 20px;")
        self.content_layout.addWidget(meta_label)
        
        self.content_layout.addStretch()
    
    def _create_group(self, title: str) -> QGroupBox:
        """Create a styled group box."""
        group = QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox {
                background-color: #18181b;
                border: 1px solid #27272a;
                border-radius: 8px;
                padding: 15px;
                margin-top: 10px;
                font-weight: bold;
                color: #f1f5f9;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        return group
    
    def _create_party_widget(self, party: dict) -> QWidget:
        """Create widget for a key party."""
        widget = QFrame()
        widget.setStyleSheet("""
            QFrame {
                background-color: #27272a;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)
        
        # Name and role
        name_label = QLabel(f"<b>{party['name']}</b> - {party['role']}")
        name_label.setStyleSheet("color: #f1f5f9;")
        layout.addWidget(name_label)
        
        # Description
        if party.get('description'):
            desc_label = QLabel(party['description'])
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #a1a1aa; font-size: 12px;")
            layout.addWidget(desc_label)
        
        # Source documents
        if party.get('source_documents'):
            sources = ", ".join(party['source_documents'])
            source_label = QLabel(f"📄 Sources: {sources}")
            source_label.setStyleSheet("color: #71717a; font-size: 11px; font-style: italic;")
            source_label.setWordWrap(True)
            layout.addWidget(source_label)
        
        return widget
    
    def _create_date_widget(self, date_info: dict) -> QWidget:
        """Create widget for a key date."""
        widget = QFrame()
        widget.setStyleSheet("""
            QFrame {
                background-color: #27272a;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)
        
        # Date and event
        date_str = date_info.get('date', 'Unknown date')
        event_label = QLabel(f"<b>{date_str}</b> - {date_info['event']}")
        event_label.setStyleSheet("color: #f1f5f9;")
        event_label.setWordWrap(True)
        layout.addWidget(event_label)
        
        # Significance
        if date_info.get('significance'):
            sig_label = QLabel(date_info['significance'])
            sig_label.setWordWrap(True)
            sig_label.setStyleSheet("color: #a1a1aa; font-size: 12px;")
            layout.addWidget(sig_label)
        
        # Source documents
        if date_info.get('source_documents'):
            sources = ", ".join(date_info['source_documents'])
            source_label = QLabel(f"📄 Sources: {sources}")
            source_label.setStyleSheet("color: #71717a; font-size: 11px; font-style: italic;")
            source_label.setWordWrap(True)
            layout.addWidget(source_label)
        
        return widget
    
    def _regenerate_overview(self):
        """Regenerate the case overview."""
        from src.ui.background_task_dialog import BackgroundTaskDialog
        
        dialog = BackgroundTaskDialog(
            incremental=False,
            single_task="case_overview_generation",
            parent=self
        )
        dialog.finished.connect(self._on_regeneration_complete)
        dialog.show()
        dialog.start_tasks()
    
    def _on_regeneration_complete(self):
        """Handle overview regeneration completion."""
        self._load_overview()
