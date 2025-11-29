"""Simple dialog to approve or discard graph updates."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from src.graph.review import GraphReviewManager


class GraphReviewDialog(QDialog):
    """Allows users to inspect and approve queued graph updates."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = GraphReviewManager()
        self.setWindowTitle("Case Graph Review")
        self.resize(720, 500)
        self._setup_ui()
        self._load_pending()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        info_label = QLabel(
            "Select a document to see the proposed nodes/edges. "
            "Approving applies it to the global case graph; discarding removes it."
        )
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)

        split_layout = QHBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self._on_selection_changed)
        split_layout.addWidget(self.list_widget, 1)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        split_layout.addWidget(self.details_text, 2)
        main_layout.addLayout(split_layout)

        button_layout = QHBoxLayout()
        self.approve_button = QPushButton("Approve Selected")
        self.approve_button.clicked.connect(self._approve_selected)
        button_layout.addWidget(self.approve_button)

        self.discard_button = QPushButton("Discard Selected")
        self.discard_button.clicked.connect(self._discard_selected)
        button_layout.addWidget(self.discard_button)
        
        # Bulk actions
        approve_all_button = QPushButton("Approve All")
        approve_all_button.setStyleSheet("background-color: #4ade80; color: #0f0f12; font-weight: 600;")
        approve_all_button.clicked.connect(self._approve_all)
        button_layout.addWidget(approve_all_button)
        
        discard_all_button = QPushButton("Discard All")
        discard_all_button.setStyleSheet("color: #f87171;")
        discard_all_button.clicked.connect(self._discard_all)
        button_layout.addWidget(discard_all_button)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._load_pending)
        button_layout.addWidget(refresh_button)

        button_layout.addStretch()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        main_layout.addLayout(button_layout)

    def _load_pending(self):
        self.list_widget.clear()
        for doc_id in self.manager.pending_documents():
            update = self.manager.fetch_update(doc_id)
            if not update:
                continue
            item = QListWidgetItem(f"{update.document_name} ({doc_id})")
            item.setData(Qt.UserRole, doc_id)
            self.list_widget.addItem(item)

        if self.list_widget.count() == 0:
            self.details_text.setPlainText("No pending updates.")

    def _on_selection_changed(self, current: QListWidgetItem, _previous: QListWidgetItem):
        if not current:
            self.details_text.clear()
            return
        doc_id = current.data(Qt.UserRole)
        update = self.manager.fetch_update(doc_id)
        if not update:
            self.details_text.setPlainText("Unable to load update.")
            return

        lines = [
            f"Document: {update.document_name}",
            f"Nodes: {len(update.nodes)}",
            f"Edges: {len(update.edges)}",
            "",
            "Parties:",
        ]
        parties = [n for n in update.nodes if n.node_type == "party"]
        if not parties:
            lines.append("  (none detected)")
        else:
            for party in parties:
                lines.append(f"  • {party.label}")

        events = [n for n in update.nodes if n.node_type == "event"]
        lines.append("")
        lines.append("Events:")
        if not events:
            lines.append("  (none detected)")
        else:
            for event in events:
                date = event.metadata.get("date", "")
                lines.append(f"  • {event.label} {date}")

        if update.notes:
            lines.append("")
            lines.append(f"Notes: {update.notes}")

        self.details_text.setPlainText("\n".join(lines))

    def _approve_selected(self):
        item = self.list_widget.currentItem()
        if not item:
            return
        doc_id = item.data(Qt.UserRole)
        if self.manager.approve_update(doc_id):
            QMessageBox.information(self, "Applied", "Graph update applied.")
            self._load_pending()
        else:
            QMessageBox.warning(self, "Error", "Could not apply update.")

    def _discard_selected(self):
        item = self.list_widget.currentItem()
        if not item:
            return
        doc_id = item.data(Qt.UserRole)
        if self.manager.discard_update(doc_id):
            QMessageBox.information(self, "Discarded", "Update discarded.")
            self._load_pending()
        else:
            QMessageBox.warning(self, "Error", "Could not discard update.")

    def _approve_all(self):
        """Approve all pending updates."""
        pending = self.manager.pending_documents()
        if not pending:
            QMessageBox.information(self, "Nothing to Approve", "No pending updates.")
            return
        
        reply = QMessageBox.question(
            self,
            "Approve All",
            f"This will approve and apply {len(pending)} graph updates.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply != QMessageBox.Yes:
            return
        
        approved = 0
        failed = 0
        for doc_id in pending:
            if self.manager.approve_update(doc_id):
                approved += 1
            else:
                failed += 1
        
        self._load_pending()
        QMessageBox.information(
            self, 
            "Bulk Approve Complete", 
            f"Approved: {approved}\nFailed: {failed}"
        )

    def _discard_all(self):
        """Discard all pending updates."""
        pending = self.manager.pending_documents()
        if not pending:
            QMessageBox.information(self, "Nothing to Discard", "No pending updates.")
            return
        
        reply = QMessageBox.warning(
            self,
            "Discard All",
            f"This will PERMANENTLY discard {len(pending)} pending updates.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        discarded = 0
        for doc_id in pending:
            if self.manager.discard_update(doc_id):
                discarded += 1
        
        self._load_pending()
        QMessageBox.information(self, "Bulk Discard Complete", f"Discarded: {discarded}")

