"""Entity management panel for viewing and editing case graph entities."""

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.graph.storage import GraphStore
from src.graph.models import GraphNode, GraphUpdate


class EntityListItem(QFrame):
    """A single entity item in the list."""

    clicked = Signal(str)  # node_id
    edit_requested = Signal(str)  # node_id

    def __init__(self, node: GraphNode, parent=None):
        super().__init__(parent)
        self.node = node
        self.setStyleSheet("""
            EntityListItem {
                background-color: #1a1a20;
                border: 1px solid #27272a;
                border-radius: 8px;
                padding: 8px;
            }
            EntityListItem:hover {
                background-color: #1e1e24;
            }
        """)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        # Header row: icon + name + type badge
        header = QHBoxLayout()

        # Type indicator - simple dot with color
        type_colors = {
            "party": "#4ade80",
            "document": "#7ba3b8",
            "event": "#8b7cf6",
            "statute": "#a07db8",
            "issue": "#f87171",
            "chunk": "#71717a",
        }
        color = type_colors.get(node.node_type, "#71717a")
        icon_label = QLabel("●")
        icon_label.setStyleSheet(f"font-size: 12pt; color: {color};")
        header.addWidget(icon_label)

        name_label = QLabel(node.label)
        name_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        header.addWidget(name_label)

        header.addStretch()

        type_badge = QLabel(node.node_type.title())
        type_badge.setStyleSheet("""
            background-color: #1e1e24;
            color: #a1a1aa;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 9pt;
        """)
        header.addWidget(type_badge)

        edit_btn = QPushButton("Edit")
        edit_btn.setMaximumWidth(50)
        edit_btn.clicked.connect(lambda: self.edit_requested.emit(node.node_id))
        header.addWidget(edit_btn)

        layout.addLayout(header)

        # Metadata row
        if node.metadata:
            meta_text = " | ".join(f"{k}: {v}" for k, v in list(node.metadata.items())[:3])
            meta_label = QLabel(meta_text)
            meta_label.setStyleSheet("color: #71717a; font-size: 9pt;")
            meta_label.setWordWrap(True)
            layout.addWidget(meta_label)

    def mousePressEvent(self, event):
        self.clicked.emit(self.node.node_id)
        super().mousePressEvent(event)


class EntityEditDialog(QDialog):
    """Dialog for editing entity details."""

    def __init__(self, node: GraphNode, parent=None):
        super().__init__(parent)
        self.node = node
        self.setWindowTitle(f"Edit Entity: {node.label}")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        form = QFormLayout()

        # Label/Name
        self.name_edit = QLineEdit(node.label)
        form.addRow("Name:", self.name_edit)

        # Type (read-only for now)
        self.type_label = QLabel(node.node_type)
        form.addRow("Type:", self.type_label)

        # Metadata display
        meta_text = "\n".join(f"{k}: {v}" for k, v in node.metadata.items())
        self.meta_edit = QPlainTextEdit(meta_text)
        self.meta_edit.setMaximumHeight(100)
        self.meta_edit.setPlaceholderText("key: value (one per line)")
        form.addRow("Metadata:", self.meta_edit)

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_updated_node(self) -> GraphNode:
        """Get the node with updated values."""
        # Parse metadata from text
        meta_text = self.meta_edit.toPlainText()
        new_meta = {}
        for line in meta_text.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                new_meta[key.strip()] = value.strip()

        return GraphNode(
            node_id=self.node.node_id,
            label=self.name_edit.text().strip() or self.node.label,
            node_type=self.node.node_type,
            metadata=new_meta,
        )


class EntityPanel(QWidget):
    """Panel for managing case graph entities."""

    entity_selected = Signal(str)  # node_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.graph_store = GraphStore()
        self._all_nodes: list[GraphNode] = []
        self._filtered_nodes: list[GraphNode] = []

        self._setup_ui()
        self.refresh_entities()

    def _setup_ui(self):
        """Set up UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header
        header = QHBoxLayout()
        title = QLabel("🔗 Case Graph Entities")
        title.setStyleSheet("font-size: 12pt; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()

        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.setToolTip("Refresh entities")
        refresh_btn.clicked.connect(self.refresh_entities)
        header.addWidget(refresh_btn)

        layout.addLayout(header)

        # Stats bar
        self.stats_label = QLabel("Loading...")
        self.stats_label.setStyleSheet("color: #71717a; font-size: 9pt;")
        layout.addWidget(self.stats_label)

        # Filters
        filter_layout = QHBoxLayout()

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search entities...")
        self.search_edit.textChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.search_edit)

        self.type_filter = QComboBox()
        self.type_filter.addItem("All Types", None)
        for node_type in ["party", "document", "event", "statute", "issue"]:
            self.type_filter.addItem(node_type.title(), node_type)
        self.type_filter.currentIndexChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.type_filter)

        layout.addLayout(filter_layout)

        # Pending reviews section
        self.pending_section = QFrame()
        self.pending_section.setStyleSheet("""
            QFrame {
                background-color: rgba(212, 165, 74, 0.1);
                border: 1px solid #fbbf24;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        pending_layout = QVBoxLayout(self.pending_section)
        pending_layout.setContentsMargins(8, 8, 8, 8)

        pending_header = QHBoxLayout()
        pending_label = QLabel("⏳ Pending Review")
        pending_label.setStyleSheet("font-weight: bold;")
        pending_header.addWidget(pending_label)
        pending_header.addStretch()

        self.pending_count_label = QLabel("0")
        pending_header.addWidget(self.pending_count_label)

        review_btn = QPushButton("Review All")
        review_btn.clicked.connect(self._review_pending)
        pending_header.addWidget(review_btn)

        pending_layout.addLayout(pending_header)
        layout.addWidget(self.pending_section)

        # Entity list (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.entity_list_widget = QWidget()
        self.entity_list_layout = QVBoxLayout(self.entity_list_widget)
        self.entity_list_layout.setContentsMargins(0, 0, 0, 0)
        self.entity_list_layout.setSpacing(6)
        self.entity_list_layout.addStretch()

        scroll.setWidget(self.entity_list_widget)
        layout.addWidget(scroll, stretch=1)

        # Action buttons
        action_layout = QHBoxLayout()

        accept_all_btn = QPushButton("Accept All Pending")
        accept_all_btn.clicked.connect(self._accept_all_pending)
        action_layout.addWidget(accept_all_btn)

        clear_btn = QPushButton("Clear Rejected")
        clear_btn.clicked.connect(self._clear_rejected)
        action_layout.addWidget(clear_btn)

        layout.addLayout(action_layout)

    def refresh_entities(self):
        """Reload entities from graph store."""
        self._all_nodes = []

        # Load accepted graph nodes
        for record in self.graph_store.load_graph_records():
            nodes = record.get("nodes", [])
            for node_data in nodes:
                node = GraphNode(**node_data)
                self._all_nodes.append(node)

        # Count pending
        pending_files = self.graph_store.list_pending()
        pending_count = len(pending_files)

        # Update UI
        self.pending_count_label.setText(str(pending_count))
        self.pending_section.setVisible(pending_count > 0)

        # Update stats
        type_counts: dict[str, int] = {}
        for node in self._all_nodes:
            type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1

        stats_parts = [f"{count} {ntype}s" for ntype, count in sorted(type_counts.items())]
        self.stats_label.setText(f"{len(self._all_nodes)} total | " + " | ".join(stats_parts))

        self._apply_filter()

    def _apply_filter(self):
        """Apply search and type filter to entity list."""
        search_text = self.search_edit.text().lower().strip()
        type_filter = self.type_filter.currentData()

        self._filtered_nodes = []
        for node in self._all_nodes:
            # Type filter
            if type_filter and node.node_type != type_filter:
                continue
            # Search filter
            if search_text:
                searchable = f"{node.label} {node.node_type}".lower()
                searchable += " ".join(str(v) for v in node.metadata.values()).lower()
                if search_text not in searchable:
                    continue
            self._filtered_nodes.append(node)

        self._rebuild_list()

    def _rebuild_list(self):
        """Rebuild the entity list widget."""
        # Clear existing items
        while self.entity_list_layout.count() > 1:  # Keep stretch
            item = self.entity_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add filtered nodes
        for node in self._filtered_nodes[:100]:  # Limit for performance
            item = EntityListItem(node)
            item.clicked.connect(self._on_entity_clicked)
            item.edit_requested.connect(self._on_edit_entity)
            self.entity_list_layout.insertWidget(
                self.entity_list_layout.count() - 1, item
            )

        # Show count if truncated
        if len(self._filtered_nodes) > 100:
            more_label = QLabel(f"... and {len(self._filtered_nodes) - 100} more")
            more_label.setStyleSheet("color: #71717a; padding: 8px;")
            self.entity_list_layout.insertWidget(
                self.entity_list_layout.count() - 1, more_label
            )

    def _on_entity_clicked(self, node_id: str):
        """Handle entity click."""
        self.entity_selected.emit(node_id)

    def _on_edit_entity(self, node_id: str):
        """Open edit dialog for entity."""
        node = next((n for n in self._all_nodes if n.node_id == node_id), None)
        if not node:
            return

        dialog = EntityEditDialog(node, self)
        if dialog.exec() == QDialog.Accepted:
            # Note: Full editing would require modifying the graph store
            # For now, just show a message
            QMessageBox.information(
                self,
                "Entity Updated",
                "Note: Entity editing is not yet fully implemented.\n"
                "Changes are displayed but not persisted."
            )
            self.refresh_entities()

    def _review_pending(self):
        """Open pending review dialog."""
        from src.ui.graph_review_dialog import GraphReviewDialog

        pending_files = self.graph_store.list_pending()
        if not pending_files:
            QMessageBox.information(self, "No Pending", "No entities pending review.")
            return

        dialog = GraphReviewDialog(self.graph_store, self)
        dialog.exec()
        self.refresh_entities()

    def _accept_all_pending(self):
        """Accept all pending graph updates."""
        pending_files = self.graph_store.list_pending()
        if not pending_files:
            return

        reply = QMessageBox.question(
            self,
            "Accept All Pending",
            f"Accept {len(pending_files)} pending graph updates?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        for pending_file in pending_files:
            doc_id = pending_file.stem
            update = self.graph_store.load_pending(doc_id)
            if update:
                self.graph_store.append_to_graph(update)
                self.graph_store.delete_pending(doc_id)

        self.refresh_entities()

    def _clear_rejected(self):
        """Clear all pending (rejected) updates."""
        pending_files = self.graph_store.list_pending()
        if not pending_files:
            return

        reply = QMessageBox.question(
            self,
            "Clear Pending",
            f"Delete {len(pending_files)} pending graph updates?\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        for pending_file in pending_files:
            pending_file.unlink()

        self.refresh_entities()

