"""Document statistics widget showing ingestion, indexing, and graph metrics."""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)

from src.config_loader import get_settings
from src.documents.catalog import DocumentCatalog
from src.retrieval.vector_store import VectorStore
from src.retrieval.fts5_index import FTS5IndexCompat
from src.graph.storage import GraphStore


class StatCard(QFrame):
    """A card displaying a single statistic."""

    def __init__(
        self,
        title: str,
        value: str = "0",
        subtitle: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setStyleSheet("""
            StatCard {
                background-color: #1a1a20;
                border: 1px solid #27272a;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        # Title
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #a1a1aa; font-size: 10pt;")
        layout.addWidget(self.title_label)

        # Value - using primary gold color for all values
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("color: #8b7cf6; font-size: 20pt; font-weight: bold;")
        layout.addWidget(self.value_label)

        # Subtitle (optional details)
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setStyleSheet("color: #71717a; font-size: 9pt;")
        self.subtitle_label.setWordWrap(True)
        layout.addWidget(self.subtitle_label)

    def update_value(self, value: str, subtitle: str = ""):
        """Update the displayed value and subtitle."""
        self.value_label.setText(value)
        if subtitle:
            self.subtitle_label.setText(subtitle)


class DocumentStatsWidget(QWidget):
    """Widget displaying document store statistics."""

    refresh_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = get_settings()
        self.catalog = DocumentCatalog()
        self._setup_ui()

        # Auto-refresh timer (every 30 seconds)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_stats)
        self.refresh_timer.start(30000)

        # Initial load
        self.refresh_stats()

    def _setup_ui(self):
        """Set up UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Header with refresh button
        header_layout = QHBoxLayout()
        title = QLabel("Store Statistics")
        title.setStyleSheet("font-size: 12pt; font-weight: bold; color: #f4f4f5;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setProperty("styleClass", "ghost")
        refresh_btn.setToolTip("Refresh statistics")
        refresh_btn.clicked.connect(self.refresh_stats)
        header_layout.addWidget(refresh_btn)

        layout.addLayout(header_layout)

        # Document stats section
        docs_section = QLabel("Documents")
        docs_section.setStyleSheet("font-weight: 600; color: #a1a1aa; font-size: 10pt; letter-spacing: 0.5px;")
        layout.addWidget(docs_section)

        docs_grid = QGridLayout()
        docs_grid.setSpacing(8)

        self.total_docs_card = StatCard("Total Documents", "0", "")
        docs_grid.addWidget(self.total_docs_card, 0, 0)

        self.indexed_docs_card = StatCard("Fully Indexed", "0", "")
        docs_grid.addWidget(self.indexed_docs_card, 0, 1)

        self.pending_docs_card = StatCard("Pending Index", "0", "")
        docs_grid.addWidget(self.pending_docs_card, 1, 0)

        self.errored_docs_card = StatCard("Errors", "0", "")
        docs_grid.addWidget(self.errored_docs_card, 1, 1)

        layout.addLayout(docs_grid)

        # Index stats section
        index_section = QLabel("Search Index")
        index_section.setStyleSheet("font-weight: 600; color: #a1a1aa; font-size: 10pt; letter-spacing: 0.5px;")
        layout.addWidget(index_section)

        index_grid = QGridLayout()
        index_grid.setSpacing(8)

        self.chunks_card = StatCard("Total Chunks", "0", "")
        index_grid.addWidget(self.chunks_card, 0, 0)

        self.bm25_card = StatCard("BM25 Chunks", "0", "")
        index_grid.addWidget(self.bm25_card, 0, 1)

        layout.addLayout(index_grid)

        # Graph stats section
        graph_section = QLabel("Case Graph")
        graph_section.setStyleSheet("font-weight: 600; color: #a1a1aa; font-size: 10pt; letter-spacing: 0.5px;")
        layout.addWidget(graph_section)

        graph_grid = QGridLayout()
        graph_grid.setSpacing(8)

        self.nodes_card = StatCard("Graph Nodes", "0", "")
        graph_grid.addWidget(self.nodes_card, 0, 0)

        self.edges_card = StatCard("Graph Edges", "0", "")
        graph_grid.addWidget(self.edges_card, 0, 1)

        self.pending_reviews_card = StatCard("Pending Review", "0", "")
        graph_grid.addWidget(self.pending_reviews_card, 1, 0)

        self.accepted_card = StatCard("Accepted", "0", "")
        graph_grid.addWidget(self.accepted_card, 1, 1)

        layout.addLayout(graph_grid)

        # Document type breakdown
        type_section = QLabel("By Document Type")
        type_section.setStyleSheet("font-weight: 600; color: #a1a1aa; font-size: 10pt; letter-spacing: 0.5px;")
        layout.addWidget(type_section)

        self.type_breakdown_label = QLabel("")
        self.type_breakdown_label.setStyleSheet("font-size: 9pt; color: #71717a;")
        self.type_breakdown_label.setWordWrap(True)
        layout.addWidget(self.type_breakdown_label)

        layout.addStretch()

    def refresh_stats(self):
        """Refresh all statistics."""
        try:
            self._update_document_stats()
            self._update_index_stats()
            self._update_graph_stats()
            self._update_type_breakdown()
        except Exception as e:
            # Don't crash on stats errors
            print(f"Stats refresh error: {e}")

    def _update_document_stats(self):
        """Update document statistics from catalog."""
        try:
            # Reload catalog from disk for fresh data
            self.catalog = DocumentCatalog()
            records = self.catalog.list_records()
            total = len(records)
            indexed = sum(1 for r in records if r.indexed)
            pending = sum(1 for r in records if not r.indexed and not r.error)
            errored = sum(1 for r in records if r.error)

            self.total_docs_card.update_value(str(total))
            self.indexed_docs_card.update_value(
                str(indexed),
                f"{(indexed/total*100):.0f}% complete" if total > 0 else ""
            )
            self.pending_docs_card.update_value(str(pending))
            self.errored_docs_card.update_value(
                str(errored),
                "Click to view errors" if errored > 0 else ""
            )
        except Exception as e:
            print(f"Document stats error: {e}")

    def _update_index_stats(self):
        """Update search index statistics."""
        try:
            # Vector store stats - optimized to avoid scanning all docs
            vector_store = VectorStore(settings=self.settings)
            vs_stats = vector_store.stats(include_documents=False)
            chunk_count = vs_stats.get("total_chunks", 0)
            
            # Get indexed document count from catalog (much faster than scanning vector DB)
            records = self.catalog.list_records()
            doc_count = sum(1 for r in records if r.indexed)

            self.chunks_card.update_value(
                f"{chunk_count:,}",
                f"from {doc_count} documents"
            )

            # FTS5 stats
            try:
                keyword_index = FTS5IndexCompat(settings=self.settings)
                keyword_stats = keyword_index.stats()
                keyword_chunks = keyword_stats.get("total_chunks", 0)
                # Count unique documents in FTS5 index
                keyword_docs = keyword_stats.get("document_count", 0)
                self.bm25_card.update_value(
                    f"{keyword_chunks:,}",
                    f"{keyword_docs} docs indexed"
                )
            except FileNotFoundError:
                self.bm25_card.update_value("0", "Not built yet")
        except Exception as e:
            print(f"Index stats error: {e}")

    def _update_graph_stats(self):
        """Update case graph statistics."""
        try:
            graph_store = GraphStore()

            # Pending reviews
            pending_files = graph_store.list_pending()
            self.pending_reviews_card.update_value(
                str(len(pending_files)),
                "awaiting user review" if pending_files else ""
            )

            # Count accepted graph elements
            total_nodes = 0
            total_edges = 0
            accepted_docs = 0

            for record in graph_store.load_graph_records():
                nodes = record.get("nodes", [])
                edges = record.get("edges", [])
                total_nodes += len(nodes)
                total_edges += len(edges)
                accepted_docs += 1

            self.nodes_card.update_value(f"{total_nodes:,}")
            self.edges_card.update_value(f"{total_edges:,}")
            self.accepted_card.update_value(
                str(accepted_docs),
                "documents processed"
            )
        except Exception:
            pass

    def _update_type_breakdown(self):
        """Update document type breakdown."""
        try:
            records = self.catalog.list_records()
            type_counts: dict[str, int] = {}

            for record in records:
                doc_type = record.category or "unknown"
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

            # Sort by count descending
            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)

            # Build display string
            lines = []
            for doc_type, count in sorted_types[:8]:  # Top 8 types
                display_name = doc_type.replace("_", " ").title()
                lines.append(f"{display_name}: {count}")

            if len(sorted_types) > 8:
                remaining = sum(count for _, count in sorted_types[8:])
                lines.append(f"Other: {remaining}")

            self.type_breakdown_label.setText("\n".join(lines) if lines else "No documents yet")
        except Exception:
            self.type_breakdown_label.setText("Unable to load")

    def get_errored_documents(self) -> list[str]:
        """Get list of documents with errors."""
        try:
            records = self.catalog.list_records()
            return [r.file_name for r in records if r.error]
        except Exception:
            return []
