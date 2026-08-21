"""Documents workspace — ingest, browse, view, edit metadata.

Layout:
  ┌─ Toolbar ────────────────────────────────────────────────────────────┐
  │ [+ Add Files] [+ Add Folder] [Rebuild]   Stats area    🔍 Search    │
  ├─ Document List ──────────┬─ Detail Panel ────────────────────────────┤
  │                          │ Name: [filename]                          │
  │ 📄 Document name         │ Category: [dropdown]  [✓ Include Graph]   │
  │    90 chunks · indexed   │ Status: Indexed · 90 chunks · 2025-01-14 │
  │    witness_statement     │                                           │
  │                          │ ── Summary ──                             │
  │ 📧 Another doc           │ [summary text or Generate button]         │
  │    45 chunks · indexed   │                                           │
  │                          │ ── Chunks ──                              │
  │                          │ [chunk 1 text preview...]                 │
  │                          │ [chunk 2 text preview...]                 │
  │                          │                                           │
  │                          │ ── Entities ──                            │
  │                          │ [Person: Dr Smith] [Org: SFO]             │
  │                          │                                           │
  │                          │ [Delete Document] [Generate Summary]      │
  └──────────────────────────┴───────────────────────────────────────────┘
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.ui7.theme import C, S, T


# ─── Ingestion Worker ──────────────────────────────────────────────────

class IngestionWorker(QThread):
    """Background document ingestion with inline parse + enrich."""

    progress = Signal(str, int, int)
    parsed = Signal(str, int)
    enriched = Signal(str, int, int)
    finished = Signal(int)
    error = Signal(str)

    def __init__(self, file_paths: list[str], parent=None) -> None:
        super().__init__(parent)
        self.file_paths = file_paths

    def run(self) -> None:
        try:
            from src.config_loader import get_settings
            from src.ingestion.ingestion_pipeline import IngestionPipeline
            from src.ingestion.chunkers import get_chunker
            from src.retrieval.embedding_service import EmbeddingService
            from src.retrieval.vector_store import VectorStore
            from src.retrieval.fts5_index import FTS5Index
            from src.documents.catalog import DocumentCatalog

            settings = get_settings()
            pipeline = IngestionPipeline()
            chunker = get_chunker(settings)
            embedder = EmbeddingService()
            vector_store = VectorStore()
            fts = FTS5Index()
            catalog = DocumentCatalog()

            total_chunks = 0

            for i, fpath in enumerate(self.file_paths):
                fname = Path(fpath).name
                self.progress.emit("Parsing", i + 1, len(self.file_paths))

                parsed = pipeline.parse_document(fpath)
                if parsed is None:
                    self.error.emit(f"Failed to parse: {fname}")
                    continue

                record = catalog.ensure_record(parsed)
                chunks = chunker.chunk_document(parsed)
                self.parsed.emit(fname, len(chunks))

                # Kanon 2 enrichment (best-effort)
                try:
                    from src.graph.enricher import KanonEnricher
                    enricher = KanonEnricher()
                    if enricher.is_available and chunks:
                        sample = [c.text for c in chunks[:10]]
                        results = enricher.enrich_batch(sample)
                        persons = sum(len(r.persons) for r in results if r)
                        locations = sum(len(r.locations) for r in results if r)
                        self.enriched.emit(fname, persons, locations)

                        from src.graph.case_graph import CaseGraph
                        graph = CaseGraph()
                        for j, r in enumerate(results):
                            if r:
                                ents, rels = enricher.to_entities(
                                    r, chunk_id=chunks[j].chunk_id if j < len(chunks) else ""
                                )
                                for e in ents:
                                    graph.add_entity(e)
                                for rel in rels:
                                    graph.add_relationship(rel)
                        graph.save()
                except Exception:
                    pass

                self.progress.emit("Embedding", i + 1, len(self.file_paths))
                embeddings = embedder.embed_batch([c.text for c in chunks])

                self.progress.emit("Indexing", i + 1, len(self.file_paths))
                vector_store.add_chunks(chunks, embeddings)
                fts.add_chunks(chunks)

                catalog.update_record(
                    fpath, indexed=True, chunk_count=len(chunks),
                    ingested_at=datetime.now().isoformat(),
                )
                total_chunks += len(chunks)

            self.finished.emit(total_chunks)
        except Exception as e:
            self.error.emit(str(e))


# ─── Rebuild Worker ───────────────────────────────────────────────────

class RebuildWorker(QThread):
    """Clear all indexes and re-ingest every catalogued document.

    Steps:
      1. Reset vector store (ChromaDB)
      2. Reset keyword index (FTS5)
      3. Re-parse → re-chunk (new strategy) → re-embed → re-index each doc
      4. Update catalog records
    """

    progress = Signal(str, int, int)   # stage, current, total
    finished = Signal(int)             # total_chunks
    error = Signal(str)

    def run(self) -> None:
        try:
            from src.config_loader import get_settings
            from src.ingestion.ingestion_pipeline import IngestionPipeline
            from src.ingestion.chunkers import get_chunker
            from src.retrieval.embedding_service import EmbeddingService
            from src.retrieval.vector_store import VectorStore
            from src.retrieval.fts5_index import FTS5Index
            from src.documents.catalog import DocumentCatalog

            settings = get_settings()
            pipeline = IngestionPipeline()
            chunker = get_chunker(settings)
            embedder = EmbeddingService()
            vector_store = VectorStore()
            fts = FTS5Index()
            catalog = DocumentCatalog()

            records = catalog.list_records()
            if not records:
                self.finished.emit(0)
                return

            # Step 1 — wipe indexes
            self.progress.emit("Clearing indexes", 0, len(records))
            vector_store.reset()
            fts.reset()

            total_chunks = 0

            for i, rec in enumerate(records):
                fpath = rec.file_path
                fname = rec.file_name

                # Skip if source file no longer exists
                if not Path(fpath).exists():
                    self.error.emit(f"Source missing: {fname}")
                    catalog.update_record(fpath, indexed=False, error="Source file missing")
                    continue

                self.progress.emit("Parsing", i + 1, len(records))
                parsed = pipeline.parse_document(fpath)
                if parsed is None:
                    self.error.emit(f"Parse failed: {fname}")
                    catalog.update_record(fpath, indexed=False, error="Parse failed")
                    continue

                self.progress.emit("Chunking", i + 1, len(records))
                chunks = chunker.chunk_document(parsed)

                self.progress.emit("Embedding", i + 1, len(records))
                embeddings = embedder.embed_batch([c.text for c in chunks])

                self.progress.emit("Indexing", i + 1, len(records))
                vector_store.add_chunks(chunks, embeddings)
                fts.add_chunks(chunks)

                catalog.update_record(
                    fpath,
                    indexed=True,
                    chunk_count=len(chunks),
                    ingested_at=datetime.now().isoformat(),
                    error=None,
                )
                total_chunks += len(chunks)

            self.finished.emit(total_chunks)
        except Exception as e:
            self.error.emit(str(e))


# ─── Document Detail Panel ─────────────────────────────────────────────

class DocumentDetailPanel(QWidget):
    """Right panel showing selected document details, chunks, entities."""

    summary_requested = Signal(str)   # file_path
    delete_requested = Signal(str)    # file_path
    metadata_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._current_path: Optional[str] = None

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background: {C.BG_BASE}; }}")

        container = QWidget()
        container.setStyleSheet(f"background: {C.BG_BASE};")
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(24, 20, 24, 20)
        self._layout.setSpacing(16)

        # ── Header ──
        self.title_label = QLabel("Select a document")
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet(f"""
            font-size: 16px; font-weight: 700; color: {C.TEXT_PRIMARY};
            background: transparent;
        """)
        self._layout.addWidget(self.title_label)

        # ── Metadata row ──
        meta_widget = QWidget()
        meta_widget.setStyleSheet("background: transparent;")
        meta_layout = QHBoxLayout(meta_widget)
        meta_layout.setContentsMargins(0, 0, 0, 0)
        meta_layout.setSpacing(12)

        meta_layout.addWidget(self._make_label("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.addItems([
            "witness_statement", "court_filing", "pleading", "statute",
            "contract", "disclosure", "email", "scanned_pdf", "unknown",
        ])
        self.category_combo.setFixedWidth(160)
        self.category_combo.currentTextChanged.connect(self._on_category_changed)
        meta_layout.addWidget(self.category_combo)

        self.graph_check = QCheckBox("Include in Graph")
        self.graph_check.setChecked(True)
        self.graph_check.stateChanged.connect(self._on_graph_toggled)
        meta_layout.addWidget(self.graph_check)

        meta_layout.addStretch(1)
        self._layout.addWidget(meta_widget)

        # ── Status row ──
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"""
            color: {C.TEXT_SECONDARY}; font-size: 11px; background: transparent;
        """)
        self._layout.addWidget(self.status_label)

        # ── Summary section ──
        self._layout.addWidget(self._make_section_header("Summary"))
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(140)
        self.summary_text.setPlaceholderText("No summary generated.")
        self.summary_text.setStyleSheet(f"""
            QTextEdit {{
                background: {C.BG_SURFACE}; border: 1px solid {C.BORDER};
                border-radius: 8px; padding: 10px; font-size: 12px; color: {C.TEXT_PRIMARY};
            }}
        """)
        self._layout.addWidget(self.summary_text)

        gen_summary_btn = QPushButton("Generate Summary")
        gen_summary_btn.setFixedWidth(160)
        gen_summary_btn.clicked.connect(lambda: self.summary_requested.emit(self._current_path or ""))
        self._layout.addWidget(gen_summary_btn)

        # ── Chunks section ──
        self._layout.addWidget(self._make_section_header("Document Content"))
        self.chunks_container = QWidget()
        self.chunks_container.setStyleSheet("background: transparent;")
        self.chunks_layout = QVBoxLayout(self.chunks_container)
        self.chunks_layout.setContentsMargins(0, 0, 0, 0)
        self.chunks_layout.setSpacing(6)
        self._layout.addWidget(self.chunks_container)

        # ── Entities section ──
        self._layout.addWidget(self._make_section_header("Entities"))
        self.entities_container = QWidget()
        self.entities_container.setStyleSheet("background: transparent;")
        self.entities_layout = QHBoxLayout(self.entities_container)
        self.entities_layout.setContentsMargins(0, 0, 0, 0)
        self.entities_layout.setSpacing(6)
        self._layout.addWidget(self.entities_container)

        self._layout.addStretch(1)

        # ── Actions ──
        actions = QHBoxLayout()
        actions.setSpacing(8)
        delete_btn = QPushButton("Delete Document")
        delete_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C.ERROR_BG}; color: {C.ERROR};
                border: 1px solid {C.ERROR}; border-radius: 6px;
                padding: 6px 14px; font-size: 12px; font-weight: 500;
            }}
            QPushButton:hover {{ background: {C.ERROR}; color: white; }}
        """)
        delete_btn.clicked.connect(lambda: self.delete_requested.emit(self._current_path or ""))
        actions.addWidget(delete_btn)
        actions.addStretch(1)
        self._layout.addLayout(actions)

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def load_document(self, record) -> None:
        """Populate panel from a DocumentRecord."""
        self._current_path = record.file_path
        self.title_label.setText(record.file_name)

        idx = self.category_combo.findText(record.category)
        if idx >= 0:
            self.category_combo.blockSignals(True)
            self.category_combo.setCurrentIndex(idx)
            self.category_combo.blockSignals(False)

        self.graph_check.blockSignals(True)
        self.graph_check.setChecked(record.include_in_graph)
        self.graph_check.blockSignals(False)

        # Status
        parts = []
        if record.indexed:
            parts.append(f"\u2713 Indexed")
        else:
            parts.append("Pending")
        parts.append(f"{record.chunk_count} chunks")
        if record.ingested_at:
            try:
                dt = datetime.fromisoformat(record.ingested_at)
                parts.append(dt.strftime("%Y-%m-%d %H:%M"))
            except Exception:
                parts.append(record.ingested_at[:16])
        if record.error:
            parts.append(f"Error: {record.error}")
        self.status_label.setText(" \u00B7 ".join(parts))

        # Load summary
        self._load_summary(record.file_path)

        # Load chunks
        self._load_chunks(record.file_path, record.file_name)

        # Load entities
        self._load_entities(record.file_name)

    def _load_summary(self, file_path: str) -> None:
        """Load document summary from summary store."""
        self.summary_text.clear()
        try:
            from src.retrieval.summary_store import SummaryStore
            store = SummaryStore()
            summaries = store.get_summaries_for_document(file_path)
            if summaries:
                text = summaries[0].summary if hasattr(summaries[0], 'summary') else str(summaries[0])
                self.summary_text.setPlainText(text)
        except Exception:
            pass

    def _load_chunks(self, file_path: str, file_name: str) -> None:
        """Load and display document chunks."""
        # Clear existing
        while self.chunks_layout.count():
            item = self.chunks_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        try:
            from src.retrieval.vector_store import VectorStore
            vs = VectorStore()
            results = vs.collection.get(
                where={"file_name": file_name},
                limit=50,
                include=["documents", "metadatas"],
            )
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])

            if not docs:
                lbl = QLabel("No chunks found.")
                lbl.setStyleSheet(f"color: {C.TEXT_TERTIARY}; font-size: 12px; background: transparent;")
                self.chunks_layout.addWidget(lbl)
                return

            for i, (text, meta) in enumerate(zip(docs, metas)):
                chunk_card = self._make_chunk_card(i + 1, text, meta)
                self.chunks_layout.addWidget(chunk_card)

            if len(docs) >= 50:
                more = QLabel(f"Showing first 50 of {len(docs)}+ chunks")
                more.setStyleSheet(f"color: {C.TEXT_TERTIARY}; font-size: 11px; background: transparent;")
                self.chunks_layout.addWidget(more)

        except Exception as e:
            lbl = QLabel(f"Could not load chunks: {e}")
            lbl.setStyleSheet(f"color: {C.TEXT_TERTIARY}; font-size: 12px; background: transparent;")
            self.chunks_layout.addWidget(lbl)

    def _make_chunk_card(self, index: int, text: str, meta: dict) -> QWidget:
        """Create an expandable chunk card — click header to toggle full text."""
        card = QWidget()
        card.setStyleSheet(f"""
            QWidget#chunkCard {{
                background: {C.BG_SURFACE};
                border: 1px solid {C.BORDER};
                border-radius: 6px;
            }}
        """)
        card.setObjectName("chunkCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # Header row (clickable toggle)
        header = QHBoxLayout()
        page = meta.get("page_number", "?")
        chars = len(text)
        header_btn = QPushButton(f"\u25B6  Chunk {index}  \u00B7  Page {page}  \u00B7  {chars} chars")
        header_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        header_btn.setStyleSheet(f"""
            QPushButton {{
                color: {C.ACCENT}; font-size: 10px; font-weight: 600;
                background: transparent; border: none; text-align: left;
                padding: 2px 0;
            }}
            QPushButton:hover {{ color: {C.TEXT_PRIMARY}; }}
        """)
        header.addWidget(header_btn)
        header.addStretch(1)
        layout.addLayout(header)

        # Preview (collapsed — first 200 chars)
        preview = text[:200].replace("\n", " ")
        if len(text) > 200:
            preview += "…"
        preview_label = QLabel(preview)
        preview_label.setWordWrap(True)
        preview_label.setStyleSheet(f"""
            color: {C.TEXT_SECONDARY}; font-size: 12px;
            line-height: 150%; background: transparent;
        """)
        layout.addWidget(preview_label)

        # Full text (hidden by default)
        full_text = QTextEdit()
        full_text.setReadOnly(True)
        full_text.setPlainText(text)
        full_text.setVisible(False)
        full_text.setMinimumHeight(100)
        full_text.setMaximumHeight(400)
        full_text.setStyleSheet(f"""
            QTextEdit {{
                background: {C.BG_BASE}; border: 1px solid {C.BORDER};
                border-radius: 4px; padding: 8px; font-size: 12px;
                color: {C.TEXT_PRIMARY};
            }}
        """)
        layout.addWidget(full_text)

        # Toggle handler
        def toggle() -> None:
            expanded = full_text.isVisible()
            full_text.setVisible(not expanded)
            preview_label.setVisible(expanded)
            arrow = "\u25BC" if not expanded else "\u25B6"
            header_btn.setText(f"{arrow}  Chunk {index}  \u00B7  Page {page}  \u00B7  {chars} chars")

        header_btn.clicked.connect(toggle)

        return card

    def _load_entities(self, file_name: str) -> None:
        """Load entities extracted from this document."""
        while self.entities_layout.count():
            item = self.entities_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        try:
            from src.graph.case_graph import CaseGraph
            graph = CaseGraph()
            entities = list(graph.entities.values())
            # Filter to entities that mention this file
            doc_entities = [
                e for e in entities
                if file_name in str(getattr(e, 'source_chunks', []))
                or file_name in str(getattr(e, 'metadata', {}))
            ]
            if not doc_entities:
                # Show all entities as fallback
                doc_entities = entities[:20]

            for e in doc_entities[:30]:
                chip = self._make_entity_chip(e)
                self.entities_layout.addWidget(chip)
            self.entities_layout.addStretch(1)

        except Exception:
            lbl = QLabel("No entities.")
            lbl.setStyleSheet(f"color: {C.TEXT_TERTIARY}; font-size: 11px; background: transparent;")
            self.entities_layout.addWidget(lbl)

    def _make_entity_chip(self, entity) -> QPushButton:
        """Entity badge chip."""
        name = getattr(entity, 'canonical_name', str(entity))
        etype_raw = getattr(entity, 'type', 'unknown')
        etype = (etype_raw.value if hasattr(etype_raw, 'value') else str(etype_raw)).lower()

        type_colors = {
            "person": C.INFO, "party": C.INFO,
            "organization": C.WARNING, "company": C.WARNING,
            "location": C.SUCCESS,
            "statute": C.ACCENT, "document": C.ACCENT,
            "event": C.ERROR,
        }
        color = type_colors.get(str(etype).lower(), C.TEXT_SECONDARY)

        chip = QPushButton(f"{name}")
        chip.setToolTip(f"{etype}: {name}")
        chip.setCursor(Qt.CursorShape.PointingHandCursor)
        chip.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {color};
                border: 1px solid {color};
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background: {C.BG_RAISED};
            }}
        """)
        return chip

    def _on_category_changed(self, category: str) -> None:
        if self._current_path:
            try:
                from src.documents.catalog import DocumentCatalog
                catalog = DocumentCatalog()
                catalog.update_record(self._current_path, category=category)
                self.metadata_changed.emit()
            except Exception:
                pass

    def _on_graph_toggled(self, state: int) -> None:
        if self._current_path:
            try:
                from src.documents.catalog import DocumentCatalog
                catalog = DocumentCatalog()
                catalog.update_record(self._current_path, include_in_graph=bool(state))
                self.metadata_changed.emit()
            except Exception:
                pass

    @staticmethod
    def _make_section_header(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"""
            color: {C.TEXT_SECONDARY}; font-size: 11px; font-weight: 700;
            letter-spacing: 1px; text-transform: uppercase;
            background: transparent; padding-top: 8px;
            border-top: 1px solid {C.BORDER};
        """)
        return lbl

    @staticmethod
    def _make_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {C.TEXT_SECONDARY}; font-size: 12px; background: transparent;")
        return lbl


# ─── Stats Bar ─────────────────────────────────────────────────────────

class DocStatsBar(QWidget):
    """Compact stats bar: total docs, indexed, chunks, graph nodes."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        self.stats: dict[str, QLabel] = {}
        for key in ("Documents", "Indexed", "Chunks", "Graph Nodes"):
            lbl = QLabel(f"{key}: --")
            lbl.setStyleSheet(f"""
                color: {C.TEXT_TERTIARY}; font-size: 10px; font-weight: 500;
                background: transparent;
            """)
            layout.addWidget(lbl)
            self.stats[key] = lbl
        layout.addStretch(1)

    def refresh(self) -> None:
        try:
            from src.documents.catalog import DocumentCatalog
            catalog = DocumentCatalog()
            records = catalog.list_records()
            total = len(records)
            indexed = sum(1 for r in records if r.indexed)
            chunks = sum(r.chunk_count for r in records)
            self.stats["Documents"].setText(f"Documents: {total}")
            self.stats["Indexed"].setText(f"Indexed: {indexed}")
            self.stats["Chunks"].setText(f"Chunks: {chunks:,}")
        except Exception:
            pass

        try:
            from src.graph.case_graph import CaseGraph
            graph = CaseGraph()
            nodes = len(graph.entities)
            self.stats["Graph Nodes"].setText(f"Graph: {nodes} nodes")
        except Exception:
            pass


# ─── Main Workspace ────────────────────────────────────────────────────

class DocumentsWorkspace(QWidget):
    """Document management workspace."""

    index_updated = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._worker: Optional[QThread] = None
        self._records: list = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Toolbar ──
        toolbar = QWidget()
        toolbar.setStyleSheet(f"""
            QWidget#docToolbar {{
                background: {C.BG_SURFACE};
                border-bottom: 1px solid {C.BORDER};
            }}
        """)
        toolbar.setObjectName("docToolbar")
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(16, 8, 16, 8)
        tb.setSpacing(8)

        add_files = QPushButton("+ Add Files")
        add_files.setProperty("class", "primary")
        add_files.clicked.connect(self._add_files)
        tb.addWidget(add_files)

        add_folder = QPushButton("+ Add Folder")
        add_folder.clicked.connect(self._add_folder)
        tb.addWidget(add_folder)

        rebuild_btn = QPushButton("Rebuild Index")
        rebuild_btn.clicked.connect(self._rebuild_index)
        tb.addWidget(rebuild_btn)

        tb.addSpacing(16)

        # Stats
        self.stats_bar = DocStatsBar()
        tb.addWidget(self.stats_bar, 1)

        # Search
        search = QLineEdit()
        search.setPlaceholderText("Search documents...")
        search.setFixedWidth(180)
        search.textChanged.connect(self._filter_list)
        tb.addWidget(search)

        layout.addWidget(toolbar)

        # ── Progress ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(3)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet(f"""
            color: {C.ACCENT}; font-size: 11px; padding: 4px 16px;
            background: {C.ACCENT_BG}; border-bottom: 1px solid {C.BORDER};
        """)
        self.progress_label.hide()
        layout.addWidget(self.progress_label)

        # ── Splitter: list + detail ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(f"QSplitter {{ background: {C.BG_BASE}; }}")

        # Document list
        self.doc_list = QListWidget()
        self.doc_list.setStyleSheet(f"""
            QListWidget {{
                background: {C.BG_BASE}; border: none;
                border-right: 1px solid {C.BORDER};
            }}
            QListWidget::item {{
                padding: 8px 12px; border: none;
                border-radius: 0; margin: 0;
            }}
            QListWidget::item:selected {{
                background: {C.ACCENT_BG};
            }}
            QListWidget::item:hover:!selected {{
                background: {C.BG_RAISED};
            }}
        """)
        self.doc_list.currentRowChanged.connect(self._on_doc_selected)
        splitter.addWidget(self.doc_list)

        # Detail panel
        self.detail_panel = DocumentDetailPanel()
        self.detail_panel.metadata_changed.connect(self._reload_catalog)
        self.detail_panel.delete_requested.connect(self._delete_document)
        splitter.addWidget(self.detail_panel)

        splitter.setSizes([380, 620])
        layout.addWidget(splitter, 1)

        # Load catalog
        self._load_catalog()

    def _load_catalog(self) -> None:
        try:
            from src.documents.catalog import DocumentCatalog
            catalog = DocumentCatalog()
            self._records = catalog.list_records()
            self.doc_list.clear()
            for rec in self._records:
                self._add_list_item(rec)
            total = sum(r.chunk_count for r in self._records)
            self.index_updated.emit(total)
            self.stats_bar.refresh()
        except Exception:
            pass

    def _reload_catalog(self) -> None:
        self._load_catalog()

    def _add_list_item(self, rec) -> None:
        item = QListWidgetItem()
        status = "\u2713" if rec.indexed else "\u2022"
        text = f"{rec.file_name}\n{rec.chunk_count} chunks \u00B7 {rec.category} \u00B7 {status}"
        item.setText(text)
        item.setData(Qt.ItemDataRole.UserRole, rec.file_path)
        self.doc_list.addItem(item)

    def _on_doc_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._records):
            return
        rec = self._records[row]
        self.detail_panel.load_document(rec)

    def _filter_list(self, text: str) -> None:
        text = text.lower()
        for i in range(self.doc_list.count()):
            item = self.doc_list.item(i)
            item.setHidden(text not in (item.text() or "").lower())

    def _add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add Documents", "",
            "Documents (*.pdf *.docx *.doc *.txt *.eml *.msg *.xlsx *.csv);;All (*.*)",
        )
        if files:
            self._start_ingestion(files)

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add Folder")
        if folder:
            exts = {".pdf", ".docx", ".doc", ".txt", ".eml", ".msg", ".xlsx", ".csv"}
            files = [str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in exts and p.is_file()]
            if files:
                self._start_ingestion(files)

    def _rebuild_index(self) -> None:
        """Clear all indexes and re-ingest every document with current settings."""
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "Busy", "An ingestion task is already running.")
            return

        count = len(self._records)
        if count == 0:
            QMessageBox.information(self, "Nothing to rebuild", "No documents in catalog.")
            return

        reply = QMessageBox.question(
            self, "Rebuild Index",
            f"This will clear all indexes and re-chunk + re-embed {count} documents "
            f"using the current chunking strategy.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.progress_label.setText(f"Rebuilding index for {count} documents...")
        self.progress_label.show()

        self._worker = RebuildWorker()
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(lambda e: self.progress_label.setText(f"Error: {e}"))
        self._worker.start()

    def _start_ingestion(self, paths: list[str]) -> None:
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.progress_label.setText(f"Ingesting {len(paths)} files...")
        self.progress_label.show()

        self._worker = IngestionWorker(paths)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(lambda e: self.progress_label.setText(f"Error: {e}"))
        self._worker.start()

    def _on_progress(self, stage: str, current: int, total: int) -> None:
        self.progress_label.setText(f"{stage}: {current}/{total}")
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)

    def _on_finished(self, total: int) -> None:
        self.progress_bar.hide()
        self.progress_label.hide()
        self._load_catalog()
        self.index_updated.emit(total)

    def _delete_document(self, file_path: str) -> None:
        reply = QMessageBox.question(
            self, "Delete Document",
            f"Delete this document from the index?\n{Path(file_path).name}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                from src.documents.catalog import DocumentCatalog
                catalog = DocumentCatalog()
                catalog.delete_record(file_path)
                self._load_catalog()
            except Exception:
                pass

    def cleanup(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.wait(3000)
