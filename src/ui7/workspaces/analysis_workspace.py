"""Analysis workspace — Overview + Entities + Graph + Timeline + Background Tasks.

Sub-tabs:
  [Overview]  [Entities]  [Graph]  [Timeline]  [Tasks]
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.ui7.theme import C, S, T


# ─── Background Task Worker ───────────────────────────────────────────

class _TaskWorker(QThread):
    """Run a named analysis task on a background thread."""

    progress = Signal(str)       # status text
    finished = Signal(str)       # result summary
    error = Signal(str)

    def __init__(self, task_name: str, parent=None) -> None:
        super().__init__(parent)
        self.task_name = task_name

    def run(self) -> None:  # noqa: C901 — simple dispatch
        try:
            from src.config.runtime_store import load_runtime_state

            state = load_runtime_state()
            model = state.get("model_name")

            if self.task_name == "graph":
                self.progress.emit("Generating case graph…")
                from src.graph.graph_generator import CaseGraphGenerator

                gen = CaseGraphGenerator()
                graph = gen.generate_full_graph(
                    model=model,
                    progress_callback=lambda msg: self.progress.emit(msg),
                )
                n_ent = len(graph.entities) if graph else 0
                n_rel = len(graph.relationships) if graph else 0
                self.finished.emit(f"Case graph: {n_ent} entities, {n_rel} relationships")

            elif self.task_name == "timeline":
                self.progress.emit("Extracting timeline events…")
                from src.graph.timeline_generator import TimelineGenerator

                gen = TimelineGenerator()
                events = gen.generate_full_timeline(
                    model=model,
                    progress_callback=lambda msg: self.progress.emit(msg),
                )
                self.finished.emit(f"Timeline: {len(events)} events extracted")

            elif self.task_name == "overview":
                self.progress.emit("Generating case overview…")
                from src.generation.case_overview_generator import CaseOverviewGenerator

                gen = CaseOverviewGenerator()
                overview = gen.generate_overview(
                    model=model,
                    progress_callback=lambda msg: self.progress.emit(msg),
                )
                self.finished.emit(f"Case overview generated ({len(str(overview))} chars)")

            elif self.task_name == "rename":
                self.progress.emit("Suggesting document names…")
                from src.generation.document_renamer import DocumentRenamer

                gen = DocumentRenamer()
                names = gen.rename_all_documents(
                    model=model,
                    progress_callback=lambda msg: self.progress.emit(msg),
                )
                self.finished.emit(f"Renamed {len(names)} documents")

            elif self.task_name == "deduplicate":
                self.progress.emit("Deduplicating entities…")
                from src.graph.case_graph import CaseGraph

                graph = CaseGraph()
                before = len(graph.entities)
                removed = graph.deduplicate()
                after = len(graph.entities)
                self.finished.emit(
                    f"Deduplicated: {before} → {after} entities ({removed} merged)"
                )

            elif self.task_name == "summaries":
                self.progress.emit("Generating document summaries…")
                from src.generation.summarizer import SummarizerService
                from src.documents.catalog import DocumentCatalog
                from src.ingestion.ingestion_pipeline import IngestionPipeline

                catalog = DocumentCatalog()
                pipeline = IngestionPipeline()
                records = catalog.list_records()
                summarizer = SummarizerService()

                docs_for_summary = []
                for rec in records:
                    if not rec.indexed:
                        continue
                    parsed = pipeline.parse_document(rec.file_path)
                    if parsed:
                        import hashlib
                        doc_id = hashlib.sha256(
                            f"{parsed.file_path}:{parsed.file_name}".encode()
                        ).hexdigest()[:16]
                        docs_for_summary.append({
                            "document_id": doc_id,
                            "text": parsed.text,
                            "file_name": parsed.file_name,
                            "doc_type": parsed.document_type,
                        })
                        self.progress.emit(f"Parsed {len(docs_for_summary)}/{len(records)} docs…")

                if not docs_for_summary:
                    self.finished.emit("No indexed documents to summarize")
                    return

                self.progress.emit(f"Summarizing {len(docs_for_summary)} documents…")
                summaries = summarizer.summarize_documents(
                    documents=docs_for_summary,
                    model=model,
                    on_progress=lambda done, total, name: self.progress.emit(
                        f"Summarizing {done}/{total}: {name}"
                    ),
                )
                self.finished.emit(f"Generated {len(summaries)} summaries")

            else:
                self.error.emit(f"Unknown task: {self.task_name}")

        except Exception as e:
            self.error.emit(f"{self.task_name} failed: {e}")


# ─── Overview Tab ──────────────────────────────────────────────────────

class OverviewTab(QWidget):
    """Case overview with key parties, dates, issues, stats."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._worker: _TaskWorker | None = None
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background: {C.BG_BASE}; }}")

        content = QWidget()
        content.setStyleSheet(f"background: {C.BG_BASE};")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(20)

        # Header + generate button
        header = QHBoxLayout()
        title = QLabel("Case Overview")
        title.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {C.TEXT_PRIMARY}; background: transparent;")
        header.addWidget(title)
        header.addStretch(1)

        self.gen_btn = QPushButton("Generate Overview")
        self.gen_btn.setProperty("class", "primary")
        self.gen_btn.clicked.connect(self._generate_overview)
        header.addWidget(self.gen_btn)
        layout.addLayout(header)

        # Summary
        self._add_section(layout, "Summary")
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(200)
        self.summary_text.setPlaceholderText("No overview generated yet. Click 'Generate Overview' to analyze your documents.")
        self.summary_text.setStyleSheet(f"""
            QTextEdit {{
                background: {C.BG_SURFACE}; border: 1px solid {C.BORDER};
                border-radius: 8px; padding: 14px; font-size: 13px; color: {C.TEXT_PRIMARY};
            }}
        """)
        layout.addWidget(self.summary_text)

        # Key Parties
        self._add_section(layout, "Key Parties")
        self.parties_container = QWidget()
        self.parties_container.setStyleSheet("background: transparent;")
        self.parties_layout = QVBoxLayout(self.parties_container)
        self.parties_layout.setContentsMargins(0, 0, 0, 0)
        self.parties_layout.setSpacing(6)
        layout.addWidget(self.parties_container)

        # Key Dates
        self._add_section(layout, "Key Dates")
        self.dates_container = QWidget()
        self.dates_container.setStyleSheet("background: transparent;")
        self.dates_layout = QVBoxLayout(self.dates_container)
        self.dates_layout.setContentsMargins(0, 0, 0, 0)
        self.dates_layout.setSpacing(6)
        layout.addWidget(self.dates_container)

        # Key Issues
        self._add_section(layout, "Key Issues")
        self.issues_text = QTextEdit()
        self.issues_text.setReadOnly(True)
        self.issues_text.setMaximumHeight(140)
        self.issues_text.setStyleSheet(f"""
            QTextEdit {{
                background: {C.BG_SURFACE}; border: 1px solid {C.BORDER};
                border-radius: 8px; padding: 12px; font-size: 12px; color: {C.TEXT_PRIMARY};
            }}
        """)
        layout.addWidget(self.issues_text)

        # Document Statistics
        self._add_section(layout, "Document Statistics")
        self.stats_grid = QWidget()
        self.stats_grid.setStyleSheet("background: transparent;")
        self.stats_layout = QHBoxLayout(self.stats_grid)
        self.stats_layout.setContentsMargins(0, 0, 0, 0)
        self.stats_layout.setSpacing(12)
        layout.addWidget(self.stats_grid)

        layout.addStretch(1)
        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._load()

    def _load(self) -> None:
        """Load existing overview data."""
        try:
            p = Path("data/case_overview.json")
            if p.exists():
                data = json.loads(p.read_text())
                self.summary_text.setPlainText(data.get("overview", ""))
                for party in data.get("key_parties", []):
                    self._add_party_card(party)
                for date_info in data.get("key_dates", []):
                    self._add_date_card(date_info)
                issues = data.get("key_issues", [])
                if issues:
                    self.issues_text.setPlainText("\n".join(f"  {i+1}. {iss}" for i, iss in enumerate(issues)))
        except Exception:
            pass

        self._load_stats()

    def _load_stats(self) -> None:
        """Load document statistics."""
        while self.stats_layout.count():
            item = self.stats_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        stats = {}
        try:
            from src.documents.catalog import DocumentCatalog
            catalog = DocumentCatalog()
            records = catalog.list_records()
            stats["Documents"] = len(records)
            stats["Indexed"] = sum(1 for r in records if r.indexed)
            stats["Chunks"] = sum(r.chunk_count for r in records)

            # Type breakdown
            types: dict[str, int] = {}
            for r in records:
                types[r.category] = types.get(r.category, 0) + 1
            for t, c in sorted(types.items(), key=lambda x: -x[1]):
                stats[t] = c
        except Exception:
            pass

        try:
            from src.graph.case_graph import CaseGraph
            g = CaseGraph()
            stats["Entities"] = len(g.entities)
            stats["Relationships"] = len(g.relationships)
            stats["Timeline Events"] = len(g.events)
        except Exception:
            pass

        for label, value in stats.items():
            card = self._make_stat_card(label, str(value))
            self.stats_layout.addWidget(card)
        self.stats_layout.addStretch(1)

    def _make_stat_card(self, label: str, value: str) -> QWidget:
        card = QWidget()
        card.setStyleSheet(f"""
            QWidget#statCard {{
                background: {C.BG_SURFACE}; border: 1px solid {C.BORDER};
                border-radius: 8px; padding: 12px 16px;
            }}
        """)
        card.setObjectName("statCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        val_lbl = QLabel(value)
        val_lbl.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {C.ACCENT}; background: transparent;")
        layout.addWidget(val_lbl)

        name_lbl = QLabel(label)
        name_lbl.setStyleSheet(f"font-size: 10px; color: {C.TEXT_TERTIARY}; background: transparent;")
        layout.addWidget(name_lbl)

        return card

    def _add_party_card(self, party: dict) -> None:
        card = QWidget()
        card.setStyleSheet(f"""
            QWidget {{ background: {C.BG_SURFACE}; border: 1px solid {C.BORDER};
                       border-radius: 6px; padding: 10px 14px; }}
        """)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        name_lbl = QLabel(party.get("name", "Unknown"))
        name_lbl.setStyleSheet(f"font-weight: 600; color: {C.TEXT_PRIMARY}; font-size: 13px; background: transparent;")
        layout.addWidget(name_lbl)

        role = party.get("role", "")
        if role:
            role_lbl = QLabel(role)
            role_lbl.setStyleSheet(f"color: {C.ACCENT}; font-size: 11px; background: transparent;")
            layout.addWidget(role_lbl)

        layout.addStretch(1)
        self.parties_layout.addWidget(card)

    def _add_date_card(self, date_info: dict) -> None:
        card = QWidget()
        card.setStyleSheet(f"""
            QWidget {{ background: {C.BG_SURFACE}; border-left: 3px solid {C.ACCENT};
                       border-radius: 0 6px 6px 0; padding: 8px 12px; }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        date_lbl = QLabel(date_info.get("date", "Unknown"))
        date_lbl.setStyleSheet(f"color: {C.ACCENT}; font-size: 11px; font-weight: 600; background: transparent;")
        layout.addWidget(date_lbl)

        event = date_info.get("event", date_info.get("description", ""))
        if event:
            ev_lbl = QLabel(event)
            ev_lbl.setWordWrap(True)
            ev_lbl.setStyleSheet(f"color: {C.TEXT_PRIMARY}; font-size: 12px; background: transparent;")
            layout.addWidget(ev_lbl)

        self.dates_layout.addWidget(card)

    def _generate_overview(self) -> None:
        # Check if summaries exist first
        try:
            from src.retrieval.summary_store import SummaryStore
            store = SummaryStore()
            import sqlite3
            conn = sqlite3.connect(str(store.db_path))
            count = conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
            conn.close()
        except Exception:
            count = 0

        if count == 0:
            self.summary_text.setPlainText(
                "No document summaries found.\n\n"
                "Go to Analysis → Tasks and run 'Generate Summaries' first, "
                "then come back to generate the case overview."
            )
            return

        self.gen_btn.setEnabled(False)
        self.gen_btn.setText("Generating...")
        self.summary_text.setPlainText("Generating case overview… this may take a moment.")

        self._worker = _TaskWorker("overview")
        self._worker.progress.connect(
            lambda msg: self.summary_text.setPlainText(msg)
        )
        self._worker.finished.connect(self._on_overview_done)
        self._worker.error.connect(self._on_overview_error)
        self._worker.start()

    def _on_overview_done(self, msg: str) -> None:
        self.gen_btn.setEnabled(True)
        self.gen_btn.setText("Generate Overview")
        self._load()  # Reload from file

    def _on_overview_error(self, msg: str) -> None:
        self.gen_btn.setEnabled(True)
        self.gen_btn.setText("Generate Overview")
        self.summary_text.setPlainText(f"Error: {msg}")

    @staticmethod
    def _add_section(layout, title: str) -> None:
        lbl = QLabel(title)
        lbl.setStyleSheet(f"""
            color: {C.TEXT_SECONDARY}; font-size: 11px; font-weight: 700;
            letter-spacing: 1px; background: transparent;
            padding-top: 8px; border-top: 1px solid {C.BORDER};
        """)
        layout.addWidget(lbl)


# ─── Entity Browser Tab ───────────────────────────────────────────────

class EntityBrowserTab(QWidget):
    """Browse, search, and filter extracted entities."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QWidget()
        toolbar.setStyleSheet(f"background: {C.BG_SURFACE}; border-bottom: 1px solid {C.BORDER};")
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(16, 8, 16, 8)
        tb.setSpacing(8)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search entities...")
        self.search.setFixedWidth(200)
        self.search.textChanged.connect(self._filter)
        tb.addWidget(self.search)

        self.type_filter = QComboBox()
        self.type_filter.addItems(["All Types", "person", "organization", "location", "statute", "document", "event"])
        self.type_filter.currentTextChanged.connect(self._filter)
        tb.addWidget(self.type_filter)

        tb.addStretch(1)

        self.count_label = QLabel("0 entities")
        self.count_label.setStyleSheet(f"color: {C.TEXT_TERTIARY}; font-size: 11px; background: transparent;")
        tb.addWidget(self.count_label)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load)
        tb.addWidget(refresh_btn)
        layout.addWidget(toolbar)

        # Entity list
        self.entity_list = QListWidget()
        self.entity_list.setStyleSheet(f"""
            QListWidget {{ background: {C.BG_BASE}; border: none; }}
            QListWidget::item {{ padding: 8px 16px; border: none; }}
            QListWidget::item:selected {{ background: {C.ACCENT_BG}; }}
            QListWidget::item:hover:!selected {{ background: {C.BG_RAISED}; }}
        """)
        layout.addWidget(self.entity_list, 1)

        self._entities: list = []
        self._load()

    def _load(self) -> None:
        self.entity_list.clear()
        self._entities = []
        try:
            from src.graph.case_graph import CaseGraph
            graph = CaseGraph()
            self._entities = list(graph.entities.values())
            self.count_label.setText(f"{len(self._entities)} entities")
            self._populate()
        except Exception:
            pass

    def _populate(self) -> None:
        self.entity_list.clear()
        search_text = self.search.text().lower()
        type_filter = self.type_filter.currentText()

        for e in self._entities:
            name = getattr(e, 'canonical_name', str(e))
            etype_raw = getattr(e, 'type', 'unknown')
            etype = (etype_raw.value if hasattr(etype_raw, 'value') else str(etype_raw)).lower()

            if search_text and search_text not in name.lower():
                continue
            if type_filter != "All Types" and type_filter.lower() != etype:
                continue

            item = QListWidgetItem(f"{name}  [{etype}]")
            item.setData(Qt.ItemDataRole.UserRole, e)
            self.entity_list.addItem(item)

    def _filter(self) -> None:
        self._populate()


# ─── Graph Tab ─────────────────────────────────────────────────────────

class GraphTab(QWidget):
    """Knowledge graph visualization using vis.js via QWebEngineView."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QWidget()
        toolbar.setStyleSheet(f"background: {C.BG_SURFACE}; border-bottom: 1px solid {C.BORDER};")
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(16, 8, 16, 8)
        tb.setSpacing(8)

        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["Force-Directed", "Hierarchical", "Radial"])
        self.layout_combo.currentTextChanged.connect(self._refresh_graph)
        tb.addWidget(QLabel("Layout:"))
        tb.addWidget(self.layout_combo)

        tb.addSpacing(12)

        # Node type filters
        from PySide6.QtWidgets import QCheckBox
        self.show_persons = QCheckBox("Persons")
        self.show_persons.setChecked(True)
        self.show_persons.stateChanged.connect(self._refresh_graph)
        tb.addWidget(self.show_persons)

        self.show_orgs = QCheckBox("Orgs")
        self.show_orgs.setChecked(True)
        self.show_orgs.stateChanged.connect(self._refresh_graph)
        tb.addWidget(self.show_orgs)

        self.show_docs = QCheckBox("Docs")
        self.show_docs.setChecked(True)
        self.show_docs.stateChanged.connect(self._refresh_graph)
        tb.addWidget(self.show_docs)

        tb.addStretch(1)

        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet(f"color: {C.TEXT_TERTIARY}; font-size: 11px; background: transparent;")
        tb.addWidget(self.stats_label)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_graph)
        tb.addWidget(refresh_btn)
        layout.addWidget(toolbar)

        # Graph view
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            self.web_view = QWebEngineView()
            self.web_view.setStyleSheet(f"background: {C.BG_BASE};")
            layout.addWidget(self.web_view, 1)
            self._has_webengine = True
        except ImportError:
            placeholder = QLabel(
                "Graph visualization requires PySide6-WebEngine.\n"
                "Install with: pip install PySide6-WebEngine"
            )
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(f"color: {C.TEXT_TERTIARY}; font-size: 13px; padding: 40px; background: {C.BG_BASE};")
            layout.addWidget(placeholder, 1)
            self._has_webengine = False

        self._refresh_graph()

    def _refresh_graph(self) -> None:
        if not self._has_webengine:
            return

        try:
            from src.graph.case_graph import CaseGraph
            graph = CaseGraph()

            nodes = []
            edges = []

            type_colors = {
                "person": "#5B8FD4", "party": "#5B8FD4",
                "organization": "#D4A843", "company": "#D4A843",
                "location": "#5CB67A",
                "statute": "#C9A96E", "document": "#C9A96E",
                "event": "#D45B5B",
            }

            for eid, entity in graph.entities.items():
                etype_raw = getattr(entity, 'type', 'unknown')
                etype = (etype_raw.value if hasattr(etype_raw, 'value') else str(etype_raw)).lower()

                # Filter
                if etype in ("person", "party") and not self.show_persons.isChecked():
                    continue
                if etype in ("organization", "company") and not self.show_orgs.isChecked():
                    continue
                if etype in ("document",) and not self.show_docs.isChecked():
                    continue

                color = type_colors.get(etype, "#9898A0")
                name = getattr(entity, 'canonical_name', str(entity))
                nodes.append({"id": eid, "label": name[:30], "color": color, "title": f"{etype}: {name}"})

            for rid, rel in graph.relationships.items():
                src = getattr(rel, 'source_entity_id', '')
                tgt = getattr(rel, 'target_entity_id', '')
                label = str(getattr(rel, 'relationship_type', ''))
                if src and tgt:
                    edges.append({"from": src, "to": tgt, "label": label[:20]})

            self.stats_label.setText(f"{len(nodes)} nodes \u00B7 {len(edges)} edges")

            # Build vis.js HTML
            nodes_json = json.dumps(nodes)
            edges_json = json.dumps(edges)

            layout_type = self.layout_combo.currentText()
            if layout_type == "Hierarchical":
                layout_config = '{ hierarchical: { direction: "UD", sortMethod: "directed" } }'
            elif layout_type == "Radial":
                layout_config = '{ hierarchical: { direction: "UD" } }'
            else:
                layout_config = '{ improvedLayout: true }'

            html = f"""<!DOCTYPE html>
<html><head>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  body {{ margin: 0; background: {C.BG_BASE}; overflow: hidden; }}
  #graph {{ width: 100vw; height: 100vh; }}
</style>
</head><body>
<div id="graph"></div>
<script>
  var nodes = new vis.DataSet({nodes_json});
  var edges = new vis.DataSet({edges_json});
  var container = document.getElementById('graph');
  var data = {{ nodes: nodes, edges: edges }};
  var options = {{
    layout: {layout_config},
    nodes: {{
      shape: 'dot', size: 12,
      font: {{ color: '{C.TEXT_PRIMARY}', size: 11, face: 'Inter, sans-serif' }},
      borderWidth: 0
    }},
    edges: {{
      color: {{ color: '{C.BORDER_STRONG}', highlight: '{C.ACCENT}' }},
      font: {{ color: '{C.TEXT_TERTIARY}', size: 9, face: 'Inter, sans-serif' }},
      arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }},
      smooth: {{ type: 'continuous' }}
    }},
    physics: {{ barnesHut: {{ gravitationalConstant: -3000, springLength: 120 }} }},
    interaction: {{ hover: true, tooltipDelay: 200 }}
  }};
  new vis.Network(container, data, options);
</script>
</body></html>"""

            self.web_view.setHtml(html)

        except Exception as e:
            self.stats_label.setText(f"Error: {e}")


# ─── Timeline Tab ─────────────────────────────────────────────────────

class TimelineTab(QWidget):
    """Chronological timeline of case events."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background: {C.BG_BASE}; }}")

        container = QWidget()
        container.setStyleSheet(f"background: {C.BG_BASE};")
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(32, 20, 32, 20)
        self._layout.setSpacing(0)

        self._load()
        self._layout.addStretch(1)

        scroll.setWidget(container)
        layout.addWidget(scroll)

    def _load(self) -> None:
        try:
            from src.graph.case_graph import CaseGraph
            graph = CaseGraph()
            events = sorted(graph.events.values(), key=lambda e: str(getattr(e, 'date', '') or ''))

            if not events:
                empty = QLabel("No timeline events yet.\nGenerate a timeline from the Tasks tab to extract dates and events from your documents.")
                empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
                empty.setWordWrap(True)
                empty.setStyleSheet(f"color: {C.TEXT_TERTIARY}; padding: 60px; font-size: 13px; background: transparent;")
                self._layout.addWidget(empty)
                return

            for event in events:
                self._add_event(event)
        except Exception:
            pass

    def _add_event(self, event) -> None:
        card = QWidget()
        card.setStyleSheet(f"""
            QWidget {{ background: {C.BG_SURFACE}; border-left: 3px solid {C.ACCENT};
                       border-radius: 0 6px 6px 0; margin: 3px 0; padding: 10px 14px; }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        date_str = str(getattr(event, 'date', getattr(event, 'date_text', 'Unknown')))
        d_lbl = QLabel(date_str)
        d_lbl.setStyleSheet(f"color: {C.ACCENT}; font-size: 11px; font-weight: 600; background: transparent;")
        layout.addWidget(d_lbl)

        desc = getattr(event, 'description', '')
        if desc:
            desc_lbl = QLabel(desc)
            desc_lbl.setWordWrap(True)
            desc_lbl.setStyleSheet(f"color: {C.TEXT_PRIMARY}; font-size: 12px; background: transparent;")
            layout.addWidget(desc_lbl)

        self._layout.addWidget(card)


# ─── Background Tasks Tab ─────────────────────────────────────────────

class TasksTab(QWidget):
    """Background task launcher: graph, timeline, overview, rename."""

    task_completed = Signal()   # emitted after any task finishes

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._worker: _TaskWorker | None = None
        self._all_queue: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(16)

        title = QLabel("Background Tasks")
        title.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {C.TEXT_PRIMARY}; background: transparent;")
        layout.addWidget(title)

        desc = QLabel("Run these tasks to extract structured information from your documents.")
        desc.setStyleSheet(f"color: {C.TEXT_SECONDARY}; font-size: 12px; background: transparent;")
        layout.addWidget(desc)

        tasks = [
            ("Generate Summaries", "Summarize all documents (required before graph/overview)", "summaries"),
            ("Generate Case Graph", "Extract entities and relationships from summaries", "graph"),
            ("Deduplicate Entities", "Merge duplicate entities by fuzzy name matching", "deduplicate"),
            ("Generate Timeline", "Extract chronological events and dates", "timeline"),
            ("Generate Case Overview", "Create a high-level summary of the entire case", "overview"),
            ("Rename Documents", "Suggest descriptive names based on content", "rename"),
        ]

        self._run_buttons: dict[str, QPushButton] = {}
        for name, description, task_id in tasks:
            card = self._make_task_card(name, description, task_id)
            layout.addWidget(card)

        # Run All
        all_card = self._make_task_card(
            "Run All Tasks", "Execute all tasks sequentially", "all"
        )
        layout.addWidget(all_card)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {C.ACCENT}; font-size: 12px; background: transparent;")
        layout.addWidget(self.status_label)

        layout.addStretch(1)

    def _make_task_card(self, name: str, description: str, task_id: str) -> QWidget:
        card = QWidget()
        card.setStyleSheet(f"""
            QWidget#taskCard {{
                background: {C.BG_SURFACE}; border: 1px solid {C.BORDER};
                border-radius: 8px;
            }}
        """)
        card.setObjectName("taskCard")
        hl = QHBoxLayout(card)
        hl.setContentsMargins(16, 12, 16, 12)
        hl.setSpacing(12)

        text_layout = QVBoxLayout()
        name_lbl = QLabel(name)
        name_lbl.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {C.TEXT_PRIMARY}; background: transparent;")
        text_layout.addWidget(name_lbl)

        desc_lbl = QLabel(description)
        desc_lbl.setStyleSheet(f"font-size: 11px; color: {C.TEXT_SECONDARY}; background: transparent;")
        text_layout.addWidget(desc_lbl)

        hl.addLayout(text_layout, 1)

        run_btn = QPushButton("Run")
        run_btn.setProperty("class", "primary")
        run_btn.setFixedWidth(80)
        run_btn.clicked.connect(lambda checked=False, tid=task_id: self._launch(tid))
        hl.addWidget(run_btn)

        self._run_buttons[task_id] = run_btn
        return card

    # ── Launcher ────────────────────────────────────────────────────

    def _launch(self, task_id: str) -> None:
        if self._worker and self._worker.isRunning():
            self.status_label.setText("A task is already running — please wait.")
            return

        if task_id == "all":
            self._all_queue = ["summaries", "graph", "deduplicate", "timeline", "overview", "rename"]
            self._launch_next_in_queue()
            return

        self._set_buttons_enabled(False)
        self._worker = _TaskWorker(task_id)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _launch_next_in_queue(self) -> None:
        if not self._all_queue:
            self.status_label.setText("All tasks complete.")
            self._set_buttons_enabled(True)
            return
        next_task = self._all_queue.pop(0)
        self._set_buttons_enabled(False)
        self._worker = _TaskWorker(next_task)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_queue_step_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    # ── Callbacks ───────────────────────────────────────────────────

    def _on_progress(self, msg: str) -> None:
        self.status_label.setText(msg)

    def _on_finished(self, msg: str) -> None:
        self.status_label.setText(f"Done — {msg}")
        self._set_buttons_enabled(True)
        self.task_completed.emit()

    def _on_queue_step_finished(self, msg: str) -> None:
        self.status_label.setText(f"Done — {msg}")
        self.task_completed.emit()
        self._launch_next_in_queue()

    def _on_error(self, msg: str) -> None:
        self.status_label.setText(f"Error: {msg}")
        self._set_buttons_enabled(True)
        self._all_queue.clear()

    def _set_buttons_enabled(self, enabled: bool) -> None:
        for btn in self._run_buttons.values():
            btn.setEnabled(enabled)


# ─── Analysis Workspace ───────────────────────────────────────────────

class AnalysisWorkspace(QWidget):
    """Analysis workspace with sub-tabs."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.overview_tab = OverviewTab()
        self.entities_tab = EntityBrowserTab()
        self.graph_tab = GraphTab()
        self.timeline_tab = TimelineTab()
        self.tasks_tab = TasksTab()

        self.tabs.addTab(self.overview_tab, "Overview")
        self.tabs.addTab(self.entities_tab, "Entities")
        self.tabs.addTab(self.graph_tab, "Graph")
        self.tabs.addTab(self.timeline_tab, "Timeline")
        self.tabs.addTab(self.tasks_tab, "Tasks")

        # Refresh data tabs when a background task completes
        self.tasks_tab.task_completed.connect(self._refresh_data_tabs)

        layout.addWidget(self.tabs)

    def _refresh_data_tabs(self) -> None:
        """Reload overview, entities, graph, timeline after a task completes."""
        try:
            self.overview_tab._load()
        except Exception:
            pass
        try:
            self.entities_tab._load()
        except Exception:
            pass
        try:
            self.graph_tab._refresh_graph()
        except Exception:
            pass
        # Timeline tab builds content at init; rebuild it
        try:
            while self.timeline_tab._layout.count():
                item = self.timeline_tab._layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.timeline_tab._load()
            self.timeline_tab._layout.addStretch(1)
        except Exception:
            pass

    def cleanup(self) -> None:
        pass
