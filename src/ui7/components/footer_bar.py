"""Bottom footer bar — live chunk count, index status, matter, active task."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from src.ui7.theme import C, S, T


class FooterBar(QWidget):
    """Slim bottom status bar with live data."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(S.FOOTER_HEIGHT)
        self.setStyleSheet(f"""
            QWidget#footer {{
                background-color: {C.BG_SURFACE};
                border-top: 1px solid {C.BORDER};
            }}
        """)
        self.setObjectName("footer")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(S.LG, 0, S.LG, 0)
        layout.setSpacing(12)

        style = f"color: {C.TEXT_TERTIARY}; font-size: 10px; background: transparent; border: none;"

        self.chunk_label = QLabel("-- chunks")
        self.chunk_label.setStyleSheet(style)
        layout.addWidget(self.chunk_label)

        self._add_sep(layout, style)

        self.doc_label = QLabel("-- documents")
        self.doc_label.setStyleSheet(style)
        layout.addWidget(self.doc_label)

        self._add_sep(layout, style)

        self.graph_label = QLabel("-- entities")
        self.graph_label.setStyleSheet(style)
        layout.addWidget(self.graph_label)

        self._add_sep(layout, style)

        self.matter_label = QLabel("Matter: Default")
        self.matter_label.setStyleSheet(style)
        layout.addWidget(self.matter_label)

        layout.addStretch(1)

        self.task_label = QLabel("")
        self.task_label.setStyleSheet(
            f"color: {C.ACCENT}; font-size: 10px; background: transparent; border: none;"
        )
        layout.addWidget(self.task_label)

        # Auto-refresh every 10 seconds
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(10_000)

        # Initial load
        QTimer.singleShot(500, self.refresh)

    def refresh(self) -> None:
        """Pull live stats from catalog and graph."""
        try:
            from src.documents.catalog import DocumentCatalog
            catalog = DocumentCatalog()
            records = catalog.list_records()
            docs = len(records)
            chunks = sum(r.chunk_count for r in records)
            self.doc_label.setText(f"{docs} documents")
            self.chunk_label.setText(f"{chunks:,} chunks")
        except Exception:
            pass

        try:
            from src.graph.case_graph import CaseGraph
            graph = CaseGraph()
            entities = len(graph.entities)
            self.graph_label.setText(f"{entities} entities")
        except Exception:
            pass

    def update_chunk_count(self, count: int) -> None:
        self.chunk_label.setText(f"{count:,} chunks")

    def update_task(self, text: str) -> None:
        self.task_label.setText(text)

    def update_matter(self, name: str) -> None:
        self.matter_label.setText(f"Matter: {name}")

    @staticmethod
    def _add_sep(layout, style: str) -> None:
        sep = QLabel("\u00B7")
        sep.setStyleSheet(style)
        layout.addWidget(sep)
