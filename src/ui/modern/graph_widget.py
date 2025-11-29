import json
import tempfile
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QCheckBox, 
    QPushButton, QLabel, QFrame
)
from PySide6.QtWebEngineWidgets import QWebEngineView

from src.graph.storage import GraphStore
from src.graph.models import GraphNode, GraphEdge

# Node configuration
NODE_COLORS = {
    "document": "#7ba3b8",
    "party": "#4ade80",
    "event": "#8b7cf6",
    "statute": "#a07db8",
    "issue": "#f87171",
    "chunk": "#6b7280",
}

class GraphWidget(QWidget):
    """Modern Graph Visualization Widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.graph_store = GraphStore()
        self._nodes = []
        self._edges = []
        self._temp_file = None
        
        self._setup_ui()
        self.refresh_graph()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QFrame()
        toolbar.setStyleSheet("background-color: #18181b; border-bottom: 1px solid #27272a; padding: 8px;")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(12, 8, 12, 8)

        # Controls
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["Force-Directed", "Hierarchical", "Radial"])
        self.layout_combo.currentIndexChanged.connect(self._render_graph)
        self.layout_combo.setStyleSheet("background-color: #27272a; color: #f1f5f9; border: none; padding: 5px; border-radius: 4px;")
        toolbar_layout.addWidget(QLabel("Layout:"))
        toolbar_layout.addWidget(self.layout_combo)

        toolbar_layout.addSpacing(20)

        self.physics_check = QCheckBox("Physics")
        self.physics_check.setChecked(True)
        self.physics_check.toggled.connect(self._toggle_physics)
        self.physics_check.setStyleSheet("color: #f1f5f9;")
        toolbar_layout.addWidget(self.physics_check)

        toolbar_layout.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setStyleSheet("background-color: #3b82f6; color: white; border: none; padding: 6px 12px; border-radius: 4px;")
        refresh_btn.clicked.connect(self.refresh_graph)
        toolbar_layout.addWidget(refresh_btn)

        layout.addWidget(toolbar)

        # Web View
        self.web_view = QWebEngineView()
        self.web_view.setStyleSheet("background-color: #0f0f12;")
        layout.addWidget(self.web_view)

    def refresh_graph(self):
        self._nodes = []
        self._edges = []
        
        # Load data
        for record in self.graph_store.load_graph_records():
            for n in record.get("nodes", []):
                self._nodes.append(GraphNode(**n))
            for e in record.get("edges", []):
                self._edges.append(GraphEdge(**e))
                
        self._render_graph()

    def _toggle_physics(self, enabled):
        js = f"if (typeof network !== 'undefined') network.setOptions({{physics: {str(enabled).lower()}}});"
        self.web_view.page().runJavaScript(js)

    def _render_graph(self):
        # Prepare data for vis.js
        nodes_data = []
        seen_nodes = set()
        for n in self._nodes:
            if n.node_id not in seen_nodes:
                seen_nodes.add(n.node_id)
                nodes_data.append({
                    "id": n.node_id,
                    "label": n.label,
                    "title": f"{n.label} ({n.node_type})",
                    "color": NODE_COLORS.get(n.node_type, "#71717a"),
                    "shape": "dot"
                })

        edges_data = []
        seen_edges = set()
        for e in self._edges:
            eid = f"{e.source}-{e.target}"
            if eid not in seen_edges:
                seen_edges.add(eid)
                edges_data.append({
                    "from": e.source,
                    "to": e.target,
                    "label": e.relation,
                    "arrows": "to",
                    "color": {"color": "#52525b"}
                })

        layout_mode = self.layout_combo.currentText()
        layout_opts = ""
        if layout_mode == "Hierarchical":
            layout_opts = "layout: { hierarchical: { direction: 'UD', sortMethod: 'directed' } },"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
            <style>
                body {{ margin: 0; background-color: #0f0f12; }}
                #mynetwork {{ width: 100vw; height: 100vh; }}
            </style>
        </head>
        <body>
            <div id="mynetwork"></div>
            <script type="text/javascript">
                var nodes = new vis.DataSet({json.dumps(nodes_data)});
                var edges = new vis.DataSet({json.dumps(edges_data)});
                var container = document.getElementById('mynetwork');
                var data = {{ nodes: nodes, edges: edges }};
                var options = {{
                    {layout_opts}
                    nodes: {{
                        font: {{ color: '#f1f5f9' }},
                        borderWidth: 0
                    }},
                    physics: {{
                        enabled: true,
                        stabilization: {{ iterations: 100 }}
                    }}
                }};
                var network = new vis.Network(container, data, options);
            </script>
        </body>
        </html>
        """

        if self._temp_file and self._temp_file.exists():
            try:
                self._temp_file.unlink()
            except:
                pass

        self._temp_file = Path(tempfile.mktemp(suffix=".html"))
        self._temp_file.write_text(html, encoding="utf-8")
        self.web_view.setUrl(QUrl.fromLocalFile(str(self._temp_file)))
