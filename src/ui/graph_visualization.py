"""Interactive graph visualization using PyVis embedded in Qt WebEngine.

Provides multiple visualization modes:
1. Force-directed layout - Best for exploring entity relationships
2. Hierarchical layout - Shows document → entity structure
3. Timeline layout - Temporal arrangement of events
4. Radial layout - Centered on selected entity
"""

import json
import tempfile
from pathlib import Path
from typing import Optional, Literal

from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QSplitter,
    QCheckBox,
)
from PySide6.QtGui import QKeySequence, QShortcut

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_AVAILABLE = True
except ImportError:
    WEBENGINE_AVAILABLE = False
    QWebEngineView = None

from src.graph.storage import GraphStore
from src.graph.models import GraphNode, GraphEdge


# Color scheme for different node types
NODE_COLORS = {
    "document": "#7ba3b8",   # Muted teal
    "party": "#4ade80",      # Sage green
    "event": "#8b7cf6",      # Warm gold
    "statute": "#a07db8",    # Muted purple
    "issue": "#f87171",      # Muted red
    "chunk": "#6b7280",      # Gray
}

NODE_SHAPES = {
    "document": "square",
    "party": "dot",
    "event": "diamond",
    "statute": "triangle",
    "issue": "star",
    "chunk": "dot",
}


class GraphVisualizationWidget(QWidget):
    """Interactive graph visualization widget."""

    node_selected = Signal(str)  # node_id
    edge_selected = Signal(str, str)  # source_id, target_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.graph_store = GraphStore()
        self._nodes: list[GraphNode] = []
        self._edges: list[GraphEdge] = []
        self._temp_file: Optional[Path] = None

        self._setup_ui()
        self.refresh_graph()

    def _setup_ui(self):
        """Set up UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QFrame()
        toolbar.setStyleSheet("""
            QFrame {
                background-color: palette(window);
                border-bottom: 1px solid palette(mid);
                padding: 8px;
            }
        """)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(12, 8, 12, 8)

        # Title
        title = QLabel("Graph View")
        title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        toolbar_layout.addWidget(title)

        toolbar_layout.addStretch()

        # Layout selector
        toolbar_layout.addWidget(QLabel("Layout:"))
        self.layout_combo = QComboBox()
        self.layout_combo.addItem("Force-Directed", "force")
        self.layout_combo.addItem("Hierarchical", "hierarchical")
        self.layout_combo.addItem("Radial", "radial")
        self.layout_combo.addItem("Circular", "circular")
        self.layout_combo.currentIndexChanged.connect(self._on_layout_changed)
        toolbar_layout.addWidget(self.layout_combo)

        # Node type filters
        toolbar_layout.addWidget(QLabel("Show:"))
        
        self.show_documents = QCheckBox("Docs")
        self.show_documents.setChecked(True)
        self.show_documents.toggled.connect(self._on_filter_changed)
        toolbar_layout.addWidget(self.show_documents)

        self.show_parties = QCheckBox("Parties")
        self.show_parties.setChecked(True)
        self.show_parties.toggled.connect(self._on_filter_changed)
        toolbar_layout.addWidget(self.show_parties)

        self.show_events = QCheckBox("Events")
        self.show_events.setChecked(True)
        self.show_events.toggled.connect(self._on_filter_changed)
        toolbar_layout.addWidget(self.show_events)

        # Physics toggle
        self.physics_checkbox = QCheckBox("Physics")
        self.physics_checkbox.setChecked(True)
        self.physics_checkbox.setToolTip("Enable/disable force simulation")
        self.physics_checkbox.toggled.connect(self._on_physics_toggled)
        toolbar_layout.addWidget(self.physics_checkbox)

        # Fullscreen button
        fullscreen_btn = QPushButton("⛶")
        fullscreen_btn.setMaximumWidth(30)
        fullscreen_btn.setToolTip("Open in fullscreen (F11)")
        fullscreen_btn.clicked.connect(self._open_fullscreen)
        toolbar_layout.addWidget(fullscreen_btn)

        # Refresh button
        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(self.refresh_graph)
        toolbar_layout.addWidget(refresh_btn)

        layout.addWidget(toolbar)
        
        # Keyboard shortcut for fullscreen
        self.fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
        self.fullscreen_shortcut.activated.connect(self._open_fullscreen)

        # Graph view area
        if WEBENGINE_AVAILABLE:
            self.web_view = QWebEngineView()
            self.web_view.setMinimumHeight(400)
            layout.addWidget(self.web_view, stretch=1)
        else:
            # Fallback for when WebEngine is not available
            fallback = QLabel(
                "Graph visualization requires PyQt6-WebEngine\n\n"
                "Install with: pip install PyQt6-WebEngine\n\n"
                "Graph data is still being tracked."
            )
            fallback.setAlignment(Qt.AlignCenter)
            fallback.setStyleSheet("color: palette(shadow); padding: 40px;")
            layout.addWidget(fallback, stretch=1)
            self.web_view = None

        # Stats bar
        self.stats_label = QLabel("Loading...")
        self.stats_label.setStyleSheet("""
            background-color: palette(base);
            border-top: 1px solid palette(mid);
            padding: 6px 12px;
            font-size: 9pt;
            color: palette(shadow);
        """)
        layout.addWidget(self.stats_label)

    def refresh_graph(self):
        """Reload graph data and render."""
        self._nodes = []
        self._edges = []

        # Load from graph store
        for record in self.graph_store.load_graph_records():
            for node_data in record.get("nodes", []):
                node = GraphNode(**node_data)
                self._nodes.append(node)
            for edge_data in record.get("edges", []):
                edge = GraphEdge(**edge_data)
                self._edges.append(edge)

        # Update stats
        type_counts = {}
        for node in self._nodes:
            type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1

        stats_parts = [f"{count} {ntype}s" for ntype, count in sorted(type_counts.items())]
        self.stats_label.setText(
            f"{len(self._nodes)} nodes | {len(self._edges)} edges | " + " | ".join(stats_parts)
        )

        self._render_graph()

    def _on_layout_changed(self):
        """Handle layout change."""
        self._render_graph()

    def _on_filter_changed(self):
        """Handle filter change."""
        self._render_graph()

    def _on_physics_toggled(self, enabled: bool):
        """Handle physics toggle."""
        if self.web_view:
            # Send message to JS to toggle physics
            js = f"if (typeof network !== 'undefined') network.setOptions({{physics: {str(enabled).lower()}}});"
            self.web_view.page().runJavaScript(js)

    def _get_filtered_nodes(self) -> list[GraphNode]:
        """Get nodes based on current filters."""
        filtered = []
        for node in self._nodes:
            if node.node_type == "document" and not self.show_documents.isChecked():
                continue
            if node.node_type == "party" and not self.show_parties.isChecked():
                continue
            if node.node_type == "event" and not self.show_events.isChecked():
                continue
            filtered.append(node)
        return filtered

    def _render_graph(self):
        """Render graph using vis.js."""
        if not self.web_view:
            return

        filtered_nodes = self._get_filtered_nodes()
        filtered_node_ids = {n.node_id for n in filtered_nodes}

        # Filter edges to only include those between visible nodes
        filtered_edges = [
            e for e in self._edges
            if e.source in filtered_node_ids and e.target in filtered_node_ids
        ]

        layout_type = self.layout_combo.currentData()
        html = self._generate_visjs_html(filtered_nodes, filtered_edges, layout_type)

        # Write to temp file and load
        if self._temp_file and self._temp_file.exists():
            self._temp_file.unlink()

        self._temp_file = Path(tempfile.mktemp(suffix=".html"))
        self._temp_file.write_text(html, encoding="utf-8")

        self.web_view.setUrl(QUrl.fromLocalFile(str(self._temp_file)))

    def _generate_visjs_html(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        layout: str = "force"
    ) -> str:
        """Generate HTML with vis.js network visualization."""
        
        # Convert nodes to vis.js format
        vis_nodes = []
        for node in nodes:
            vis_node = {
                "id": node.node_id,
                "label": self._truncate(node.label, 25),
                "title": f"{node.label}\n{node.node_type}\n" + "\n".join(
                    f"{k}: {v}" for k, v in node.metadata.items()
                ),
                "color": NODE_COLORS.get(node.node_type, "#71717a"),
                "shape": NODE_SHAPES.get(node.node_type, "dot"),
                "size": 20 if node.node_type == "document" else 15,
                "font": {"size": 12, "color": "#f4f4f5"},
            }
            
            # For hierarchical layout, set levels
            if layout == "hierarchical":
                if node.node_type == "document":
                    vis_node["level"] = 0
                elif node.node_type == "party":
                    vis_node["level"] = 1
                elif node.node_type == "event":
                    vis_node["level"] = 2
                else:
                    vis_node["level"] = 3
            
            vis_nodes.append(vis_node)

        # Convert edges to vis.js format
        vis_edges = []
        for edge in edges:
            vis_edge = {
                "from": edge.source,
                "to": edge.target,
                "label": edge.relation,
                "arrows": "to",
                "color": {"color": "#999", "highlight": "#3b82f6"},
                "font": {"size": 10, "color": "#666", "align": "middle"},
            }
            vis_edges.append(vis_edge)

        # Layout options
        if layout == "hierarchical":
            layout_options = """
                layout: {
                    hierarchical: {
                        direction: 'UD',
                        sortMethod: 'directed',
                        levelSeparation: 100,
                        nodeSpacing: 150
                    }
                },
            """
        elif layout == "radial":
            layout_options = """
                layout: {
                    improvedLayout: true
                },
            """
        elif layout == "circular":
            # vis.js doesn't have native circular, use custom positioning
            layout_options = """
                layout: {
                    improvedLayout: false
                },
            """
        else:  # force-directed
            layout_options = """
                layout: {
                    improvedLayout: true
                },
            """

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Case Graph</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Barlow Condensed', 'Segoe UI', Roboto, sans-serif;
            background: #0f0f12;
            color: #f4f4f5;
        }}
        #graph {{
            width: 100vw;
            height: 100vh;
        }}
        #error {{
            display: none;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            color: #a1a1aa;
        }}
        #error h2 {{
            color: #8b7cf6;
            margin-bottom: 10px;
        }}
        .legend {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: #1a1a20;
            padding: 12px 16px;
            border-radius: 10px;
            border: 1px solid #27272a;
            font-size: 11px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 6px 0;
        }}
        .legend-color {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 10px;
        }}
    </style>
</head>
<body>
    <div id="graph"></div>
    <div id="error">
        <h2>Graph Library Loading</h2>
        <p>Waiting for vis.js to load from CDN...</p>
    </div>
    <div class="legend">
        <div class="legend-item"><div class="legend-color" style="background: #7ba3b8"></div>Documents</div>
        <div class="legend-item"><div class="legend-color" style="background: #4ade80"></div>Parties</div>
        <div class="legend-item"><div class="legend-color" style="background: #8b7cf6"></div>Events</div>
        <div class="legend-item"><div class="legend-color" style="background: #a07db8"></div>Statutes</div>
        <div class="legend-item"><div class="legend-color" style="background: #f87171"></div>Issues</div>
    </div>
    <script>
        // Check if vis.js loaded
        if (typeof vis === 'undefined') {{
            document.getElementById('error').style.display = 'block';
            document.getElementById('graph').style.display = 'none';
            // Retry loading after delay
            setTimeout(function() {{
                if (typeof vis !== 'undefined') {{
                    document.getElementById('error').style.display = 'none';
                    document.getElementById('graph').style.display = 'block';
                    initGraph();
                }} else {{
                    document.getElementById('error').innerHTML = '<h2>Graph Library Failed</h2><p>Could not load vis.js. Check your internet connection.</p>';
                }}
            }}, 3000);
        }} else {{
            initGraph();
        }}
        
        function initGraph() {{
        var nodes = new vis.DataSet({json.dumps(vis_nodes)});
        var edges = new vis.DataSet({json.dumps(vis_edges)});
        
        var container = document.getElementById('graph');
        var data = {{ nodes: nodes, edges: edges }};
        
        var options = {{
            {layout_options}
            physics: {{
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -50,
                    centralGravity: 0.01,
                    springLength: 100,
                    springConstant: 0.08
                }},
                stabilization: {{
                    iterations: 200
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                zoomView: true,
                dragView: true
            }},
            nodes: {{
                borderWidth: 2,
                borderWidthSelected: 3,
                shadow: true
            }},
            edges: {{
                smooth: {{
                    type: 'curvedCW',
                    roundness: 0.2
                }},
                shadow: true
            }}
        }};
        
        var network = new vis.Network(container, data, options);
        
        // Handle node click
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                console.log('Node clicked:', nodeId);
                // Could send to Qt via channel if needed
            }}
        }});
        
        // Handle double-click to focus
        network.on('doubleClick', function(params) {{
            if (params.nodes.length > 0) {{
                network.focus(params.nodes[0], {{
                    scale: 1.5,
                    animation: true
                }});
            }}
        }});

        // Apply circular layout if selected
        {"" if layout != "circular" else '''
        // Circular layout
        var nodeIds = nodes.getIds();
        var radius = Math.min(container.clientWidth, container.clientHeight) / 3;
        var angleStep = (2 * Math.PI) / nodeIds.length;
        
        nodeIds.forEach(function(nodeId, index) {
            var angle = index * angleStep;
            nodes.update({
                id: nodeId,
                x: radius * Math.cos(angle),
                y: radius * Math.sin(angle)
            });
        });
        network.setOptions({physics: false});
        '''}
        }} // end initGraph
    </script>
</body>
</html>
"""
        return html

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text with ellipsis."""
        if len(text) <= max_len:
            return text
        return text[:max_len-3] + "..."

    def focus_on_node(self, node_id: str):
        """Focus the view on a specific node."""
        if self.web_view:
            js = f"""
            if (typeof network !== 'undefined') {{
                network.focus('{node_id}', {{
                    scale: 1.5,
                    animation: {{
                        duration: 500,
                        easingFunction: 'easeInOutQuad'
                    }}
                }});
                network.selectNodes(['{node_id}']);
            }}
            """
            self.web_view.page().runJavaScript(js)

    def highlight_path(self, node_ids: list[str]):
        """Highlight a path through the graph."""
        if self.web_view and node_ids:
            js = f"""
            if (typeof network !== 'undefined') {{
                network.selectNodes({json.dumps(node_ids)});
            }}
            """
            self.web_view.page().runJavaScript(js)

    def export_image(self, path: str):
        """Export current view as image."""
        if self.web_view:
            # Capture via Qt
            self.web_view.grab().save(path)

    def cleanup(self):
        """Clean up temporary files."""
        if self._temp_file and self._temp_file.exists():
            self._temp_file.unlink()

    def _open_fullscreen(self):
        """Open graph in fullscreen dialog."""
        dialog = FullscreenGraphDialog(
            nodes=self._get_filtered_nodes(),
            edges=self._edges,
            layout=self.layout_combo.currentData(),
            parent=self.window()
        )
        dialog.exec()


class FullscreenGraphDialog(QDialog):
    """Fullscreen dialog for immersive graph exploration."""

    def __init__(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        layout: str = "force",
        parent=None
    ):
        super().__init__(parent)
        self.nodes = nodes
        self.edges = edges
        self.current_layout = layout
        self._temp_file: Optional[Path] = None

        self.setWindowTitle("Case Graph - Fullscreen")
        self.setModal(False)
        
        self._setup_ui()
        self._render_graph()
        
        # Show fullscreen
        self.showFullScreen()

    def _setup_ui(self):
        """Set up fullscreen UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Floating toolbar at top
        toolbar = QFrame()
        toolbar.setStyleSheet("""
            QFrame {
                background-color: rgba(15, 23, 42, 0.95);
                border-bottom: 1px solid #334155;
            }
        """)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(16, 12, 16, 12)

        # Title
        title = QLabel("Case Graph Explorer")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #f1f5f9;")
        toolbar_layout.addWidget(title)

        toolbar_layout.addStretch()

        # Layout selector
        layout_label = QLabel("Layout:")
        layout_label.setStyleSheet("color: #94a3b8;")
        toolbar_layout.addWidget(layout_label)

        self.layout_combo = QComboBox()
        self.layout_combo.addItem("Force-Directed", "force")
        self.layout_combo.addItem("Hierarchical", "hierarchical")
        self.layout_combo.addItem("Radial", "radial")
        self.layout_combo.addItem("Circular", "circular")
        self.layout_combo.setStyleSheet("""
            QComboBox {
                background-color: #1e293b;
                color: #f1f5f9;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 6px 12px;
            }
        """)
        # Set current layout
        idx = self.layout_combo.findData(self.current_layout)
        if idx >= 0:
            self.layout_combo.setCurrentIndex(idx)
        self.layout_combo.currentIndexChanged.connect(self._on_layout_changed)
        toolbar_layout.addWidget(self.layout_combo)

        toolbar_layout.addSpacing(20)

        # Stats
        self.stats_label = QLabel(f"{len(self.nodes)} nodes • {len(self.edges)} edges")
        self.stats_label.setStyleSheet("color: #64748b; font-size: 10pt;")
        toolbar_layout.addWidget(self.stats_label)

        toolbar_layout.addSpacing(20)

        # Help text
        help_label = QLabel("Scroll to zoom • Drag to pan • Double-click to focus • ESC to exit")
        help_label.setStyleSheet("color: #64748b; font-size: 9pt;")
        toolbar_layout.addWidget(help_label)

        toolbar_layout.addStretch()

        # Close button
        close_btn = QPushButton("✕ Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
        """)
        close_btn.clicked.connect(self.close)
        toolbar_layout.addWidget(close_btn)

        layout.addWidget(toolbar)

        # Web view for graph
        if WEBENGINE_AVAILABLE:
            self.web_view = QWebEngineView()
            layout.addWidget(self.web_view, stretch=1)
        else:
            fallback = QLabel("WebEngine not available")
            fallback.setAlignment(Qt.AlignCenter)
            fallback.setStyleSheet("color: #94a3b8; font-size: 14pt;")
            layout.addWidget(fallback, stretch=1)
            self.web_view = None

        # Escape to close
        self.escape_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.escape_shortcut.activated.connect(self.close)

    def _on_layout_changed(self):
        """Handle layout change."""
        self.current_layout = self.layout_combo.currentData()
        self._render_graph()

    def _render_graph(self):
        """Render the graph."""
        if not self.web_view:
            return

        # Filter edges to visible nodes
        node_ids = {n.node_id for n in self.nodes}
        filtered_edges = [e for e in self.edges if e.source in node_ids and e.target in node_ids]

        html = self._generate_fullscreen_html(self.nodes, filtered_edges, self.current_layout)

        # Write to temp file
        if self._temp_file and self._temp_file.exists():
            self._temp_file.unlink()

        self._temp_file = Path(tempfile.mktemp(suffix=".html"))
        self._temp_file.write_text(html, encoding="utf-8")

        self.web_view.setUrl(QUrl.fromLocalFile(str(self._temp_file)))

    def _generate_fullscreen_html(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        layout: str
    ) -> str:
        """Generate fullscreen-optimized HTML."""
        
        # Convert to vis.js format
        vis_nodes = []
        for node in nodes:
            vis_node = {
                "id": node.node_id,
                "label": node.label[:30] + "..." if len(node.label) > 30 else node.label,
                "title": f"<b>{node.label}</b><br>Type: {node.node_type}<br>" + 
                        "<br>".join(f"{k}: {v}" for k, v in list(node.metadata.items())[:5]),
                "color": {
                    "background": NODE_COLORS.get(node.node_type, "#71717a"),
                    "border": "#27272a",
                    "highlight": {"background": "#d4b57a", "border": "#8b7cf6"},
                },
                "shape": NODE_SHAPES.get(node.node_type, "dot"),
                "size": 25 if node.node_type == "document" else 20,
                "font": {"size": 14, "color": "#f1f5f9"},
                "shadow": True,
            }
            
            if layout == "hierarchical":
                level_map = {"document": 0, "party": 1, "event": 2}
                vis_node["level"] = level_map.get(node.node_type, 3)
            
            vis_nodes.append(vis_node)

        vis_edges = []
        for edge in edges:
            vis_edges.append({
                "from": edge.source,
                "to": edge.target,
                "label": edge.relation,
                "arrows": "to",
                "color": {"color": "#3f3f46", "highlight": "#8b7cf6"},
                "font": {"size": 11, "color": "#a1a1aa", "align": "middle"},
                "smooth": {"type": "curvedCW", "roundness": 0.15},
            })

        # Layout options
        layout_options = self._get_layout_options(layout)

        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Case Graph - Fullscreen</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Barlow Condensed', 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0f12 0%, #1f1b18 100%);
            overflow: hidden;
            color: #f4f4f5;
        }}
        #graph {{ width: 100vw; height: 100vh; }}
        
        /* Floating search box */
        .search-box {{
            position: fixed;
            top: 80px;
            left: 20px;
            background: rgba(36, 32, 25, 0.95);
            padding: 12px;
            border-radius: 10px;
            border: 1px solid #27272a;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            z-index: 1000;
        }}
        .search-box input {{
            background: #0f0f12;
            border: 1px solid #27272a;
            border-radius: 6px;
            color: #f4f4f5;
            padding: 10px 14px;
            font-size: 12px;
            width: 200px;
        }}
        .search-box input:focus {{
            outline: none;
            border-color: #8b7cf6;
        }}
        .search-box input::placeholder {{
            color: #71717a;
        }}
        
        /* Floating legend */
        .legend {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(36, 32, 25, 0.95);
            padding: 16px 20px;
            border-radius: 10px;
            border: 1px solid #27272a;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            z-index: 1000;
        }}
        .legend h4 {{
            color: #8b7cf6;
            margin-bottom: 12px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
            color: #a1a1aa;
            font-size: 11px;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
            border: 2px solid #27272a;
        }}
        
        /* Node info tooltip */
        .node-info {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(36, 32, 25, 0.95);
            padding: 16px;
            border-radius: 10px;
            border: 1px solid #27272a;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            z-index: 1000;
            max-width: 300px;
            display: none;
        }}
        .node-info h4 {{
            color: #8b7cf6;
            margin-bottom: 8px;
        }}
        .node-info p {{
            color: #a1a1aa;
            font-size: 11px;
            margin: 4px 0;
        }}
    </style>
</head>
<body>
    <div class="search-box">
        <input type="text" id="searchInput" placeholder="Search nodes..." onkeyup="searchNodes()">
    </div>
    
    <div id="graph"></div>
    
    <div class="legend">
        <h4>Node Types</h4>
        <div class="legend-item"><div class="legend-color" style="background: #7ba3b8"></div>Documents</div>
        <div class="legend-item"><div class="legend-color" style="background: #4ade80"></div>Parties</div>
        <div class="legend-item"><div class="legend-color" style="background: #8b7cf6"></div>Events</div>
        <div class="legend-item"><div class="legend-color" style="background: #a07db8"></div>Statutes</div>
        <div class="legend-item"><div class="legend-color" style="background: #f87171"></div>Issues</div>
    </div>
    
    <div class="node-info" id="nodeInfo">
        <h4 id="nodeTitle">Node</h4>
        <p id="nodeType">Type: -</p>
        <p id="nodeDetails">-</p>
    </div>
    
    <script>
        var nodes = new vis.DataSet({json.dumps(vis_nodes)});
        var edges = new vis.DataSet({json.dumps(vis_edges)});
        var allNodes = nodes.get();
        
        var container = document.getElementById('graph');
        var data = {{ nodes: nodes, edges: edges }};
        
        var options = {{
            {layout_options}
            physics: {{
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -80,
                    centralGravity: 0.005,
                    springLength: 150,
                    springConstant: 0.05,
                    damping: 0.4
                }},
                stabilization: {{
                    iterations: 300,
                    fit: true
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                zoomView: true,
                dragView: true,
                multiselect: true,
                navigationButtons: true,
                keyboard: {{
                    enabled: true,
                    bindToWindow: true
                }}
            }},
            nodes: {{
                borderWidth: 2,
                borderWidthSelected: 4,
                shadow: {{
                    enabled: true,
                    color: 'rgba(0,0,0,0.3)',
                    size: 10
                }}
            }},
            edges: {{
                shadow: true
            }}
        }};
        
        var network = new vis.Network(container, data, options);
        
        // Show node info on select
        network.on('selectNode', function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                document.getElementById('nodeInfo').style.display = 'block';
                document.getElementById('nodeTitle').textContent = node.label;
                document.getElementById('nodeType').textContent = 'ID: ' + node.id;
            }}
        }});
        
        network.on('deselectNode', function() {{
            document.getElementById('nodeInfo').style.display = 'none';
        }});
        
        // Double-click to focus
        network.on('doubleClick', function(params) {{
            if (params.nodes.length > 0) {{
                network.focus(params.nodes[0], {{
                    scale: 1.5,
                    animation: {{ duration: 500, easingFunction: 'easeInOutQuad' }}
                }});
            }}
        }});
        
        // Search function
        function searchNodes() {{
            var query = document.getElementById('searchInput').value.toLowerCase();
            if (!query) {{
                // Reset all nodes
                allNodes.forEach(function(node) {{
                    nodes.update({{ id: node.id, opacity: 1, font: {{ color: '#f1f5f9' }} }});
                }});
                return;
            }}
            
            allNodes.forEach(function(node) {{
                var matches = node.label.toLowerCase().includes(query);
                nodes.update({{
                    id: node.id,
                    opacity: matches ? 1 : 0.2,
                    font: {{ color: matches ? '#60a5fa' : '#475569' }}
                }});
                
                if (matches) {{
                    network.selectNodes([node.id]);
                }}
            }});
        }}
    </script>
</body>
</html>
"""

    def _get_layout_options(self, layout: str) -> str:
        """Get vis.js layout options string."""
        if layout == "hierarchical":
            return """
                layout: {
                    hierarchical: {
                        direction: 'UD',
                        sortMethod: 'directed',
                        levelSeparation: 120,
                        nodeSpacing: 180
                    }
                },
            """
        elif layout == "circular":
            return """
                layout: { improvedLayout: false },
            """
        elif layout == "radial":
            return """
                layout: { improvedLayout: true },
            """
        else:  # force
            return """
                layout: { improvedLayout: true },
            """

    def closeEvent(self, event):
        """Clean up on close."""
        if self._temp_file and self._temp_file.exists():
            self._temp_file.unlink()
        super().closeEvent(event)

