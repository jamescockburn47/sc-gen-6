"""Persistence helpers for the RAG graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.graph.models import GraphNode, GraphEdge, GraphUpdate


class GraphStore:
    """Lightweight JSON-backed store for graph nodes and pending reviews."""

    def __init__(self, base_path: str | Path | None = None):
        self.base_path = Path(base_path or "data/graph")
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "pending").mkdir(exist_ok=True)
        self.accepted_path = self.base_path / "graph.jsonl"

    # ------------------------------------------------------------------#
    # Pending queue
    # ------------------------------------------------------------------#
    def queue_update(self, update: GraphUpdate) -> Path:
        """Save a pending update for user review."""
        # Sanitize document_id to be a valid filename
        safe_id = self._sanitize_filename(update.document_id)
        pending_file = self.base_path / "pending" / f"{safe_id}.json"
        payload = {
            "document_id": update.document_id,
            "document_name": update.document_name,
            "auto_generated": update.auto_generated,
            "notes": update.notes,
            "nodes": [node.__dict__ for node in update.nodes],
            "edges": [edge.__dict__ for edge in update.edges],
        }
        pending_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return pending_file

    def _sanitize_filename(self, document_id: str) -> str:
        """Convert document_id to a safe filename."""
        import hashlib
        # Use hash of full path + original filename for uniqueness and readability
        from pathlib import Path
        name = Path(document_id).stem  # Get filename without extension
        path_hash = hashlib.md5(document_id.encode()).hexdigest()[:8]
        # Remove any remaining unsafe characters
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        return f"{safe_name}_{path_hash}"

    def list_pending(self) -> list[Path]:
        """Return pending update files."""
        return sorted((self.base_path / "pending").glob("*.json"))

    def load_pending(self, document_id: str) -> GraphUpdate | None:
        """Load pending update by original document_id (will sanitize)."""
        safe_id = self._sanitize_filename(document_id)
        return self.load_pending_by_filename(safe_id)

    def load_pending_by_filename(self, sanitized_id: str) -> GraphUpdate | None:
        """Load pending update by sanitized filename (no further sanitization)."""
        path = self.base_path / "pending" / f"{sanitized_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        nodes = [GraphNode(**node) for node in data.get("nodes", [])]
        edges = [GraphEdge(**edge) for edge in data.get("edges", [])]
        return GraphUpdate(
            document_id=data["document_id"],
            document_name=data["document_name"],
            nodes=nodes,
            edges=edges,
            auto_generated=data.get("auto_generated", True),
            notes=data.get("notes"),
        )

    def delete_pending(self, document_id: str) -> None:
        """Delete pending update by original document_id (will sanitize)."""
        safe_id = self._sanitize_filename(document_id)
        self.delete_pending_by_filename(safe_id)

    def delete_pending_by_filename(self, sanitized_id: str) -> None:
        """Delete pending update by sanitized filename (no further sanitization)."""
        path = self.base_path / "pending" / f"{sanitized_id}.json"
        if path.exists():
            path.unlink()

    # ------------------------------------------------------------------#
    # Accepted graph
    # ------------------------------------------------------------------#
    def append_to_graph(self, update: GraphUpdate) -> None:
        """Append nodes/edges to the accepted graph log."""
        record = {
            "document_id": update.document_id,
            "document_name": update.document_name,
            "nodes": [node.__dict__ for node in update.nodes],
            "edges": [edge.__dict__ for edge in update.edges],
        }
        with self.accepted_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record))
            f.write("\n")

    def load_graph_records(self) -> Iterable[dict]:
        """Iterate over accepted graph records."""
        if not self.accepted_path.exists():
            return []
        with self.accepted_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

