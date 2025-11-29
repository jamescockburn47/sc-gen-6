"""User review workflow for graph updates."""

from __future__ import annotations

from typing import Iterable

from src.graph.models import GraphUpdate
from src.graph.storage import GraphStore


class GraphReviewManager:
    """Coordinates pending graph updates and approvals."""

    def __init__(self, store: GraphStore | None = None):
        self.store = store or GraphStore()

    def queue_update(self, update: GraphUpdate) -> None:
        self.store.queue_update(update)

    def pending_documents(self) -> list[str]:
        """Return list of pending document IDs (sanitized filenames)."""
        return [path.stem for path in self.store.list_pending()]

    def fetch_update(self, sanitized_id: str) -> GraphUpdate | None:
        """Fetch pending update by sanitized filename (from pending_documents)."""
        return self.store.load_pending_by_filename(sanitized_id)

    def approve_update(self, sanitized_id: str) -> bool:
        """Approve update by sanitized filename."""
        update = self.store.load_pending_by_filename(sanitized_id)
        if not update:
            return False
        self.store.append_to_graph(update)
        self.store.delete_pending_by_filename(sanitized_id)
        return True

    def discard_update(self, sanitized_id: str) -> bool:
        """Discard update by sanitized filename."""
        update = self.store.load_pending_by_filename(sanitized_id)
        if not update:
            return False
        self.store.delete_pending_by_filename(sanitized_id)
        return True

    def iter_graph(self) -> Iterable[dict]:
        return self.store.load_graph_records()

