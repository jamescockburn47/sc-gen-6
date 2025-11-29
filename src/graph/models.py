"""Graph data models for litigation-aware RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


GraphNodeType = Literal["document", "chunk", "party", "event", "statute", "issue"]
GraphEdgeType = Literal[
    "references",
    "supports",
    "contradicts",
    "mentions",
    "chronology",
    "statutory_basis",
]


@dataclass
class GraphNode:
    """Node representing a document, chunk, or entity."""

    node_id: str
    label: str
    node_type: GraphNodeType
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Typed edge between nodes."""

    source: str
    target: str
    relation: GraphEdgeType
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphUpdate:
    """Update payload produced during ingestion."""

    document_id: str
    document_name: str
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    auto_generated: bool = True
    notes: str | None = None

