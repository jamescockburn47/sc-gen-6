"""Document chunkers with adaptive strategies per document type.

Factory function ``get_chunker()`` returns the best available chunker
based on ``config.chunking.strategy``, falling back gracefully when
dependencies (LegalBERT, torch, etc.) are not installed.

Strategies
----------
* ``semantic``  – LegalBERT-based similarity-drop splitting with
  contextual enrichment.  Best quality for legal text.
* ``agentic``   – LLM-guided multi-pass splitting (requires Ollama).
* ``robust``    – Fast character-based RCTS.  Always available.
"""

from __future__ import annotations

from typing import Optional, Protocol

from src.config_loader import Settings, get_settings
from src.schema import Chunk, ParsedDocument


class Chunker(Protocol):
    """Minimal interface every chunker must satisfy."""

    def chunk_document(self, document: ParsedDocument) -> list[Chunk]: ...


def get_chunker(settings: Optional[Settings] = None) -> Chunker:
    """Return the best available chunker based on config.

    Falls back automatically:
        semantic → robust
        agentic → semantic → robust

    Args:
        settings: Optional ``Settings`` override.

    Returns:
        A chunker instance satisfying the ``Chunker`` protocol.
    """
    settings = settings or get_settings()
    strategy = getattr(settings.chunking, "strategy", "semantic").lower()

    if strategy == "agentic":
        try:
            from src.ingestion.chunkers.agentic_chunker import AgenticChunker

            return AgenticChunker(settings=settings)
        except Exception as exc:
            print(f"[Chunker] Agentic chunker unavailable ({exc}), trying semantic…")
            strategy = "semantic"

    if strategy == "semantic":
        try:
            from src.ingestion.chunkers.semantic_chunker import SemanticHybridChunker

            chunker = SemanticHybridChunker(settings=settings)
            return chunker
        except Exception as exc:
            print(f"[Chunker] Semantic chunker unavailable ({exc}), falling back to robust")

    # Robust is the guaranteed fallback — pure Python, no ML deps
    from src.ingestion.chunkers.adaptive_chunker import RobustChunker

    return RobustChunker(settings=settings)


# Legacy alias
from src.ingestion.chunkers.adaptive_chunker import AdaptiveChunker

__all__ = ["AdaptiveChunker", "Chunker", "get_chunker"]
