"""Reciprocal Rank Fusion (RRF) for merging search results.

Pure function implementation for fusing results from multiple retrievers.
"""

from typing import Any


class ScoredChunk:
    """A chunk with score from a retriever."""

    def __init__(self, chunk_id: str, score: float, source: str = "unknown"):
        """Initialize scored chunk.

        Args:
            chunk_id: Chunk identifier
            score: Score from retriever
            source: Source retriever name (e.g., "semantic", "bm25")
        """
        self.chunk_id = chunk_id
        self.score = score
        self.source = source
        self.rank = 0  # Will be set during fusion
        self.provenance_scores: dict[str, float] = {}  # Scores from each source


def fuse(
    results_a: list[dict[str, Any]], results_b: list[dict[str, Any]], k: int = 60
) -> list[ScoredChunk]:
    """Fuse two result lists using Reciprocal Rank Fusion (RRF).

    RRF formula: score = sum(1 / (k + rank)) for each result list

    Args:
        results_a: First result list, each dict must have "id" and optionally "score"
        results_b: Second result list, each dict must have "id" and optionally "score"
        k: RRF constant (default 60)

    Returns:
        List of ScoredChunk objects sorted by combined RRF score (descending)
    """
    # Convert to ScoredChunk objects with ranks
    scored_a = []
    for rank, result in enumerate(results_a, start=1):
        chunk_id = result.get("id") or result.get("chunk_id")
        score = result.get("score", 0.0)
        if chunk_id:
            chunk = ScoredChunk(chunk_id, score, source="semantic")
            chunk.rank = rank
            scored_a.append(chunk)

    scored_b = []
    for rank, result in enumerate(results_b, start=1):
        chunk_id = result.get("id") or result.get("chunk_id")
        score = result.get("score", 0.0)
        if chunk_id:
            chunk = ScoredChunk(chunk_id, score, source="bm25")
            chunk.rank = rank
            scored_b.append(chunk)

    # Build map of chunk_id -> RRF score
    rrf_scores: dict[str, ScoredChunk] = {}

    # Add scores from first list
    for chunk in scored_a:
        if chunk.chunk_id not in rrf_scores:
            rrf_scores[chunk.chunk_id] = ScoredChunk(
                chunk.chunk_id, 0.0, source=chunk.source
            )
        rrf_scores[chunk.chunk_id].score += 1.0 / (k + chunk.rank)
        # Keep provenance scores
        if not hasattr(rrf_scores[chunk.chunk_id], "provenance_scores"):
            rrf_scores[chunk.chunk_id].provenance_scores = {}
        rrf_scores[chunk.chunk_id].provenance_scores[chunk.source] = chunk.score

    # Add scores from second list
    for chunk in scored_b:
        if chunk.chunk_id not in rrf_scores:
            rrf_scores[chunk.chunk_id] = ScoredChunk(
                chunk.chunk_id, 0.0, source=chunk.source
            )
        rrf_scores[chunk.chunk_id].score += 1.0 / (k + chunk.rank)
        # Keep provenance scores
        if not hasattr(rrf_scores[chunk.chunk_id], "provenance_scores"):
            rrf_scores[chunk.chunk_id].provenance_scores = {}
        rrf_scores[chunk.chunk_id].provenance_scores[chunk.source] = chunk.score

    # Sort by RRF score descending
    fused_results = sorted(rrf_scores.values(), key=lambda x: x.score, reverse=True)

    return fused_results

