"""Tests for RRF fusion."""

import pytest

from src.retrieval.rrf import ScoredChunk, fuse


def test_fuse_basic():
    """Test basic RRF fusion."""
    results_a = [{"id": "chunk_1", "score": 0.9}, {"id": "chunk_2", "score": 0.8}]
    results_b = [{"id": "chunk_2", "score": 0.7}, {"id": "chunk_3", "score": 0.6}]

    fused = fuse(results_a, results_b, k=60)

    assert len(fused) == 3  # chunk_1, chunk_2, chunk_3
    assert fused[0].chunk_id == "chunk_2"  # Should be top (appears in both)
    assert fused[0].score > fused[1].score  # Scores should be descending


def test_fuse_uneven_lengths():
    """Test RRF fusion with uneven result lists."""
    results_a = [{"id": "chunk_1", "score": 0.9}]
    results_b = [
        {"id": "chunk_2", "score": 0.8},
        {"id": "chunk_3", "score": 0.7},
        {"id": "chunk_4", "score": 0.6},
    ]

    fused = fuse(results_a, results_b, k=60)

    assert len(fused) == 4
    assert fused[0].chunk_id == "chunk_1"  # Top rank in first list


def test_fuse_ties():
    """Test RRF fusion handles ties correctly."""
    results_a = [{"id": "chunk_1", "score": 0.9}, {"id": "chunk_2", "score": 0.8}]
    results_b = [{"id": "chunk_3", "score": 0.9}, {"id": "chunk_4", "score": 0.8}]

    fused = fuse(results_a, results_b, k=60)

    # All should be present
    chunk_ids = [chunk.chunk_id for chunk in fused]
    assert "chunk_1" in chunk_ids
    assert "chunk_2" in chunk_ids
    assert "chunk_3" in chunk_ids
    assert "chunk_4" in chunk_ids


def test_fuse_missing_ids():
    """Test RRF fusion handles missing IDs gracefully."""
    results_a = [{"id": "chunk_1", "score": 0.9}, {"score": 0.8}]  # Missing ID
    results_b = [{"id": "chunk_2", "score": 0.7}]

    fused = fuse(results_a, results_b, k=60)

    # Should only include chunks with valid IDs
    chunk_ids = [chunk.chunk_id for chunk in fused]
    assert "chunk_1" in chunk_ids
    assert "chunk_2" in chunk_ids
    assert len(fused) == 2


def test_scored_chunk():
    """Test ScoredChunk class."""
    chunk = ScoredChunk("chunk_1", 0.9, source="semantic")
    assert chunk.chunk_id == "chunk_1"
    assert chunk.score == 0.9
    assert chunk.source == "semantic"
    assert chunk.rank == 0  # Default rank




