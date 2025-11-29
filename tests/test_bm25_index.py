"""Tests for BM25 index."""

import tempfile
from pathlib import Path

import pytest

from src.config_loader import Settings
from src.retrieval.bm25_index import BM25Index
from src.schema import Chunk


@pytest.fixture
def temp_index_path(tmp_path):
    """Create temporary index path."""
    return tmp_path / "test_bm25_index"


@pytest.fixture
def bm25_index(temp_index_path):
    """Create BM25 index instance."""
    return BM25Index(index_path=temp_index_path)


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            file_name="witness_statement.pdf",
            text="John Smith testified on January 15, 2023. He stated that the events occurred in 2022.",
            page_number=1,
            paragraph_number=1,
            document_type="witness_statement",
        ),
        Chunk(
            chunk_id="chunk_2",
            document_id="doc_1",
            file_name="witness_statement.pdf",
            text="The meeting took place on Page 1.2 of the document. See Para 3.4 for details.",
            page_number=1,
            paragraph_number=2,
            document_type="witness_statement",
        ),
        Chunk(
            chunk_id="chunk_3",
            document_id="doc_2",
            file_name="contract.pdf",
            text="The contract was signed on 2023.12.25. Version 2.1.0 of the agreement applies.",
            page_number=1,
            paragraph_number=1,
            document_type="contract",
        ),
    ]


def test_bm25_index_initialization(temp_index_path):
    """Test BM25 index can be initialized."""
    index = BM25Index(index_path=temp_index_path)
    assert index is not None
    assert index.bm25 is None
    assert len(index.chunks) == 0


def test_tokenize_preserves_citations(bm25_index: BM25Index):
    """Test that tokenization preserves dots in citations."""
    text = "See Page 1.2 and Para 3.4 for details."
    tokens = bm25_index._tokenize(text)

    # Should preserve "1.2" and "3.4" as single tokens or with dots
    text_lower = text.lower()
    assert any("1.2" in token or "page 1.2" in " ".join(tokens) for token in tokens)
    assert any("3.4" in token or "para 3.4" in " ".join(tokens) for token in tokens)


def test_tokenize_preserves_dates(bm25_index: BM25Index):
    """Test that tokenization preserves dots in dates."""
    text = "The event occurred on 2023.12.25 and 1.2.2023."
    tokens = bm25_index._tokenize(text)

    # Should preserve date patterns
    text_lower = " ".join(tokens).lower()
    assert "2023.12.25" in text_lower or any("2023" in t and "12" in t and "25" in t for t in tokens)


def test_build_index(bm25_index: BM25Index, sample_chunks):
    """Test building index from chunks."""
    bm25_index.build_index(sample_chunks)

    assert bm25_index.bm25 is not None
    assert len(bm25_index.chunks) == len(sample_chunks)
    assert len(bm25_index.chunk_id_to_index) == len(sample_chunks)


def test_build_index_empty(bm25_index: BM25Index):
    """Test building index with empty chunks raises error."""
    with pytest.raises(ValueError, match="empty chunk list"):
        bm25_index.build_index([])


def test_search(bm25_index: BM25Index, sample_chunks):
    """Test searching the index."""
    bm25_index.build_index(sample_chunks)

    # Search for a name
    results = bm25_index.search("John Smith", top_k=5)
    assert len(results) > 0
    assert results[0][0] == "chunk_1"  # Should find chunk with John Smith

    # Search for a date
    results = bm25_index.search("January 15, 2023", top_k=5)
    assert len(results) > 0


def test_search_empty_query(bm25_index: BM25Index, sample_chunks):
    """Test searching with empty query."""
    bm25_index.build_index(sample_chunks)
    results = bm25_index.search("", top_k=5)
    assert results == []


def test_search_not_built(bm25_index: BM25Index):
    """Test searching without building index raises error."""
    with pytest.raises(RuntimeError, match="not built"):
        bm25_index.search("test")


def test_save_and_load(bm25_index: BM25Index, sample_chunks):
    """Test saving and loading index."""
    # Build and save
    bm25_index.build_index(sample_chunks)
    bm25_index.save()

    # Create new index and load
    new_index = BM25Index(index_path=bm25_index.index_path)
    new_index.load()

    assert new_index.bm25 is not None
    assert len(new_index.chunks) == len(sample_chunks)

    # Verify search works after load
    results = new_index.search("John Smith", top_k=5)
    assert len(results) > 0


def test_delete_document(bm25_index: BM25Index, sample_chunks):
    """Test deleting a document."""
    bm25_index.build_index(sample_chunks)
    assert len(bm25_index.chunks) == 3

    bm25_index.delete_document("doc_1")
    assert len(bm25_index.chunks) == 1  # Only doc_2 should remain
    assert bm25_index.chunks[0].document_id == "doc_2"


def test_delete_chunks(bm25_index: BM25Index, sample_chunks):
    """Test deleting specific chunks."""
    bm25_index.build_index(sample_chunks)
    assert len(bm25_index.chunks) == 3

    bm25_index.delete_chunks(["chunk_1", "chunk_2"])
    assert len(bm25_index.chunks) == 1
    assert bm25_index.chunks[0].chunk_id == "chunk_3"


def test_stats(bm25_index: BM25Index, sample_chunks):
    """Test getting statistics."""
    stats = bm25_index.stats()
    assert stats["total_chunks"] == 0
    assert stats["index_built"] is False

    bm25_index.build_index(sample_chunks)
    stats = bm25_index.stats()
    assert stats["total_chunks"] == 3
    assert stats["index_built"] is True


def test_acceptance_criteria_name_search(bm25_index: BM25Index, sample_chunks):
    """Test acceptance criteria: searching for known name returns expected chunks."""
    bm25_index.build_index(sample_chunks)

    # Search for "John Smith" - should return chunk_1
    results = bm25_index.search("John Smith", top_k=5)
    assert len(results) > 0, "Should find results for 'John Smith'"
    assert results[0][0] == "chunk_1", "Top result should be chunk_1"

    print("✓ Acceptance criteria PASSED: name search returns expected chunk")


def test_acceptance_criteria_date_search(bm25_index: BM25Index, sample_chunks):
    """Test acceptance criteria: searching for known date returns expected chunks."""
    bm25_index.build_index(sample_chunks)

    # Search for date - should return relevant chunks
    results = bm25_index.search("2023", top_k=5)
    assert len(results) > 0, "Should find results for '2023'"

    # Should find chunks mentioning 2023
    chunk_ids = [result[0] for result in results]
    assert "chunk_1" in chunk_ids or "chunk_3" in chunk_ids

    print("✓ Acceptance criteria PASSED: date search returns expected chunks")




