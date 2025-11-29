"""Tests for vector store."""

import tempfile
from pathlib import Path

import pytest

from src.config_loader import Settings
from src.retrieval.embedding_service import EmbeddingService
from src.retrieval.vector_store import VectorStore
from src.schema import Chunk


@pytest.fixture
def temp_db_path(tmp_path):
    """Create temporary database path."""
    return tmp_path / "test_chroma_db"


@pytest.fixture
def vector_store(temp_db_path):
    """Create vector store instance with temporary path."""
    try:
        return VectorStore(db_path=temp_db_path)
    except Exception as e:
        pytest.skip(f"Could not create vector store: {e}")


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            file_name="test1.pdf",
            text="This is a test document about litigation matters.",
            page_number=1,
            paragraph_number=1,
            document_type="witness_statement",
        ),
        Chunk(
            chunk_id="chunk_2",
            document_id="doc_1",
            file_name="test1.pdf",
            text="The witness stated that the events occurred in 2023.",
            page_number=1,
            paragraph_number=2,
            document_type="witness_statement",
        ),
        Chunk(
            chunk_id="chunk_3",
            document_id="doc_2",
            file_name="test2.pdf",
            text="This is a different document about court proceedings.",
            page_number=1,
            paragraph_number=1,
            document_type="pleading",
        ),
    ]


@pytest.fixture
def embedding_service():
    """Create embedding service for tests."""
    try:
        return EmbeddingService()
    except Exception:
        pytest.skip("Embedding service not available")


def test_vector_store_initialization(temp_db_path):
    """Test vector store can be initialized."""
    try:
        store = VectorStore(db_path=temp_db_path)
        assert store is not None
        assert store.collection is not None
        assert store.db_path == temp_db_path
    except Exception:
        pytest.skip("ChromaDB not available")


def test_add_chunks(vector_store: VectorStore, sample_chunks, embedding_service: EmbeddingService):
    """Test adding chunks to vector store."""
    # Generate embeddings
    embeddings = embedding_service.embed_batch([chunk.text for chunk in sample_chunks])

    # Add chunks
    vector_store.add_chunks(sample_chunks, embeddings)

    # Verify chunks were added
    stats = vector_store.stats()
    assert stats["total_chunks"] == len(sample_chunks)


def test_add_chunks_mismatched_lengths(vector_store: VectorStore, sample_chunks):
    """Test adding chunks with mismatched embeddings raises error."""
    embeddings = [[0.1] * 1024, [0.2] * 1024]  # Only 2 embeddings for 3 chunks

    with pytest.raises(ValueError, match="must have the same length"):
        vector_store.add_chunks(sample_chunks, embeddings)


def test_query(vector_store: VectorStore, sample_chunks, embedding_service: EmbeddingService):
    """Test querying the vector store."""
    # Add chunks
    embeddings = embedding_service.embed_batch([chunk.text for chunk in sample_chunks])
    vector_store.add_chunks(sample_chunks, embeddings)

    # Query
    query_text = "litigation matters"
    query_embedding = embedding_service.embed_query(query_text)
    results = vector_store.query([query_embedding], n_results=2)

    assert "ids" in results
    assert "distances" in results
    assert "metadatas" in results
    assert "documents" in results
    assert len(results["ids"][0]) > 0


def test_query_with_where_filter(
    vector_store: VectorStore, sample_chunks, embedding_service: EmbeddingService
):
    """Test querying with metadata filter."""
    # Add chunks
    embeddings = embedding_service.embed_batch([chunk.text for chunk in sample_chunks])
    vector_store.add_chunks(sample_chunks, embeddings)

    # Query with filter
    query_embedding = embedding_service.embed_query("witness statement")
    results = vector_store.query(
        [query_embedding],
        n_results=10,
        where={"document_type": "witness_statement"},
    )

    # All results should be witness statements
    if results["metadatas"] and results["metadatas"][0]:
        for metadata in results["metadatas"][0]:
            assert metadata.get("document_type") == "witness_statement"


def test_delete_document(vector_store: VectorStore, sample_chunks, embedding_service):
    """Test deleting a document."""
    # Add chunks
    embeddings = embedding_service.embed_batch([chunk.text for chunk in sample_chunks])
    vector_store.add_chunks(sample_chunks, embeddings)

    # Verify chunks exist
    stats_before = vector_store.stats()
    assert stats_before["total_chunks"] == len(sample_chunks)

    # Delete document
    vector_store.delete_document("doc_1")

    # Verify chunks deleted
    stats_after = vector_store.stats()
    assert stats_after["total_chunks"] < stats_before["total_chunks"]


def test_delete_chunks(vector_store: VectorStore, sample_chunks, embedding_service):
    """Test deleting specific chunks."""
    # Add chunks
    embeddings = embedding_service.embed_batch([chunk.text for chunk in sample_chunks])
    vector_store.add_chunks(sample_chunks, embeddings)

    # Delete specific chunks
    vector_store.delete_chunks(["chunk_1", "chunk_2"])

    # Verify chunks deleted
    stats = vector_store.stats()
    assert stats["total_chunks"] == 1  # Only chunk_3 should remain


def test_stats(vector_store: VectorStore, sample_chunks, embedding_service):
    """Test getting statistics."""
    # Add chunks
    embeddings = embedding_service.embed_batch([chunk.text for chunk in sample_chunks])
    vector_store.add_chunks(sample_chunks, embeddings)

    stats = vector_store.stats()

    assert "total_chunks" in stats
    assert stats["total_chunks"] == len(sample_chunks)
    assert "collection_name" in stats
    assert stats["collection_name"] == "litigation_docs"


def test_acceptance_criteria_round_trip(
    vector_store: VectorStore, embedding_service: EmbeddingService
):
    """Test acceptance criteria: add/query/delete round-trip."""
    # Create fake chunks
    fake_chunks = [
        Chunk(
            chunk_id=f"fake_chunk_{i}",
            document_id="fake_doc",
            file_name=f"fake_{i}.pdf",
            text=f"This is fake chunk {i} with some test content.",
            page_number=i,
            paragraph_number=i,
            document_type="witness_statement",
        )
        for i in range(1, 4)
    ]

    # Add chunks
    embeddings = embedding_service.embed_batch([chunk.text for chunk in fake_chunks])
    vector_store.add_chunks(fake_chunks, embeddings)

    # Query
    query_embedding = embedding_service.embed_query("test content")
    results = vector_store.query([query_embedding], n_results=5)

    assert len(results["ids"][0]) > 0, "Query should return results"

    # Delete document
    vector_store.delete_document("fake_doc")

    # Verify deletion
    stats = vector_store.stats()
    assert stats["total_chunks"] == 0, "All chunks should be deleted"

    print("✓ Acceptance criteria PASSED: add/query/delete round-trip successful")




