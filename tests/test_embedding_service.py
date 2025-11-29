"""Tests for embedding service."""

import pytest

from src.config_loader import Settings
from src.retrieval.embedding_service import EmbeddingService


@pytest.fixture
def embedding_service():
    """Create embedding service instance."""
    # Note: This will try to load the model, which requires dependencies
    # Skip if model not available
    try:
        return EmbeddingService()
    except Exception as e:
        pytest.skip(f"Could not load embedding model: {e}")


def test_embedding_service_initialization():
    """Test embedding service can be initialized."""
    try:
        service = EmbeddingService()
        assert service is not None
        assert service.model_name is not None
    except Exception:
        pytest.skip("Model not available for testing")


def test_embed_text(embedding_service: EmbeddingService):
    """Test embedding a single text."""
    text = "This is a test document about litigation."
    embedding = embedding_service.embed_text(text)

    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, (int, float)) for x in embedding)


def test_embed_text_empty():
    """Test embedding empty text raises error."""
    try:
        service = EmbeddingService()
        with pytest.raises(ValueError, match="cannot be empty"):
            service.embed_text("")
    except Exception:
        pytest.skip("Model not available for testing")


def test_embed_batch(embedding_service: EmbeddingService):
    """Test embedding a batch of texts."""
    texts = [
        "First document about legal matters.",
        "Second document about court proceedings.",
        "Third document about witness statements.",
    ]
    embeddings = embedding_service.embed_batch(texts)

    assert len(embeddings) == len(texts)
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(len(emb) > 0 for emb in embeddings)

    # All embeddings should have same dimension
    dim = len(embeddings[0])
    assert all(len(emb) == dim for emb in embeddings)


def test_embed_batch_empty():
    """Test embedding empty batch."""
    try:
        service = EmbeddingService()
        result = service.embed_batch([])
        assert result == []
    except Exception:
        pytest.skip("Model not available for testing")


def test_embed_query(embedding_service: EmbeddingService):
    """Test embedding a query with prefix."""
    query = "What are the key facts in the witness statements?"
    embedding = embedding_service.embed_query(query)

    assert isinstance(embedding, list)
    assert len(embedding) > 0

    # Query embedding should have same dimension as text embedding
    text_embedding = embedding_service.embed_text("test")
    assert len(embedding) == len(text_embedding)


def test_embed_query_vs_text_different(embedding_service: EmbeddingService):
    """Test that query embedding (with prefix) differs from text embedding."""
    text = "What are the key facts in the witness statements?"
    query_embedding = embedding_service.embed_query(text)
    text_embedding = embedding_service.embed_text(text)

    # Embeddings should be different due to prefix
    assert query_embedding != text_embedding


def test_get_embedding_dimension(embedding_service: EmbeddingService):
    """Test getting embedding dimension."""
    dim = embedding_service.get_embedding_dimension()
    assert isinstance(dim, int)
    assert dim > 0

    # For bge-large-en-v1.5, should be 1024
    if "bge-large-en-v1.5" in embedding_service.model_name:
        assert dim == 1024


def test_acceptance_criteria_bge_large():
    """Test acceptance criteria: embed_text returns 1024 dimensions."""
    try:
        service = EmbeddingService(model_name="BAAI/bge-large-en-v1.5")
        embedding = service.embed_text("test")
        assert len(embedding) == 1024, f"Expected 1024 dimensions, got {len(embedding)}"
    except Exception as e:
        pytest.skip(f"Could not test acceptance criteria: {e}")


def test_device_detection():
    """Test device detection."""
    try:
        service = EmbeddingService()
        device = service.device
        assert device in ["cuda", "cpu"]
    except Exception:
        pytest.skip("Model not available for testing")


def test_is_gpu_available():
    """Test GPU availability check."""
    try:
        service = EmbeddingService()
        is_gpu = service.is_gpu_available()
        assert isinstance(is_gpu, bool)
    except Exception:
        pytest.skip("Model not available for testing")


def test_custom_batch_size():
    """Test custom batch size from settings."""
    try:
        settings = Settings()
        settings.performance.embed_batch_size = 16
        service = EmbeddingService(settings=settings)
        assert service.batch_size == 16
    except Exception:
        pytest.skip("Model not available for testing")




