"""Tests for adaptive chunker."""

from pathlib import Path

import pytest

from src.config_loader import Settings
from src.ingestion.chunkers.adaptive_chunker import AdaptiveChunker
from src.schema import DocumentType, ParsedDocument


@pytest.fixture
def sample_witness_statement() -> ParsedDocument:
    """Create a sample witness statement document."""
    # Create text with multiple paragraphs (simulating ~2000 tokens)
    text = "\n\n".join([f"Paragraph {i}: " + "This is a witness statement. " * 50 for i in range(1, 21)])
    return ParsedDocument(
        file_path="test_witness.pdf",
        file_name="test_witness.pdf",
        document_type="witness_statement",
        text=text,
        pages=[1] * 20,
        paragraphs=[
            {
                "text": f"Paragraph {i}: " + "This is a witness statement. " * 50,
                "page": 1,
                "paragraph": i,
                "char_start": i * 1000,
                "char_end": (i + 1) * 1000,
            }
            for i in range(1, 21)
        ],
    )


@pytest.fixture
def sample_statute() -> ParsedDocument:
    """Create a sample statute document."""
    text = "\n\n".join([f"Section {i}: " + "Legal text here. " * 30 for i in range(1, 11)])
    return ParsedDocument(
        file_path="test_statute.pdf",
        file_name="test_statute.pdf",
        document_type="statute",
        text=text,
        pages=[1] * 10,
        paragraphs=[
            {
                "text": f"Section {i}: " + "Legal text here. " * 30,
                "page": 1,
                "paragraph": i,
                "char_start": i * 500,
                "char_end": (i + 1) * 500,
            }
            for i in range(1, 11)
        ],
    )


def test_chunker_initialization():
    """Test chunker can be initialized."""
    chunker = AdaptiveChunker()
    assert chunker is not None
    assert chunker.settings is not None


def test_chunker_with_custom_settings():
    """Test chunker accepts custom settings."""
    settings = Settings()
    chunker = AdaptiveChunker(settings=settings)
    assert chunker.settings == settings


def test_chunk_size_witness_statement():
    """Test witness statement chunk size is 1024 tokens."""
    chunker = AdaptiveChunker()
    size = chunker._get_chunk_size("witness_statement")
    assert size == 1024


def test_chunk_size_statute():
    """Test statute chunk size is 512 tokens."""
    chunker = AdaptiveChunker()
    size = chunker._get_chunk_size("statute")
    assert size == 512


def test_overlap_witness_statement():
    """Test witness statement overlap is 500 tokens."""
    chunker = AdaptiveChunker()
    overlap = chunker._get_overlap("witness_statement")
    assert overlap == 512


def test_overlap_statute():
    """Test statute overlap is 200 tokens."""
    chunker = AdaptiveChunker()
    overlap = chunker._get_overlap("statute")
    assert overlap == 256


def test_chunk_document_witness_statement(sample_witness_statement: ParsedDocument):
    """Test chunking witness statement produces chunks with correct size."""
    chunker = AdaptiveChunker()
    chunks = chunker.chunk_document(sample_witness_statement)

    assert len(chunks) > 0

    # Check chunk sizes (approximate: 1024 tokens * 4 chars/token = 4096 chars)
    target_size_chars = 1024 * chunker.CHARS_PER_TOKEN
    for chunk in chunks:
        # Chunks should be roughly target size (allow 20% tolerance)
        assert len(chunk.text) <= target_size_chars * 1.2
        assert chunk.document_type == "witness_statement"
        assert chunk.file_name == "test_witness.pdf"


def test_chunk_document_statute(sample_statute: ParsedDocument):
    """Test chunking statute produces chunks with correct size."""
    chunker = AdaptiveChunker()
    chunks = chunker.chunk_document(sample_statute)

    assert len(chunks) > 0

    # Check chunk sizes (approximate: 512 tokens * 4 chars/token = 2048 chars)
    target_size_chars = 512 * chunker.CHARS_PER_TOKEN
    for chunk in chunks:
        assert len(chunk.text) <= target_size_chars * 1.2
        assert chunk.document_type == "statute"


def test_chunk_preserves_metadata(sample_witness_statement: ParsedDocument):
    """Test chunks preserve page/paragraph metadata."""
    chunker = AdaptiveChunker()
    chunks = chunker.chunk_document(sample_witness_statement)

    # At least some chunks should have metadata
    chunks_with_metadata = [c for c in chunks if c.page_number is not None or c.paragraph_number is not None]
    assert len(chunks_with_metadata) > 0


def test_chunk_char_offsets(sample_witness_statement: ParsedDocument):
    """Test chunks have valid char offsets."""
    chunker = AdaptiveChunker()
    chunks = chunker.chunk_document(sample_witness_statement)

    for chunk in chunks:
        assert chunk.char_start >= 0
        assert chunk.char_end > chunk.char_start
        assert abs((chunk.char_end - chunk.char_start) - len(chunk.text)) <= 5


def test_overlap_boundary_continuity(sample_witness_statement: ParsedDocument):
    """Test that chunks have proper overlap for boundary continuity."""
    chunker = AdaptiveChunker()
    chunks = chunker.chunk_document(sample_witness_statement)

    if len(chunks) < 2:
        pytest.skip("Need at least 2 chunks to test overlap")

    # Check overlap between consecutive chunks
    expected_overlap_chars = 500 * chunker.CHARS_PER_TOKEN  # witness_statement overlap

    for i in range(len(chunks) - 1):
        chunk1 = chunks[i]
        chunk2 = chunks[i + 1]

        # Check if chunks overlap or are adjacent
        # Overlap exists if chunk2 starts before chunk1 ends
        if chunk2.char_start < chunk1.char_end:
            overlap_size = chunk1.char_end - chunk2.char_start
            # Overlap should be roughly expected size (allow 50% tolerance)
            assert overlap_size >= expected_overlap_chars * 0.5, (
                f"Chunk {i+1}->{i+2}: overlap {overlap_size} chars "
                f"is less than expected {expected_overlap_chars * 0.5}"
            )


def test_chunk_document_id_consistency(sample_witness_statement: ParsedDocument):
    """Test that all chunks from same document have same document_id."""
    chunker = AdaptiveChunker()
    chunks = chunker.chunk_document(sample_witness_statement)

    if len(chunks) == 0:
        pytest.skip("No chunks generated")

    document_id = chunks[0].document_id
    for chunk in chunks:
        assert chunk.document_id == document_id


def test_chunk_unique_ids(sample_witness_statement: ParsedDocument):
    """Test that each chunk has a unique chunk_id."""
    chunker = AdaptiveChunker()
    chunks = chunker.chunk_document(sample_witness_statement)

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found"




