"""Adaptive chunker with RCTS (Recursive Character Text Splitting).

Uses document-type-aware settings from config.yaml to chunk documents
with high overlap, preserving page/paragraph/section metadata and char offsets.
"""

import hashlib
import uuid
from pathlib import Path
from typing import Optional

from src.config_loader import Settings, get_settings
from src.schema import Chunk, DocumentType, ParsedDocument


class AdaptiveChunker:
    """Adaptive chunker with RCTS and document-type-aware settings.

    Chunks documents using Recursive Character Text Splitting with
    configurable separators, sizes, and overlaps per document type.
    Preserves metadata (page, paragraph, section, char offsets).
    """

    # Approximate tokens per character for English (rough estimate: 1 token ≈ 4 chars)
    CHARS_PER_TOKEN = 4

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize chunker with settings.

        Args:
            settings: Settings instance. If None, loads from config.
        """
        self.settings = settings or get_settings()

    def chunk_document(self, document: ParsedDocument) -> list[Chunk]:
        """Chunk a parsed document into chunks with metadata.

        Args:
            document: ParsedDocument to chunk

        Returns:
            List of Chunk objects with preserved metadata
        """
        doc_type = document.document_type

        # Get chunking parameters for this document type
        chunk_size_tokens = self._get_chunk_size(doc_type)
        overlap_tokens = self._get_overlap(doc_type)
        separators = self.settings.chunking.separators

        # Convert token sizes to character sizes (approximate)
        chunk_size_chars = chunk_size_tokens * self.CHARS_PER_TOKEN
        overlap_chars = overlap_tokens * self.CHARS_PER_TOKEN

        # Generate document ID
        document_id = self._generate_document_id(document)

        # Build text segments with metadata mapping
        text_segments = self._build_text_segments(document)

        # Perform RCTS chunking
        chunks = self._rcts_chunk(
            text_segments=text_segments,
            chunk_size_chars=chunk_size_chars,
            overlap_chars=overlap_chars,
            separators=separators,
            document_id=document_id,
            document=document,
        )

        return chunks

    def _get_chunk_size(self, doc_type: DocumentType) -> int:
        """Get chunk size in tokens for document type.
        
        Uses config values where available, with sensible defaults for
        extended document types (skeleton arguments, expert reports, etc.)
        """
        sizes = self.settings.chunking.sizes
        
        # Primary mappings from config
        size_map = {
            # Core types with config values
            "witness_statement": sizes.witness_statement,
            "court_filing": sizes.court_filing,
            "pleading": sizes.pleading,
            "statute": sizes.statute,
            "contract": sizes.contract,
            "disclosure": sizes.disclosure,
            "email": sizes.email,
            
            # Extended types with sensible defaults
            # Legal arguments - larger chunks to preserve reasoning flow
            "skeleton_argument": sizes.pleading,  # Similar to pleadings
            
            # Expert reports - larger chunks for technical content
            "expert_report": sizes.witness_statement,  # Technical detail like WS
            "schedule_of_loss": sizes.contract,  # Structured like contracts
            "medical_report": sizes.witness_statement,  # Technical detail
            
            # Case law - preserve paragraphs
            "case_law": sizes.court_filing,
            
            # Correspondence - smaller chunks
            "letter": sizes.email,
            
            # Lists and forms - medium
            "disclosure_list": sizes.disclosure,
            "court_form": sizes.disclosure,
            "case_management": sizes.court_filing,
            "chronology": sizes.disclosure,  # Often tabular
            
            # Tribunal/regulatory - like court filings
            "tribunal_document": sizes.court_filing,
            "regulatory_document": sizes.court_filing,
            
            # Fallbacks
            "scanned_pdf": sizes.disclosure,
            "unknown": sizes.disclosure,
        }
        return size_map.get(doc_type, sizes.disclosure)

    def _get_overlap(self, doc_type: DocumentType) -> int:
        """Get overlap in tokens for document type.
        
        Uses config values where available, with sensible defaults for
        extended document types.
        """
        overlaps = self.settings.chunking.overlaps
        
        overlap_map = {
            # Core types with config values
            "witness_statement": overlaps.witness_statement,
            "court_filing": overlaps.court_filing,
            "pleading": overlaps.pleading,
            "statute": overlaps.statute,
            "contract": overlaps.contract,
            "disclosure": overlaps.disclosure,
            "email": overlaps.email,
            
            # Extended types with sensible defaults
            "skeleton_argument": overlaps.pleading,
            "expert_report": overlaps.witness_statement,
            "schedule_of_loss": overlaps.contract,
            "medical_report": overlaps.witness_statement,
            "case_law": overlaps.court_filing,
            "letter": overlaps.email,
            "disclosure_list": overlaps.disclosure,
            "court_form": overlaps.disclosure,
            "case_management": overlaps.court_filing,
            "chronology": overlaps.disclosure,
            "tribunal_document": overlaps.court_filing,
            "regulatory_document": overlaps.court_filing,
            
            # Fallbacks
            "scanned_pdf": overlaps.disclosure,
            "unknown": overlaps.disclosure,
        }
        return overlap_map.get(doc_type, overlaps.disclosure)

    def _generate_document_id(self, document: ParsedDocument) -> str:
        """Generate a unique document ID."""
        # Use file path hash for consistency
        file_hash = hashlib.md5(document.file_path.encode()).hexdigest()[:8]
        return f"doc_{file_hash}"

    def _build_text_segments(self, document: ParsedDocument) -> list[dict]:
        """Build text segments with metadata mapping.

        Creates segments from paragraphs with their metadata (page, para, section).

        Args:
            document: ParsedDocument

        Returns:
            List of segment dicts with text and metadata
        """
        segments = []

        if document.paragraphs:
            # Use paragraph structure
            for para in document.paragraphs:
                segments.append(
                    {
                        "text": para.get("text", ""),
                        "page": para.get("page"),
                        "paragraph": para.get("paragraph"),
                        "section": para.get("section") or para.get("section_header"),
                        "char_start": para.get("char_start", 0),
                        "char_end": para.get("char_end", 0),
                    }
                )
        else:
            # Fallback: split by pages or create single segment
            if document.pages:
                # Split text by pages (rough approximation)
                text = document.text
                page_count = len(set(document.pages)) if document.pages else 1
                chunk_size = len(text) // max(page_count, 1)

                for i, page_num in enumerate(set(document.pages) if document.pages else [1]):
                    start_idx = i * chunk_size
                    end_idx = start_idx + chunk_size if i < page_count - 1 else len(text)
                    segments.append(
                        {
                            "text": text[start_idx:end_idx],
                            "page": page_num,
                            "paragraph": None,
                            "section": None,
                            "char_start": start_idx,
                            "char_end": end_idx,
                        }
                    )
            else:
                # Single segment
                segments.append(
                    {
                        "text": document.text,
                        "page": None,
                        "paragraph": None,
                        "section": None,
                        "char_start": 0,
                        "char_end": len(document.text),
                    }
                )

        return segments

    def _rcts_chunk(
        self,
        text_segments: list[dict],
        chunk_size_chars: int,
        overlap_chars: int,
        separators: list[str],
        document_id: str,
        document: ParsedDocument,
    ) -> list[Chunk]:
        """Perform Recursive Character Text Splitting.

        Args:
            text_segments: List of text segments with metadata
            chunk_size_chars: Target chunk size in characters
            overlap_chars: Overlap size in characters
            separators: List of separators to try (in order)
            document_id: Document ID
            document: Original ParsedDocument

        Returns:
            List of Chunk objects
        """
        # First, combine all segments into full text with metadata tracking
        full_text = ""
        char_to_metadata = {}  # Map char position to metadata

        for segment in text_segments:
            segment_text = segment["text"]
            start_pos = len(full_text)
            full_text += segment_text + "\n\n"
            end_pos = len(full_text)

            # Map character positions to metadata
            for pos in range(start_pos, end_pos):
                char_to_metadata[pos] = {
                    "page": segment.get("page"),
                    "paragraph": segment.get("paragraph"),
                    "section": segment.get("section"),
                }

        # Now perform RCTS on full text
        chunks = self._rcts_split_text(
            text=full_text,
            chunk_size_chars=chunk_size_chars,
            overlap_chars=overlap_chars,
            separators=separators,
            char_to_metadata=char_to_metadata,
            document_id=document_id,
            document=document,
        )

        return chunks

    def _rcts_split_text(
        self,
        text: str,
        chunk_size_chars: int,
        overlap_chars: int,
        separators: list[str],
        char_to_metadata: dict[int, dict],
        document_id: str,
        document: ParsedDocument,
        depth: int = 0,  # Track recursion depth
    ) -> list[Chunk]:
        """Recursively split text using separators.

        Args:
            text: Text to split
            chunk_size_chars: Target chunk size
            overlap_chars: Overlap size
            separators: List of separators to try
            char_to_metadata: Mapping of char positions to metadata
            document_id: Document ID
            document: Original document
            depth: Current recursion depth

        Returns:
            List of Chunk objects
        """
        # Prevent infinite recursion on problematic text
        if depth > 100:
            # Fallback to simple character slicing
            chunks = []
            i = 0
            while i < len(text):
                chunk_end = min(i + chunk_size_chars, len(text))
                chunk_text = text[i:chunk_end]
                if chunk_text.strip():
                    metadata = char_to_metadata.get(i, {})
                    chunks.append(self._create_chunk(
                        text=chunk_text,
                        document_id=document_id,
                        document=document,
                        char_start=i,
                        metadata=metadata,
                    ))
                i += chunk_size_chars - overlap_chars
            return chunks

        if len(text) <= chunk_size_chars:
            # Text fits in one chunk
            metadata = char_to_metadata.get(0, {})
            return [
                self._create_chunk(
                    text=text,
                    document_id=document_id,
                    document=document,
                    char_start=0,
                    metadata=metadata,
                )
            ]

        # Try to split by separators
        for separator in separators:
            if separator in text:
                # Find split point
                split_pos = self._find_split_point(text, chunk_size_chars, separator)
                if split_pos > 0:
                    # Split at this position
                    chunk1_text = text[:split_pos].rstrip()
                    # Get overlap for next chunk
                    overlap_start = max(0, split_pos - overlap_chars)
                    overlap_text = self._get_overlap_text(
                        chunk1_text, overlap_chars, separators
                    )
                    chunk2_start = overlap_start

                    # Create first chunk
                    metadata1 = char_to_metadata.get(0, {})
                    chunk1 = self._create_chunk(
                        text=chunk1_text,
                        document_id=document_id,
                        document=document,
                        char_start=0,
                        metadata=metadata1,
                    )

                    # Recursively process remaining text with overlap
                    remaining_text = text[chunk2_start:]
                    remaining_metadata = {
                        k - chunk2_start: v
                        for k, v in char_to_metadata.items()
                        if k >= chunk2_start
                    }
                    remaining_chunks = self._rcts_split_text(
                        text=remaining_text,
                        chunk_size_chars=chunk_size_chars,
                        overlap_chars=overlap_chars,
                        separators=separators,
                        char_to_metadata=remaining_metadata,
                        document_id=document_id,
                        document=document,
                        depth=depth + 1,  # Increment depth
                    )

                    # Adjust char_start for remaining chunks
                    for chunk in remaining_chunks:
                        chunk.char_start += chunk2_start
                        chunk.char_end += chunk2_start

                    return [chunk1] + remaining_chunks

        # Fallback: character-level split with overlap
        chunks = []
        i = 0
        while i < len(text):
            chunk_end = min(i + chunk_size_chars, len(text))
            chunk_text = text[i:chunk_end]

            if chunk_text.strip():
                metadata = char_to_metadata.get(i, {})
                chunk = self._create_chunk(
                    text=chunk_text,
                    document_id=document_id,
                    document=document,
                    char_start=i,
                    metadata=metadata,
                )
                chunks.append(chunk)

            # Move to next chunk with overlap (back up by overlap_chars)
            if chunk_end < len(text):
                i = max(i + 1, chunk_end - overlap_chars)
            else:
                break

        return chunks

    def _find_split_point(self, text: str, target_size: int, separator: str) -> int:
        """Find the best split point near target_size using separator.

        Args:
            text: Text to split
            target_size: Target chunk size
            separator: Separator to use

        Returns:
            Split position, or 0 if no good split found
        """
        # Look for separator near target_size
        search_start = max(0, target_size - 200)  # Allow 200 chars before target
        search_end = min(len(text), target_size + 200)  # Allow 200 chars after target

        # Find last occurrence of separator in search range
        search_text = text[search_start:search_end]
        last_sep_pos = search_text.rfind(separator)

        if last_sep_pos != -1:
            return search_start + last_sep_pos + len(separator)

        return 0


    def _get_overlap_text(self, text: str, overlap_chars: int, separators: list[str]) -> str:
        """Get overlap text from the end of a chunk.

        Tries to preserve sentence/paragraph boundaries.
        """
        if len(text) <= overlap_chars:
            return text

        overlap_start = len(text) - overlap_chars

        # Try to start overlap at a separator boundary
        for separator in separators:
            idx = text.rfind(separator, overlap_start - 100, overlap_start + 100)
            if idx != -1 and idx >= overlap_start:
                return text[idx + len(separator) :]

        # Fall back to character boundary
        return text[overlap_start:]

    def _create_chunk(
        self,
        text: str,
        document_id: str,
        document: ParsedDocument,
        char_start: int,
        metadata: dict,
    ) -> Chunk:
        """Create a Chunk object with metadata.

        Args:
            text: Chunk text
            document_id: Document ID
            document: Original ParsedDocument
            char_start: Character start offset
            metadata: Metadata dict with page/paragraph/section

        Returns:
            Chunk object
        """
        # Create deterministic ID based on document ID, position, and content
        # This ensures re-ingesting the same file produces the same chunk IDs
        # and allows for efficient updates/deduplication
        content_hash = hashlib.md5(f"{char_start}:{text}".encode("utf-8")).hexdigest()[:12]
        chunk_id = f"{document_id}_{content_hash}"
        char_end = char_start + len(text)

        # Build chunk metadata from document metadata
        # This ensures email from/to/subject, dates, etc. are preserved
        chunk_metadata = {}
        if document.metadata:
            # Copy relevant metadata fields
            for key in ["from", "to", "cc", "subject", "date", "message_id",
                       "document_date", "parties", "case_number", "author"]:
                if key in document.metadata and document.metadata[key]:
                    chunk_metadata[key] = str(document.metadata[key])

        return Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            file_name=document.file_name,
            text=text.strip(),
            page_number=metadata.get("page"),
            paragraph_number=metadata.get("paragraph"),
            section_header=metadata.get("section"),
            char_start=char_start,
            char_end=char_end,
            document_type=document.document_type,
            metadata=chunk_metadata,
        )


def main():
    """CLI demo for adaptive chunker.

    Usage:
        python -m src.ingestion.chunkers.adaptive_chunker <file_path>
    """
    import sys
    from src.ingestion.ingestion_pipeline import IngestionPipeline

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.chunkers.adaptive_chunker <file_path>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Parse document
    pipeline = IngestionPipeline()
    document = pipeline.parse_document(file_path)

    if not document:
        print(f"Error: Failed to parse {file_path}")
        sys.exit(1)

    # Chunk document
    chunker = AdaptiveChunker()
    chunks = chunker.chunk_document(document)

    # Print stats
    print(f"\n{'='*60}")
    print(f"Chunking Statistics for: {document.file_name}")
    print(f"{'='*60}")
    print(f"Document Type: {document.document_type}")
    print(f"Original Text Length: {len(document.text)} chars")
    print(f"Number of Chunks: {len(chunks)}")
    print(f"\nChunk Size Settings:")
    print(f"  Target Size: {chunker._get_chunk_size(document.document_type)} tokens")
    print(f"  Overlap: {chunker._get_overlap(document.document_type)} tokens")
    print(f"\nChunk Details:")
    print(f"{'ID':<12} {'Size':<8} {'Page':<6} {'Para':<6} {'Section':<20}")
    print("-" * 60)

    for i, chunk in enumerate(chunks[:10], 1):  # Show first 10
        size_tokens = len(chunk.text) // chunker.CHARS_PER_TOKEN
        page_str = str(chunk.page_number) if chunk.page_number else "-"
        para_str = str(chunk.paragraph_number) if chunk.paragraph_number else "-"
        section_str = (chunk.section_header[:18] + "..") if chunk.section_header else "-"
        print(f"{chunk.chunk_id[:10]:<12} {size_tokens:<8} {page_str:<6} {para_str:<6} {section_str:<20}")

    if len(chunks) > 10:
        print(f"\n... and {len(chunks) - 10} more chunks")

    # Verify overlaps
    print(f"\n{'='*60}")
    print("Overlap Verification:")
    print(f"{'='*60}")
    overlap_issues = 0
    for i in range(len(chunks) - 1):
        chunk1 = chunks[i]
        chunk2 = chunks[i + 1]

        # Check if there's overlap
        expected_overlap = chunker._get_overlap(document.document_type) * chunker.CHARS_PER_TOKEN
        chunk1_end = chunk1.char_end
        chunk2_start = chunk2.char_start

        if chunk2_start < chunk1_end:
            overlap_size = chunk1_end - chunk2_start
            if overlap_size < expected_overlap * 0.5:  # Allow 50% tolerance
                overlap_issues += 1
                print(
                    f"Warning: Chunk {i+1}->{i+2} overlap is {overlap_size} chars "
                    f"(expected ~{expected_overlap})"
                )

    if overlap_issues == 0:
        print("✓ All overlaps verified successfully")
    else:
        print(f"⚠ Found {overlap_issues} overlap issues")


if __name__ == "__main__":
    main()

