"""Robust chunker with paragraph/sentence awareness but document-type agnostic.

Uses fixed chunk sizes with RCTS (Recursive Character Text Splitting).
Enforces maximum chunk size to prevent malformed documents creating huge chunks.
Preserves page/paragraph/sentence metadata.
"""

import hashlib
import re
from pathlib import Path
from typing import Optional

from src.config_loader import Settings, get_settings
from src.schema import Chunk, ParsedDocument


class RobustChunker:
    """Document-agnostic chunker with paragraph/sentence awareness.
    
    Key features:
    - Fixed chunk sizes (not document-type dependent)
    - Maximum chunk size enforcement (prevents huge chunks from malformed docs)
    - Paragraph/sentence/page boundary awareness
    - Deterministic chunk IDs for deduplication
    - High overlap for context preservation
    """

    # Fixed chunking parameters (in tokens, approx 4 chars/token)
    DEFAULT_CHUNK_SIZE = 512  # tokens (~2048 chars)
    DEFAULT_OVERLAP = 128     # tokens (~512 chars) - 25% overlap
    MAX_CHUNK_SIZE = 1024     # tokens (~4096 chars) - absolute max
    CHARS_PER_TOKEN = 4

    # Separators in priority order (try to split at natural boundaries)
    SEPARATORS = [
        "\n\n\n",    # Major section breaks
        "\n\n",      # Paragraph breaks
        "\n",        # Line breaks
        ". ",        # Sentence ends
        "? ",        # Question ends
        "! ",        # Exclamation ends
        "; ",        # Semicolon
        ", ",        # Comma
        " ",         # Word boundary (last resort)
    ]

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize chunker with optional settings override."""
        self.settings = settings or get_settings()
        
        # Use config if available, else defaults
        self.chunk_size_chars = self.DEFAULT_CHUNK_SIZE * self.CHARS_PER_TOKEN
        self.overlap_chars = self.DEFAULT_OVERLAP * self.CHARS_PER_TOKEN
        self.max_chunk_chars = self.MAX_CHUNK_SIZE * self.CHARS_PER_TOKEN

    def chunk_document(self, document: ParsedDocument) -> list[Chunk]:
        """Chunk a parsed document into chunks with metadata.
        
        Args:
            document: ParsedDocument to chunk
            
        Returns:
            List of Chunk objects with preserved metadata
        """
        # Generate stable document ID
        document_id = self._generate_document_id(document)
        
        # Get full text with paragraph metadata mapping
        full_text, char_to_metadata = self._build_text_with_metadata(document)
        
        if not full_text.strip():
            return []
        
        # Perform robust chunking with max size enforcement
        chunks = self._chunk_text(
            text=full_text,
            char_to_metadata=char_to_metadata,
            document_id=document_id,
            document=document,
        )
        
        return chunks

    def _generate_document_id(self, document: ParsedDocument) -> str:
        """Generate stable document ID from file path."""
        file_hash = hashlib.md5(document.file_path.encode()).hexdigest()[:8]
        return f"doc_{file_hash}"

    def _build_text_with_metadata(self, document: ParsedDocument) -> tuple[str, dict]:
        """Build full text with character-to-metadata mapping.
        
        Returns:
            (full_text, char_to_metadata) where char_to_metadata maps
            character positions to their page/paragraph metadata.
        """
        char_to_metadata = {}
        
        if document.paragraphs:
            # Use paragraph structure
            full_text = ""
            for para in document.paragraphs:
                para_text = para.get("text", "")
                if not para_text:
                    continue
                    
                start_pos = len(full_text)
                full_text += para_text + "\n\n"
                end_pos = len(full_text)
                
                # Map each character to its metadata
                meta = {
                    "page": para.get("page"),
                    "paragraph": para.get("paragraph"),
                    "section": para.get("section") or para.get("section_header"),
                }
                for pos in range(start_pos, end_pos):
                    char_to_metadata[pos] = meta
        else:
            # Fallback: use raw text, infer page from position
            full_text = document.text
            page_count = max(1, len(set(document.pages)) if document.pages else 1)
            chars_per_page = max(1, len(full_text) // page_count)
            
            for pos in range(len(full_text)):
                page_num = (pos // chars_per_page) + 1
                char_to_metadata[pos] = {"page": page_num, "paragraph": None, "section": None}
        
        return full_text, char_to_metadata

    def _chunk_text(
        self,
        text: str,
        char_to_metadata: dict,
        document_id: str,
        document: ParsedDocument,
    ) -> list[Chunk]:
        """Chunk text with robust splitting and max size enforcement.
        
        Key behavior:
        - Tries to split at natural boundaries (paragraphs, sentences)
        - Enforces maximum chunk size (prevents huge chunks)
        - Uses overlap for context continuity
        """
        chunks = []
        text_len = len(text)
        pos = 0
        chunk_index = 0
        
        while pos < text_len:
            # Determine chunk end (target size, but respect max)
            target_end = min(pos + self.chunk_size_chars, text_len)
            max_end = min(pos + self.max_chunk_chars, text_len)
            
            # Find best split point near target
            split_pos = self._find_best_split(text, pos, target_end, max_end)
            
            # Extract chunk text
            chunk_text = text[pos:split_pos].strip()
            
            if chunk_text:
                # Get metadata from chunk start position
                meta = char_to_metadata.get(pos, {})
                
                # Create chunk with deterministic ID
                chunk = self._create_chunk(
                    text=chunk_text,
                    document_id=document_id,
                    document=document,
                    chunk_index=chunk_index,
                    char_start=pos,
                    char_end=split_pos,
                    metadata=meta,
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move position with overlap (back up by overlap amount)
            if split_pos >= text_len:
                break
            
            # Calculate next position with overlap
            next_pos = split_pos - self.overlap_chars
            # Ensure we always make forward progress
            next_pos = max(next_pos, pos + (self.chunk_size_chars // 2))
            pos = min(next_pos, split_pos)
        
        return chunks

    def _find_best_split(self, text: str, start: int, target: int, max_end: int) -> int:
        """Find the best split point, preferring natural boundaries.
        
        Args:
            text: Full text
            start: Chunk start position
            target: Target end position
            max_end: Absolute maximum end position
            
        Returns:
            Best split position
        """
        if target >= len(text):
            return len(text)
        
        # Search window around target (+/- 20% of chunk size)
        window = self.chunk_size_chars // 5
        search_start = max(start, target - window)
        search_end = min(max_end, target + window)
        
        # Try each separator in priority order
        for separator in self.SEPARATORS:
            # Find last occurrence of separator in search window
            search_text = text[search_start:search_end]
            last_sep = search_text.rfind(separator)
            
            if last_sep != -1:
                split_pos = search_start + last_sep + len(separator)
                # Ensure we got at least some content
                if split_pos > start + 100:
                    return split_pos
        
        # No good separator found - use target (but respect max)
        return min(target, max_end)

    def _create_chunk(
        self,
        text: str,
        document_id: str,
        document: ParsedDocument,
        chunk_index: int,
        char_start: int,
        char_end: int,
        metadata: dict,
    ) -> Chunk:
        """Create a Chunk object with deterministic ID."""
        # Deterministic ID: document + index ensures stable ordering
        # Content hash added for change detection
        content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        chunk_id = f"{document_id}_c{chunk_index:04d}_{content_hash}"
        
        # Copy relevant document metadata
        chunk_metadata = {}
        if document.metadata:
            for key in ["from", "to", "cc", "subject", "date", "message_id",
                       "document_date", "parties", "case_number", "author"]:
                if key in document.metadata and document.metadata[key]:
                    chunk_metadata[key] = str(document.metadata[key])
        
        return Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            file_name=document.file_name,
            text=text,
            page_number=metadata.get("page"),
            paragraph_number=metadata.get("paragraph"),
            section_header=metadata.get("section"),
            char_start=char_start,
            char_end=char_end,
            document_type=document.document_type,
            metadata=chunk_metadata,
        )


# Backwards compatibility - alias for existing code
AdaptiveChunker = RobustChunker


def main():
    """CLI demo for robust chunker."""
    import sys
    from src.ingestion.ingestion_pipeline import IngestionPipeline

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.chunkers.adaptive_chunker <file_path>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Parse and chunk
    pipeline = IngestionPipeline()
    document = pipeline.parse_document(file_path)

    if not document:
        print(f"Error: Failed to parse {file_path}")
        sys.exit(1)

    chunker = RobustChunker()
    chunks = chunker.chunk_document(document)

    # Print stats
    print(f"\n{'='*60}")
    print(f"Chunking: {document.file_name}")
    print(f"{'='*60}")
    print(f"Text length: {len(document.text)} chars")
    print(f"Chunks: {len(chunks)}")
    print(f"Target size: {chunker.chunk_size_chars} chars ({chunker.DEFAULT_CHUNK_SIZE} tokens)")
    print(f"Max size: {chunker.max_chunk_chars} chars ({chunker.MAX_CHUNK_SIZE} tokens)")
    print(f"Overlap: {chunker.overlap_chars} chars ({chunker.DEFAULT_OVERLAP} tokens)")
    
    print(f"\n{'ID':<25} {'Size':<8} {'Page':<6}")
    print("-" * 50)
    
    for chunk in chunks[:10]:
        size = len(chunk.text)
        page = chunk.page_number or "-"
        print(f"{chunk.chunk_id:<25} {size:<8} {page:<6}")
    
    if len(chunks) > 10:
        print(f"... and {len(chunks) - 10} more chunks")
    
    # Verify no huge chunks
    max_chunk = max(len(c.text) for c in chunks) if chunks else 0
    if max_chunk > chunker.max_chunk_chars:
        print(f"\n⚠ WARNING: Found chunk exceeding max size ({max_chunk} chars)")
    else:
        print(f"\n✓ All chunks within limits (max: {max_chunk} chars)")


if __name__ == "__main__":
    main()
