"""Citation parser and verifier for LLM outputs.

Parses citations from LLM responses, verifies they map to provided chunks,
and flags uncited sentences. Implements fail-closed verification.

Supports two citation formats:
1. Verbose: [Source: filename | Page X, Para Y | "Section"]
2. Inline: [[@Source-N]] - compact format for click-to-navigate
"""

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from src.schema import Citation


@dataclass
class InlineCitation:
    """Inline citation marker for click-to-navigate.
    
    Attributes:
        marker: The inline marker text (e.g., "[[@Source-1]]")
        source_id: Numeric source ID (1-indexed)
        chunk_id: Associated chunk ID
        file_name: Source file name
        page_number: Page number in source
        paragraph_number: Paragraph number (optional)
        char_start: Character position in source document
        char_end: Character position end in source document
        position_in_response: Position of marker in LLM response
        confidence: Reranker confidence score
    """
    marker: str
    source_id: int
    chunk_id: str
    file_name: str
    page_number: int
    paragraph_number: Optional[int] = None
    char_start: int = 0
    char_end: int = 0
    position_in_response: int = 0
    confidence: float = 1.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "marker": self.marker,
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "file_name": self.file_name,
            "page_number": self.page_number,
            "paragraph_number": self.paragraph_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "position_in_response": self.position_in_response,
            "confidence": self.confidence,
        }


@dataclass
class CitationVerificationResult:
    """Result of citation verification.

    Attributes:
        response: Original LLM response text
        verified: Whether all citations are valid
        missing: List of citation strings that couldn't be mapped to chunks
        citations: List of verified Citation objects
        uncited_sentences: List of sentences without citations
        all_cited: Whether every sentence has a citation
        inline_citations: List of inline citations with positions for navigation
        source_map: Mapping of source_id to chunk metadata for UI
    """

    response: str
    verified: bool
    missing: list[str]
    citations: list[Citation]
    uncited_sentences: list[str]
    all_cited: bool
    inline_citations: list[InlineCitation] = field(default_factory=list)
    source_map: dict[int, dict[str, Any]] = field(default_factory=dict)


class CitationParser:
    """Parser for citations in LLM output."""

    # Regex pattern for verbose citation format: [Source: filename | Page X, Para Y | "Section"]
    VERBOSE_CITATION_PATTERN = re.compile(
        r'\[Source:\s*([^|]+)\s*\|\s*(?:Page\s+(\d+)(?:,\s*Para\s+(\d+))?)?\s*(?:\|\s*"([^"]+)")?\s*\]',
        re.IGNORECASE,
    )
    
    # Regex pattern for inline citation format: [[@Source-N]] (Hyperlink-style)
    INLINE_CITATION_PATTERN = re.compile(
        r'\[\[@Source-(\d+)\]\]',
        re.IGNORECASE,
    )

    def parse_citations(self, text: str) -> list[dict[str, Any]]:
        """Parse verbose citations from text.

        Args:
            text: Text containing citations

        Returns:
            List of citation dicts with keys: file_name, page_number, paragraph_number, section_title, raw_text
        """
        citations = []
        for match in self.VERBOSE_CITATION_PATTERN.finditer(text):
            file_name = match.group(1).strip()
            page_str = match.group(2)
            para_str = match.group(3)
            section_title = match.group(4)

            citations.append(
                {
                    "file_name": file_name,
                    "page_number": int(page_str) if page_str else None,
                    "paragraph_number": int(para_str) if para_str else None,
                    "section_title": section_title,
                    "raw_text": match.group(0),
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                }
            )

        return citations
    
    def parse_inline_citations(self, text: str) -> list[dict[str, Any]]:
        """Parse inline citations from text.
        
        Args:
            text: Text containing inline citations like [[@Source-1]]
            
        Returns:
            List of citation dicts with keys: source_id, raw_text, start_pos, end_pos
        """
        citations = []
        for match in self.INLINE_CITATION_PATTERN.finditer(text):
            source_id = int(match.group(1))
            citations.append({
                "source_id": source_id,
                "raw_text": match.group(0),
                "start_pos": match.start(),
                "end_pos": match.end(),
            })
        return citations

    def extract_sentences(self, text: str) -> list[dict[str, Any]]:
        """Extract sentences from text with positions.

        Args:
            text: Text to extract sentences from

        Returns:
            List of sentence dicts with keys: text, start_pos, end_pos
        """
        # Pattern to match sentences ending with . ! ? followed by space or end of string
        # Use non-greedy matching to allow dots inside the sentence (e.g. filenames)
        # But ensure we stop at the first likely sentence boundary
        sentence_pattern = re.compile(r'.+?[.!?](?:\s+|$)', re.MULTILINE | re.DOTALL)

        sentences = []
        for match in sentence_pattern.finditer(text):
            sentence_text = match.group(0).strip()
            if sentence_text and len(sentence_text) > 3:  # Ignore very short fragments
                sentences.append(
                    {
                        "text": sentence_text,
                        "start_pos": match.start(),
                        "end_pos": match.end(),
                    }
                )

        return sentences


class CitationVerifier:
    """Verifier for citations against provided chunks."""

    def __init__(self, chunks: list[dict[str, Any]], confidence_threshold: float = 0.70):
        """Initialize verifier with chunks and threshold.

        Args:
            chunks: List of chunk dicts with 'chunk_id', 'metadata' (file_name, page_number, etc.)
            confidence_threshold: Minimum confidence score for chunks
        """
        self.chunks = chunks
        self.confidence_threshold = confidence_threshold
        self.parser = CitationParser()

        # Build lookup map: (file_name, page, para) -> chunk
        self.chunk_map: dict[tuple[str, int | None, int | None], dict[str, Any]] = {}
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            file_name = metadata.get("file_name", "")
            page = metadata.get("page_number")
            para = metadata.get("paragraph_number")
            score = chunk.get("score", 1.0)

            # Only include chunks meeting confidence threshold
            if score >= confidence_threshold:
                key = (file_name, page, para)
                self.chunk_map[key] = chunk

    def verify(
        self, response: str, check_all_sentences: bool = True
    ) -> CitationVerificationResult:
        """Verify citations in LLM response.

        Args:
            response: LLM response text
            check_all_sentences: If True, check that every sentence has a citation

        Returns:
            CitationVerificationResult with verification status
        """
        # Parse citations
        parsed_citations = self.parser.parse_citations(response)

        # Verify each citation maps to a chunk
        verified_citations = []
        missing_citations = []

        for parsed in parsed_citations:
            file_name = parsed["file_name"]
            page = parsed["page_number"]
            para = parsed["paragraph_number"]

            # Try to find matching chunk
            key = (file_name, page, para)
            chunk = self.chunk_map.get(key)

            if chunk:
                # Create Citation object
                citation = Citation(
                    file_name=file_name,
                    page_number=page or 1,  # Default to 1 if not specified
                    paragraph_number=para,
                    section_title=parsed.get("section_title"),
                    chunk_id=chunk.get("chunk_id"),
                    confidence=chunk.get("score", 1.0),
                )
                verified_citations.append(citation)
            else:
                missing_citations.append(parsed["raw_text"])

        # Check for uncited sentences
        uncited_sentences = []
        if check_all_sentences:
            sentences = self.parser.extract_sentences(response)
            citation_positions = [
                (c["start_pos"], c["end_pos"]) for c in parsed_citations
            ]

            for sentence in sentences:
                # Check if sentence overlaps with any citation
                sentence_start = sentence["start_pos"]
                sentence_end = sentence["end_pos"]

                has_citation = False
                for cit_start, cit_end in citation_positions:
                    # Citation overlaps with sentence if it's within or near sentence
                    if cit_start >= sentence_start - 10 and cit_end <= sentence_end + 10:
                        has_citation = True
                        break

                if not has_citation:
                    # Skip refusal messages and meta-text
                    sentence_text = sentence["text"].lower()
                    if not any(
                        skip_phrase in sentence_text
                        for skip_phrase in [
                            "not found in provided documents",
                            "cannot answer",
                            "sufficient relevant information",
                        ]
                    ):
                        uncited_sentences.append(sentence["text"])

        # Determine verification status
        verified = len(missing_citations) == 0 and len(uncited_sentences) == 0
        all_cited = len(uncited_sentences) == 0

        return CitationVerificationResult(
            response=response,
            verified=verified,
            missing=missing_citations,
            citations=verified_citations,
            uncited_sentences=uncited_sentences,
            all_cited=all_cited,
        )

    def should_refuse_before_generation(self) -> tuple[bool, str]:
        """Check if generation should be refused due to low-confidence chunks.

        Returns:
            Tuple of (should_refuse, reason)
        """
        if not self.chunks:
            return True, "No chunks provided"

        # Check if scores are RRF scores (small values < 1.0) or reranker scores (0-1 range)
        # RRF scores are typically 0.01-0.1, reranker scores are 0-1
        max_score = max((c.get("score", 0.0) for c in self.chunks), default=0.0)

        # If max score is very small (< 0.1), these are RRF scores - skip confidence check
        # RRF fusion already ranked them, so trust the ranking
        if max_score < 0.1:
            # Using RRF scores - no confidence filtering needed
            return False, ""

        # Using reranker scores - apply confidence threshold
        chunks_above_threshold = [
            c for c in self.chunks if c.get("score", 0.0) >= self.confidence_threshold
        ]

        if not chunks_above_threshold:
            return (
                True,
                f"No chunks meet confidence threshold {self.confidence_threshold}",
            )

        return False, ""

    def format_source_list(self) -> str:
        """Format list of sources for prompt.

        Returns:
            Formatted string listing all sources
        """
        sources = []
        seen = set()

        for chunk in self.chunks:
            metadata = chunk.get("metadata", {})
            file_name = metadata.get("file_name", "unknown")
            page = metadata.get("page_number")
            para = metadata.get("paragraph_number")
            section = metadata.get("section_header")

            # Build citation string
            parts = [f"Source: {file_name}"]
            if page:
                parts.append(f"Page {page}")
                if para:
                    parts.append(f"Para {para}")
            if section:
                parts.append(f'"{section}"')

            citation_str = " | ".join(parts)
            key = (file_name, page, para)

            if key not in seen:
                sources.append(f"[{citation_str}]")
                seen.add(key)

        return "\n".join(sources)
    
    def build_source_map(self) -> dict[int, dict[str, Any]]:
        """Build a mapping of source IDs to chunk metadata.
        
        Used for inline citations: [[@Source-N]] -> chunk metadata
        
        Returns:
            Dict mapping source_id (1-indexed) to chunk metadata
        """
        source_map = {}
        for i, chunk in enumerate(self.chunks):
            source_id = i + 1  # 1-indexed
            metadata = chunk.get("metadata", {})
            source_map[source_id] = {
                "source_id": source_id,
                "chunk_id": chunk.get("chunk_id", ""),
                "file_name": metadata.get("file_name", "unknown"),
                "page_number": metadata.get("page_number"),
                "paragraph_number": metadata.get("paragraph_number"),
                "section_header": metadata.get("section_header"),
                "char_start": metadata.get("char_start", 0),
                "char_end": metadata.get("char_end", 0),
                "score": chunk.get("score", 1.0),
                "text_preview": chunk.get("text", "")[:200],
            }
        return source_map
    
    def verify_with_inline(
        self, response: str, check_all_sentences: bool = True
    ) -> CitationVerificationResult:
        """Verify citations including inline format and build source map.
        
        Args:
            response: LLM response text (may contain [[@Source-N]] markers)
            check_all_sentences: If True, check that every sentence has a citation
            
        Returns:
            CitationVerificationResult with inline_citations and source_map populated
        """
        # First do standard verification
        result = self.verify(response, check_all_sentences)
        
        # Build source map
        source_map = self.build_source_map()
        
        # Parse inline citations
        inline_parsed = self.parser.parse_inline_citations(response)
        inline_citations = []
        
        for parsed in inline_parsed:
            source_id = parsed["source_id"]
            if source_id in source_map:
                source = source_map[source_id]
                inline_citations.append(InlineCitation(
                    marker=parsed["raw_text"],
                    source_id=source_id,
                    chunk_id=source["chunk_id"],
                    file_name=source["file_name"],
                    page_number=source["page_number"] or 1,
                    paragraph_number=source["paragraph_number"],
                    char_start=source["char_start"],
                    char_end=source["char_end"],
                    position_in_response=parsed["start_pos"],
                    confidence=source["score"],
                ))
            else:
                # Invalid source reference
                result.missing.append(parsed["raw_text"])
        
        # Update result with inline data
        result.inline_citations = inline_citations
        result.source_map = source_map
        
        return result
    
    def format_source_list_with_ids(self) -> str:
        """Format list of sources for prompt with inline citation IDs.
        
        Use this when you want the LLM to use [[@Source-N]] format.

        Returns:
            Formatted string listing all sources with IDs
        """
        sources = []
        
        for i, chunk in enumerate(self.chunks):
            source_id = i + 1  # 1-indexed
            metadata = chunk.get("metadata", {})
            file_name = metadata.get("file_name", "unknown")
            page = metadata.get("page_number")
            para = metadata.get("paragraph_number")
            section = metadata.get("section_header")

            # Build citation string with ID
            parts = [f"Source-{source_id}: {file_name}"]
            if page:
                parts.append(f"Page {page}")
                if para:
                    parts.append(f"Para {para}")
            if section:
                parts.append(f'"{section}"')

            citation_str = " | ".join(parts)
            sources.append(f"[{citation_str}]")

        return "\n".join(sources)


def verify_citations(
    response: str,
    chunks: list[dict[str, Any]],
    confidence_threshold: float = 0.70,
    check_all_sentences: bool = True,
) -> CitationVerificationResult:
    """Convenience function to verify citations.

    Args:
        response: LLM response text
        chunks: List of chunk dicts provided to LLM
        confidence_threshold: Minimum confidence threshold
        check_all_sentences: Whether to check all sentences have citations

    Returns:
        CitationVerificationResult
    """
    verifier = CitationVerifier(chunks, confidence_threshold)
    return verifier.verify(response, check_all_sentences)




