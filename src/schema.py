"""Data schemas for documents, chunks, citations, and retrieval parameters."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# Document types supported by the system
# Expanded for UK litigation (civil fraud & competition law)
DocumentType = Literal[
    # Core pleadings
    "witness_statement",
    "court_filing",
    "pleading",
    "skeleton_argument",
    # Expert & technical
    "expert_report",
    "schedule_of_loss",
    # Legal sources
    "statute",
    "case_law",
    "contract",
    # Correspondence & disclosure
    "email",
    "letter",
    "disclosure",
    "disclosure_list",
    # Court forms & management
    "court_form",
    "case_management",
    "chronology",
    # Specialist
    "medical_report",
    "tribunal_document",
    "regulatory_document",
    # Fallback
    "scanned_pdf",
    "unknown",
]


@dataclass
class ParsedDocument:
    """Parsed document with metadata and content.

    Attributes:
        file_path: Path to the source file
        file_name: Name of the file
        document_type: Type of document (witness_statement, pleading, etc.)
        text: Full extracted text content
        pages: List of page numbers (1-indexed) for each paragraph
        paragraphs: List of paragraph text with metadata
        metadata: Additional metadata (headers, sections, dates, etc.)
        parsed_at: Timestamp when document was parsed
    """

    file_path: str
    file_name: str
    document_type: DocumentType
    text: str
    pages: list[int] = field(default_factory=list)
    paragraphs: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    parsed_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate document after initialization."""
        if not self.text.strip():
            raise ValueError("Document text cannot be empty")
        valid_types = [
            # Core pleadings
            "witness_statement",
            "court_filing",
            "pleading",
            "skeleton_argument",
            # Expert & technical
            "expert_report",
            "schedule_of_loss",
            # Legal sources
            "statute",
            "case_law",
            "contract",
            # Correspondence & disclosure
            "email",
            "letter",
            "disclosure",
            "disclosure_list",
            # Court forms & management
            "court_form",
            "case_management",
            "chronology",
            # Specialist
            "medical_report",
            "tribunal_document",
            "regulatory_document",
            # Fallback
            "scanned_pdf",
            "unknown",
        ]
        if self.document_type not in valid_types:
            raise ValueError(f"Invalid document type: {self.document_type}")


class Chunk(BaseModel):
    """A chunk of text with location metadata.

    Attributes:
        chunk_id: Unique identifier for the chunk
        document_id: Identifier of the source document
        file_name: Name of the source file
        text: Chunk text content
        page_number: Page number (1-indexed, None if not applicable)
        paragraph_number: Paragraph number (1-indexed, None if not applicable)
        section_header: Section or heading text (if applicable)
        char_start: Character offset start in original document
        char_end: Character offset end in original document
        document_type: Type of source document
        metadata: Additional chunk metadata
    """

    chunk_id: str
    document_id: str
    file_name: str
    text: str
    page_number: Optional[int] = None
    paragraph_number: Optional[int] = None
    section_header: Optional[str] = None
    char_start: int = 0
    char_end: int = 0
    document_type: DocumentType
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("page_number", "paragraph_number")
    @classmethod
    def validate_positive(cls, v: Optional[int]) -> Optional[int]:
        """Ensure page and paragraph numbers are positive if provided."""
        if v is not None and v < 1:
            raise ValueError("Page and paragraph numbers must be >= 1")
        return v

    @field_validator("char_end")
    @classmethod
    def validate_char_range(cls, v: int, info) -> int:
        """Ensure char_end >= char_start."""
        if "char_start" in info.data and v < info.data["char_start"]:
            raise ValueError("char_end must be >= char_start")
        return v


class Citation(BaseModel):
    """A citation reference to a source document.

    Attributes:
        file_name: Name of the source file
        page_number: Page number (1-indexed)
        paragraph_number: Paragraph number (1-indexed, optional)
        section_title: Section or heading title (optional)
        chunk_id: Associated chunk ID (optional)
        confidence: Confidence score from reranker (0.0-1.0)
    """

    file_name: str
    page_number: int
    paragraph_number: Optional[int] = None
    section_title: Optional[str] = None
    chunk_id: Optional[str] = None
    confidence: float = 1.0

    @field_validator("page_number", "paragraph_number")
    @classmethod
    def validate_positive(cls, v: Optional[int]) -> Optional[int]:
        """Ensure page and paragraph numbers are positive."""
        if v is not None and v < 1:
            raise ValueError("Page and paragraph numbers must be >= 1")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    def format(self) -> str:
        """Format citation as string: [Source: filename | Page X, Para Y | "Section"]."""
        parts = [f"Source: {self.file_name}"]
        if self.page_number:
            para_part = f"Page {self.page_number}"
            if self.paragraph_number:
                para_part += f", Para {self.paragraph_number}"
            parts.append(para_part)
        if self.section_title:
            parts.append(f'"{self.section_title}"')
        return f"[{' | '.join(parts)}]"


class RetrievalParams(BaseModel):
    """Parameters for retrieval and reranking.

    Attributes:
        semantic_top_n: Number of results from semantic search
        keyword_top_n: Number of results from BM25 search
        rerank_top_k: Number of candidates to rerank
        context_to_llm: Number of chunks to send to LLM
        confidence_threshold: Minimum reranker score threshold
        rrf_k: RRF fusion parameter
        doc_type_filter: Optional document type filter
        date_range: Optional date range filter (tuple of (start, end))
    """

    semantic_top_n: int = 50
    keyword_top_n: int = 50
    rerank_top_k: int = 20
    context_to_llm: int = 5
    confidence_threshold: float = 0.70
    rrf_k: int = 60
    doc_type_filter: Optional[DocumentType] = None
    date_range: Optional[tuple[datetime, datetime]] = None

    @field_validator("semantic_top_n", "keyword_top_n", "rerank_top_k", "context_to_llm")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Ensure counts are positive."""
        if v < 1:
            raise ValueError("Counts must be >= 1")
        return v

    @field_validator("confidence_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Ensure confidence threshold is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        return v

