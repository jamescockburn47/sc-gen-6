"""PDF parser using PyMuPDF (fitz)."""

import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from src.ingestion.parsers.base_parser import BaseParser
from src.schema import DocumentType, ParsedDocument


class PDFParser(BaseParser):
    """Parser for PDF documents using PyMuPDF.

    Extracts text with page numbers, detects headers, and identifies
    document type. Handles both text-based and scanned PDFs (though
    scanned PDFs should use OCRParser).
    """

    def can_parse(self, file_path: str | Path) -> bool:
        """Check if file is a PDF."""
        return Path(file_path).suffix.lower() == ".pdf"

    def parse(
        self, file_path: str | Path, document_type: Optional[DocumentType] = None
    ) -> ParsedDocument:
        """Parse PDF document.

        Args:
            file_path: Path to PDF file
            document_type: Optional document type override

        Returns:
            ParsedDocument with text, pages, and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If PDF parsing fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            doc = fitz.open(file_path)
            text_parts = []
            pages = []
            paragraphs = []
            headers = []
            full_text = ""

            # Get page count before processing
            page_count = len(doc)

            # Extract text from each page
            for page_num in range(page_count):
                page = doc[page_num]
                page_text = page.get_text()

                if not page_text.strip():
                    # Empty page, might be scanned - skip for now (use OCR parser)
                    continue

                # Extract headers (text in larger font sizes, typically at top)
                page_blocks = page.get_text("dict")
                page_headers = self._extract_headers(page_blocks)

                # Split into paragraphs
                page_paragraphs = self._split_into_paragraphs(page_text)

                # Build paragraph list with metadata
                para_start = len(full_text)
                for para_idx, para_text in enumerate(page_paragraphs):
                    if para_text.strip():
                        para_end = para_start + len(para_text)
                        paragraphs.append(
                            {
                                "text": para_text,
                                "page": page_num + 1,  # 1-indexed
                                "paragraph": len(paragraphs) + 1,
                                "char_start": para_start,
                                "char_end": para_end,
                            }
                        )
                        pages.append(page_num + 1)
                        text_parts.append(para_text)
                        full_text += para_text + "\n\n"
                        para_start = para_end + 2  # +2 for "\n\n"

                headers.extend(page_headers)

            doc.close()

            if not full_text.strip():
                raise ValueError(f"PDF appears to be empty or scanned: {file_path}")

            # Detect document type if not provided
            if document_type is None:
                document_type = self.detect_document_type(file_path, full_text, {"headers": headers})

            # Extract enhanced metadata
            extracted_metadata = self._extract_metadata(full_text)
            
            metadata = {
                "page_count": page_count,
                "headers": headers,
                "is_scanned": False,
                **extracted_metadata,  # Add extracted dates, case numbers, etc.
            }

            return ParsedDocument(
                file_path=str(file_path),
                file_name=file_path.name,
                document_type=document_type,
                text=full_text,
                pages=pages,
                paragraphs=paragraphs,
                metadata=metadata,
            )

        except Exception as e:
            raise Exception(f"Failed to parse PDF {file_path}: {str(e)}") from e

    def _extract_headers(self, page_blocks: dict) -> list[str]:
        """Extract headers from page blocks.

        Headers are typically larger font sizes or at the top of pages.
        """
        headers = []
        if "blocks" not in page_blocks:
            return headers

        for block in page_blocks["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                if "spans" not in line:
                    continue
                for span in line["spans"]:
                    # Heuristic: headers are often larger fonts (>12pt) or bold
                    font_size = span.get("size", 0)
                    if font_size > 12 or span.get("flags", 0) & 16:  # 16 = bold flag
                        text = span.get("text", "").strip()
                        if text and len(text) < 200:  # Headers are usually short
                            headers.append(text)

        return headers[:10]  # Limit to first 10 headers

    def _extract_metadata(self, text: str) -> dict:
        """Extract dates, case numbers, parties, and other metadata from text.
        
        Args:
            text: Full document text
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}
        text_start = text[:5000]  # First ~2 pages for metadata extraction
        
        # Extract dates (UK format: DD/MM/YYYY, DD.MM.YYYY, DD Month YYYY)
        date_patterns = [
            # DD/MM/YYYY or DD.MM.YYYY
            r'\b(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})\b',
            # DD Month YYYY (e.g., 15 January 2024)
            r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
            # Month DD, YYYY
            r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
        ]
        
        dates_found = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text_start, re.IGNORECASE)
            dates_found.extend(matches[:5])  # Limit per pattern
        
        if dates_found:
            metadata["document_date"] = dates_found[0]  # First date found
            if len(dates_found) > 1:
                metadata["dates_mentioned"] = ", ".join(dates_found[:5])
        
        # Extract UK case/claim numbers
        case_patterns = [
            # High Court: [Year] EWHC [Number] ([Division])
            r'\[(\d{4})\]\s*EWHC\s*(\d+)\s*\([A-Za-z]+\)',
            # County Court claim numbers
            r'\b([A-Z]{2}\d{2}[A-Z]\d{5})\b',
            # Generic claim number: CL-YYYY-NNNNNN
            r'\b(CL[-\s]?\d{4}[-\s]?\d{4,6})\b',
            # Case No: XXXXX
            r'(?:Case|Claim)\s*(?:No|Number)[.:\s]*([A-Z0-9/-]+)',
        ]
        
        for pattern in case_patterns:
            match = re.search(pattern, text_start, re.IGNORECASE)
            if match:
                if len(match.groups()) > 1:
                    metadata["case_number"] = f"[{match.group(1)}] EWHC {match.group(2)}"
                else:
                    metadata["case_number"] = match.group(1)
                break
        
        # Extract parties (Between: X and Y)
        party_patterns = [
            r'Between[:\s]*\n?\s*(.+?)\s*(?:Claimant|Applicant)',
            r'(.+?)\s+(?:v\.?|versus)\s+(.+?)(?:\n|$)',
        ]
        
        for pattern in party_patterns:
            match = re.search(pattern, text_start, re.IGNORECASE)
            if match:
                if len(match.groups()) > 1:
                    metadata["parties"] = f"{match.group(1).strip()} v {match.group(2).strip()}"
                else:
                    metadata["parties"] = match.group(1).strip()
                break
        
        # Extract author/witness name (from witness statements)
        author_patterns = [
            r'(?:I,?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+),?\s+(?:make this statement|say as follows)',
            r'Statement of[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'Witness[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, text_start)
            if match:
                metadata["author"] = match.group(1).strip()
                break
        
        return metadata

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs.

        Uses double newlines as primary separator, falls back to single
        newlines if needed.
        """
        # First try double newlines
        paragraphs = re.split(r"\n\s*\n", text)
        if len(paragraphs) > 1:
            return [p.strip() for p in paragraphs if p.strip()]

        # Fall back to single newlines
        paragraphs = text.split("\n")
        return [p.strip() for p in paragraphs if p.strip()]

    def is_scanned(self, file_path: str | Path) -> bool:
        """Check if PDF is scanned (image-based) rather than text-based.

        Args:
            file_path: Path to PDF file

        Returns:
            True if PDF appears to be scanned
        """
        try:
            doc = fitz.open(file_path)
            # Check first few pages
            scanned_count = 0
            total_pages = min(3, len(doc))
            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text()
                if not text.strip():
                    scanned_count += 1
            doc.close()
            # If most pages have no text, likely scanned
            return scanned_count >= total_pages * 0.8
        except Exception:
            return False




