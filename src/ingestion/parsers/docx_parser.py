"""DOCX parser using python-docx."""

import re
from pathlib import Path
from typing import Optional

from docx import Document
from docx.document import Document as DocumentType
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table
from docx.text.paragraph import Paragraph

from src.ingestion.parsers.base_parser import BaseParser
from src.schema import DocumentType, ParsedDocument


class DOCXParser(BaseParser):
    """Parser for DOCX documents using python-docx.

    Extracts text with headings converted to sections, preserves
    paragraph structure, and extracts metadata.
    """

    def can_parse(self, file_path: str | Path) -> bool:
        """Check if file is a DOCX."""
        return Path(file_path).suffix.lower() in [".docx", ".doc"]

    def parse(
        self, file_path: str | Path, document_type: Optional[DocumentType] = None
    ) -> ParsedDocument:
        """Parse DOCX document.

        Args:
            file_path: Path to DOCX file
            document_type: Optional document type override

        Returns:
            ParsedDocument with text, sections, and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If DOCX parsing fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"DOCX file not found: {file_path}")

        try:
            doc = Document(file_path)
            text_parts = []
            paragraphs = []
            sections = []
            current_section = None
            full_text = ""

            # Process document elements (paragraphs and tables)
            para_start = 0
            para_num = 0

            for element in doc.element.body:
                if isinstance(element, CT_P):
                    # It's a paragraph
                    para = Paragraph(element, doc)
                    para_text = para.text.strip()

                    if not para_text:
                        continue

                    # Check if it's a heading
                    style_name = para.style.name.lower() if para.style else ""
                    is_heading = "heading" in style_name or para.style.name.startswith("Heading")

                    if is_heading:
                        current_section = para_text
                        sections.append(para_text)
                        # Add section header to text
                        text_parts.append(f"\n\n## {para_text}\n\n")
                        full_text += f"\n\n## {para_text}\n\n"
                    else:
                        para_num += 1
                        para_end = para_start + len(para_text)
                        paragraphs.append(
                            {
                                "text": para_text,
                                "page": 1,  # DOCX doesn't have explicit pages
                                "paragraph": para_num,
                                "section": current_section,
                                "char_start": para_start,
                                "char_end": para_end,
                            }
                        )
                        text_parts.append(para_text)
                        full_text += para_text + "\n\n"
                        para_start = para_end + 2  # +2 for "\n\n"

                elif isinstance(element, CT_Tbl):
                    # It's a table - extract as text
                    table = Table(element, doc)
                    table_text = self._table_to_text(table)
                    if table_text.strip():
                        para_num += 1
                        para_end = para_start + len(table_text)
                        paragraphs.append(
                            {
                                "text": table_text,
                                "page": 1,
                                "paragraph": para_num,
                                "section": current_section,
                                "is_table": True,
                                "char_start": para_start,
                                "char_end": para_end,
                            }
                        )
                        text_parts.append(table_text)
                        full_text += table_text + "\n\n"
                        para_start = para_end + 2

            if not full_text.strip():
                raise ValueError(f"DOCX appears to be empty: {file_path}")

            # Detect document type if not provided
            if document_type is None:
                document_type = self.detect_document_type(file_path, full_text, {"sections": sections})

            # Extract enhanced metadata (dates, case numbers, parties)
            extracted_metadata = self._extract_metadata(full_text)
            
            metadata = {
                "sections": sections,
                "paragraph_count": para_num,
                "has_tables": any(p.get("is_table", False) for p in paragraphs),
                **extracted_metadata,
            }

            return ParsedDocument(
                file_path=str(file_path),
                file_name=file_path.name,
                document_type=document_type,
                text=full_text,
                pages=[1] * len(paragraphs),  # DOCX doesn't have explicit pages
                paragraphs=paragraphs,
                metadata=metadata,
            )

        except Exception as e:
            raise Exception(f"Failed to parse DOCX {file_path}: {str(e)}") from e

    def _table_to_text(self, table: Table) -> str:
        """Convert a table to text representation.

        Args:
            table: python-docx Table object

        Returns:
            Text representation of table
        """
        rows_text = []
        for row in table.rows:
            cells_text = []
            for cell in row.cells:
                cells_text.append(cell.text.strip())
            rows_text.append(" | ".join(cells_text))
        return "\n".join(rows_text)

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
            r'\b(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})\b',
            r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
            r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
        ]
        
        dates_found = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text_start, re.IGNORECASE)
            dates_found.extend(matches[:5])
        
        if dates_found:
            metadata["document_date"] = dates_found[0]
            if len(dates_found) > 1:
                metadata["dates_mentioned"] = ", ".join(dates_found[:5])
        
        # Extract UK case/claim numbers
        case_patterns = [
            r'\[(\d{4})\]\s*EWHC\s*(\d+)\s*\([A-Za-z]+\)',
            r'\b([A-Z]{2}\d{2}[A-Z]\d{5})\b',
            r'\b(CL[-\s]?\d{4}[-\s]?\d{4,6})\b',
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
        
        # Extract parties
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
        
        # Extract author/witness name
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




