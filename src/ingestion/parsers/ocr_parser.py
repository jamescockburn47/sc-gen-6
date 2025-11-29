"""OCR parser for scanned PDFs using Tesseract."""

import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

from src.ingestion.parsers.base_parser import BaseParser
from src.ingestion.parsers.pdf_parser import PDFParser
from src.schema import DocumentType, ParsedDocument


class OCRParser(BaseParser):
    """Parser for scanned PDFs using Tesseract OCR.

    Detects scanned PDFs, performs OCR, and filters low-confidence tokens.
    Falls back to PDFParser for text-based PDFs.
    """

    def __init__(self, confidence_threshold: float = 0.6):
        """Initialize OCR parser.

        Args:
            confidence_threshold: Minimum confidence (0.0-1.0) for tokens.
                                Tokens below this are dropped.
        """
        self.confidence_threshold = confidence_threshold
        self.pdf_parser = PDFParser()

    def can_parse(self, file_path: str | Path) -> bool:
        """Check if file is a PDF (OCR parser handles scanned PDFs)."""
        return Path(file_path).suffix.lower() == ".pdf"

    def parse(
        self, file_path: str | Path, document_type: Optional[DocumentType] = None
    ) -> ParsedDocument:
        """Parse scanned PDF using OCR.

        First checks if PDF is scanned. If not, delegates to PDFParser.
        If scanned, performs OCR with confidence filtering.

        Args:
            file_path: Path to PDF file
            document_type: Optional document type override

        Returns:
            ParsedDocument with OCR'd text

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If OCR fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        # Check if PDF is scanned
        if not self.pdf_parser.is_scanned(file_path):
            # Not scanned, use regular PDF parser
            return self.pdf_parser.parse(file_path, document_type)

        try:
            # Open PDF and extract images
            doc = fitz.open(file_path)
            text_parts = []
            pages = []
            paragraphs = []
            full_text = ""
            low_confidence_regions = []

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Perform OCR with confidence data
                ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

                # Extract text with confidence filtering
                page_text, page_low_conf = self._extract_text_with_confidence(
                    ocr_data, page_num + 1
                )

                if page_text.strip():
                    # Split into paragraphs
                    page_paragraphs = self._split_into_paragraphs(page_text)

                    para_start = len(full_text)
                    for para_text in page_paragraphs:
                        if para_text.strip():
                            para_end = para_start + len(para_text)
                            paragraphs.append(
                                {
                                    "text": para_text,
                                    "page": page_num + 1,
                                    "paragraph": len(paragraphs) + 1,
                                    "char_start": para_start,
                                    "char_end": para_end,
                                    "is_ocr": True,
                                }
                            )
                            pages.append(page_num + 1)
                            text_parts.append(para_text)
                            full_text += para_text + "\n\n"
                            para_start = para_end + 2

                    if page_low_conf:
                        low_confidence_regions.extend(page_low_conf)

            doc.close()

            if not full_text.strip():
                raise ValueError(f"OCR produced no text from PDF: {file_path}")

            # Default to scanned_pdf type
            if document_type is None:
                document_type = "scanned_pdf"

            metadata = {
                "is_scanned": True,
                "is_ocr": True,
                "low_confidence_regions": low_confidence_regions,
                "confidence_threshold": self.confidence_threshold,
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
            raise Exception(f"Failed to perform OCR on PDF {file_path}: {str(e)}") from e

    def _extract_text_with_confidence(
        self, ocr_data: dict, page_num: int
    ) -> tuple[str, list[dict]]:
        """Extract text from OCR data, filtering low-confidence tokens.

        Args:
            ocr_data: Tesseract OCR output dictionary
            page_num: Page number for logging

        Returns:
            Tuple of (text, low_confidence_regions)
        """
        text_parts = []
        low_confidence_regions = []
        current_line = []
        current_line_conf = []

        n_boxes = len(ocr_data["text"])

        for i in range(n_boxes):
            text = ocr_data["text"][i].strip()
            conf = float(ocr_data["conf"][i])

            if text:
                if conf >= self.confidence_threshold * 100:  # Tesseract uses 0-100 scale
                    current_line.append(text)
                    current_line_conf.append(conf)
                else:
                    # Low confidence token - record but don't include
                    low_confidence_regions.append(
                        {
                            "page": page_num,
                            "text": text,
                            "confidence": conf / 100.0,
                            "bbox": (
                                ocr_data["left"][i],
                                ocr_data["top"][i],
                                ocr_data["width"][i],
                                ocr_data["height"][i],
                            ),
                        }
                    )

            # Check for line break
            if i < n_boxes - 1:
                # Check if next token is on a new line
                if ocr_data["top"][i + 1] > ocr_data["top"][i] + ocr_data["height"][i] * 0.5:
                    if current_line:
                        text_parts.append(" ".join(current_line))
                        current_line = []
                        current_line_conf = []

        # Add last line
        if current_line:
            text_parts.append(" ".join(current_line))

        return "\n".join(text_parts), low_confidence_regions

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]




