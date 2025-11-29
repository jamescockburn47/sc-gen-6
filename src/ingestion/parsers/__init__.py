"""Document parsers for various file formats."""

from src.ingestion.parsers.base_parser import BaseParser
from src.ingestion.parsers.docx_parser import DOCXParser
from src.ingestion.parsers.email_parser import EmailParser
from src.ingestion.parsers.ocr_parser import OCRParser
from src.ingestion.parsers.pdf_parser import PDFParser
from src.ingestion.parsers.spreadsheet_parser import SpreadsheetParser

__all__ = [
    "BaseParser",
    "PDFParser",
    "DOCXParser",
    "EmailParser",
    "SpreadsheetParser",
    "OCRParser",
]
