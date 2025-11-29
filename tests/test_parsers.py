"""Tests for document parsers."""

from pathlib import Path

import pytest

from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.ingestion.parsers.docx_parser import DOCXParser
from src.ingestion.parsers.email_parser import EmailParser
from src.ingestion.parsers.pdf_parser import PDFParser
from src.ingestion.parsers.spreadsheet_parser import SpreadsheetParser
from src.schema import ParsedDocument


def test_pdf_parser_can_parse():
    """Test PDF parser recognizes PDF files."""
    parser = PDFParser()
    assert parser.can_parse("test.pdf")
    assert parser.can_parse("test.PDF")
    assert not parser.can_parse("test.docx")


def test_docx_parser_can_parse():
    """Test DOCX parser recognizes DOCX files."""
    parser = DOCXParser()
    assert parser.can_parse("test.docx")
    assert parser.can_parse("test.doc")
    assert not parser.can_parse("test.pdf")


def test_email_parser_can_parse():
    """Test email parser recognizes email files."""
    parser = EmailParser()
    assert parser.can_parse("test.eml")
    assert parser.can_parse("test.msg")
    assert not parser.can_parse("test.pdf")


def test_spreadsheet_parser_can_parse():
    """Test spreadsheet parser recognizes spreadsheet files."""
    parser = SpreadsheetParser()
    assert parser.can_parse("test.xlsx")
    assert parser.can_parse("test.xls")
    assert not parser.can_parse("test.pdf")


def test_ingestion_pipeline_get_parser():
    """Test pipeline selects correct parser."""
    pipeline = IngestionPipeline()
    
    assert pipeline.get_parser("test.pdf") is not None
    assert pipeline.get_parser("test.docx") is not None
    assert pipeline.get_parser("test.eml") is not None
    assert pipeline.get_parser("test.xlsx") is not None
    assert pipeline.get_parser("test.unknown") is None


def test_ingestion_pipeline_parse_nonexistent_file():
    """Test pipeline handles nonexistent files gracefully."""
    pipeline = IngestionPipeline()
    result = pipeline.parse_document("nonexistent_file.pdf")
    assert result is None


def test_ingestion_pipeline_ingest_empty_folder(tmp_path):
    """Test pipeline handles empty folder."""
    pipeline = IngestionPipeline()
    result = pipeline.ingest_folder(tmp_path)
    assert result == []


def test_parsed_document_validation():
    """Test ParsedDocument validation."""
    # Valid document
    doc = ParsedDocument(
        file_path="test.pdf",
        file_name="test.pdf",
        document_type="witness_statement",
        text="Sample text content",
    )
    assert doc.text == "Sample text content"
    assert doc.document_type == "witness_statement"

    # Invalid: empty text
    with pytest.raises(ValueError, match="cannot be empty"):
        ParsedDocument(
            file_path="test.pdf",
            file_name="test.pdf",
            document_type="witness_statement",
            text="",
        )

    # Invalid: unknown document type
    with pytest.raises(ValueError, match="Invalid document type"):
        ParsedDocument(
            file_path="test.pdf",
            file_name="test.pdf",
            document_type="invalid_type",  # type: ignore
            text="Sample text",
        )




