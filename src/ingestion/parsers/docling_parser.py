"""Docling-based PDF parser for high-quality document understanding.

Used as fallback when PyMuPDF fails or produces low-quality output.
Docling uses AI models for layout analysis and table extraction.
"""

from pathlib import Path
from typing import Optional
import re

from loguru import logger

from src.ingestion.parsers.base_parser import BaseParser
from src.schema import DocumentType, ParsedDocument


class DoclingParser(BaseParser):
    """AI-powered PDF parser using Docling.
    
    Provides high-quality document understanding with:
    - Layout analysis (columns, headers, footers)
    - Table structure recognition
    - Reading order detection
    - Markdown output for LLM-ready text
    
    Note: Slower than PyMuPDF but more accurate for complex layouts.
    """

    _converter = None  # Lazy-loaded singleton
    
    def __init__(self):
        """Initialize parser (converter loaded lazily on first use)."""
        pass
    
    @classmethod
    def release_gpu_memory(cls):
        """Release GPU memory by unloading both Docling converters.
        
        Call this after parsing is complete and before starting other
        GPU operations to free up VRAM.
        """
        has_converters = cls._converter_fast is not None or cls._converter_accurate is not None
        
        if has_converters:
            logger.info("[Docling] Releasing GPU memory...")
            cls._converter_fast = None
            cls._converter_accurate = None
            cls._converter = None  # Legacy cleanup
            
            # Force PyTorch to release GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info("[Docling] PyTorch GPU cache cleared")
            except Exception as e:
                logger.warning(f"[Docling] Failed to clear PyTorch cache: {e}")
            
            # Run garbage collection to release Python objects
            import gc
            gc.collect()
            logger.success("[Docling] GPU memory released")
    
    # Processing modes
    MODE_FAST = "fast"      # TableFormerMode.FAST - faster, less accurate
    MODE_ACCURATE = "accurate"  # TableFormerMode.ACCURATE - slower, better for complex tables
    
    _converter_fast = None   # Lazy-loaded singleton for FAST mode
    _converter_accurate = None  # Lazy-loaded singleton for ACCURATE mode
    _current_mode = MODE_FAST  # Default to FAST
    
    @classmethod
    def set_mode(cls, mode: str):
        """Set the processing mode (fast/accurate)."""
        if mode in (cls.MODE_FAST, cls.MODE_ACCURATE):
            cls._current_mode = mode
            logger.info(f"[Docling] Mode set to: {mode.upper()}")
    
    @classmethod
    def _detect_complexity(cls, file_path: Path) -> str:
        """Detect document complexity to route to appropriate mode.
        
        Returns MODE_FAST for simple docs, MODE_ACCURATE for complex ones.
        """
        import os
        file_size = os.path.getsize(file_path)
        
        # Heuristics for complexity:
        # - Large files (>5MB) likely have many pages/tables
        # - Files with "schedule", "accounts", "financial" in name likely have tables
        filename_lower = file_path.name.lower()
        
        complex_keywords = ["schedule", "account", "financial", "appendix", "annex", "table"]
        has_complex_keyword = any(kw in filename_lower for kw in complex_keywords)
        is_large = file_size > 5 * 1024 * 1024  # >5MB
        
        if has_complex_keyword or is_large:
            logger.info(f"[Docling] Complex document detected: {file_path.name} (size={file_size/1024:.0f}KB)")
            return cls.MODE_ACCURATE
        
        return cls.MODE_FAST
    
    @classmethod
    def _get_converter(cls, mode: str = None):
        """Get or create the Docling converter for the specified mode."""
        if mode is None:
            mode = cls._current_mode
            
        # Check if we already have the converter for this mode
        if mode == cls.MODE_FAST and cls._converter_fast is not None:
            return cls._converter_fast
        if mode == cls.MODE_ACCURATE and cls._converter_accurate is not None:
            return cls._converter_accurate
            
        try:
            import torch
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                AcceleratorOptions,
                AcceleratorDevice,
                TableStructureOptions,
            )
            from docling.datamodel.pipeline_options import TableFormerMode
            
            # Select TableFormer mode
            table_mode = TableFormerMode.FAST if mode == cls.MODE_FAST else TableFormerMode.ACCURATE
            
            # Check if GPU is available
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"[Docling] GPU detected: {device_name}")
                
                accelerator = AcceleratorOptions(
                    device=AcceleratorDevice.CUDA,
                )
                
                # Configure pipeline with selected mode
                table_options = TableStructureOptions(
                    mode=table_mode,
                    do_cell_matching=True,
                )
                
                pipeline_options = PdfPipelineOptions(
                    accelerator_options=accelerator,
                    do_ocr=False,
                    force_backend_text=True,
                    do_table_structure=True,
                    table_structure_options=table_options,
                    layout_batch_size=16,  # Increased for GPU
                    table_batch_size=8,    # Increased for FAST mode
                )
                
                pdf_format_option = PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
                
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: pdf_format_option,
                    }
                )
                logger.success(f"[Docling] Ready (GPU: {device_name}, Mode: {mode.upper()}, OCR: OFF)")
            else:
                logger.info("[Docling] No GPU detected, using CPU mode")
                
                table_options = TableStructureOptions(
                    mode=table_mode,
                )
                
                pipeline_options = PdfPipelineOptions(
                    do_ocr=False,
                    force_backend_text=True,
                    do_table_structure=True,
                    table_structure_options=table_options,
                )
                
                pdf_format_option = PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
                
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: pdf_format_option,
                    }
                )
                logger.info(f"[Docling] Ready (CPU mode, Mode: {mode.upper()})")
            
            # Cache the converter
            if mode == cls.MODE_FAST:
                cls._converter_fast = converter
            else:
                cls._converter_accurate = converter
                
            return converter
            
        except ImportError as e:
            logger.error(f"[Docling] Failed to import: {e}")
            raise ImportError(
                "Docling not installed. Install with: pip install docling"
            ) from e


    def can_parse(self, file_path: str | Path) -> bool:
        """Check if file is a PDF."""
        return Path(file_path).suffix.lower() == ".pdf"

    def parse(
        self, file_path: str | Path, document_type: Optional[DocumentType] = None
    ) -> ParsedDocument:
        """Parse PDF using Docling AI models.
        
        Args:
            file_path: Path to PDF file
            document_type: Optional document type override
            
        Returns:
            ParsedDocument with structured text and metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        logger.info(f"[Docling] Parsing {file_path.name}...")
        
        # Intelligent routing: detect complexity and select appropriate mode
        detected_mode = self._detect_complexity(file_path)
        
        # Log GPU/CPU stats before parsing (lightweight, non-blocking)
        try:
            from src.system.gpu_monitor import log_performance
            log_performance(f"BEFORE {file_path.name}")
        except ImportError:
            pass
        
        import time
        start_time = time.time()
        
        try:
            # Get converter with appropriate mode
            converter = self._get_converter(mode=detected_mode)
            
            # Convert PDF to Docling document
            result = converter.convert(str(file_path))
            doc = result.document
            
            # Log GPU/CPU stats after parsing
            elapsed = time.time() - start_time
            try:
                from src.system.gpu_monitor import log_performance
                log_performance(f"AFTER {file_path.name} ({elapsed:.1f}s)")
            except ImportError:
                pass
            
            # Export to Markdown for LLM-ready text
            markdown_text = doc.export_to_markdown()
            
            # Extract page count
            page_count = len(result.pages) if hasattr(result, 'pages') else 1
            
            # Split into paragraphs (by double newline in markdown)
            paragraphs = self._extract_paragraphs(markdown_text)
            
            # Extract headers from markdown (lines starting with #)
            headers = self._extract_headers_from_markdown(markdown_text)
            
            # Detect document type if not provided
            if document_type is None:
                document_type = self.detect_document_type(
                    file_path, markdown_text, {"headers": headers}
                )
            
            metadata = {
                "page_count": page_count,
                "headers": headers,
                "is_scanned": False,
                "parser": "docling",
                "has_tables": "| " in markdown_text,  # Simple table detection
            }
            
            # Build pages list (approximate - Docling merges pages)
            pages = list(range(1, page_count + 1)) * max(1, len(paragraphs) // page_count)
            pages = pages[:len(paragraphs)]
            
            logger.success(
                f"[Docling] Parsed {file_path.name}: "
                f"{len(markdown_text)} chars, {len(paragraphs)} paragraphs"
            )
            
            return ParsedDocument(
                file_path=str(file_path),
                file_name=file_path.name,
                document_type=document_type,
                text=markdown_text,
                pages=pages,
                paragraphs=paragraphs,
                metadata=metadata,
            )
            
        except Exception as e:
            logger.error(f"[Docling] Failed to parse {file_path}: {e}")
            raise Exception(f"Docling parsing failed for {file_path}: {e}") from e
    
    def _extract_paragraphs(self, markdown_text: str) -> list[dict]:
        """Extract paragraphs from markdown text."""
        # Split by double newlines
        raw_paragraphs = re.split(r'\n\s*\n', markdown_text)
        
        paragraphs = []
        char_pos = 0
        
        for i, para_text in enumerate(raw_paragraphs):
            para_text = para_text.strip()
            if not para_text:
                continue
                
            para_end = char_pos + len(para_text)
            paragraphs.append({
                "text": para_text,
                "page": 1,  # Docling doesn't preserve page info in export
                "paragraph": i + 1,
                "char_start": char_pos,
                "char_end": para_end,
            })
            char_pos = para_end + 2  # +2 for "\n\n"
        
        return paragraphs
    
    def _extract_headers_from_markdown(self, markdown_text: str) -> list[str]:
        """Extract headers from markdown (lines starting with #)."""
        headers = []
        for line in markdown_text.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                # Remove # prefixes and clean up
                header_text = re.sub(r'^#+\s*', '', line).strip()
                if header_text and len(header_text) < 200:
                    headers.append(header_text)
        return headers[:15]  # Limit to first 15 headers
