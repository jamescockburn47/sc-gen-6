"""Plain text file parser for .txt, .html, .htm, .md files.

Optimized for Enron-style email text files with RFC 2822 headers.
"""

from pathlib import Path
from typing import Optional
import re
from datetime import datetime

from src.ingestion.parsers.base_parser import BaseParser
from src.schema import DocumentType, ParsedDocument


class TextParser(BaseParser):
    """Parser for plain text files including TXT, HTML, and Markdown.
    
    Specially handles email text files with RFC 2822-style headers
    (Message-ID, From, To, Date, Subject, Cc, Bcc, etc.)
    """
    
    SUPPORTED_EXTENSIONS = {".txt", ".html", ".htm", ".md", ".markdown", ".rst", ".log"}
    
    # RFC 2822 email headers to extract
    EMAIL_HEADERS = {
        "message-id", "from", "to", "date", "subject", 
        "cc", "bcc", "reply-to", "x-from", "x-to",
        "content-type", "mime-version", "x-folder"
    }
    
    def can_parse(self, file_path: str | Path) -> bool:
        """Check if file is a supported text format."""
        path = Path(file_path)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def parse(
        self, file_path: str | Path, document_type: Optional[DocumentType] = None
    ) -> ParsedDocument:
        """Parse a plain text file.
        
        Args:
            file_path: Path to text file
            document_type: Optional document type override
            
        Returns:
            ParsedDocument with extracted content and metadata
        """
        path = Path(file_path)
        
        # Try multiple encodings
        text = None
        for encoding in ["utf-8", "utf-16", "latin-1", "cp1252"]:
            try:
                text = path.read_text(encoding=encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if text is None:
            # Fallback: read with error replacement
            text = path.read_text(encoding="utf-8", errors="replace")
        
        # Extract metadata from email-style headers
        metadata = self._extract_email_headers(text)
        
        # Get body text (after headers)
        body_text = self._extract_body(text)
        
        # Split into paragraphs
        paragraphs = []
        for para in re.split(r'\n\s*\n', body_text):
            para = para.strip()
            if para:
                paragraphs.append({"text": para})
        
        # Determine document type
        if document_type is None:
            document_type = self._detect_type(path, metadata)
        
        # Add file metadata
        metadata["file_size"] = path.stat().st_size
        metadata["modified_date"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        metadata["parser"] = "text_parser"
        
        return ParsedDocument(
            file_path=str(path.resolve()),
            file_name=path.name,
            text=text,
            paragraphs=paragraphs,
            document_type=document_type,
            metadata=metadata,
        )
    
    def _extract_email_headers(self, text: str) -> dict:
        """Extract RFC 2822-style email headers.
        
        Handles multi-line header values (continuation lines start with whitespace).
        """
        metadata = {}
        lines = text.split("\n")
        i = 0
        current_header = None
        current_value = []
        
        while i < len(lines):
            line = lines[i]
            
            # Check for empty line - marks end of headers
            if not line.strip():
                # Save last header if any
                if current_header and current_value:
                    metadata[current_header] = " ".join(current_value).strip()
                break
            
            # Check for continuation line (starts with whitespace)
            if line.startswith((' ', '\t')) and current_header:
                current_value.append(line.strip())
                i += 1
                continue
            
            # Check for new header
            header_match = re.match(r'^([A-Za-z][A-Za-z0-9-]*):\s*(.*)$', line)
            if header_match:
                # Save previous header
                if current_header and current_value:
                    metadata[current_header] = " ".join(current_value).strip()
                
                # Start new header
                header_name = header_match.group(1).lower()
                header_value = header_match.group(2)
                
                # Only extract known email headers
                if header_name in self.EMAIL_HEADERS:
                    current_header = header_name
                    current_value = [header_value] if header_value else []
                else:
                    current_header = None
                    current_value = []
            else:
                # Not a header line, stop looking
                if current_header and current_value:
                    metadata[current_header] = " ".join(current_value).strip()
                break
            
            i += 1
            
            # Safety limit: don't scan more than 50 lines for headers
            if i > 50:
                break
        
        # Clean up extracted values
        for key in list(metadata.keys()):
            value = metadata[key]
            # Remove angle brackets from Message-ID
            if key == "message-id" and value.startswith("<"):
                metadata[key] = value.strip("<>")
            # Parse date into ISO format if possible
            if key == "date":
                metadata["date_raw"] = value
                parsed_date = self._parse_date(value)
                if parsed_date:
                    metadata["date"] = parsed_date
        
        return metadata
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse various date formats into ISO format."""
        from email.utils import parsedate_to_datetime
        try:
            dt = parsedate_to_datetime(date_str)
            return dt.isoformat()
        except (ValueError, TypeError):
            pass
        
        # Try other common formats
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%d %b %Y %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%d/%m/%Y",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.isoformat()
            except ValueError:
                continue
        
        return None
    
    def _extract_body(self, text: str) -> str:
        """Extract body text, skipping headers."""
        lines = text.split("\n")
        body_start = 0
        
        # Find first blank line (end of headers)
        for i, line in enumerate(lines):
            if not line.strip():
                body_start = i + 1
                break
            # Stop if we've gone too far without finding headers
            if i > 50:
                body_start = 0
                break
        
        return "\n".join(lines[body_start:]).strip()
    
    def _detect_type(self, path: Path, metadata: dict) -> DocumentType:
        """Detect document type from content and metadata."""
        # If has email headers, it's an email
        if any(k in metadata for k in ["from", "to", "message-id", "subject"]):
            return "email"
        
        suffix = path.suffix.lower()
        if suffix in {".html", ".htm"}:
            return "disclosure"
        elif suffix == ".md":
            return "disclosure"
        else:
            return "disclosure"  # Default for plain text
