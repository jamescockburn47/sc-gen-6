"""Email parser for .eml and .msg files."""

import email
import re
from datetime import datetime
from email.header import decode_header
from pathlib import Path
from typing import Optional

from src.ingestion.parsers.base_parser import BaseParser
from src.schema import DocumentType, ParsedDocument


class EmailParser(BaseParser):
    """Parser for email files (.eml, .msg).

    Extracts headers (From, To, Subject, Date) and body text.
    Handles both plain text and HTML emails.
    """

    def can_parse(self, file_path: str | Path) -> bool:
        """Check if file is an email."""
        suffix = Path(file_path).suffix.lower()
        return suffix in [".eml", ".msg", ".email"]

    def parse(
        self, file_path: str | Path, document_type: Optional[DocumentType] = None
    ) -> ParsedDocument:
        """Parse email file.

        Args:
            file_path: Path to email file
            document_type: Optional document type override (should be "email")

        Returns:
            ParsedDocument with email headers and body

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If email parsing fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Email file not found: {file_path}")

        # Force email type
        if document_type is None:
            document_type = "email"

        try:
            with open(file_path, "rb") as f:
                msg = email.message_from_bytes(f.read())

            # Extract headers
            headers = self._extract_headers(msg)

            # Extract body
            body_text = self._extract_body(msg)

            # Combine headers and body
            full_text = self._format_email(headers, body_text)

            # Build paragraphs (header + body paragraphs)
            paragraphs = []
            para_start = 0

            # Header paragraph
            header_text = self._format_headers_text(headers)
            if header_text:
                para_end = para_start + len(header_text)
                paragraphs.append(
                    {
                        "text": header_text,
                        "page": 1,
                        "paragraph": 1,
                        "is_header": True,
                        "char_start": para_start,
                        "char_end": para_end,
                    }
                )
                para_start = para_end + 2

            # Body paragraphs
            body_paragraphs = self._split_into_paragraphs(body_text)
            for para_idx, para_text in enumerate(body_paragraphs):
                if para_text.strip():
                    para_end = para_start + len(para_text)
                    paragraphs.append(
                        {
                            "text": para_text,
                            "page": 1,
                            "paragraph": len(paragraphs) + 1,
                            "char_start": para_start,
                            "char_end": para_end,
                        }
                    )
                    para_start = para_end + 2

            metadata = {
                "from": headers.get("From", ""),
                "to": headers.get("To", ""),
                "cc": headers.get("Cc", ""),
                "bcc": headers.get("Bcc", ""),
                "subject": headers.get("Subject", ""),
                "date": headers.get("Date", ""),
                "message_id": headers.get("Message-ID", ""),
            }

            return ParsedDocument(
                file_path=str(file_path),
                file_name=file_path.name,
                document_type=document_type,
                text=full_text,
                pages=[1] * len(paragraphs),
                paragraphs=paragraphs,
                metadata=metadata,
            )

        except Exception as e:
            raise Exception(f"Failed to parse email {file_path}: {str(e)}") from e

    def _extract_headers(self, msg: email.message.Message) -> dict[str, str]:
        """Extract email headers.

        Args:
            msg: Email message object

        Returns:
            Dictionary of header fields
        """
        headers = {}
        for key in ["From", "To", "Cc", "Bcc", "Subject", "Date", "Message-ID"]:
            value = msg.get(key)
            if value:
                # Decode header if needed
                decoded_parts = decode_header(value)
                decoded_str = ""
                for part, encoding in decoded_parts:
                    if isinstance(part, bytes):
                        decoded_str += part.decode(encoding or "utf-8", errors="ignore")
                    else:
                        decoded_str += part
                headers[key] = decoded_str
        return headers

    def _extract_body(self, msg: email.message.Message) -> str:
        """Extract email body text.

        Handles both plain text and HTML emails. For HTML, extracts
        text content (strips tags).

        Args:
            msg: Email message object

        Returns:
            Body text
        """
        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                # Skip attachments
                if "attachment" in content_disposition:
                    continue

                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode("utf-8", errors="ignore")
                        break
                elif content_type == "text/html":
                    # Fallback to HTML if no plain text
                    if not body:
                        payload = part.get_payload(decode=True)
                        if payload:
                            html_text = payload.decode("utf-8", errors="ignore")
                            # Simple HTML tag removal
                            body = re.sub(r"<[^>]+>", "", html_text)
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode("utf-8", errors="ignore")

        return body.strip()

    def _format_email(self, headers: dict[str, str], body: str) -> str:
        """Format email as text with headers and body.

        Args:
            headers: Email headers dictionary
            body: Email body text

        Returns:
            Formatted email text
        """
        header_text = self._format_headers_text(headers)
        return f"{header_text}\n\n{body}"

    def _format_headers_text(self, headers: dict[str, str]) -> str:
        """Format headers as text.

        Args:
            headers: Email headers dictionary

        Returns:
            Formatted header text
        """
        lines = []
        for key in ["From", "To", "Cc", "Bcc", "Subject", "Date"]:
            if key in headers:
                lines.append(f"{key}: {headers[key]}")
        return "\n".join(lines)

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]




