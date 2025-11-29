"""Persistent catalog for document metadata (labels, categories, graph flags)."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

from src.schema import DocumentType, ParsedDocument


DEFAULT_CATEGORY: DocumentType = "disclosure"
GRAPH_DOC_TYPES = {
    "witness_statement",
    "court_filing",
    "pleading",
    "statute",
    "contract",
}


@dataclass
class DocumentRecord:
    """User-facing metadata for a single document."""

    file_path: str
    file_name: str
    label: str
    category: DocumentType
    include_in_graph: bool = True
    indexed: bool = False  # Whether chunks have been indexed
    error: str | None = None  # Error message if ingestion failed
    chunk_count: int = 0  # Number of chunks created
    ingested_at: str | None = None  # ISO timestamp

    def to_dict(self) -> dict:
        return asdict(self)


class DocumentCatalog:
    """Simple JSON-backed catalog storing document metadata."""

    CONFIGURED_CATEGORIES: tuple[DocumentType, ...] = (
        "witness_statement",
        "court_filing",
        "pleading",
        "statute",
        "contract",
        "disclosure",
        "email",
        "scanned_pdf",
        "unknown",
    )

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path or "data/documents/catalog.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._records: dict[str, DocumentRecord] = {}
        self._load()

    # ------------------------------------------------------------------#
    # Public API
    # ------------------------------------------------------------------#
    def ensure_record(self, document: ParsedDocument) -> DocumentRecord:
        """Ensure metadata exists for document; return the record."""
        with self._lock:
            record = self._records.get(document.file_path)
            if record is None:
                label = self._suggest_label(document)
                category = self._suggest_category(document.document_type)
                include = self._default_include_in_graph(category)
                record = DocumentRecord(
                    file_path=document.file_path,
                    file_name=document.file_name,
                    label=label,
                    category=category,
                    include_in_graph=include,
                )
                self._records[document.file_path] = record
            else:
                # keep metadata but refresh file name
                record.file_name = document.file_name
            self._save()
            return record

    def update_record(
        self,
        file_path_or_record: str | DocumentRecord,
        *,
        label: Optional[str] = None,
        category: Optional[DocumentType] = None,
        include_in_graph: Optional[bool] = None,
        indexed: Optional[bool] = None,
        error: Optional[str] = None,
        chunk_count: Optional[int] = None,
        ingested_at: Optional[str] = None,
    ) -> DocumentRecord:
        """Update an existing record.
        
        Can pass either a file_path string or a DocumentRecord object.
        If passing a DocumentRecord, its values are used directly.
        """
        with self._lock:
            # Handle both string path and record object
            if isinstance(file_path_or_record, DocumentRecord):
                # Direct update from record object
                record = file_path_or_record
                file_path = record.file_path
                self._records[file_path] = record
            else:
                file_path = file_path_or_record
                record = self._records.get(file_path)
                if record is None:
                    raise ValueError(f"Document not found in catalog: {file_path}")
                
                # Apply field updates
                if label is not None:
                    record.label = label.strip() or record.label
                if category is not None:
                    record.category = category
                if include_in_graph is not None:
                    record.include_in_graph = include_in_graph
                if indexed is not None:
                    record.indexed = indexed
                if error is not None:
                    record.error = error
                if chunk_count is not None:
                    record.chunk_count = chunk_count
                if ingested_at is not None:
                    record.ingested_at = ingested_at
            
            self._save()
            return record

    def get(self, file_path: str) -> Optional[DocumentRecord]:
        return self._records.get(file_path)
    
    def get_record(self, file_path: str) -> Optional[DocumentRecord]:
        """Alias for get() for clarity."""
        return self._records.get(file_path)
    
    def list_records(self) -> list[DocumentRecord]:
        """Return all records as a list."""
        return list(self._records.values())

    def all_records(self) -> Iterable[DocumentRecord]:
        return list(self._records.values())

    def delete_record(self, file_path: str) -> None:
        with self._lock:
            if file_path in self._records:
                del self._records[file_path]
                self._save()

    def default_include_flag(self, doc_type: DocumentType) -> bool:
        """Expose default graph flag for UI helpers."""
        return self._default_include_in_graph(doc_type)

    # ------------------------------------------------------------------#
    # Internals
    # ------------------------------------------------------------------#
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = []
        for item in data:
            try:
                record = DocumentRecord(**item)
            except TypeError:
                continue
            self._records[record.file_path] = record

    def _save(self) -> None:
        data = [record.to_dict() for record in self._records.values()]
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _suggest_label(self, document: ParsedDocument) -> str:
        meta_title = document.metadata.get("title")
        if isinstance(meta_title, str) and meta_title.strip():
            return meta_title.strip()
        # try to capture heading from first paragraph
        if document.paragraphs:
            first = document.paragraphs[0].get("text", "")
            if first:
                first_line = first.strip().split("\n")[0]
                if 8 <= len(first_line) <= 80:
                    return first_line
        return document.file_name

    def _suggest_category(self, doc_type: DocumentType) -> DocumentType:
        if doc_type in self.CONFIGURED_CATEGORIES:
            return doc_type
        return DEFAULT_CATEGORY

    def _default_include_in_graph(self, doc_type: DocumentType) -> bool:
        return doc_type in GRAPH_DOC_TYPES

