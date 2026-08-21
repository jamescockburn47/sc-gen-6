"""SQLite-backed document catalog with atomic operations and pagination.

Replaces JSON catalog for large document sets (500k+) with:
- Atomic writes (no corruption on crash)
- Pagination (don't load all records into memory)
- Fast counts without full load
- Proper indexing for queries
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from src.documents.catalog import DocumentRecord, DEFAULT_CATEGORY, GRAPH_DOC_TYPES
from src.schema import DocumentType, ParsedDocument


class SQLiteCatalog:
    """SQLite-backed document catalog with atomic operations."""

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
        """Initialize catalog.
        
        Args:
            path: Path to documents directory. catalog.db will be created inside.
        """
        if path is None:
            from src.config_loader import get_settings
            settings = get_settings()
            docs_path = getattr(settings.paths, 'documents', 'data/documents')
            self.db_path = Path(docs_path) / "catalog.db"
        else:
            p = Path(path)
            if p.suffix == '.db':
                self.db_path = p
            else:
                self.db_path = p / "catalog.db"
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
        
        # Migrate from JSON if exists
        self._migrate_from_json()

    def _init_db(self) -> None:
        """Create tables with proper schema and indexes."""
        conn = self._get_conn()
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS documents (
                file_path TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                label TEXT,
                category TEXT DEFAULT 'unknown',
                include_in_graph INTEGER DEFAULT 1,
                indexed INTEGER DEFAULT 0,
                error TEXT,
                chunk_count INTEGER DEFAULT 0,
                ingested_at TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_indexed ON documents(indexed);
            CREATE INDEX IF NOT EXISTS idx_category ON documents(category);
            CREATE INDEX IF NOT EXISTS idx_file_name ON documents(file_name);
        ''')
        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a connection with row factory and WAL mode for crash safety."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        
        # Enable crash-safe settings (WAL is more resistant to corruption)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        
        return conn

    def _migrate_from_json(self) -> None:
        """Migrate from JSON catalog if it exists and DB is empty."""
        json_path = self.db_path.parent / "catalog.json"
        if not json_path.exists():
            return
        
        # Check if DB already has data
        if self.count() > 0:
            return
        
        try:
            import json
            data = json.loads(json_path.read_text(encoding='utf-8'))
            if not data:
                return
            
            print(f"[SQLiteCatalog] Migrating {len(data)} records from JSON...")
            
            conn = self._get_conn()
            with conn:
                for item in data:
                    conn.execute('''
                        INSERT OR IGNORE INTO documents 
                        (file_path, file_name, label, category, include_in_graph, 
                         indexed, error, chunk_count, ingested_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        item.get('file_path'),
                        item.get('file_name'),
                        item.get('label'),
                        item.get('category', 'unknown'),
                        1 if item.get('include_in_graph', True) else 0,
                        1 if item.get('indexed', False) else 0,
                        item.get('error'),
                        item.get('chunk_count', 0),
                        item.get('ingested_at'),
                    ))
            conn.close()
            
            # Rename old JSON to .bak
            backup_path = json_path.with_suffix('.json.migrated')
            json_path.rename(backup_path)
            print(f"[SQLiteCatalog] Migration complete. Old JSON backed up to {backup_path.name}")
            
        except Exception as e:
            print(f"[SQLiteCatalog] Migration failed: {e}")

    # ------------------------------------------------------------------#
    # Public API - Compatible with DocumentCatalog
    # ------------------------------------------------------------------#
    
    def count(self) -> int:
        """Fast count without loading records."""
        conn = self._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.close()
        return count

    def count_indexed(self) -> int:
        """Count indexed documents."""
        conn = self._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM documents WHERE indexed = 1").fetchone()[0]
        conn.close()
        return count

    def count_pending(self) -> int:
        """Count documents pending indexing."""
        conn = self._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM documents WHERE indexed = 0").fetchone()[0]
        conn.close()
        return count

    def list_records(self, offset: int = 0, limit: int = 100) -> list[DocumentRecord]:
        """Paginated record retrieval."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM documents ORDER BY label LIMIT ? OFFSET ?",
            (limit, offset)
        )
        records = [self._row_to_record(row) for row in cursor]
        conn.close()
        return records

    def list_pending(self, limit: int = 100) -> list[DocumentRecord]:
        """Get documents pending indexing."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM documents WHERE indexed = 0 LIMIT ?",
            (limit,)
        )
        records = [self._row_to_record(row) for row in cursor]
        conn.close()
        return records

    def ensure_record(self, document: ParsedDocument) -> DocumentRecord:
        """Ensure metadata exists for document; return the record."""
        with self._lock:
            existing = self.get(document.file_path)
            if existing:
                # Update file_name if changed
                if existing.file_name != document.file_name:
                    self._update_field(document.file_path, 'file_name', document.file_name)
                    existing.file_name = document.file_name
                return existing
            
            # Create new record
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
            
            conn = self._get_conn()
            with conn:
                conn.execute('''
                    INSERT INTO documents 
                    (file_path, file_name, label, category, include_in_graph)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    record.file_path,
                    record.file_name,
                    record.label,
                    record.category,
                    1 if record.include_in_graph else 0,
                ))
            conn.close()
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
        """Update an existing record."""
        with self._lock:
            if isinstance(file_path_or_record, DocumentRecord):
                # Direct update from record object
                record = file_path_or_record
                file_path = record.file_path
                conn = self._get_conn()
                with conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO documents 
                        (file_path, file_name, label, category, include_in_graph, 
                         indexed, error, chunk_count, ingested_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        record.file_path,
                        record.file_name,
                        record.label,
                        record.category,
                        1 if record.include_in_graph else 0,
                        1 if record.indexed else 0,
                        record.error,
                        record.chunk_count,
                        record.ingested_at,
                    ))
                conn.close()
                return record
            
            file_path = file_path_or_record
            record = self.get(file_path)
            if record is None:
                raise ValueError(f"Document not found in catalog: {file_path}")
            
            # Build update query
            updates = []
            params = []
            
            if label is not None:
                updates.append("label = ?")
                params.append(label.strip() or record.label)
            if category is not None:
                updates.append("category = ?")
                params.append(category)
            if include_in_graph is not None:
                updates.append("include_in_graph = ?")
                params.append(1 if include_in_graph else 0)
            if indexed is not None:
                updates.append("indexed = ?")
                params.append(1 if indexed else 0)
            if error is not None:
                updates.append("error = ?")
                params.append(error if error else None)
            if chunk_count is not None:
                updates.append("chunk_count = ?")
                params.append(chunk_count)
            if ingested_at is not None:
                updates.append("ingested_at = ?")
                params.append(ingested_at)
            
            if updates:
                params.append(file_path)
                conn = self._get_conn()
                with conn:
                    conn.execute(
                        f"UPDATE documents SET {', '.join(updates)} WHERE file_path = ?",
                        params
                    )
                conn.close()
            
            return self.get(file_path)

    def get(self, file_path: str) -> Optional[DocumentRecord]:
        """Get record by file path."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM documents WHERE file_path = ?",
            (file_path,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_record(row)
        return None

    def get_record(self, file_path: str) -> Optional[DocumentRecord]:
        """Alias for get() for clarity."""
        return self.get(file_path)

    def all_records(self) -> Iterator[DocumentRecord]:
        """Iterate all records without loading into memory."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM documents ORDER BY label")
        for row in cursor:
            yield self._row_to_record(row)
        conn.close()

    def list_all_records(self) -> list[DocumentRecord]:
        """Return all records as list (use with caution for large datasets)."""
        return list(self.all_records())

    def delete_record(self, file_path: str) -> None:
        """Delete a record."""
        with self._lock:
            conn = self._get_conn()
            with conn:
                conn.execute("DELETE FROM documents WHERE file_path = ?", (file_path,))
            conn.close()

    def delete_all(self) -> None:
        """Delete all records."""
        with self._lock:
            conn = self._get_conn()
            with conn:
                conn.execute("DELETE FROM documents")
            conn.close()

    def default_include_flag(self, doc_type: DocumentType) -> bool:
        """Expose default graph flag for UI helpers."""
        return self._default_include_in_graph(doc_type)

    # ------------------------------------------------------------------#
    # Batch operations for large ingestion
    # ------------------------------------------------------------------#
    
    def bulk_insert(self, records: list[DocumentRecord]) -> int:
        """Insert multiple records in a single transaction."""
        with self._lock:
            conn = self._get_conn()
            inserted = 0
            with conn:
                for record in records:
                    try:
                        conn.execute('''
                            INSERT OR IGNORE INTO documents 
                            (file_path, file_name, label, category, include_in_graph, 
                             indexed, error, chunk_count, ingested_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            record.file_path,
                            record.file_name,
                            record.label,
                            record.category,
                            1 if record.include_in_graph else 0,
                            1 if record.indexed else 0,
                            record.error,
                            record.chunk_count,
                            record.ingested_at,
                        ))
                        inserted += 1
                    except Exception:
                        pass
            conn.close()
            return inserted

    def mark_indexed(self, file_paths: list[str], chunk_counts: Optional[list[int]] = None) -> None:
        """Mark multiple documents as indexed in a single transaction."""
        with self._lock:
            conn = self._get_conn()
            now = datetime.now().isoformat()
            with conn:
                for i, path in enumerate(file_paths):
                    chunk_count = chunk_counts[i] if chunk_counts else 0
                    conn.execute(
                        "UPDATE documents SET indexed = 1, ingested_at = ?, chunk_count = ? WHERE file_path = ?",
                        (now, chunk_count, path)
                    )
            conn.close()

    # ------------------------------------------------------------------#
    # Internals
    # ------------------------------------------------------------------#
    
    def _row_to_record(self, row: sqlite3.Row) -> DocumentRecord:
        """Convert database row to DocumentRecord."""
        return DocumentRecord(
            file_path=row['file_path'],
            file_name=row['file_name'],
            label=row['label'] or row['file_name'],
            category=row['category'] or 'unknown',
            include_in_graph=bool(row['include_in_graph']),
            indexed=bool(row['indexed']),
            error=row['error'],
            chunk_count=row['chunk_count'] or 0,
            ingested_at=row['ingested_at'],
        )

    def _update_field(self, file_path: str, field: str, value) -> None:
        """Update a single field."""
        conn = self._get_conn()
        with conn:
            conn.execute(f"UPDATE documents SET {field} = ? WHERE file_path = ?", (value, file_path))
        conn.close()

    def _suggest_label(self, document: ParsedDocument) -> str:
        """Suggest a label from document metadata."""
        meta_title = document.metadata.get("title")
        if isinstance(meta_title, str) and meta_title.strip():
            return meta_title.strip()[:200]
        if document.paragraphs:
            first = document.paragraphs[0].get("text", "")
            if first:
                first_line = first.strip().split("\n")[0]
                if 8 <= len(first_line) <= 200:
                    return first_line
        return document.file_name

    def _suggest_category(self, doc_type: DocumentType) -> DocumentType:
        """Suggest category from document type."""
        if doc_type in self.CONFIGURED_CATEGORIES:
            return doc_type
        return DEFAULT_CATEGORY

    def _default_include_in_graph(self, doc_type: DocumentType) -> bool:
        """Determine if document type should be included in graph."""
        return doc_type in GRAPH_DOC_TYPES
