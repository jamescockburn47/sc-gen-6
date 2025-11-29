"""SQLite FTS5 keyword index with auto-sync triggers.

Replaces BM25 pickle-based index with SQLite FTS5 for:
- Automatic index updates on insert/update/delete
- Richer metadata indexing (title, author, page, keywords)
- Single database file for easier backup/migration
- Better query syntax (phrase search, prefix, boolean)
"""

import sqlite3
import re
from pathlib import Path
from typing import Any, Optional
from dataclasses import asdict

from src.config_loader import Settings, get_settings
from src.schema import Chunk


class FTS5Index:
    """SQLite FTS5 full-text search index.
    
    Uses SQLite FTS5 virtual table with auto-sync triggers for
    automatic index maintenance. Supports rich metadata indexing.
    """

    DB_FILE = "fts5_index.db"

    def __init__(
        self,
        index_path: Optional[str | Path] = None,
        settings: Optional[Settings] = None,
    ):
        """Initialize FTS5 index.

        Args:
            index_path: Path to index directory. If None, uses path from config.
            settings: Settings instance. If None, loads from config.
        """
        self.settings = settings or get_settings()
        self.index_path = Path(index_path or self.settings.paths.keyword_index)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.index_path / self.DB_FILE
        
        # Thread-local storage for connections (SQLite requires per-thread connections)
        import threading
        self._local = threading.local()
        
        # Initialize database schema
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent read performance
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema with FTS5 virtual table and triggers."""
        conn = self._get_conn()
        
        # Create chunks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,
                document_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                page INTEGER,
                paragraph INTEGER,
                start_char INTEGER,
                end_char INTEGER,
                document_type TEXT,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for common queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_name ON chunks(file_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id)")
        
        # Create FTS5 virtual table
        # Using 'unicode61' tokenizer for better international support
        # 'remove_diacritics=0' preserves accents for legal document accuracy
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                document_id UNINDEXED,
                file_name,
                chunk_text,
                document_type,
                content='chunks',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 0'
            )
        """)
        
        # Create auto-sync triggers
        # Trigger: After INSERT on chunks
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai
            AFTER INSERT ON chunks
            BEGIN
                INSERT INTO chunks_fts(
                    rowid, chunk_id, document_id, file_name, chunk_text, document_type
                ) VALUES (
                    new.id, new.chunk_id, new.document_id, new.file_name, 
                    new.chunk_text, new.document_type
                );
            END
        """)
        
        # Trigger: After DELETE on chunks
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ad
            AFTER DELETE ON chunks
            BEGIN
                INSERT INTO chunks_fts(
                    chunks_fts, rowid, chunk_id, document_id, file_name, 
                    chunk_text, document_type
                ) VALUES (
                    'delete', old.id, old.chunk_id, old.document_id, 
                    old.file_name, old.chunk_text, old.document_type
                );
            END
        """)
        
        # Trigger: After UPDATE on chunks
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_au
            AFTER UPDATE ON chunks
            BEGIN
                INSERT INTO chunks_fts(
                    chunks_fts, rowid, chunk_id, document_id, file_name, 
                    chunk_text, document_type
                ) VALUES (
                    'delete', old.id, old.chunk_id, old.document_id, 
                    old.file_name, old.chunk_text, old.document_type
                );
                INSERT INTO chunks_fts(
                    rowid, chunk_id, document_id, file_name, chunk_text, document_type
                ) VALUES (
                    new.id, new.chunk_id, new.document_id, new.file_name, 
                    new.chunk_text, new.document_type
                );
            END
        """)
        
        conn.commit()

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to the index.
        
        Uses INSERT OR REPLACE for upsert behavior.
        FTS5 index is automatically updated via triggers.

        Args:
            chunks: List of Chunk objects to add
        """
        if not chunks:
            return
        
        conn = self._get_conn()
        
        import json
        
        for chunk in chunks:
            # Serialize metadata to JSON
            metadata_json = json.dumps(chunk.metadata) if chunk.metadata else "{}"
            
            # Handle both old and new attribute names for char positions
            char_start = getattr(chunk, 'char_start', None) or getattr(chunk, 'start_char', 0)
            char_end = getattr(chunk, 'char_end', None) or getattr(chunk, 'end_char', 0)
            
            conn.execute("""
                INSERT OR REPLACE INTO chunks (
                    chunk_id, document_id, file_name, chunk_text,
                    page, paragraph, start_char, end_char,
                    document_type, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.chunk_id,
                chunk.document_id,
                chunk.file_name,
                chunk.text,
                chunk.page_number,
                chunk.paragraph_number,
                char_start,
                char_end,
                chunk.document_type,
                metadata_json,
            ))
        
        conn.commit()

    def search(
        self,
        query: str,
        top_k: int = 10,
        selected_documents: Optional[list[str]] = None,
        doc_type_filter: Optional[str] = None,
    ) -> list[tuple[str, float]]:
        """Search the FTS5 index.

        Args:
            query: Query string (supports FTS5 syntax: phrases, prefix*, boolean)
            top_k: Maximum number of results to return
            selected_documents: Optional list of file names to filter by
            doc_type_filter: Optional document type filter

        Returns:
            List of tuples (chunk_id, score) sorted by relevance
        """
        if not query.strip():
            return []
        
        conn = self._get_conn()
        
        # Escape special FTS5 characters for safe querying
        # But preserve common operators for power users
        safe_query = self._prepare_query(query)
        
        # Build WHERE clause for filters
        where_parts = []
        params = [safe_query]
        
        if selected_documents:
            placeholders = ",".join("?" * len(selected_documents))
            where_parts.append(f"c.file_name IN ({placeholders})")
            params.extend(selected_documents)
        
        if doc_type_filter:
            where_parts.append("c.document_type = ?")
            params.append(doc_type_filter)
        
        where_clause = ""
        if where_parts:
            where_clause = "AND " + " AND ".join(where_parts)
        
        # Query with BM25 ranking
        # bm25() returns negative values (lower is better), so we negate for sorting
        sql = f"""
            SELECT 
                c.chunk_id,
                -bm25(chunks_fts) as score
            FROM chunks_fts fts
            JOIN chunks c ON c.id = fts.rowid
            WHERE chunks_fts MATCH ?
            {where_clause}
            ORDER BY score DESC
            LIMIT ?
        """
        params.append(top_k)
        
        try:
            cursor = conn.execute(sql, params)
            results = [(row["chunk_id"], row["score"]) for row in cursor.fetchall()]
            return results
        except sqlite3.OperationalError as e:
            # FTS5 query syntax error - try simpler query
            print(f"FTS5 query error: {e}, falling back to simple search")
            return self._simple_search(query, top_k, selected_documents)

    def _prepare_query(self, query: str) -> str:
        """Prepare query for FTS5 search.
        
        Handles special characters and converts to FTS5 syntax.
        """
        # Remove characters that break FTS5 syntax
        # Keep: alphanumeric, spaces, quotes, asterisk (prefix), hyphen
        cleaned = re.sub(r'[^\w\s"*\-]', ' ', query)
        
        # Collapse multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if not cleaned:
            return '""'  # Empty query
        
        # If query has no special operators, wrap each word for OR matching
        if '"' not in cleaned and '*' not in cleaned:
            words = cleaned.split()
            if len(words) > 1:
                # Use OR for multiple words (more results)
                cleaned = " OR ".join(words)
        
        return cleaned

    def _simple_search(
        self,
        query: str,
        top_k: int,
        selected_documents: Optional[list[str]] = None,
    ) -> list[tuple[str, float]]:
        """Fallback simple LIKE search when FTS5 fails."""
        conn = self._get_conn()
        
        # Simple LIKE search on chunk_text
        words = query.lower().split()
        if not words:
            return []
        
        # Build LIKE conditions for each word
        like_conditions = " AND ".join(
            "LOWER(chunk_text) LIKE ?" for _ in words
        )
        params = [f"%{word}%" for word in words]
        
        where_parts = [like_conditions]
        
        if selected_documents:
            placeholders = ",".join("?" * len(selected_documents))
            where_parts.append(f"file_name IN ({placeholders})")
            params.extend(selected_documents)
        
        sql = f"""
            SELECT chunk_id, 1.0 as score
            FROM chunks
            WHERE {" AND ".join(where_parts)}
            LIMIT ?
        """
        params.append(top_k)
        
        cursor = conn.execute(sql, params)
        return [(row[0], row[1]) for row in cursor.fetchall()]

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document.
        
        FTS5 index is automatically updated via triggers.

        Args:
            document_id: Document ID to delete
        """
        conn = self._get_conn()
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.commit()

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Delete specific chunks by ID.

        Args:
            chunk_ids: List of chunk IDs to delete
        """
        if not chunk_ids:
            return
        
        conn = self._get_conn()
        placeholders = ",".join("?" * len(chunk_ids))
        conn.execute(f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
        conn.commit()

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """Get a single chunk by ID.

        Args:
            chunk_id: Chunk ID to retrieve

        Returns:
            Chunk data as dictionary, or None if not found
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?",
            (chunk_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_chunks_by_document(self, document_id: str) -> list[dict]:
        """Get all chunks for a document.

        Args:
            document_id: Document ID

        Returns:
            List of chunk dictionaries
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY id",
            (document_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def rebuild_fts(self) -> None:
        """Rebuild FTS5 index from chunks table.
        
        Use this if the FTS index gets out of sync.
        """
        conn = self._get_conn()
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        conn.commit()

    def optimize(self) -> None:
        """Optimize the FTS5 index for better query performance."""
        conn = self._get_conn()
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
        conn.commit()

    def stats(self) -> dict[str, Any]:
        """Get index statistics.

        Returns:
            Dictionary with index statistics
        """
        conn = self._get_conn()
        
        # Total chunks
        cursor = conn.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]
        
        # Unique documents
        cursor = conn.execute("SELECT COUNT(DISTINCT document_id) FROM chunks")
        doc_count = cursor.fetchone()[0]
        
        # Unique files
        cursor = conn.execute("SELECT COUNT(DISTINCT file_name) FROM chunks")
        file_count = cursor.fetchone()[0]
        
        # Document types breakdown
        cursor = conn.execute("""
            SELECT document_type, COUNT(*) as cnt
            FROM chunks
            GROUP BY document_type
        """)
        doc_types = {row[0] or "unknown": row[1] for row in cursor.fetchall()}
        
        return {
            "total_chunks": total_chunks,
            "document_count": doc_count,
            "file_count": file_count,
            "document_types": doc_types,
            "index_path": str(self.db_path),
            "index_type": "FTS5",
        }

    def reset(self) -> None:
        """Reset the index by deleting all data."""
        conn = self._get_conn()
        conn.execute("DELETE FROM chunks")
        conn.commit()

    def close(self) -> None:
        """Close thread-local database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# Compatibility wrapper to match BM25Index interface
class FTS5IndexCompat(FTS5Index):
    """FTS5 index with BM25Index-compatible interface.
    
    Drop-in replacement for BM25Index that uses SQLite FTS5.
    """
    
    @property
    def chunks(self) -> list[Chunk]:
        """Get all chunks (for compatibility)."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM chunks ORDER BY id")
        
        import json
        chunks = []
        for row in cursor.fetchall():
            metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            chunk = Chunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                file_name=row["file_name"],
                text=row["chunk_text"],
                char_start=row["start_char"] or 0,
                char_end=row["end_char"] or 0,
                page_number=row["page"],
                paragraph_number=row["paragraph"],
                document_type=row["document_type"] or "unknown",
                metadata=metadata,
            )
            chunks.append(chunk)
        
        return chunks

    def build_index(self, chunks: list[Chunk]) -> None:
        """Build index from chunks (replaces all existing).
        
        For compatibility with BM25Index interface.
        """
        self.reset()
        self.add_chunks(chunks)

    def save(self) -> None:
        """Save is automatic with SQLite - this is a no-op."""
        # FTS5 auto-saves via SQLite
        pass

    def load(self) -> None:
        """Load is automatic with SQLite - this is a no-op."""
        # Database is opened on demand
        if not self.db_path.exists():
            raise FileNotFoundError(f"FTS5 index not found: {self.db_path}")

