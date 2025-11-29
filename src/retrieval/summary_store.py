"""Document Summary Store with FTS5 indexing.

Stores and retrieves document summaries at multiple levels:
- Document-level: Overview of entire document
- Section-level: Key points from sections
- Chunk-level: Brief summary of individual chunks

Summaries are indexed with FTS5 for semantic search enhancement.
Inspired by Hyperlink's document_summaries table approach.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Literal
from dataclasses import dataclass, field, asdict

from src.config_loader import Settings, get_settings


SummaryType = Literal["overview", "key_points", "entities", "timeline", "custom"]
SummaryLevel = Literal["document", "section", "chunk"]


@dataclass
class DocumentSummary:
    """A summary of a document or part of a document.
    
    Attributes:
        summary_id: Unique identifier
        document_id: Source document identifier
        file_name: Source file name
        summary_type: Type of summary (overview, key_points, etc.)
        summary_level: Level of summary (document, section, chunk)
        content: The summary text
        chunk_id: Associated chunk ID (for chunk-level summaries)
        section_header: Section title (for section-level summaries)
        metadata: Additional metadata (model used, tokens, etc.)
        created_at: Timestamp of creation
    """
    summary_id: str
    document_id: str
    file_name: str
    summary_type: SummaryType
    summary_level: SummaryLevel
    content: str
    chunk_id: Optional[str] = None
    section_header: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)


class SummaryStore:
    """SQLite-based storage for document summaries with FTS5 indexing.
    
    Uses same FTS5 approach as the main keyword index for consistency.
    Summaries can be included in search results to improve retrieval.
    
    Thread-safe: Creates a new connection per thread.
    """
    
    DB_FILE = "summaries.db"
    
    def __init__(
        self,
        store_path: Optional[str | Path] = None,
        settings: Optional[Settings] = None,
    ):
        """Initialize summary store.
        
        Args:
            store_path: Path to store directory. If None, uses data/summaries.
            settings: Settings instance. If None, loads from config.
        """
        self.settings = settings or get_settings()
        
        if store_path:
            self.store_path = Path(store_path)
        else:
            self.store_path = Path(self.settings.paths.vector_db).parent / "summaries"
        
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.store_path / self.DB_FILE
        
        # Thread-local storage for connections
        import threading
        self._local = threading.local()
        
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get or create thread-local database connection."""
        # Each thread gets its own connection
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn
    
    def _init_db(self) -> None:
        """Initialize database schema with FTS5 virtual table."""
        conn = self._get_conn()
        
        # Create summaries table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_id TEXT UNIQUE NOT NULL,
                document_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                summary_type TEXT NOT NULL,
                summary_level TEXT NOT NULL,
                content TEXT NOT NULL,
                chunk_id TEXT,
                section_header TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_summaries_document_id ON document_summaries(document_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_summaries_file_name ON document_summaries(file_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_summaries_type ON document_summaries(summary_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_summaries_level ON document_summaries(summary_level)")
        
        # Create FTS5 virtual table for summary search
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS document_summaries_fts USING fts5(
                summary_id UNINDEXED,
                document_id UNINDEXED,
                file_name,
                summary_type,
                summary_level UNINDEXED,
                content,
                section_header,
                content='document_summaries',
                content_rowid='id',
                tokenize='unicode61'
            )
        """)
        
        # Auto-sync triggers for FTS5
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS summaries_ai
            AFTER INSERT ON document_summaries
            BEGIN
                INSERT INTO document_summaries_fts(
                    rowid, summary_id, document_id, file_name, 
                    summary_type, summary_level, content, section_header
                ) VALUES (
                    new.id, new.summary_id, new.document_id, new.file_name,
                    new.summary_type, new.summary_level, new.content, new.section_header
                );
            END
        """)
        
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS summaries_ad
            AFTER DELETE ON document_summaries
            BEGIN
                INSERT INTO document_summaries_fts(
                    document_summaries_fts, rowid, summary_id, document_id, 
                    file_name, summary_type, summary_level, content, section_header
                ) VALUES (
                    'delete', old.id, old.summary_id, old.document_id, 
                    old.file_name, old.summary_type, old.summary_level, 
                    old.content, old.section_header
                );
            END
        """)
        
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS summaries_au
            AFTER UPDATE ON document_summaries
            BEGIN
                INSERT INTO document_summaries_fts(
                    document_summaries_fts, rowid, summary_id, document_id, 
                    file_name, summary_type, summary_level, content, section_header
                ) VALUES (
                    'delete', old.id, old.summary_id, old.document_id, 
                    old.file_name, old.summary_type, old.summary_level, 
                    old.content, old.section_header
                );
                INSERT INTO document_summaries_fts(
                    rowid, summary_id, document_id, file_name, 
                    summary_type, summary_level, content, section_header
                ) VALUES (
                    new.id, new.summary_id, new.document_id, new.file_name,
                    new.summary_type, new.summary_level, new.content, new.section_header
                );
            END
        """)
        
        conn.commit()
    
    def add_summary(self, summary: DocumentSummary) -> None:
        """Add or update a document summary.
        
        Args:
            summary: DocumentSummary to store
        """
        conn = self._get_conn()
        metadata_json = json.dumps(summary.metadata)
        
        conn.execute("""
            INSERT OR REPLACE INTO document_summaries (
                summary_id, document_id, file_name, summary_type, summary_level,
                content, chunk_id, section_header, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary.summary_id,
            summary.document_id,
            summary.file_name,
            summary.summary_type,
            summary.summary_level,
            summary.content,
            summary.chunk_id,
            summary.section_header,
            metadata_json,
            summary.created_at,
        ))
        conn.commit()
    
    def add_summaries(self, summaries: list[DocumentSummary]) -> None:
        """Add multiple summaries in a batch.
        
        Args:
            summaries: List of summaries to store
        """
        if not summaries:
            return
        
        conn = self._get_conn()
        
        for summary in summaries:
            metadata_json = json.dumps(summary.metadata)
            conn.execute("""
                INSERT OR REPLACE INTO document_summaries (
                    summary_id, document_id, file_name, summary_type, summary_level,
                    content, chunk_id, section_header, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                summary.summary_id,
                summary.document_id,
                summary.file_name,
                summary.summary_type,
                summary.summary_level,
                summary.content,
                summary.chunk_id,
                summary.section_header,
                metadata_json,
                summary.created_at,
            ))
        
        conn.commit()
    
    def get_summary(self, summary_id: str) -> Optional[DocumentSummary]:
        """Get a summary by ID.
        
        Args:
            summary_id: Summary ID
            
        Returns:
            DocumentSummary or None if not found
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM document_summaries WHERE summary_id = ?",
            (summary_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return self._row_to_summary(row)
    
    def get_document_summaries(
        self,
        document_id: str,
        summary_type: Optional[SummaryType] = None,
        summary_level: Optional[SummaryLevel] = None,
    ) -> list[DocumentSummary]:
        """Get all summaries for a document.
        
        Args:
            document_id: Document ID
            summary_type: Optional filter by type
            summary_level: Optional filter by level
            
        Returns:
            List of summaries
        """
        conn = self._get_conn()
        
        query = "SELECT * FROM document_summaries WHERE document_id = ?"
        params = [document_id]
        
        if summary_type:
            query += " AND summary_type = ?"
            params.append(summary_type)
        
        if summary_level:
            query += " AND summary_level = ?"
            params.append(summary_level)
        
        cursor = conn.execute(query, params)
        return [self._row_to_summary(row) for row in cursor.fetchall()]
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        summary_type: Optional[SummaryType] = None,
        summary_level: Optional[SummaryLevel] = None,
    ) -> list[tuple[DocumentSummary, float]]:
        """Search summaries using FTS5.
        
        Args:
            query: Search query
            top_k: Maximum results to return
            summary_type: Optional filter by type
            summary_level: Optional filter by level
            
        Returns:
            List of (summary, score) tuples
        """
        conn = self._get_conn()
        
        # Prepare query for FTS5
        import re
        safe_query = re.sub(r'[^\w\s"*\-]', ' ', query)
        safe_query = re.sub(r'\s+', ' ', safe_query).strip()
        
        if not safe_query:
            return []
        
        # Build query with filters
        where_parts = []
        params = [safe_query]
        
        if summary_type:
            where_parts.append("s.summary_type = ?")
            params.append(summary_type)
        
        if summary_level:
            where_parts.append("s.summary_level = ?")
            params.append(summary_level)
        
        where_clause = ""
        if where_parts:
            where_clause = "AND " + " AND ".join(where_parts)
        
        sql = f"""
            SELECT s.*, -bm25(document_summaries_fts) as score
            FROM document_summaries_fts fts
            JOIN document_summaries s ON s.id = fts.rowid
            WHERE document_summaries_fts MATCH ?
            {where_clause}
            ORDER BY score DESC
            LIMIT ?
        """
        params.append(top_k)
        
        try:
            cursor = conn.execute(sql, params)
            results = []
            for row in cursor.fetchall():
                summary = self._row_to_summary(row)
                score = row["score"]
                results.append((summary, score))
            return results
        except sqlite3.OperationalError:
            return []
    
    def delete_document_summaries(self, document_id: str) -> int:
        """Delete all summaries for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Number of summaries deleted
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM document_summaries WHERE document_id = ?",
            (document_id,)
        )
        conn.commit()
        return cursor.rowcount
    
    def stats(self) -> dict[str, Any]:
        """Get summary store statistics.
        
        Returns:
            Dictionary with statistics
        """
        conn = self._get_conn()
        
        cursor = conn.execute("SELECT COUNT(*) FROM document_summaries")
        total = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(DISTINCT document_id) FROM document_summaries")
        doc_count = cursor.fetchone()[0]
        
        cursor = conn.execute("""
            SELECT summary_type, COUNT(*) as cnt
            FROM document_summaries
            GROUP BY summary_type
        """)
        by_type = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor = conn.execute("""
            SELECT summary_level, COUNT(*) as cnt
            FROM document_summaries
            GROUP BY summary_level
        """)
        by_level = {row[0]: row[1] for row in cursor.fetchall()}
        
        return {
            "total_summaries": total,
            "documents_with_summaries": doc_count,
            "by_type": by_type,
            "by_level": by_level,
            "index_path": str(self.db_path),
        }
    
    def _row_to_summary(self, row: sqlite3.Row) -> DocumentSummary:
        """Convert database row to DocumentSummary."""
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        return DocumentSummary(
            summary_id=row["summary_id"],
            document_id=row["document_id"],
            file_name=row["file_name"],
            summary_type=row["summary_type"],
            summary_level=row["summary_level"],
            content=row["content"],
            chunk_id=row["chunk_id"],
            section_header=row["section_header"],
            metadata=metadata,
            created_at=row["created_at"],
        )
    
    def close(self) -> None:
        """Close thread-local database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
    
    def __del__(self):
        self.close()


def generate_document_summary(
    document_id: str,
    document_text: str,
    file_name: str,
    llm_service: Any,
    summary_type: SummaryType = "overview",
    max_length: int = 500,
) -> DocumentSummary:
    """Generate a document summary using the LLM.
    
    Args:
        document_id: Document identifier
        document_text: Full document text (will be truncated if too long)
        file_name: Source file name
        llm_service: LLM service instance
        summary_type: Type of summary to generate
        max_length: Target summary length in words
        
    Returns:
        Generated DocumentSummary
    """
    import hashlib
    
    # Truncate text if too long (approximately 8000 tokens = 32000 chars)
    max_chars = 32000
    text_to_summarize = document_text[:max_chars]
    if len(document_text) > max_chars:
        text_to_summarize += "\n\n[Document truncated for summarization...]"
    
    # Build prompt based on summary type
    if summary_type == "overview":
        prompt = f"""Provide a concise overview of the following document in approximately {max_length} words.
Focus on: main purpose, key parties involved, important dates, and critical facts.

Document: {file_name}

---
{text_to_summarize}
---

Summary:"""
    elif summary_type == "key_points":
        prompt = f"""Extract the key points from the following document as a bulleted list.
Include: main arguments, important facts, dates, amounts, and conclusions.

Document: {file_name}

---
{text_to_summarize}
---

Key Points:"""
    elif summary_type == "entities":
        prompt = f"""Extract all named entities from the following document.
Categories: People, Organizations, Dates, Amounts, Locations, Case References.

Document: {file_name}

---
{text_to_summarize}
---

Entities:"""
    elif summary_type == "timeline":
        prompt = f"""Create a chronological timeline of events from the following document.
Format: Date - Event Description

Document: {file_name}

---
{text_to_summarize}
---

Timeline:"""
    else:
        prompt = f"""Summarize the following document in approximately {max_length} words.

Document: {file_name}

---
{text_to_summarize}
---

Summary:"""
    
    # Generate summary
    content = llm_service.generate(
        prompt=prompt,
        system_prompt="You are a legal document summarization assistant. Be accurate and concise.",
        temperature=0.3,
        max_tokens=max_length * 2,  # Approximate tokens from words
    )
    
    # Create summary object
    summary_id = hashlib.sha256(f"{document_id}:{summary_type}:document".encode()).hexdigest()[:16]
    
    return DocumentSummary(
        summary_id=summary_id,
        document_id=document_id,
        file_name=file_name,
        summary_type=summary_type,
        summary_level="document",
        content=content.strip(),
        metadata={
            "model": llm_service.get_default_model(),
            "generated_at": datetime.now().isoformat(),
            "input_length": len(document_text),
        },
    )

