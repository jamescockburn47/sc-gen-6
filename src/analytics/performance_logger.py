"""Async performance logger with zero overhead on query execution."""

import sqlite3
import threading
import queue
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib
from typing import Optional


@dataclass
class LLMMetrics:
    """Metrics from a single LLM request/response."""
    timestamp: datetime
    model: str
    provider: str
    query_type: Optional[str] = None
    system_prompt_hash: Optional[str] = None
    
    # Tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    # Timing (ms)
    prompt_ms: float = 0.0
    completion_ms: float = 0.0
    tokens_per_second: float = 0.0
    
    # Context
    context_chunks: int = 0
    context_chars: int = 0
    query_chars: int = 0
    response_chars: int = 0
    
    error: Optional[str] = None


class AsyncPerformanceLogger:
    """Async logger with zero overhead on query execution."""
    
    def __init__(self, db_path: Path = Path("data/performance.db")):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Queue for async logging (max 1000 pending)
        self.log_queue: queue.Queue = queue.Queue(maxsize=1000)
        
        # Background worker thread
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def log_request(self, metrics: LLMMetrics):
        """Queue metrics for async logging (returns immediately - 0ms overhead)."""
        try:
            # Non-blocking put - returns instantly
            self.log_queue.put_nowait(("log", metrics))
        except queue.Full:
            # If queue full, drop oldest to prevent memory issues
            # This should rarely happen (1000 pending writes = something wrong)
            pass
    
    def shutdown(self):
        """Graceful shutdown (for testing)."""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
    
    def _worker(self):
        """Background worker that writes to database."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            
            # Initialize schema on first run
            self._init_db(conn)
            
            while self.running:
                try:
                    # Wait up to 1 second for next item
                    action, data = self.log_queue.get(timeout=1.0)
                    
                    if action == "log":
                        self._write_metric(conn, data)
                        conn.commit()
                        
                except queue.Empty:
                    # Timeout - just continue loop
                    continue
                except Exception as e:
                    print(f"[PERF LOG ERROR] {e}")
        finally:
            if conn:
                conn.close()
    
    def _init_db(self, conn):
        """Initialize database schema."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                model TEXT NOT NULL,
                provider TEXT NOT NULL,
                query_type TEXT,
                system_prompt_hash TEXT,
                
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                
                prompt_ms REAL,
                completion_ms REAL,
                tokens_per_second REAL,
                
                context_chunks INTEGER,
                context_chars INTEGER,
                query_chars INTEGER,
                response_chars INTEGER,
                
                error TEXT
            )
        """)
        
        # Indexes for fast queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON llm_performance(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON llm_performance(model)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_query_type ON llm_performance(query_type)")
        
        # Table for auto-generated insights
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                insight_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        conn.commit()
    
    def _write_metric(self, conn, metrics: LLMMetrics):
        """Write single metric to database."""
        conn.execute("""
            INSERT INTO llm_performance (
                timestamp, model, provider, query_type, system_prompt_hash,
                prompt_tokens, completion_tokens,
                prompt_ms, completion_ms, tokens_per_second,
                context_chunks, context_chars, query_chars, response_chars,
                error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.timestamp, metrics.model, metrics.provider, 
            metrics.query_type, metrics.system_prompt_hash,
            metrics.prompt_tokens, metrics.completion_tokens,
            metrics.prompt_ms, metrics.completion_ms, metrics.tokens_per_second,
            metrics.context_chunks, metrics.context_chars, 
            metrics.query_chars, metrics.response_chars,
            metrics.error
        ))
