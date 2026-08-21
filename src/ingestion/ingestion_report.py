"""Ingestion report tracking and display.

Tracks success/failure/skipped documents during ingestion.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from typing import Optional


@dataclass
class IngestionResult:
    """Result for a single document ingestion."""
    file_path: str
    file_name: str
    status: str  # "success", "failed", "skipped"
    error: Optional[str] = None
    chunk_count: int = 0
    avg_chunk_chars: int = 0
    duration_ms: float = 0
    document_type: Optional[str] = None


@dataclass
class IngestionReport:
    """Collects and summarizes ingestion results."""
    
    results: list[IngestionResult] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def start(self):
        """Mark ingestion start time."""
        self.started_at = datetime.now()
        self.results = []
    
    def add_result(self, result: IngestionResult):
        """Add a document result."""
        self.results.append(result)
    
    def add_success(
        self, 
        file_path: str, 
        chunk_count: int, 
        avg_chunk_chars: int = 0,
        duration_ms: float = 0,
        document_type: str = None
    ):
        """Record a successful ingestion."""
        self.results.append(IngestionResult(
            file_path=str(file_path),
            file_name=Path(file_path).name,
            status="success",
            chunk_count=chunk_count,
            avg_chunk_chars=avg_chunk_chars,
            duration_ms=duration_ms,
            document_type=document_type,
        ))
    
    def add_failed(self, file_path: str, error: str, duration_ms: float = 0):
        """Record a failed ingestion."""
        self.results.append(IngestionResult(
            file_path=str(file_path),
            file_name=Path(file_path).name,
            status="failed",
            error=error,
            duration_ms=duration_ms,
        ))
    
    def add_skipped(self, file_path: str, reason: str = "Unsupported format"):
        """Record a skipped file."""
        self.results.append(IngestionResult(
            file_path=str(file_path),
            file_name=Path(file_path).name,
            status="skipped",
            error=reason,
        ))
    
    def complete(self):
        """Mark ingestion complete."""
        self.completed_at = datetime.now()
    
    @property
    def total_count(self) -> int:
        return len(self.results)
    
    @property
    def success_count(self) -> int:
        return len([r for r in self.results if r.status == "success"])
    
    @property
    def failed_count(self) -> int:
        return len([r for r in self.results if r.status == "failed"])
    
    @property
    def skipped_count(self) -> int:
        return len([r for r in self.results if r.status == "skipped"])
    
    @property
    def total_chunks(self) -> int:
        return sum(r.chunk_count for r in self.results)
    
    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0
    
    def get_successful(self) -> list[IngestionResult]:
        return [r for r in self.results if r.status == "success"]
    
    def get_failed(self) -> list[IngestionResult]:
        return [r for r in self.results if r.status == "failed"]
    
    def get_skipped(self) -> list[IngestionResult]:
        return [r for r in self.results if r.status == "skipped"]
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        return {
            "total": self.total_count,
            "success": self.success_count,
            "failed": self.failed_count,
            "skipped": self.skipped_count,
            "total_chunks": self.total_chunks,
            "duration_seconds": self.duration_seconds,
            "success_rate": f"{(self.success_count / self.total_count * 100):.1f}%" if self.total_count > 0 else "0%",
        }
    
    def save_to_json(self, path: Path):
        """Save report to JSON file."""
        data = {
            "summary": self.get_summary(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "results": [
                {
                    "file_name": r.file_name,
                    "status": r.status,
                    "error": r.error,
                    "chunk_count": r.chunk_count,
                    "document_type": r.document_type,
                }
                for r in self.results
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
