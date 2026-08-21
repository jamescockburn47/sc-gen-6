"""Ingestion checkpoint system for resumable document processing.

Enables crash-resistant ingestion of large document sets (500k+) by:
- Saving progress every N documents
- Tracking phases (parsing, chunking, indexing)
- Enabling resume from crash point
- Recording failed files for retry
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional


@dataclass
class IngestionCheckpoint:
    """Persistent checkpoint for resumable ingestion."""
    
    # Progress tracking
    total_files: int = 0
    processed_files: int = 0
    indexed_files: int = 0
    
    # Current state
    phase: Literal["idle", "parsing", "chunking", "indexing", "complete"] = "idle"
    current_file: str = ""
    
    # Error tracking  
    failed_files: list[str] = field(default_factory=list)
    error_count: int = 0
    
    # Timing
    started_at: str = ""
    updated_at: str = ""
    completed_at: str = ""
    
    # Configuration
    batch_size: int = 100  # Save checkpoint every N files
    
    @property
    def progress_percent(self) -> float:
        """Get progress as percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100

    @property
    def is_complete(self) -> bool:
        """Check if ingestion is complete."""
        return self.phase == "complete"

    @property
    def is_active(self) -> bool:
        """Check if ingestion is in progress."""
        return self.phase in ("parsing", "chunking", "indexing")

    @property
    def can_resume(self) -> bool:
        """Check if there's resumable progress."""
        return self.processed_files > 0 and not self.is_complete

    def start(self, total_files: int) -> None:
        """Start a new ingestion run."""
        self.total_files = total_files
        self.processed_files = 0
        self.indexed_files = 0
        self.phase = "parsing"
        self.current_file = ""
        self.failed_files = []
        self.error_count = 0
        self.started_at = datetime.now().isoformat()
        self.updated_at = self.started_at
        self.completed_at = ""

    def advance_phase(self, phase: Literal["parsing", "chunking", "indexing", "complete"]) -> None:
        """Move to next phase."""
        self.phase = phase
        self.updated_at = datetime.now().isoformat()
        if phase == "complete":
            self.completed_at = self.updated_at

    def record_progress(self, file_path: str, success: bool = True) -> None:
        """Record processing of a file."""
        self.current_file = file_path
        self.processed_files += 1
        self.updated_at = datetime.now().isoformat()
        
        if not success:
            self.failed_files.append(file_path)
            self.error_count += 1

    def record_indexed(self, count: int = 1) -> None:
        """Record indexed files."""
        self.indexed_files += count
        self.updated_at = datetime.now().isoformat()

    def save(self, path: Path) -> None:
        """Atomic save using temp file + rename."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        temp_path = path.with_suffix('.tmp')
        data = asdict(self)
        
        temp_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        
        # Atomic rename (works on same filesystem)
        if path.exists():
            path.unlink()
        temp_path.rename(path)

    @classmethod
    def load(cls, path: Path) -> Optional["IngestionCheckpoint"]:
        """Load checkpoint from file."""
        path = Path(path)
        if not path.exists():
            return None
        
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            return cls(**data)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[Checkpoint] Failed to load: {e}")
            return None

    @classmethod
    def get_path(cls, matter_path: Path) -> Path:
        """Get checkpoint path for a matter."""
        return Path(matter_path) / "ingestion_checkpoint.json"

    def to_status_string(self) -> str:
        """Get human-readable status string."""
        if self.phase == "idle":
            return "Ready to ingest"
        elif self.phase == "complete":
            return f"Complete: {self.processed_files:,} files processed"
        else:
            phase_name = self.phase.title()
            return f"{phase_name}: {self.processed_files:,}/{self.total_files:,} ({self.progress_percent:.1f}%)"


class CheckpointedIngestion:
    """Wrapper for checkpoint-aware ingestion operations."""
    
    def __init__(self, matter_path: Path):
        self.matter_path = Path(matter_path)
        self.checkpoint_path = IngestionCheckpoint.get_path(matter_path)
        self._checkpoint: Optional[IngestionCheckpoint] = None
    
    @property
    def checkpoint(self) -> IngestionCheckpoint:
        """Get or create checkpoint."""
        if self._checkpoint is None:
            self._checkpoint = IngestionCheckpoint.load(self.checkpoint_path)
            if self._checkpoint is None:
                self._checkpoint = IngestionCheckpoint()
        return self._checkpoint
    
    def start_fresh(self, file_paths: list[Path]) -> None:
        """Start a fresh ingestion run."""
        self._checkpoint = IngestionCheckpoint()
        self._checkpoint.start(len(file_paths))
        self._checkpoint.save(self.checkpoint_path)
    
    def resume(self) -> int:
        """Resume from checkpoint, returns number of files already processed."""
        cp = self.checkpoint
        if cp.can_resume:
            print(f"[Checkpoint] Resuming from {cp.processed_files}/{cp.total_files}")
            return cp.processed_files
        return 0
    
    def record_parsed(self, file_path: str, success: bool = True) -> None:
        """Record a parsed file."""
        cp = self.checkpoint
        cp.record_progress(file_path, success)
        
        # Save checkpoint every batch_size files
        if cp.processed_files % cp.batch_size == 0:
            cp.save(self.checkpoint_path)
    
    def start_chunking(self) -> None:
        """Transition to chunking phase."""
        self.checkpoint.advance_phase("chunking")
        self.checkpoint.save(self.checkpoint_path)
    
    def start_indexing(self) -> None:
        """Transition to indexing phase."""
        self.checkpoint.advance_phase("indexing")
        self.checkpoint.save(self.checkpoint_path)
    
    def record_indexed(self, count: int = 1) -> None:
        """Record indexed files."""
        self.checkpoint.record_indexed(count)
        
        # Save checkpoint periodically
        if self.checkpoint.indexed_files % self.checkpoint.batch_size == 0:
            self.checkpoint.save(self.checkpoint_path)
    
    def complete(self) -> None:
        """Mark ingestion as complete."""
        self.checkpoint.advance_phase("complete")
        self.checkpoint.save(self.checkpoint_path)
    
    def reset(self) -> None:
        """Reset checkpoint (start over)."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        self._checkpoint = None
    
    def get_failed_files(self) -> list[str]:
        """Get list of failed files for retry."""
        return self.checkpoint.failed_files
