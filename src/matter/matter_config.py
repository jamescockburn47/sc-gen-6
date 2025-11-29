"""Matter configuration data model."""

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional
import uuid


@dataclass
class MatterConfig:
    """
    Configuration for a single matter/case.
    
    Each matter has isolated document storage, indexes, and case graph.
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""  # "Smith v Acme Ltd"
    reference: str = ""  # "CL-2024-001234"
    client: str = ""  # "John Smith"
    
    # Parties involved
    parties: dict[str, str] = field(default_factory=dict)  # {"claimant": "John Smith", ...}
    
    # Relevant date range for the matter
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    
    # Path information (relative to matters directory)
    base_path: str = ""  # Will be set when creating matter
    
    # Stats (cached)
    document_count: int = 0
    chunk_count: int = 0
    entity_count: int = 0
    
    # User notes
    notes: str = ""
    
    @property
    def documents_path(self) -> str:
        """Path to documents directory."""
        return f"{self.base_path}/documents"
    
    @property
    def chroma_path(self) -> str:
        """Path to Chroma vector database."""
        return f"{self.base_path}/chroma_db"
    
    @property
    def bm25_path(self) -> str:
        """Path to BM25 index."""
        return f"{self.base_path}/bm25_index"
    
    @property
    def graph_path(self) -> str:
        """Path to case graph directory."""
        return f"{self.base_path}/case_graph"
    
    @property
    def exports_path(self) -> str:
        """Path to exports directory."""
        return f"{self.base_path}/exports"
    
    @property
    def display_name(self) -> str:
        """Display name for UI."""
        if self.name:
            return self.name
        if self.reference:
            return self.reference
        return f"Matter {self.id}"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "reference": self.reference,
            "client": self.client,
            "parties": self.parties,
            "date_range_start": self.date_range_start.isoformat() if self.date_range_start else None,
            "date_range_end": self.date_range_end.isoformat() if self.date_range_end else None,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "base_path": self.base_path,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "entity_count": self.entity_count,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatterConfig":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            reference=data.get("reference", ""),
            client=data.get("client", ""),
            parties=data.get("parties", {}),
            date_range_start=date.fromisoformat(data["date_range_start"]) if data.get("date_range_start") else None,
            date_range_end=date.fromisoformat(data["date_range_end"]) if data.get("date_range_end") else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            base_path=data.get("base_path", ""),
            document_count=data.get("document_count", 0),
            chunk_count=data.get("chunk_count", 0),
            entity_count=data.get("entity_count", 0),
            notes=data.get("notes", ""),
        )
    
    def save(self, path: Path) -> None:
        """Save config to file.
        
        Args:
            path: Path to matter.json file
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "MatterConfig":
        """Load config from file.
        
        Args:
            path: Path to matter.json file
            
        Returns:
            MatterConfig instance
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


