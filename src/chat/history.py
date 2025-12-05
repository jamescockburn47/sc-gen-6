"""Chat history persistence for SC Gen 6.

Stores query/response history in JSON format for sidebar display and session persistence.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ChatEntry:
    """A single chat history entry."""
    
    id: str
    query: str
    response_preview: str  # First ~100 chars of response
    timestamp: str  # ISO format
    model: str
    chunk_count: int
    duration_ms: int = 0
    
    # Full data stored separately to keep list lightweight
    full_response: str = ""
    sources: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        query: str,
        response: str,
        model: str,
        chunk_count: int,
        duration_ms: int = 0,
        sources: list = None,
        metrics: dict = None,
    ) -> "ChatEntry":
        """Create a new chat entry with auto-generated ID and timestamp."""
        preview = response[:100].replace("\n", " ")
        if len(response) > 100:
            preview += "..."
        
        return cls(
            id=str(uuid.uuid4())[:8],
            query=query,
            response_preview=preview,
            timestamp=datetime.now().isoformat(),
            model=model,
            chunk_count=chunk_count,
            duration_ms=duration_ms,
            full_response=response,
            sources=sources or [],
            metrics=metrics or {},
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ChatEntry":
        """Create from dictionary."""
        return cls(**data)
    
    @property
    def display_time(self) -> str:
        """Get human-readable time for display."""
        try:
            dt = datetime.fromisoformat(self.timestamp)
            now = datetime.now()
            diff = now - dt
            
            if diff.days == 0:
                if diff.seconds < 60:
                    return "Just now"
                elif diff.seconds < 3600:
                    mins = diff.seconds // 60
                    return f"{mins}m ago"
                else:
                    hours = diff.seconds // 3600
                    return f"{hours}h ago"
            elif diff.days == 1:
                return "Yesterday"
            elif diff.days < 7:
                return f"{diff.days}d ago"
            else:
                return dt.strftime("%d %b")
        except Exception:
            return ""
    
    @property
    def query_preview(self) -> str:
        """Get truncated query for display."""
        preview = self.query[:50].replace("\n", " ")
        if len(self.query) > 50:
            preview += "..."
        return preview


class ChatHistory:
    """JSON-backed chat history store."""
    
    DEFAULT_PATH = Path("data/chat_history.json")
    MAX_ENTRIES = 100  # Keep last 100 entries
    
    def __init__(self, path: Optional[Path] = None):
        self.path = path or self.DEFAULT_PATH
        self._entries: list[ChatEntry] = []
        self._load()
    
    def _load(self):
        """Load history from disk."""
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._entries = [ChatEntry.from_dict(e) for e in data]
            except Exception as e:
                print(f"Failed to load chat history: {e}")
                self._entries = []
        else:
            self._entries = []
    
    def _save(self):
        """Save history to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump([e.to_dict() for e in self._entries], f, indent=2)
        except Exception as e:
            print(f"Failed to save chat history: {e}")
    
    def add(self, entry: ChatEntry) -> None:
        """Add a new entry to history."""
        self._entries.insert(0, entry)  # Most recent first
        
        # Trim to max entries
        if len(self._entries) > self.MAX_ENTRIES:
            self._entries = self._entries[:self.MAX_ENTRIES]
        
        self._save()
    
    def get(self, entry_id: str) -> Optional[ChatEntry]:
        """Get entry by ID."""
        for entry in self._entries:
            if entry.id == entry_id:
                return entry
        return None
    
    def get_recent(self, limit: int = 20) -> list[ChatEntry]:
        """Get most recent entries."""
        return self._entries[:limit]
    
    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID."""
        for i, entry in enumerate(self._entries):
            if entry.id == entry_id:
                del self._entries[i]
                self._save()
                return True
        return False
    
    def clear(self) -> None:
        """Clear all history."""
        self._entries = []
        self._save()
    
    def __len__(self) -> int:
        return len(self._entries)
    
    def __iter__(self):
        return iter(self._entries)



