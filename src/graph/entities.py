"""Entity and relationship data models for case graph."""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional
import uuid


class EntityType(str, Enum):
    """Types of entities that can be extracted."""
    PERSON = "person"
    ORGANIZATION = "organization"
    DATE = "date"
    AMOUNT = "amount"
    LOCATION = "location"
    CASE_REF = "case_ref"
    DOCUMENT_REF = "document_ref"
    EVENT = "event"
    EMAIL_ADDRESS = "email_address"
    PHONE = "phone"


class RelationshipType(str, Enum):
    """Types of relationships between entities."""
    # Communication
    SENT_TO = "sent_to"
    COPIED_ON = "copied_on"
    REPLIED_TO = "replied_to"
    
    # Employment/Affiliation
    EMPLOYED_BY = "employed_by"
    DIRECTOR_OF = "director_of"
    REPRESENTS = "represents"
    
    # Legal
    PARTY_TO = "party_to"
    WITNESS_FOR = "witness_for"
    EXPERT_FOR = "expert_for"
    
    # Events
    ATTENDED = "attended"
    SIGNED = "signed"
    AUTHORED = "authored"
    
    # General
    MENTIONED_WITH = "mentioned_with"
    RELATED_TO = "related_to"


class EventType(str, Enum):
    """Types of timeline events."""
    MEETING = "meeting"
    COMMUNICATION = "communication"
    TRANSACTION = "transaction"
    FILING = "filing"
    HEARING = "hearing"
    DEADLINE = "deadline"
    CONTRACT = "contract"
    INCIDENT = "incident"
    OTHER = "other"


@dataclass
class Entity:
    """An entity extracted from documents."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EntityType = EntityType.PERSON
    canonical_name: str = ""
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_chunks: list[str] = field(default_factory=list)
    confidence: float = 1.0
    user_verified: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_alias(self, alias: str) -> None:
        """Add an alias if not already present."""
        normalized = alias.strip()
        if normalized and normalized.lower() not in [a.lower() for a in self.aliases]:
            self.aliases.append(normalized)
            self.updated_at = datetime.now()
    
    def add_chunk(self, chunk_id: str) -> None:
        """Link a chunk to this entity."""
        if chunk_id not in self.source_chunks:
            self.source_chunks.append(chunk_id)
            self.updated_at = datetime.now()
    
    def matches(self, text: str) -> bool:
        """Check if text matches this entity's name or aliases."""
        text_lower = text.lower().strip()
        if self.canonical_name.lower() == text_lower:
            return True
        return any(alias.lower() == text_lower for alias in self.aliases)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "canonical_name": self.canonical_name,
            "aliases": self.aliases,
            "metadata": self.metadata,
            "source_chunks": self.source_chunks,
            "confidence": self.confidence,
            "user_verified": self.user_verified,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            type=EntityType(data["type"]),
            canonical_name=data["canonical_name"],
            aliases=data.get("aliases", []),
            metadata=data.get("metadata", {}),
            source_chunks=data.get("source_chunks", []),
            confidence=data.get("confidence", 1.0),
            user_verified=data.get("user_verified", False),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


@dataclass
class Relationship:
    """A relationship between two entities."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_entity_id: str = ""
    target_entity_id: str = ""
    relationship_type: RelationshipType = RelationshipType.RELATED_TO
    properties: dict[str, Any] = field(default_factory=dict)
    source_chunks: list[str] = field(default_factory=list)
    confidence: float = 1.0
    user_verified: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "relationship_type": self.relationship_type.value,
            "properties": self.properties,
            "source_chunks": self.source_chunks,
            "confidence": self.confidence,
            "user_verified": self.user_verified,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Relationship":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            source_entity_id=data["source_entity_id"],
            target_entity_id=data["target_entity_id"],
            relationship_type=RelationshipType(data["relationship_type"]),
            properties=data.get("properties", {}),
            source_chunks=data.get("source_chunks", []),
            confidence=data.get("confidence", 1.0),
            user_verified=data.get("user_verified", False),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class TimelineEvent:
    """An event on the case timeline."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    date: Optional[date] = None
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None
    date_text: str = ""  # Original text: "mid-January 2024"
    description: str = ""
    entities_involved: list[str] = field(default_factory=list)  # Entity IDs
    source_chunks: list[str] = field(default_factory=list)
    source_documents: list[str] = field(default_factory=list)  # Document file names
    event_type: EventType = EventType.OTHER
    user_verified: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def sort_date(self) -> date:
        """Get date for sorting (uses exact date, then range start, then today)."""
        if self.date:
            return self.date
        if self.date_range_start:
            return self.date_range_start
        return date.today()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "date": self.date.isoformat() if self.date else None,
            "date_range_start": self.date_range_start.isoformat() if self.date_range_start else None,
            "date_range_end": self.date_range_end.isoformat() if self.date_range_end else None,
            "date_text": self.date_text,
            "description": self.description,
            "entities_involved": self.entities_involved,
            "source_chunks": self.source_chunks,
            "source_documents": self.source_documents,
            "event_type": self.event_type.value,
            "user_verified": self.user_verified,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TimelineEvent":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            date=date.fromisoformat(data["date"]) if data.get("date") else None,
            date_range_start=date.fromisoformat(data["date_range_start"]) if data.get("date_range_start") else None,
            date_range_end=date.fromisoformat(data["date_range_end"]) if data.get("date_range_end") else None,
            date_text=data.get("date_text", ""),
            description=data.get("description", ""),
            entities_involved=data.get("entities_involved", []),
            source_chunks=data.get("source_chunks", []),
            source_documents=data.get("source_documents", []),
            event_type=EventType(data.get("event_type", "other")),
            user_verified=data.get("user_verified", False),
            created_at=datetime.fromisoformat(data["created_at"]),
        )



