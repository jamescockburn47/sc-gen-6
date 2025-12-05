"""Case graph storage and operations."""

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional
from difflib import SequenceMatcher

from src.graph.entities import (
    Entity,
    EntityType,
    Relationship,
    RelationshipType,
    TimelineEvent,
    EventType,
)

logger = logging.getLogger(__name__)


class CaseGraph:
    """
    Case knowledge graph for entities, relationships, and timeline.
    
    Stores extracted entities from documents with:
    - Entity definitions with aliases
    - Relationships between entities
    - Timeline events
    - Chunk-to-entity mappings
    - User corrections overlay
    """
    
    def __init__(self, path: Path | str):
        """Initialize case graph.
        
        Args:
            path: Directory path for graph storage
        """
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage
        self.entities: dict[str, Entity] = {}
        self.relationships: dict[str, Relationship] = {}
        self.events: dict[str, TimelineEvent] = {}
        
        # Index: chunk_id -> list of entity_ids
        self.chunk_to_entities: dict[str, list[str]] = {}
        
        # User edits tracking
        self.user_edits: list[dict] = []
        
        # Load existing data
        self.load()
    
    # ========== Entity Operations ==========
    
    def add_entity(self, entity: Entity) -> str:
        """Add or update an entity.
        
        Args:
            entity: Entity to add
            
        Returns:
            Entity ID
        """
        self.entities[entity.id] = entity
        
        # Update chunk index
        for chunk_id in entity.source_chunks:
            if chunk_id not in self.chunk_to_entities:
                self.chunk_to_entities[chunk_id] = []
            if entity.id not in self.chunk_to_entities[chunk_id]:
                self.chunk_to_entities[chunk_id].append(entity.id)
        
        return entity.id
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)
    
    def find_entities(
        self,
        query: str,
        entity_type: Optional[EntityType] = None,
        min_confidence: float = 0.0,
    ) -> list[Entity]:
        """Find entities matching a query.
        
        Args:
            query: Search text (matches name or aliases)
            entity_type: Optional type filter
            min_confidence: Minimum confidence score
            
        Returns:
            List of matching entities
        """
        query_lower = query.lower().strip()
        results = []
        
        for entity in self.entities.values():
            if entity.confidence < min_confidence:
                continue
            if entity_type and entity.type != entity_type:
                continue
            
            # Check canonical name
            if query_lower in entity.canonical_name.lower():
                results.append(entity)
                continue
            
            # Check aliases
            if any(query_lower in alias.lower() for alias in entity.aliases):
                results.append(entity)
        
        return results
    
    def find_similar_entity(
        self,
        name: str,
        entity_type: EntityType,
        threshold: float = 0.85,
    ) -> Optional[tuple[Entity, float]]:
        """Find existing entity similar to given name.
        
        Uses fuzzy matching for entity resolution.
        
        Args:
            name: Name to match
            entity_type: Type of entity
            threshold: Minimum similarity score (0-1)
            
        Returns:
            Tuple of (entity, similarity) if found, None otherwise
        """
        name_lower = name.lower().strip()
        best_match = None
        best_score = 0.0
        
        for entity in self.entities.values():
            if entity.type != entity_type:
                continue
            
            # Check canonical name
            score = SequenceMatcher(None, name_lower, entity.canonical_name.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = entity
            
            # Check aliases
            for alias in entity.aliases:
                score = SequenceMatcher(None, name_lower, alias.lower()).ratio()
                if score > best_score:
                    best_score = score
                    best_match = entity
        
        if best_match and best_score >= threshold:
            return (best_match, best_score)
        return None
    
    def merge_entities(self, keep_id: str, merge_ids: list[str]) -> Entity:
        """Merge multiple entities into one.
        
        Args:
            keep_id: ID of entity to keep
            merge_ids: IDs of entities to merge into keep_id
            
        Returns:
            Merged entity
        """
        keep_entity = self.entities[keep_id]
        
        for merge_id in merge_ids:
            if merge_id == keep_id:
                continue
            merge_entity = self.entities.get(merge_id)
            if not merge_entity:
                continue
            
            # Add aliases
            keep_entity.add_alias(merge_entity.canonical_name)
            for alias in merge_entity.aliases:
                keep_entity.add_alias(alias)
            
            # Add chunks
            for chunk_id in merge_entity.source_chunks:
                keep_entity.add_chunk(chunk_id)
                # Update chunk index
                if chunk_id in self.chunk_to_entities:
                    if merge_id in self.chunk_to_entities[chunk_id]:
                        self.chunk_to_entities[chunk_id].remove(merge_id)
                    if keep_id not in self.chunk_to_entities[chunk_id]:
                        self.chunk_to_entities[chunk_id].append(keep_id)
            
            # Merge metadata
            keep_entity.metadata.update(merge_entity.metadata)
            
            # Update relationships
            for rel in self.relationships.values():
                if rel.source_entity_id == merge_id:
                    rel.source_entity_id = keep_id
                if rel.target_entity_id == merge_id:
                    rel.target_entity_id = keep_id
            
            # Update events
            for event in self.events.values():
                if merge_id in event.entities_involved:
                    event.entities_involved.remove(merge_id)
                    if keep_id not in event.entities_involved:
                        event.entities_involved.append(keep_id)
            
            # Remove merged entity
            del self.entities[merge_id]
        
        # Track edit
        self.user_edits.append({
            "action": "merge",
            "keep_id": keep_id,
            "merge_ids": merge_ids,
            "timestamp": datetime.now().isoformat(),
        })
        
        return keep_entity
    
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity.
        
        Args:
            entity_id: ID of entity to delete
            
        Returns:
            True if deleted, False if not found
        """
        if entity_id not in self.entities:
            return False
        
        # Remove from chunk index
        for chunk_ids in self.chunk_to_entities.values():
            if entity_id in chunk_ids:
                chunk_ids.remove(entity_id)
        
        # Remove related relationships
        self.relationships = {
            k: v for k, v in self.relationships.items()
            if v.source_entity_id != entity_id and v.target_entity_id != entity_id
        }
        
        # Remove from events
        for event in self.events.values():
            if entity_id in event.entities_involved:
                event.entities_involved.remove(entity_id)
        
        del self.entities[entity_id]
        return True
    
    def get_entities_in_chunks(self, chunk_ids: list[str]) -> list[Entity]:
        """Get all entities mentioned in given chunks.
        
        Args:
            chunk_ids: List of chunk IDs
            
        Returns:
            List of entities (deduplicated)
        """
        entity_ids = set()
        for chunk_id in chunk_ids:
            if chunk_id in self.chunk_to_entities:
                entity_ids.update(self.chunk_to_entities[chunk_id])
        
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]
    
    def get_all_entities(
        self,
        entity_type: Optional[EntityType] = None,
        verified_only: bool = False,
    ) -> list[Entity]:
        """Get all entities, optionally filtered.
        
        Args:
            entity_type: Optional type filter
            verified_only: Only return user-verified entities
            
        Returns:
            List of entities
        """
        results = []
        for entity in self.entities.values():
            if entity_type and entity.type != entity_type:
                continue
            if verified_only and not entity.user_verified:
                continue
            results.append(entity)
        
        return sorted(results, key=lambda e: e.canonical_name.lower())
    
    # ========== Relationship Operations ==========
    
    def add_relationship(self, relationship: Relationship) -> str:
        """Add a relationship.
        
        Args:
            relationship: Relationship to add
            
        Returns:
            Relationship ID
        """
        self.relationships[relationship.id] = relationship
        return relationship.id
    
    def get_relationships(
        self,
        entity_id: Optional[str] = None,
        relationship_type: Optional[RelationshipType] = None,
    ) -> list[Relationship]:
        """Get relationships, optionally filtered.
        
        Args:
            entity_id: Filter by entity involvement
            relationship_type: Filter by type
            
        Returns:
            List of relationships
        """
        results = []
        for rel in self.relationships.values():
            if relationship_type and rel.relationship_type != relationship_type:
                continue
            if entity_id:
                if rel.source_entity_id != entity_id and rel.target_entity_id != entity_id:
                    continue
            results.append(rel)
        return results
    
    # ========== Timeline Operations ==========
    
    def add_event(self, event: TimelineEvent) -> str:
        """Add a timeline event.
        
        Args:
            event: Event to add
            
        Returns:
            Event ID
        """
        self.events[event.id] = event
        return event.id
    
    def get_timeline(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        event_type: Optional[EventType] = None,
        entity_id: Optional[str] = None,
    ) -> list[TimelineEvent]:
        """Get timeline events, optionally filtered.
        
        Args:
            start_date: Filter events after this date
            end_date: Filter events before this date
            event_type: Filter by event type
            entity_id: Filter by entity involvement
            
        Returns:
            List of events, sorted chronologically
        """
        results = []
        for event in self.events.values():
            if event_type and event.event_type != event_type:
                continue
            if entity_id and entity_id not in event.entities_involved:
                continue
            
            # Date filtering
            event_date = event.sort_date
            if start_date and event_date < start_date:
                continue
            if end_date and event_date > end_date:
                continue
            
            results.append(event)
        
        return sorted(results, key=lambda e: e.sort_date)
    
    def get_events_for_chunk(self, chunk_id: str) -> list[TimelineEvent]:
        """Get events linked to a specific chunk."""
        return [e for e in self.events.values() if chunk_id in e.source_chunks]
    
    # ========== Search Enhancement ==========
    
    def expand_query(self, query: str) -> list[str]:
        """Expand query with entity aliases.
        
        Args:
            query: Original query text
            
        Returns:
            List of expanded query terms
        """
        terms = [query]
        
        # Find entities mentioned in query
        for entity in self.find_entities(query):
            terms.append(entity.canonical_name)
            terms.extend(entity.aliases)
        
        # Deduplicate while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
        
        return unique_terms
    
    def get_graph_context(
        self,
        chunk_ids: list[str],
        max_entities: int = 10,
        max_events: int = 5,
    ) -> str:
        """Generate concise graph context for LLM.
        
        Args:
            chunk_ids: Chunk IDs from retrieval
            max_entities: Maximum entities to include
            max_events: Maximum events to include
            
        Returns:
            Formatted context string (compact, ~500 tokens max)
        """
        lines = []
        
        # Get entities from chunks
        entities = self.get_entities_in_chunks(chunk_ids)
        if entities:
            lines.append("## Key Entities:")
            for entity in entities[:max_entities]:
                aliases = ", ".join(entity.aliases[:3])
                role = entity.metadata.get("role", "")
                if aliases:
                    lines.append(f"- **{entity.canonical_name}** ({entity.type.value}): also known as {aliases}" + (f" - {role}" if role else ""))
                else:
                    lines.append(f"- **{entity.canonical_name}** ({entity.type.value})" + (f" - {role}" if role else ""))
        
        # Get related events
        entity_ids = [e.id for e in entities[:max_entities]]
        events = []
        for eid in entity_ids:
            events.extend(self.get_timeline(entity_id=eid))
        
        # Deduplicate and sort
        seen_events = set()
        unique_events = []
        for event in sorted(events, key=lambda e: e.sort_date):
            if event.id not in seen_events:
                seen_events.add(event.id)
                unique_events.append(event)
        
        if unique_events:
            lines.append("\n## Timeline Context:")
            for event in unique_events[:max_events]:
                date_str = event.date_text or (event.date.strftime("%d %b %Y") if event.date else "Unknown date")
                lines.append(f"- {date_str}: {event.description}")
        
        return "\n".join(lines)
    
    # ========== Persistence ==========
    
    def save(self) -> None:
        """Save graph to disk."""
        # Entities
        entities_path = self.path / "entities.json"
        with open(entities_path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self.entities.values()], f, indent=2)
        
        # Relationships
        relationships_path = self.path / "relationships.json"
        with open(relationships_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.relationships.values()], f, indent=2)
        
        # Events
        events_path = self.path / "timeline.json"
        with open(events_path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self.events.values()], f, indent=2)
        
        # Chunk index
        index_path = self.path / "chunk_links.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(self.chunk_to_entities, f, indent=2)
        
        # User edits
        edits_path = self.path / "user_edits.json"
        with open(edits_path, "w", encoding="utf-8") as f:
            json.dump(self.user_edits, f, indent=2)
        
        logger.info(f"Saved case graph: {len(self.entities)} entities, {len(self.relationships)} relationships, {len(self.events)} events")
    
    def load(self) -> None:
        """Load graph from disk."""
        # Entities
        entities_path = self.path / "entities.json"
        if entities_path.exists():
            with open(entities_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.entities = {e["id"]: Entity.from_dict(e) for e in data}
        
        # Relationships
        relationships_path = self.path / "relationships.json"
        if relationships_path.exists():
            with open(relationships_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.relationships = {r["id"]: Relationship.from_dict(r) for r in data}
        
        # Events
        events_path = self.path / "timeline.json"
        if events_path.exists():
            with open(events_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.events = {e["id"]: TimelineEvent.from_dict(e) for e in data}
        
        # Chunk index
        index_path = self.path / "chunk_links.json"
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                self.chunk_to_entities = json.load(f)
        
        # User edits
        edits_path = self.path / "user_edits.json"
        if edits_path.exists():
            with open(edits_path, "r", encoding="utf-8") as f:
                self.user_edits = json.load(f)
        
        logger.info(f"Loaded case graph: {len(self.entities)} entities, {len(self.relationships)} relationships, {len(self.events)} events")
    
    def stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        return {
            "entity_count": len(self.entities),
            "relationship_count": len(self.relationships),
            "event_count": len(self.events),
            "chunks_indexed": len(self.chunk_to_entities),
            "entities_by_type": {
                t.value: sum(1 for e in self.entities.values() if e.type == t)
                for t in EntityType
            },
            "verified_entities": sum(1 for e in self.entities.values() if e.user_verified),
            "user_edits": len(self.user_edits),
        }



