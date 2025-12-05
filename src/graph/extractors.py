"""Entity and relationship extraction from documents."""

import re
import logging
from datetime import date, datetime
from typing import Any, Optional

from src.graph.entities import (
    Entity,
    EntityType,
    Relationship,
    RelationshipType,
    TimelineEvent,
    EventType,
)
from src.graph.case_graph import CaseGraph
from src.schema import Chunk, ParsedDocument

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extract entities, relationships, and events from documents.
    
    Uses pattern matching and heuristics for extraction.
    Future: integrate spaCy NER or local LLM for improved accuracy.
    """
    
    # Regex patterns for UK legal documents
    PATTERNS = {
        # UK dates
        "date_uk": r'\b(\d{1,2})\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s*(\d{4})\b',
        "date_numeric": r'\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\b',
        
        # Amounts
        "amount_gbp": r'£\s*([\d,]+(?:\.\d{2})?)\b',
        "amount_usd": r'\$\s*([\d,]+(?:\.\d{2})?)\b',
        "amount_eur": r'€\s*([\d,]+(?:\.\d{2})?)\b',
        
        # Case references
        "case_ewhc": r'\[(\d{4})\]\s*EWHC\s*(\d+)\s*\(([A-Za-z]+)\)',
        "case_ewca": r'\[(\d{4})\]\s*EWCA\s*(Civ|Crim)\s*(\d+)',
        "case_uksc": r'\[(\d{4})\]\s*UKSC\s*(\d+)',
        "claim_number": r'\b([A-Z]{2}\d{2}[A-Z]\d{5})\b',
        
        # Email addresses
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        
        # Phone numbers (UK)
        "phone_uk": r'\b(?:0|\+44)\s*\d{2,4}\s*\d{3,4}\s*\d{3,4}\b',
        
        # Company patterns
        "company_ltd": r'\b([A-Z][A-Za-z\s&]+)\s+(?:Limited|Ltd|LLP|PLC|plc)\b',
        
        # Person titles
        "person_titled": r'\b(Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Dame|Lord|Lady)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        
        # Document references
        "exhibit": r'\b(Exhibit\s+[A-Z0-9]+)\b',
        "appendix": r'\b(Appendix\s+[A-Z0-9]+)\b',
        "schedule": r'\b(Schedule\s+[A-Z0-9]+)\b',
    }
    
    MONTH_MAP = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    def __init__(self, graph: CaseGraph):
        """Initialize extractor with case graph.
        
        Args:
            graph: Case graph to populate
        """
        self.graph = graph
    
    def extract_from_document(
        self,
        document: ParsedDocument,
        chunks: list[Chunk],
    ) -> dict[str, int]:
        """Extract entities from a parsed document and its chunks.
        
        Args:
            document: Parsed document
            chunks: Document chunks
            
        Returns:
            Stats dict with counts of extracted items
        """
        stats = {
            "entities_added": 0,
            "entities_linked": 0,
            "relationships_added": 0,
            "events_added": 0,
        }
        
        # Extract from document metadata (high confidence)
        stats["entities_added"] += self._extract_from_metadata(document)
        
        # Extract from each chunk
        for chunk in chunks:
            chunk_stats = self._extract_from_chunk(chunk)
            stats["entities_added"] += chunk_stats["entities_added"]
            stats["entities_linked"] += chunk_stats["entities_linked"]
            stats["relationships_added"] += chunk_stats["relationships_added"]
            stats["events_added"] += chunk_stats["events_added"]
        
        return stats
    
    def _extract_from_metadata(self, document: ParsedDocument) -> int:
        """Extract entities from document metadata.
        
        Metadata entities have higher confidence than text extraction.
        """
        count = 0
        metadata = document.metadata or {}
        
        # Email From/To
        if document.document_type == "email":
            for field, role in [("from", "sender"), ("to", "recipient"), ("cc", "cc")]:
                value = metadata.get(field)
                if value:
                    # Could be multiple addresses separated by comma
                    for addr in value.split(","):
                        addr = addr.strip()
                        if addr:
                            self._add_or_link_entity(
                                name=addr,
                                entity_type=EntityType.PERSON,
                                source_chunk=None,
                                metadata={"role": role, "email": addr},
                                confidence=0.9,
                            )
                            count += 1
        
        # Author from witness statements
        author = metadata.get("author")
        if author:
            self._add_or_link_entity(
                name=author,
                entity_type=EntityType.PERSON,
                source_chunk=None,
                metadata={"role": "author"},
                confidence=0.95,
            )
            count += 1
        
        # Parties
        parties = metadata.get("parties")
        if parties:
            # Parse "X v Y" format
            if " v " in parties.lower():
                parts = re.split(r'\s+v\.?\s+', parties, flags=re.IGNORECASE)
                for i, part in enumerate(parts):
                    role = "claimant" if i == 0 else "defendant"
                    self._add_or_link_entity(
                        name=part.strip(),
                        entity_type=EntityType.PERSON,  # Could be org, will be corrected
                        source_chunk=None,
                        metadata={"role": role},
                        confidence=0.85,
                    )
                    count += 1
        
        # Case number
        case_number = metadata.get("case_number")
        if case_number:
            self._add_or_link_entity(
                name=case_number,
                entity_type=EntityType.CASE_REF,
                source_chunk=None,
                confidence=0.95,
            )
            count += 1
        
        # Document date as event
        doc_date = metadata.get("document_date")
        if doc_date:
            parsed_date = self._parse_date(doc_date)
            if parsed_date:
                event = TimelineEvent(
                    date=parsed_date,
                    date_text=doc_date,
                    description=f"Document created: {document.file_name}",
                    event_type=EventType.FILING,
                )
                self.graph.add_event(event)
        
        return count
    
    def _extract_from_chunk(self, chunk: Chunk) -> dict[str, int]:
        """Extract entities from a single chunk."""
        stats = {
            "entities_added": 0,
            "entities_linked": 0,
            "relationships_added": 0,
            "events_added": 0,
        }
        
        text = chunk.text
        
        # Extract dates
        for match in re.finditer(self.PATTERNS["date_uk"], text, re.IGNORECASE):
            day, month, year = match.groups()
            try:
                parsed_date = date(int(year), self.MONTH_MAP[month.lower()], int(day))
                date_text = match.group(0)
                
                # Check if this date has surrounding event context
                context = self._get_context(text, match.start(), match.end(), 100)
                if self._looks_like_event(context):
                    event = TimelineEvent(
                        date=parsed_date,
                        date_text=date_text,
                        description=self._extract_event_description(context),
                        source_chunks=[chunk.chunk_id],
                        event_type=self._classify_event(context),
                    )
                    self.graph.add_event(event)
                    stats["events_added"] += 1
            except ValueError:
                pass
        
        # Extract amounts
        for pattern_name in ["amount_gbp", "amount_usd", "amount_eur"]:
            for match in re.finditer(self.PATTERNS[pattern_name], text):
                amount_str = match.group(0)
                entity = self._add_or_link_entity(
                    name=amount_str,
                    entity_type=EntityType.AMOUNT,
                    source_chunk=chunk.chunk_id,
                    confidence=0.9,
                )
                if entity:
                    stats["entities_added" if entity.source_chunks == [chunk.chunk_id] else "entities_linked"] += 1
        
        # Extract case references
        for pattern_name in ["case_ewhc", "case_ewca", "case_uksc", "claim_number"]:
            for match in re.finditer(self.PATTERNS[pattern_name], text):
                ref = match.group(0)
                entity = self._add_or_link_entity(
                    name=ref,
                    entity_type=EntityType.CASE_REF,
                    source_chunk=chunk.chunk_id,
                    confidence=0.95,
                )
                if entity:
                    stats["entities_added" if entity.source_chunks == [chunk.chunk_id] else "entities_linked"] += 1
        
        # Extract companies
        for match in re.finditer(self.PATTERNS["company_ltd"], text):
            company = match.group(0)
            entity = self._add_or_link_entity(
                name=company,
                entity_type=EntityType.ORGANIZATION,
                source_chunk=chunk.chunk_id,
                confidence=0.85,
            )
            if entity:
                stats["entities_added" if entity.source_chunks == [chunk.chunk_id] else "entities_linked"] += 1
        
        # Extract titled persons
        for match in re.finditer(self.PATTERNS["person_titled"], text):
            person = match.group(0)
            entity = self._add_or_link_entity(
                name=person,
                entity_type=EntityType.PERSON,
                source_chunk=chunk.chunk_id,
                confidence=0.8,
            )
            if entity:
                stats["entities_added" if entity.source_chunks == [chunk.chunk_id] else "entities_linked"] += 1
        
        # Extract document references
        for pattern_name in ["exhibit", "appendix", "schedule"]:
            for match in re.finditer(self.PATTERNS[pattern_name], text, re.IGNORECASE):
                ref = match.group(0)
                entity = self._add_or_link_entity(
                    name=ref,
                    entity_type=EntityType.DOCUMENT_REF,
                    source_chunk=chunk.chunk_id,
                    confidence=0.9,
                )
                if entity:
                    stats["entities_added" if entity.source_chunks == [chunk.chunk_id] else "entities_linked"] += 1
        
        # Extract email addresses
        for match in re.finditer(self.PATTERNS["email"], text):
            email = match.group(0)
            entity = self._add_or_link_entity(
                name=email,
                entity_type=EntityType.EMAIL_ADDRESS,
                source_chunk=chunk.chunk_id,
                confidence=0.95,
            )
            if entity:
                stats["entities_added" if entity.source_chunks == [chunk.chunk_id] else "entities_linked"] += 1
        
        return stats
    
    def _add_or_link_entity(
        self,
        name: str,
        entity_type: EntityType,
        source_chunk: Optional[str],
        metadata: Optional[dict] = None,
        confidence: float = 0.8,
    ) -> Optional[Entity]:
        """Add new entity or link to existing similar entity.
        
        Args:
            name: Entity name
            entity_type: Type of entity
            source_chunk: Chunk ID where found
            metadata: Additional metadata
            confidence: Extraction confidence
            
        Returns:
            Entity (new or existing)
        """
        # Try to find similar existing entity
        existing = self.graph.find_similar_entity(name, entity_type, threshold=0.85)
        
        if existing:
            entity, similarity = existing
            # Link chunk to existing entity
            if source_chunk:
                entity.add_chunk(source_chunk)
            return entity
        else:
            # Create new entity
            entity = Entity(
                type=entity_type,
                canonical_name=name,
                source_chunks=[source_chunk] if source_chunk else [],
                metadata=metadata or {},
                confidence=confidence,
            )
            self.graph.add_entity(entity)
            return entity
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse date from string."""
        if not date_str:
            return None
        
        # Try UK date format: DD/MM/YYYY
        match = re.match(r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})', date_str)
        if match:
            try:
                return date(int(match.group(3)), int(match.group(2)), int(match.group(1)))
            except ValueError:
                pass
        
        # Try UK text format: DD Month YYYY
        match = re.match(r'(\d{1,2})\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s*(\d{4})', date_str, re.IGNORECASE)
        if match:
            try:
                return date(int(match.group(3)), self.MONTH_MAP[match.group(2).lower()], int(match.group(1)))
            except ValueError:
                pass
        
        return None
    
    def _get_context(self, text: str, start: int, end: int, window: int) -> str:
        """Get surrounding context for a match."""
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)
        return text[ctx_start:ctx_end]
    
    def _looks_like_event(self, context: str) -> bool:
        """Check if context describes an event."""
        event_words = [
            'meeting', 'met', 'agreed', 'signed', 'sent', 'received',
            'discussed', 'called', 'attended', 'filed', 'submitted',
            'hearing', 'trial', 'conference', 'deadline', 'due',
        ]
        context_lower = context.lower()
        return any(word in context_lower for word in event_words)
    
    def _extract_event_description(self, context: str) -> str:
        """Extract event description from context."""
        # Take first sentence containing the date
        sentences = re.split(r'[.!?]', context)
        for sent in sentences:
            if len(sent.strip()) > 10:
                return sent.strip()[:200]
        return context[:200]
    
    def _classify_event(self, context: str) -> EventType:
        """Classify event type from context."""
        context_lower = context.lower()
        
        if any(w in context_lower for w in ['meeting', 'met', 'attended', 'conference']):
            return EventType.MEETING
        if any(w in context_lower for w in ['email', 'sent', 'received', 'wrote', 'letter']):
            return EventType.COMMUNICATION
        if any(w in context_lower for w in ['signed', 'contract', 'agreement', 'executed']):
            return EventType.CONTRACT
        if any(w in context_lower for w in ['filed', 'submitted', 'issued', 'served']):
            return EventType.FILING
        if any(w in context_lower for w in ['hearing', 'trial', 'court']):
            return EventType.HEARING
        if any(w in context_lower for w in ['deadline', 'due', 'expir']):
            return EventType.DEADLINE
        if any(w in context_lower for w in ['paid', 'transfer', 'transaction', 'payment']):
            return EventType.TRANSACTION
        
        return EventType.OTHER



