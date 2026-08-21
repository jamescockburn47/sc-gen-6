"""Chunk-based timeline extraction using LLM.

Extracts timeline events directly from document chunks for better accuracy
than summary-based extraction. Chunks contain full context with dates.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Optional

from src.config_loader import Settings, get_settings
from src.graph.entities import TimelineEvent, EventType
from src.graph.case_graph import CaseGraph
from src.llm.client import get_llm_client
from src.config.llm_config import load_llm_config

logger = logging.getLogger(__name__)


CHUNK_TIMELINE_PROMPT = """You are analyzing a legal document chunk to extract timeline events.

Extract any chronological events with dates from this text. For each event found:
- date: ISO format (YYYY-MM-DD) if exact date is clear
- date_text: the original date text as written
- description: clear description of what happened (2-3 sentences)
- event_type: one of [meeting, communication, transaction, filing, hearing, deadline, contract, incident, other]
- entities: names of people/orgs involved

If NO events with dates are found, return an empty array [].

Document: {file_name}
Chunk text:
---
{chunk_text}
---

Return ONLY a JSON array (no markdown):
[
  {{
    "date": "2024-01-15",
    "date_text": "15 January 2024",
    "description": "Contract signed between ABC Corp and XYZ Ltd for the supply of goods.",
    "event_type": "contract",
    "entities": ["ABC Corp", "XYZ Ltd"]
  }}
]
"""


# Date patterns to identify chunks with potential events
DATE_PATTERNS = [
    r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
    r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
    r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b',
]


class ChunkTimelineExtractor:
    """Extract timeline events directly from document chunks using LLM."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.graph = CaseGraph(Path("data/case_graph"))
        # load() is called in CaseGraph.__init__ automatically
    
    def find_date_chunks(self) -> list[dict[str, Any]]:
        """Find all chunks that contain date patterns.
        
        Returns:
            List of chunk dicts with text, document info
        """
        from src.retrieval.vector_store import VectorStore
        
        vector_store = VectorStore(settings=self.settings)
        
        # Get all chunks from ChromaDB
        try:
            collection = vector_store.collection
            results = collection.get(include=["documents", "metadatas"])
            
            date_chunks = []
            pattern = re.compile('|'.join(DATE_PATTERNS), re.IGNORECASE)
            
            for i, doc in enumerate(results.get("documents", [])):
                if doc and pattern.search(doc):
                    metadata = results.get("metadatas", [{}])[i] or {}
                    chunk_id = results.get("ids", [])[i] if results.get("ids") else None
                    
                    date_chunks.append({
                        "chunk_id": chunk_id,
                        "text": doc,
                        "file_name": metadata.get("file_name", "Unknown"),
                        "document_id": metadata.get("document_id", ""),
                    })
            
            logger.info(f"Found {len(date_chunks)} chunks with date patterns")
            return date_chunks
            
        except Exception as e:
            logger.error(f"Error finding date chunks: {e}")
            return []
    
    def extract_events_from_chunk(
        self,
        chunk: dict[str, Any],
        llm_client: Any,
    ) -> list[TimelineEvent]:
        """Extract timeline events from a single chunk using LLM.
        
        Args:
            chunk: Chunk dict with text, file_name, etc.
            llm_client: LLM client for generation
            
        Returns:
            List of extracted TimelineEvent objects
        """
        events = []
        
        try:
            prompt = CHUNK_TIMELINE_PROMPT.format(
                file_name=chunk.get("file_name", "Unknown"),
                chunk_text=chunk.get("text", "")[:3000]  # Limit chunk size
            )
            
            response = llm_client.generate(prompt, temperature=0.2)
            
            # Clean response (remove markdown if any)
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)
            
            # Parse JSON
            events_data = json.loads(response)
            
            if not isinstance(events_data, list):
                events_data = [events_data] if events_data else []
            
            for event_dict in events_data:
                # Parse date
                event_date = None
                if event_dict.get("date"):
                    try:
                        event_date = date.fromisoformat(event_dict["date"])
                    except ValueError:
                        logger.warning(f"Invalid date format: {event_dict['date']}")
                
                # Map event type
                event_type_str = event_dict.get("event_type", "other").lower()
                try:
                    event_type = EventType(event_type_str)
                except ValueError:
                    event_type = EventType.OTHER
                
                event = TimelineEvent(
                    date=event_date,
                    date_text=event_dict.get("date_text", ""),
                    description=event_dict.get("description", ""),
                    event_type=event_type,
                    entities_involved=event_dict.get("entities", []),
                    source_chunks=[chunk.get("chunk_id", "")],
                    source_documents=[chunk.get("file_name", "")],
                )
                events.append(event)
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
        except Exception as e:
            logger.error(f"Error extracting from chunk: {e}")
        
        return events
    
    def extract_all_events(
        self,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        max_chunks: Optional[int] = None,
    ) -> list[TimelineEvent]:
        """Extract timeline events from all chunks with dates.
        
        Args:
            progress_callback: Optional callback(message, current, total)
            max_chunks: Limit number of chunks to process
            
        Returns:
            List of all extracted events
        """
        # Get chunks with date patterns
        date_chunks = self.find_date_chunks()
        
        if max_chunks:
            date_chunks = date_chunks[:max_chunks]
        
        if not date_chunks:
            logger.warning("No chunks with date patterns found")
            return []
        
        # Initialize LLM client
        config = load_llm_config()
        llm_client = get_llm_client(config)
        
        all_events: list[TimelineEvent] = []
        total = len(date_chunks)
        
        print(f"[TIMELINE] Processing {total} chunks with date patterns...")
        
        for idx, chunk in enumerate(date_chunks):
            if progress_callback:
                progress_callback(
                    f"Extracting events from {chunk.get('file_name', 'Unknown')}",
                    idx,
                    total
                )
            
            print(f"[TIMELINE] [{idx+1}/{total}] Processing chunk from {chunk.get('file_name', 'Unknown')[:40]}...")
            
            events = self.extract_events_from_chunk(chunk, llm_client)
            
            if events:
                print(f"[TIMELINE]   Found {len(events)} events")
                all_events.extend(events)
        
        # Deduplicate by date + similar description
        deduplicated = self._deduplicate_events(all_events)
        
        print(f"[TIMELINE] Extraction complete: {len(all_events)} events found, {len(deduplicated)} after deduplication")
        
        return deduplicated
    
    def _deduplicate_events(self, events: list[TimelineEvent]) -> list[TimelineEvent]:
        """Remove duplicate events using fuzzy matching on date and description.
        
        Deduplication rules:
        1. Events on same date with >70% description similarity are merged
        2. Events within 3 days with >85% description similarity are merged
        3. Source documents are merged for duplicates
        
        Args:
            events: List of events to deduplicate
            
        Returns:
            Deduplicated list with source documents merged
        """
        from difflib import SequenceMatcher
        from datetime import timedelta
        
        if not events:
            return []
        
        # Sort by date for consistent processing
        events = sorted(events, key=lambda e: e.sort_date)
        
        unique: list[TimelineEvent] = []
        
        for event in events:
            is_duplicate = False
            merge_with_idx = None
            
            event_date = event.date
            event_desc = (event.description or "").lower().strip()
            
            # Compare against existing unique events
            for idx, existing in enumerate(unique):
                existing_date = existing.date
                existing_desc = (existing.description or "").lower().strip()
                
                # Skip if no descriptions to compare
                if not event_desc or not existing_desc:
                    continue
                
                # Calculate description similarity
                similarity = SequenceMatcher(None, event_desc, existing_desc).ratio()
                
                # Check if dates are close
                date_match = False
                if event_date and existing_date:
                    date_diff = abs((event_date - existing_date).days)
                    if date_diff == 0:
                        date_match = True
                        required_similarity = 0.70  # Same day: 70% match
                    elif date_diff <= 3:
                        date_match = True
                        required_similarity = 0.85  # Within 3 days: 85% match
                elif event.date_text and existing.date_text:
                    # Compare date text if no ISO dates
                    date_text_sim = SequenceMatcher(None, 
                        event.date_text.lower(), 
                        existing.date_text.lower()
                    ).ratio()
                    if date_text_sim > 0.8:
                        date_match = True
                        required_similarity = 0.75
                
                # Check for duplicate
                if date_match and similarity >= required_similarity:
                    is_duplicate = True
                    merge_with_idx = idx
                    break
            
            if is_duplicate and merge_with_idx is not None:
                # Merge source documents
                for doc in event.source_documents:
                    if doc not in unique[merge_with_idx].source_documents:
                        unique[merge_with_idx].source_documents.append(doc)
                for chunk in event.source_chunks:
                    if chunk not in unique[merge_with_idx].source_chunks:
                        unique[merge_with_idx].source_chunks.append(chunk)
                # Keep longer description if significantly longer
                if len(event.description or "") > len(unique[merge_with_idx].description or "") * 1.3:
                    unique[merge_with_idx].description = event.description
            else:
                unique.append(event)
        
        return unique
    
    def save_events_to_graph(self, events: list[TimelineEvent]) -> int:
        """Save extracted events to the case graph.
        
        Args:
            events: List of events to save
            
        Returns:
            Number of events saved
        """
        count = 0
        for event in events:
            self.graph.add_event(event)
            count += 1
        
        self.graph.save()
        logger.info(f"Saved {count} timeline events to graph")
        return count


def run_timeline_extraction(
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    max_chunks: Optional[int] = None,
) -> dict[str, Any]:
    """Run the full timeline extraction pipeline.
    
    Args:
        progress_callback: Optional progress callback
        max_chunks: Limit chunks processed (for testing)
        
    Returns:
        Stats dict with extraction results
    """
    extractor = ChunkTimelineExtractor()
    
    # Extract events
    events = extractor.extract_all_events(
        progress_callback=progress_callback,
        max_chunks=max_chunks,
    )
    
    # Save to graph
    saved_count = extractor.save_events_to_graph(events)
    
    # Return stats
    stats = {
        "total_events": len(events),
        "saved_count": saved_count,
        "event_types": {},
        "date_range": None,
    }
    
    # Count by type
    for event in events:
        type_name = event.event_type.value
        stats["event_types"][type_name] = stats["event_types"].get(type_name, 0) + 1
    
    # Get date range
    dated_events = [e for e in events if e.date]
    if dated_events:
        min_date = min(e.date for e in dated_events)
        max_date = max(e.date for e in dated_events)
        stats["date_range"] = f"{min_date} to {max_date}"
    
    return stats


if __name__ == "__main__":
    # Run extraction when executed directly
    print("Starting chunk-based timeline extraction...")
    stats = run_timeline_extraction()
    print(f"\nResults: {json.dumps(stats, indent=2, default=str)}")
