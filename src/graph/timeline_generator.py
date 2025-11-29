"""Timeline generator from document summaries.

Extracts chronological events from document summaries using LLM,
with source document tracking.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any, Callable, Optional

from src.generation.summary_based_generator import SummaryBasedGenerator
from src.graph.entities import TimelineEvent, EventType
from src.retrieval.summary_store import DocumentSummary

logger = logging.getLogger(__name__)


TIMELINE_EXTRACTION_PROMPT = """You are analyzing a legal case document summary to extract timeline events.

Extract all chronological events from this summary. For each event, provide:
- date: ISO format (YYYY-MM-DD) if exact date known, otherwise null
- date_text: original date description (e.g., "mid-January 2024", "Q2 2023")
- description: brief description of the event
- event_type: one of [meeting, communication, transaction, filing, hearing, deadline, contract, incident, other]
- entities: list of entity names involved
- confidence: 0.0-1.0

Document: {file_name}
Summary:
{summary_content}

Return ONLY a JSON array of events:
[
  {{
    "date": "2024-01-15",
    "date_text": "January 15, 2024",
    "description": "Contract signed between ABC Corp and XYZ Ltd",
    "event_type": "contract",
    "entities": ["ABC Corp", "XYZ Ltd"],
    "confidence": 0.95
  }},
  {{
    "date": null,
    "date_text": "mid-2023",
    "description": "Initial meeting between parties",
    "event_type": "meeting",
    "entities": ["John Smith", "Jane Doe"],
    "confidence": 0.8
  }}
]
"""


class TimelineGenerator(SummaryBasedGenerator):
    """Generate timeline from document summaries."""
    
    def generate_events_from_summaries(
        self,
        summaries: list[DocumentSummary],
        model: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> list[TimelineEvent]:
        """Extract timeline events from summaries using LLM.
        
        Each event includes source_documents field tracking which
        document(s) the event was extracted from.
        
        Args:
            summaries: List of document summaries
            model: Model to use for extraction
            progress_callback: Optional progress callback
            
        Returns:
            List of extracted timeline events
        """
        llm_client = self.get_llm_client(model)
        all_events: list[TimelineEvent] = []
        
        total = len(summaries)
        for idx, summary in enumerate(summaries):
            if progress_callback:
                progress_callback(f"Extracting events from {summary.file_name}", idx, total)
            
            try:
                # Format prompt
                prompt = TIMELINE_EXTRACTION_PROMPT.format(
                    file_name=summary.file_name,
                    summary_content=summary.content
                )
                
                # Call LLM
                response = llm_client.generate(prompt, temperature=0.3)
                
                # Parse JSON response
                events_data = json.loads(response)
                
                # Create TimelineEvent objects
                for event_dict in events_data:
                    # Parse date
                    event_date = None
                    if event_dict.get("date"):
                        try:
                            event_date = date.fromisoformat(event_dict["date"])
                        except ValueError:
                            logger.warning(f"Invalid date format: {event_dict['date']}")
                    
                    event = TimelineEvent(
                        date=event_date,
                        date_text=event_dict.get("date_text", ""),
                        description=event_dict.get("description", ""),
                        event_type=EventType(event_dict.get("event_type", "other")),
                        entities_involved=event_dict.get("entities", []),
                        source_documents=[summary.file_name],  # Track source document
                    )
                    all_events.append(event)
                    
            except Exception as e:
                logger.error(f"Error extracting events from {summary.file_name}: {e}")
                continue
        
        return all_events
    
    def generate_full_timeline(
        self,
        model: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        incremental: bool = False,
    ) -> list[TimelineEvent]:
        """Generate complete timeline from all summaries.
        
        Args:
            model: Model to use for generation
            progress_callback: Optional progress callback
            incremental: If True, only process new summaries
            
        Returns:
            List of all timeline events, sorted chronologically
        """
        # Get summaries
        if incremental:
            # Get timestamp of last timeline generation
            from pathlib import Path
            timeline_file = Path("data/timeline_metadata.json")
            last_update = None
            
            if timeline_file.exists():
                try:
                    import json
                    with open(timeline_file) as f:
                        metadata = json.load(f)
                        last_update = metadata.get("last_generated")
                except Exception:
                    pass
            
            summaries = self.get_new_summaries(
                last_processed_time=last_update,
                summary_type="overview"
            )
            logger.info(f"Incremental update: processing {len(summaries)} new summaries")
        else:
            summaries = self.get_all_summaries(summary_type="overview")
            logger.info(f"Full generation: processing {len(summaries)} summaries")
        
        if not summaries:
            logger.warning("No summaries found for timeline generation")
            return []
        
        # Extract events
        events = self.generate_events_from_summaries(
            summaries,
            model,
            progress_callback
        )
        
        # Sort by date
        events.sort(key=lambda e: e.sort_date)
        
        # Save metadata
        from pathlib import Path
        timeline_file = Path("data/timeline_metadata.json")
        timeline_file.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "last_generated": datetime.now().isoformat(),
            "total_events": len(events),
            "total_summaries_processed": len(summaries)
        }
        
        with open(timeline_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Import events into graph store
        from src.graph.case_graph import CaseGraph
        graph = CaseGraph()
        try:
            graph.load()
        except Exception:
            pass
        
        for event in events:
            graph.add_timeline_event(event)
        
        graph.save()
        
        logger.info(f"Timeline generation complete: {len(events)} events")
        
        return events
