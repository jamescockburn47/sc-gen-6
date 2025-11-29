"""Case graph generator from document summaries.

Extracts entities and relationships from document summaries using LLM,
generating a comprehensive case graph.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Callable, Optional

from src.generation.summary_based_generator import SummaryBasedGenerator
from src.graph.case_graph import CaseGraph
from src.graph.entities import Entity, EntityType, Relationship, RelationshipType
from src.retrieval.summary_store import DocumentSummary

logger = logging.getLogger(__name__)


ENTITY_EXTRACTION_PROMPT = """You are analyzing a legal case document summary to extract entities.

Extract all relevant entities from this summary. For each entity, provide:
- type: one of [person, organization, location, date, amount, case_ref, document_ref, event, email_address, phone]
- name: canonical name
- aliases: list of alternative names/spellings
- confidence: 0.0-1.0

Document: {file_name}
Summary:
{summary_content}

Return ONLY a JSON array of entities:
[
  {{"type": "person", "name": "John Smith", "aliases": ["J. Smith", "Smith"], "confidence": 0.95}},
  {{"type": "organization", "name": "ABC Corporation", "aliases": ["ABC Corp"], "confidence": 0.9}}
]
"""

RELATIONSHIP_EXTRACTION_PROMPT = """You are analyzing a legal case document summary to extract relationships between entities.

Known entities: {entity_names}

Extract relationships from this summary. For each relationship, provide:
- source: entity name (must be from known entities)
- target: entity name (must be from known entities)
- type: one of [sent_to, employed_by, party_to, witness_for, represents, attended, signed, authored, related_to]
- confidence: 0.0-1.0

Document: {file_name}
Summary:
{summary_content}

Return ONLY a JSON array of relationships:
[
  {{"source": "John Smith", "target": "ABC Corporation", "type": "employed_by", "confidence": 0.9}},
  {{"source": "Jane Doe", "target": "John Smith", "type": "sent_to", "confidence": 0.85}}
]
"""


class CaseGraphGenerator(SummaryBasedGenerator):
    """Generate case graph from document summaries."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.case_graph = CaseGraph()
    
    def generate_entities_from_summaries(
        self,
        summaries: list[DocumentSummary],
        model: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> list[Entity]:
        """Extract entities from summaries using LLM.
        
        Args:
            summaries: List of document summaries
            model: Model to use for extraction
            progress_callback: Optional progress callback
            
        Returns:
            List of extracted entities
        """
        llm_client = self.get_llm_client(model)
        all_entities: list[Entity] = []
        
        total = len(summaries)
        for idx, summary in enumerate(summaries):
            if progress_callback:
                progress_callback(f"Extracting entities from {summary.file_name}", idx, total)
            
            try:
                # Format prompt
                prompt = ENTITY_EXTRACTION_PROMPT.format(
                    file_name=summary.file_name,
                    summary_content=summary.content
                )
                
                # Call LLM
                response = llm_client.generate(prompt, temperature=0.3)
                
                # Parse JSON response
                entities_data = json.loads(response)
                
                # Create Entity objects
                for entity_dict in entities_data:
                    entity = Entity(
                        type=EntityType(entity_dict["type"]),
                        canonical_name=entity_dict["name"],
                        aliases=entity_dict.get("aliases", []),
                        confidence=entity_dict.get("confidence", 0.8),
                        metadata={"source_document": summary.file_name}
                    )
                    all_entities.append(entity)
                    
            except Exception as e:
                logger.error(f"Error extracting entities from {summary.file_name}: {e}")
                continue
        
        return all_entities
    
    def generate_relationships_from_summaries(
        self,
        summaries: list[DocumentSummary],
        entities: list[Entity],
        model: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> list[Relationship]:
        """Extract relationships between entities from summaries.
        
        Args:
            summaries: List of document summaries
            entities: Known entities to link
            model: Model to use for extraction
            progress_callback: Optional progress callback
            
        Returns:
            List of extracted relationships
        """
        llm_client = self.get_llm_client(model)
        all_relationships: list[Relationship] = []
        
        # Build entity name lookup
        entity_lookup = {e.canonical_name: e for e in entities}
        for entity in entities:
            for alias in entity.aliases:
                entity_lookup[alias] = entity
        
        entity_names = ", ".join([e.canonical_name for e in entities[:50]])  # Limit for prompt
        
        total = len(summaries)
        for idx, summary in enumerate(summaries):
            if progress_callback:
                progress_callback(f"Extracting relationships from {summary.file_name}", idx, total)
            
            try:
                # Format prompt
                prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
                    entity_names=entity_names,
                    file_name=summary.file_name,
                    summary_content=summary.content
                )
                
                # Call LLM
                response = llm_client.generate(prompt, temperature=0.3)
                
                # Parse JSON response
                relationships_data = json.loads(response)
                
                # Create Relationship objects
                for rel_dict in relationships_data:
                    source_entity = entity_lookup.get(rel_dict["source"])
                    target_entity = entity_lookup.get(rel_dict["target"])
                    
                    if source_entity and target_entity:
                        relationship = Relationship(
                            source_entity_id=source_entity.id,
                            target_entity_id=target_entity.id,
                            relationship_type=RelationshipType(rel_dict["type"]),
                            confidence=rel_dict.get("confidence", 0.8),
                            properties={"source_document": summary.file_name}
                        )
                        all_relationships.append(relationship)
                        
            except Exception as e:
                logger.error(f"Error extracting relationships from {summary.file_name}: {e}")
                continue
        
        return all_relationships
    
    def generate_full_graph(
        self,
        model: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        incremental: bool = False,
    ) -> CaseGraph:
        """Generate complete case graph from all summaries.
        
        Args:
            model: Model to use for generation
            progress_callback: Optional progress callback
            incremental: If True, only process new summaries
            
        Returns:
            Complete case graph
        """
        # Get summaries
        if incremental:
            # Load existing graph to get last update time
            try:
                self.case_graph.load()
                last_update = self.case_graph.metadata.get("last_generated", None)
                summaries = self.get_new_summaries(last_processed_time=last_update)
                logger.info(f"Incremental update: processing {len(summaries)} new summaries")
            except Exception:
                summaries = self.get_all_summaries(summary_type="overview")
                logger.info(f"Full generation: processing {len(summaries)} summaries")
        else:
            summaries = self.get_all_summaries(summary_type="overview")
            logger.info(f"Full generation: processing {len(summaries)} summaries")
        
        if not summaries:
            logger.warning("No summaries found for graph generation")
            return self.case_graph
        
        # Extract entities
        if progress_callback:
            progress_callback("Extracting entities...", 0, 2)
        
        entities = self.generate_entities_from_summaries(
            summaries,
            model,
            progress_callback
        )
        
        # Deduplicate and add to graph
        for entity in entities:
            self.case_graph.add_or_merge_entity(entity)
        
        # Extract relationships
        if progress_callback:
            progress_callback("Extracting relationships...", 1, 2)
        
        relationships = self.generate_relationships_from_summaries(
            summaries,
            entities,
            model,
            progress_callback
        )
        
        # Add to graph
        for relationship in relationships:
            self.case_graph.add_relationship(relationship)
        
        # Update metadata
        self.case_graph.metadata["last_generated"] = datetime.now().isoformat()
        self.case_graph.metadata["total_summaries_processed"] = len(summaries)
        
        # Save graph
        self.case_graph.save()
        
        logger.info(f"Graph generation complete: {len(entities)} entities, {len(relationships)} relationships")
        
        return self.case_graph
