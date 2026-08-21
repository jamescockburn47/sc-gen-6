"""Kanon 2 Enricher integration for knowledge graph extraction.

Uses the Isaacus Kanon 2 Enricher API to transform unstructured legal text
into rich, hierarchical knowledge graphs (ILGS schema) — persons, locations,
terms, citations, cross-references, segments, and more.

Key properties:
  - Non-generative, graph-first architecture
  - Zero hallucination by design
  - Efficient enough for batch processing
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

from src.graph.entities import (
    Entity,
    EntityType,
    Relationship,
    RelationshipType,
    TimelineEvent,
)


# ---------------------------------------------------------------------------
# Data models for ILGS (Isaacus Legal Graph Schema) enrichment output
# ---------------------------------------------------------------------------

@dataclass
class ILGSPerson:
    """Person extracted by Kanon 2 Enricher."""
    id: str
    name_span: tuple[int, int]
    name_text: str
    person_type: str  # "natural" | "corporate"
    role: str  # "defendant", "complainant", "non_party", etc.
    parent: Optional[str] = None
    children: list[str] = field(default_factory=list)
    residence: Optional[str] = None  # location ID
    mentions: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class ILGSLocation:
    """Location extracted by Kanon 2 Enricher."""
    id: str
    name_span: tuple[int, int]
    name_text: str
    location_type: str  # "address", "city", "country", etc.
    parent: Optional[str] = None
    children: list[str] = field(default_factory=list)
    mentions: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class ILGSSegment:
    """Document segment extracted by Kanon 2 Enricher."""
    id: str
    kind: str  # "unit", "heading", etc.
    segment_type: Optional[str] = None
    category: str = "main"
    title: Optional[str] = None
    parent: Optional[str] = None
    children: list[str] = field(default_factory=list)
    level: int = 0
    span: tuple[int, int] = (0, 0)


@dataclass
class ILGSDocument:
    """Full enrichment result from Kanon 2 Enricher."""
    version: str = "ilgs@1"
    text: str = ""
    title: Optional[str] = None
    subtitle: Optional[str] = None
    doc_type: str = "decision"
    jurisdiction: str = ""
    segments: list[ILGSSegment] = field(default_factory=list)
    persons: list[ILGSPerson] = field(default_factory=list)
    locations: list[ILGSLocation] = field(default_factory=list)
    crossreferences: list[dict[str, Any]] = field(default_factory=list)
    dates: list[dict[str, Any]] = field(default_factory=list)
    terms: list[dict[str, Any]] = field(default_factory=list)
    quotes: list[dict[str, Any]] = field(default_factory=list)
    emails: list[dict[str, Any]] = field(default_factory=list)
    external_documents: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Enricher service
# ---------------------------------------------------------------------------

class KanonEnricher:
    """Client for the Isaacus Kanon 2 Enricher API.

    Transforms unstructured legal text into structured ILGS knowledge graphs.

    When an anonymisation gateway is attached, all text is pseudonymised
    before being sent to the external API.
    """

    API_URL = "https://api.isaacus.com/v1/enrichments"
    MODEL = "kanon-2-enricher"
    MAX_BATCH_SIZE = 8  # API enforces max 8 texts per request

    def __init__(self, api_key: Optional[str] = None, anonymisation_gateway: Optional[Any] = None):
        """Initialize with API key from param or environment.

        Args:
            api_key: Isaacus API key. Falls back to ISAACUS_API_KEY env var.
            anonymisation_gateway: Optional CloudExportGateway for anonymisation.
        """
        self.api_key = api_key or os.environ.get("ISAACUS_API_KEY", "")
        if not self.api_key:
            logger.warning("No ISAACUS_API_KEY set — enrichment will be unavailable")
        self._total_tokens_used = 0
        self._total_requests = 0
        self._gateway = anonymisation_gateway

    def set_anonymisation_gateway(self, gateway: Any) -> None:
        """Attach or replace the anonymisation gateway.

        Args:
            gateway: CloudExportGateway instance for anonymising outbound text.
        """
        self._gateway = gateway
        logger.info("KanonEnricher: anonymisation gateway attached")

    @property
    def is_available(self) -> bool:
        """Check if the enricher is configured and allowed.

        Returns False if:
        - No API key is set
        - An anonymisation gateway is attached and has Kanon disabled
          (the Kanon API is an external US-based service that receives
          raw legal text — it should be disabled when handling
          sensitive/privileged data or when data sovereignty is required)
        """
        if not self.api_key:
            return False
        # If a gateway is attached, check whether Kanon is explicitly enabled
        if self._gateway is not None:
            if not getattr(self._gateway, "kanon_enricher_enabled", True):
                logger.info(
                    "KanonEnricher: disabled by anonymisation gateway policy "
                    "(external API blocked for data sovereignty)"
                )
                return False
        return True

    @property
    def usage_stats(self) -> dict[str, int]:
        """Return usage statistics."""
        return {
            "total_tokens": self._total_tokens_used,
            "total_requests": self._total_requests,
        }

    def enrich_text(self, text: str) -> Optional[ILGSDocument]:
        """Enrich a single text with Kanon 2 Enricher.

        Args:
            text: Legal text to enrich.

        Returns:
            ILGSDocument with extracted entities, or None on failure.
        """
        results = self.enrich_batch([text])
        return results[0] if results else None

    def enrich_batch(self, texts: list[str]) -> list[Optional[ILGSDocument]]:
        """Enrich a batch of texts.

        If an anonymisation gateway is attached, texts are pseudonymised
        before being sent to the external Isaacus API.

        Args:
            texts: List of legal texts to enrich.

        Returns:
            List of ILGSDocument results (None for failed items).
        """
        import httpx

        if not self.is_available:
            logger.warning("Kanon 2 Enricher not configured — skipping enrichment")
            return [None] * len(texts)

        # --- Anonymise texts if gateway is configured ---
        if self._gateway:
            try:
                anonymised_texts = []
                for t in texts:
                    anon_text = self._gateway.export_text(t)
                    anonymised_texts.append(anon_text)
                texts_to_send = anonymised_texts
                logger.info(f"KanonEnricher: anonymised {len(texts)} texts via gateway")
            except Exception as e:
                logger.error(f"KanonEnricher: anonymisation failed — {e}")
                return [None] * len(texts)
        else:
            texts_to_send = texts

        results: list[Optional[ILGSDocument]] = [None] * len(texts)

        # Process in batches
        for batch_start in range(0, len(texts_to_send), self.MAX_BATCH_SIZE):
            batch = texts_to_send[batch_start:batch_start + self.MAX_BATCH_SIZE]
            try:
                t0 = time.perf_counter()
                response = httpx.post(
                    self.API_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.MODEL,
                        "texts": batch,
                    },
                    timeout=120.0,
                )
                elapsed = time.perf_counter() - t0

                if response.status_code != 200:
                    logger.error(
                        f"Kanon 2 Enricher API error {response.status_code}: "
                        f"{response.text[:500]}"
                    )
                    continue

                data = response.json()
                self._total_requests += 1
                self._total_tokens_used += data.get("usage", {}).get("input_tokens", 0)

                for item in data.get("results", []):
                    idx = item.get("index", 0)
                    doc = item.get("document", {})
                    abs_idx = batch_start + idx
                    if abs_idx < len(results):
                        results[abs_idx] = self._parse_ilgs(doc)

                logger.info(
                    f"Enriched {len(batch)} texts in {elapsed:.1f}s "
                    f"({data.get('usage', {}).get('input_tokens', 0)} tokens)"
                )

            except httpx.TimeoutException:
                logger.error(f"Kanon 2 Enricher timeout on batch {batch_start}")
            except Exception as e:
                logger.error(f"Kanon 2 Enricher error: {e}")

        return results

    def _parse_ilgs(self, doc: dict[str, Any]) -> ILGSDocument:
        """Parse raw API response into ILGSDocument."""
        text = doc.get("text", "")

        persons = []
        for p in doc.get("persons", []):
            name_span = (p["name"]["start"], p["name"]["end"])
            persons.append(ILGSPerson(
                id=p["id"],
                name_span=name_span,
                name_text=text[name_span[0]:name_span[1]],
                person_type=p.get("type", "natural"),
                role=p.get("role", "non_party"),
                parent=p.get("parent"),
                children=p.get("children", []),
                residence=p.get("residence"),
                mentions=[(m["start"], m["end"]) for m in p.get("mentions", [])],
            ))

        locations = []
        for loc in doc.get("locations", []):
            name_span = (loc["name"]["start"], loc["name"]["end"])
            locations.append(ILGSLocation(
                id=loc["id"],
                name_span=name_span,
                name_text=text[name_span[0]:name_span[1]],
                location_type=loc.get("type", "unknown"),
                parent=loc.get("parent"),
                children=loc.get("children", []),
                mentions=[(m["start"], m["end"]) for m in loc.get("mentions", [])],
            ))

        segments = []
        for seg in doc.get("segments", []):
            span = seg.get("span", {})
            segments.append(ILGSSegment(
                id=seg["id"],
                kind=seg.get("kind", "unit"),
                segment_type=seg.get("type"),
                category=seg.get("category", "main"),
                title=seg.get("title"),
                parent=seg.get("parent"),
                children=seg.get("children", []),
                level=seg.get("level", 0),
                span=(span.get("start", 0), span.get("end", 0)),
            ))

        return ILGSDocument(
            version=doc.get("version", "ilgs@1"),
            text=text,
            title=doc.get("title"),
            subtitle=doc.get("subtitle"),
            doc_type=doc.get("type", "decision"),
            jurisdiction=doc.get("jurisdiction", ""),
            segments=segments,
            persons=persons,
            locations=locations,
            crossreferences=doc.get("crossreferences", []),
            dates=doc.get("dates", []),
            terms=doc.get("terms", []),
            quotes=doc.get("quotes", []),
            emails=doc.get("emails", []),
            external_documents=doc.get("external_documents", []),
            raw=doc,
        )

    # ------------------------------------------------------------------
    # Conversion to SCGen6 entity model
    # ------------------------------------------------------------------

    def to_entities(
        self,
        ilgs_doc: ILGSDocument,
        chunk_id: str = "",
    ) -> tuple[list[Entity], list[Relationship]]:
        """Convert ILGS enrichment into SCGen6 Entity/Relationship objects.

        Args:
            ilgs_doc: Enrichment result from Kanon 2.
            chunk_id: Source chunk ID for provenance.

        Returns:
            Tuple of (entities, relationships).
        """
        entities: list[Entity] = []
        relationships: list[Relationship] = []
        ilgs_to_entity: dict[str, str] = {}  # ILGS ID → Entity UUID

        # --- Persons → Entities ---
        for person in ilgs_doc.persons:
            etype = (
                EntityType.ORGANIZATION
                if person.person_type == "corporate"
                else EntityType.PERSON
            )
            entity = Entity(
                type=etype,
                canonical_name=person.name_text,
                metadata={
                    "ilgs_id": person.id,
                    "role": person.role,
                    "person_type": person.person_type,
                    "jurisdiction": ilgs_doc.jurisdiction,
                    "source": "kanon-2-enricher",
                },
                source_chunks=[chunk_id] if chunk_id else [],
                confidence=1.0,  # Non-generative = high confidence
            )
            entities.append(entity)
            ilgs_to_entity[person.id] = entity.id

        # --- Locations → Entities ---
        for location in ilgs_doc.locations:
            entity = Entity(
                type=EntityType.LOCATION,
                canonical_name=location.name_text,
                metadata={
                    "ilgs_id": location.id,
                    "location_type": location.location_type,
                    "source": "kanon-2-enricher",
                },
                source_chunks=[chunk_id] if chunk_id else [],
                confidence=1.0,
            )
            entities.append(entity)
            ilgs_to_entity[location.id] = entity.id

        # --- Person→Location residence relationships ---
        for person in ilgs_doc.persons:
            if person.residence and person.residence in ilgs_to_entity:
                person_entity_id = ilgs_to_entity.get(person.id, "")
                loc_entity_id = ilgs_to_entity[person.residence]
                if person_entity_id:
                    relationships.append(Relationship(
                        source_entity_id=person_entity_id,
                        target_entity_id=loc_entity_id,
                        relationship_type=RelationshipType.RELATED_TO,
                        properties={"relation": "residence"},
                        source_chunks=[chunk_id] if chunk_id else [],
                        confidence=1.0,
                    ))

        # --- Parent→Child relationships ---
        for person in ilgs_doc.persons:
            if person.parent and person.parent in ilgs_to_entity:
                child_id = ilgs_to_entity.get(person.id, "")
                parent_id = ilgs_to_entity[person.parent]
                if child_id:
                    relationships.append(Relationship(
                        source_entity_id=child_id,
                        target_entity_id=parent_id,
                        relationship_type=RelationshipType.EMPLOYED_BY,
                        properties={"relation": "subsidiary_of"},
                        source_chunks=[chunk_id] if chunk_id else [],
                        confidence=1.0,
                    ))

        return entities, relationships

    def enrich_and_convert(
        self,
        text: str,
        chunk_id: str = "",
    ) -> tuple[Optional[ILGSDocument], list[Entity], list[Relationship]]:
        """Enrich text and convert to SCGen6 entities in one call.

        Args:
            text: Legal text to enrich.
            chunk_id: Source chunk ID.

        Returns:
            Tuple of (ilgs_doc, entities, relationships).
        """
        ilgs_doc = self.enrich_text(text)
        if ilgs_doc is None:
            return None, [], []
        entities, relationships = self.to_entities(ilgs_doc, chunk_id)
        return ilgs_doc, entities, relationships
