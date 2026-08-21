"""Multi-layer PII detection engine.

Combines four detection layers for maximum coverage:
  1. Microsoft Presidio (NER + pattern recognisers)
  2. spaCy NER (transformer-based named entity recognition)
  3. Rule-based patterns (UK legal, abuse-specific, financial)
  4. Local LLM contextual analysis (optional, for special category data)

Entities are merged, de-duplicated, and scored across layers.
Overlapping detections from multiple layers boost confidence.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from loguru import logger

from .citation_guard import ProtectedCitation, extract_protected_citations, is_in_protected_span
from .models import (
    CATEGORY_RISK_MAP,
    DetectionLayer,
    PIICategory,
    PIIEntity,
    RiskLevel,
)
from .patterns.abuse_specific import scan_abuse_patterns
from .patterns.financial import scan_financial_patterns
from .patterns.uk_legal import PatternMatch, scan_uk_legal_patterns
from .recognisers.presidio_config import PresidioDetector
from .recognisers.spacy_legal import SpacyLegalDetector


# Confidence boost when multiple layers agree on the same span
MULTI_LAYER_BOOST = 0.10

# LLM prompt for contextual PII detection
_LLM_DETECTION_PROMPT = """You are a PII detection specialist for UK legal documents involving sensitive cases (including child abuse, sexual offences, and domestic violence).

Analyse the following text and identify ALL personally identifiable information that could lead to identification of any individual, especially victims and vulnerable persons.

For each PII entity found, output a JSON array of objects with:
- "text": the exact PII text
- "category": one of: person_name, address, postcode, phone_number, email_address, ni_number, date_of_birth, age, school_name, institution_name, relationship_descriptor, medical_identifier, location, date, case_reference, organisation, victim_identifier, perpetrator_identifier
- "reason": brief explanation of why this is PII

CRITICAL: In abuse cases, even indirect identifiers matter:
- Relationship chains ("the defendant's daughter") can identify victims
- School names + ages can narrow identification
- Care home names + dates can identify investigations
- Specific injury descriptions may be identifying

TEXT TO ANALYSE:
---
{text}
---

Respond with ONLY a JSON array. If no PII found, respond with [].
"""


class PIIDetector:
    """Multi-layer PII detection engine.

    Orchestrates detection across all four layers and merges results.

    Args:
        use_presidio: Enable Presidio detection layer.
        use_spacy: Enable spaCy NER layer.
        use_patterns: Enable rule-based pattern layer.
        use_llm: Enable local LLM contextual layer.
        spacy_model: spaCy model name.
        confidence_threshold: Minimum confidence to keep.
        llm_service: Optional LLM service for contextual detection.
    """

    def __init__(
        self,
        use_presidio: bool = True,
        use_spacy: bool = True,
        use_patterns: bool = True,
        use_llm: bool = True,
        spacy_model: str = "en_core_web_trf",
        confidence_threshold: float = 0.5,
        llm_service: Optional[Any] = None,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._use_llm = use_llm
        self._llm_service = llm_service

        # Initialise detection layers
        self._presidio: Optional[PresidioDetector] = None
        self._spacy: Optional[SpacyLegalDetector] = None
        self._use_patterns = use_patterns

        if use_presidio:
            self._presidio = PresidioDetector()
            if not self._presidio.is_available:
                logger.warning("Presidio unavailable — layer disabled")
                self._presidio = None

        if use_spacy:
            self._spacy = SpacyLegalDetector(model_name=spacy_model)
            if not self._spacy.is_available:
                logger.warning("spaCy NER unavailable — layer disabled")
                self._spacy = None

        active_layers = sum([
            self._presidio is not None,
            self._spacy is not None,
            self._use_patterns,
            self._use_llm and self._llm_service is not None,
        ])
        logger.info(f"PIIDetector initialised with {active_layers} active layers")

    @property
    def active_layers(self) -> list[str]:
        """List of active detection layer names."""
        layers = []
        if self._presidio:
            layers.append("presidio")
        if self._spacy:
            layers.append("spacy_ner")
        if self._use_patterns:
            layers.append("rule_patterns")
        if self._use_llm and self._llm_service:
            layers.append("local_llm")
        return layers

    def detect(
        self,
        text: str,
        include_context: bool = True,
        enable_llm_layer: bool = False,
    ) -> list[PIIEntity]:
        """Detect all PII entities in text using all active layers.

        Published case citations (detected by the citation guard) are
        protected — any PII entity that overlaps with a reported case
        name or neutral citation is excluded. This preserves the law
        while removing the identifying facts.

        Args:
            text: Text to scan for PII.
            include_context: Whether to include surrounding context snippets.
            enable_llm_layer: Whether to use local LLM for this specific call
                (expensive, typically only for special category data).

        Returns:
            Merged, de-duplicated list of PIIEntity instances sorted by position.
        """
        # Step 0: Citation guard — find published case law to protect
        protected_citations = extract_protected_citations(text)
        self._last_protected_citations = protected_citations

        all_entities: list[PIIEntity] = []

        # Layer 1: Presidio
        if self._presidio:
            try:
                presidio_entities = self._presidio.detect(text)
                all_entities.extend(presidio_entities)
                logger.debug(f"Presidio detected {len(presidio_entities)} entities")
            except Exception as e:
                logger.error(f"Presidio layer failed: {e}")

        # Layer 2: spaCy NER
        if self._spacy:
            try:
                spacy_entities = self._spacy.detect(text)
                all_entities.extend(spacy_entities)
                logger.debug(f"spaCy NER detected {len(spacy_entities)} entities")
            except Exception as e:
                logger.error(f"spaCy NER layer failed: {e}")

        # Layer 3: Rule-based patterns
        if self._use_patterns:
            try:
                pattern_entities = self._detect_patterns(text)
                all_entities.extend(pattern_entities)
                logger.debug(f"Rule patterns detected {len(pattern_entities)} entities")
            except Exception as e:
                logger.error(f"Pattern layer failed: {e}")

        # Layer 4: Local LLM contextual analysis (optional)
        if enable_llm_layer and self._use_llm and self._llm_service:
            try:
                llm_entities = self._detect_with_llm(text)
                all_entities.extend(llm_entities)
                logger.debug(f"Local LLM detected {len(llm_entities)} entities")
            except Exception as e:
                logger.error(f"LLM detection layer failed: {e}")

        # Merge and de-duplicate
        merged = self._merge_entities(all_entities)

        # Apply confidence threshold
        filtered = [e for e in merged if e.confidence >= self._confidence_threshold]

        # Filter out entities that fall within protected citation spans
        if protected_citations:
            pre_filter_count = len(filtered)
            filtered = [
                e for e in filtered
                if not is_in_protected_span(e.start, e.end, protected_citations)
            ]
            guarded = pre_filter_count - len(filtered)
            if guarded:
                logger.info(
                    f"Citation guard preserved {guarded} entities "
                    f"(published case law — not PII)"
                )

        # Sort by position
        filtered.sort(key=lambda e: e.start)

        logger.info(
            f"PII detection complete: {len(filtered)} entities "
            f"(from {len(all_entities)} raw detections across {len(self.active_layers)} layers)"
        )

        return filtered

    @property
    def last_protected_citations(self) -> list[ProtectedCitation]:
        """Citations protected in the most recent detect() call."""
        return getattr(self, "_last_protected_citations", [])

    def _detect_patterns(self, text: str) -> list[PIIEntity]:
        """Run all rule-based pattern scanners."""
        entities: list[PIIEntity] = []

        for scanner in (scan_uk_legal_patterns, scan_abuse_patterns, scan_financial_patterns):
            matches: list[PatternMatch] = scanner(text)
            for m in matches:
                # Context snippet
                ctx_start = max(0, m.start - 50)
                ctx_end = min(len(text), m.end + 50)

                entities.append(PIIEntity(
                    category=m.category,
                    original_text=m.text,
                    start=m.start,
                    end=m.end,
                    confidence=m.confidence,
                    detection_layer=DetectionLayer.RULE_PATTERN,
                    context=text[ctx_start:ctx_end],
                    metadata={"pattern_name": m.pattern_name},
                ))

        return entities

    def _detect_with_llm(self, text: str) -> list[PIIEntity]:
        """Use local LLM for contextual PII detection.

        This is the most expensive layer and should only be used for
        special category data (abuse cases, etc.).
        """
        if not self._llm_service:
            return []

        # Truncate to reasonable size for LLM context
        max_chars = 4000
        truncated = text[:max_chars] if len(text) > max_chars else text

        prompt = _LLM_DETECTION_PROMPT.format(text=truncated)

        try:
            # Use generate_with_context or a simpler method
            response = self._llm_service.generate(
                prompt=prompt,
                max_tokens=2048,
                temperature=0.1,  # Low temp for consistent detection
            )

            if not response:
                return []

            return self._parse_llm_response(response, text)

        except Exception as e:
            logger.error(f"LLM PII detection failed: {e}")
            return []

    def _parse_llm_response(self, response: str, original_text: str) -> list[PIIEntity]:
        """Parse LLM JSON response into PIIEntity instances."""
        import json

        entities: list[PIIEntity] = []

        # Extract JSON array from response
        try:
            # Find JSON array in response
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if not json_match:
                return []

            items = json.loads(json_match.group())
            if not isinstance(items, list):
                return []

            for item in items:
                if not isinstance(item, dict):
                    continue

                pii_text = item.get("text", "")
                if not pii_text:
                    continue

                # Find the entity position in the original text
                idx = original_text.find(pii_text)
                if idx == -1:
                    # Try case-insensitive
                    lower_text = original_text.lower()
                    idx = lower_text.find(pii_text.lower())

                if idx == -1:
                    continue

                # Map category string to enum
                cat_str = item.get("category", "custom")
                try:
                    category = PIICategory(cat_str)
                except ValueError:
                    category = PIICategory.CUSTOM

                ctx_start = max(0, idx - 50)
                ctx_end = min(len(original_text), idx + len(pii_text) + 50)

                entities.append(PIIEntity(
                    category=category,
                    original_text=pii_text,
                    start=idx,
                    end=idx + len(pii_text),
                    confidence=0.75,  # Moderate confidence for LLM detections
                    detection_layer=DetectionLayer.LOCAL_LLM,
                    context=original_text[ctx_start:ctx_end],
                    metadata={
                        "reason": item.get("reason", ""),
                        "llm_category": cat_str,
                    },
                ))

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse LLM detection response: {e}")

        return entities

    def _merge_entities(self, entities: list[PIIEntity]) -> list[PIIEntity]:
        """Merge overlapping entities from different layers.

        When multiple layers detect the same span:
        - Keep the entity with the highest confidence
        - Boost confidence for multi-layer agreement
        - Prefer more specific categories over generic ones
        """
        if not entities:
            return []

        # Sort by start position, then by span length (longer first)
        entities.sort(key=lambda e: (e.start, -(e.end - e.start)))

        merged: list[PIIEntity] = []

        for entity in entities:
            # Check if this overlaps with an already-merged entity
            overlap_found = False
            for i, existing in enumerate(merged):
                if self._spans_overlap(entity, existing):
                    overlap_found = True
                    # Merge: boost confidence, prefer longer span, prefer more specific category
                    if entity.detection_layer != existing.detection_layer:
                        # Multi-layer agreement — boost confidence
                        existing.confidence = min(
                            1.0, existing.confidence + MULTI_LAYER_BOOST
                        )

                    # Keep the longer/more specific entity
                    if (entity.end - entity.start) > (existing.end - existing.start):
                        existing.original_text = entity.original_text
                        existing.start = entity.start
                        existing.end = entity.end
                        existing.context = entity.context

                    # Prefer more specific category
                    if self._category_specificity(entity.category) > self._category_specificity(existing.category):
                        existing.category = entity.category

                    # Take higher confidence
                    if entity.confidence > existing.confidence:
                        existing.confidence = entity.confidence

                    # Track all detection layers
                    layers = existing.metadata.get("detection_layers", [existing.detection_layer.value])
                    if entity.detection_layer.value not in layers:
                        layers.append(entity.detection_layer.value)
                    existing.metadata["detection_layers"] = layers

                    break

            if not overlap_found:
                entity.metadata["detection_layers"] = [entity.detection_layer.value]
                merged.append(entity)

        return merged

    @staticmethod
    def _spans_overlap(a: PIIEntity, b: PIIEntity) -> bool:
        """Check if two entity spans overlap significantly."""
        overlap_start = max(a.start, b.start)
        overlap_end = min(a.end, b.end)
        if overlap_start >= overlap_end:
            return False

        overlap_len = overlap_end - overlap_start
        shorter_len = min(a.end - a.start, b.end - b.start)

        # Consider overlapping if more than 50% of the shorter span overlaps
        return overlap_len >= shorter_len * 0.5

    @staticmethod
    def _category_specificity(category: PIICategory) -> int:
        """Score how specific a category is (higher = more specific)."""
        specificity = {
            PIICategory.VICTIM_IDENTIFIER: 10,
            PIICategory.PERPETRATOR_IDENTIFIER: 10,
            PIICategory.WITNESS_NAME: 9,
            PIICategory.JUDGE_NAME: 9,
            PIICategory.SOLICITOR_NAME: 9,
            PIICategory.BARRISTER_NAME: 9,
            PIICategory.SCHOOL_NAME: 8,
            PIICategory.INSTITUTION_NAME: 8,
            PIICategory.NI_NUMBER: 8,
            PIICategory.MEDICAL_IDENTIFIER: 8,
            PIICategory.RELATIONSHIP_DESCRIPTOR: 7,
            PIICategory.PERSON_NAME: 6,
            PIICategory.ADDRESS: 6,
            PIICategory.EMAIL_ADDRESS: 6,
            PIICategory.PHONE_NUMBER: 6,
            PIICategory.DATE_OF_BIRTH: 6,
            PIICategory.AGE: 5,
            PIICategory.POSTCODE: 5,
            PIICategory.CASE_REFERENCE: 4,
            PIICategory.LOCATION: 4,
            PIICategory.DATE: 3,
            PIICategory.MONETARY_AMOUNT: 3,
            PIICategory.ORGANISATION: 2,
            PIICategory.CUSTOM: 1,
        }
        return specificity.get(category, 1)

    def get_detection_summary(self, entities: list[PIIEntity]) -> dict[str, Any]:
        """Generate a summary of detected PII for reporting.

        Args:
            entities: List of detected PIIEntity instances.

        Returns:
            Summary dict with counts by category, risk level, and layer.
        """
        by_category: dict[str, int] = {}
        by_risk: dict[str, int] = {}
        by_layer: dict[str, int] = {}

        for e in entities:
            by_category[e.category.value] = by_category.get(e.category.value, 0) + 1
            by_risk[e.risk_level.value] = by_risk.get(e.risk_level.value, 0) + 1
            by_layer[e.detection_layer.value] = by_layer.get(e.detection_layer.value, 0) + 1

        return {
            "total_entities": len(entities),
            "by_category": by_category,
            "by_risk_level": by_risk,
            "by_detection_layer": by_layer,
            "has_critical": any(e.risk_level == RiskLevel.CRITICAL for e in entities),
            "avg_confidence": (
                sum(e.confidence for e in entities) / len(entities)
                if entities
                else 0.0
            ),
        }
