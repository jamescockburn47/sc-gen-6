"""spaCy NER wrapper for legal document entity detection.

Provides a dedicated NER layer that catches entities Presidio may miss,
particularly in dense legal prose where names are embedded in complex
sentence structures.

Uses en_core_web_trf (transformer-based) for highest accuracy,
falling back to en_core_web_sm if unavailable.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from ..models import PIICategory, PIIEntity, DetectionLayer

# spaCy is optional — graceful fallback
try:
    import spacy
    from spacy.language import Language

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning(
        "spaCy not installed — NER detection layer disabled. "
        "Install with: pip install spacy && python -m spacy download en_core_web_trf"
    )


# Mapping from spaCy NER labels to our PIICategory
SPACY_TO_CATEGORY: dict[str, PIICategory] = {
    "PERSON": PIICategory.PERSON_NAME,
    "ORG": PIICategory.ORGANISATION,
    "GPE": PIICategory.LOCATION,           # Geopolitical entity (countries, cities)
    "LOC": PIICategory.LOCATION,           # Non-GPE locations
    "FAC": PIICategory.INSTITUTION_NAME,   # Buildings, airports, highways
    "DATE": PIICategory.DATE,
    "TIME": PIICategory.DATE,
    "MONEY": PIICategory.MONETARY_AMOUNT,
    "CARDINAL": PIICategory.CUSTOM,        # Numerals (context-gated)
    "ORDINAL": PIICategory.CUSTOM,
    "NORP": PIICategory.CUSTOM,            # Nationality/religion/political group
}

# Entity types we always want to capture from spaCy
RELEVANT_SPACY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "DATE", "MONEY"}


class SpacyLegalDetector:
    """spaCy NER detector tuned for UK legal documents.

    Args:
        model_name: spaCy model to load. Defaults to en_core_web_sm
            (en_core_web_trf segfaults on ROCm PyTorch builds).
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self._nlp: Optional[Language] = None
        self._model_name = model_name

        if SPACY_AVAILABLE:
            self._init_model(model_name)

    def _init_model(self, model_name: str) -> None:
        """Load the spaCy model.

        Tries the requested model first, then falls back through
        smaller models. The transformer model (en_core_web_trf) may
        segfault on ROCm PyTorch builds, so we catch that gracefully.
        """
        fallback_chain = [model_name, "en_core_web_sm", "en_core_web_md"]
        # De-duplicate while preserving order
        seen: set[str] = set()
        models_to_try = []
        for m in fallback_chain:
            if m not in seen:
                seen.add(m)
                models_to_try.append(m)

        for name in models_to_try:
            try:
                self._nlp = spacy.load(name)
                self._model_name = name
                logger.info(f"spaCy NER model loaded: {name}")
                return
            except OSError:
                logger.warning(f"spaCy model '{name}' not available, trying next")
            except Exception as e:
                logger.warning(f"spaCy model '{name}' failed to load: {e}, trying next")

        logger.error(
            "No spaCy model available. Install with: "
            "python -m spacy download en_core_web_sm"
        )
        self._nlp = None

    @property
    def is_available(self) -> bool:
        """Whether spaCy NER is ready."""
        return self._nlp is not None

    def detect(self, text: str, max_length: int = 1_000_000) -> list[PIIEntity]:
        """Detect named entities in text using spaCy NER.

        Args:
            text: Text to analyse.
            max_length: Maximum text length to process (safety limit).

        Returns:
            List of PIIEntity instances.
        """
        if not self._nlp:
            return []

        # Truncate very long texts to avoid OOM
        if len(text) > max_length:
            text = text[:max_length]

        try:
            doc = self._nlp(text)
            entities: list[PIIEntity] = []

            for ent in doc.ents:
                if ent.label_ not in RELEVANT_SPACY_LABELS:
                    continue

                category = SPACY_TO_CATEGORY.get(ent.label_, PIICategory.CUSTOM)

                # Skip very short entities (likely false positives)
                if len(ent.text.strip()) < 2:
                    continue

                # Confidence heuristic: spaCy doesn't provide confidence scores
                # directly, so we estimate based on entity type and context
                confidence = self._estimate_confidence(ent)

                # Surrounding context
                ctx_start = max(0, ent.start_char - 50)
                ctx_end = min(len(text), ent.end_char + 50)

                entities.append(PIIEntity(
                    category=category,
                    original_text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=confidence,
                    detection_layer=DetectionLayer.SPACY_NER,
                    context=text[ctx_start:ctx_end],
                    metadata={
                        "spacy_label": ent.label_,
                        "model": self._model_name,
                    },
                ))

            return entities

        except Exception as e:
            logger.error(f"spaCy NER detection failed: {e}")
            return []

    @staticmethod
    def _estimate_confidence(ent) -> float:
        """Estimate confidence for a spaCy entity.

        spaCy doesn't provide per-entity confidence, so we use heuristics:
        - PERSON entities in legal text are high confidence
        - Short entities or entities that look like common words are lower
        - Entities with title case are more likely to be genuine names
        """
        base_confidence = {
            "PERSON": 0.85,
            "ORG": 0.80,
            "GPE": 0.82,
            "LOC": 0.75,
            "FAC": 0.72,
            "DATE": 0.78,
            "MONEY": 0.90,
        }.get(ent.label_, 0.65)

        text = ent.text.strip()

        # Boost for title case (proper nouns)
        if text.istitle() and ent.label_ in ("PERSON", "ORG", "GPE"):
            base_confidence = min(1.0, base_confidence + 0.05)

        # Penalise very short entities
        if len(text) <= 3:
            base_confidence *= 0.7

        # Penalise single-word organisations (often false positives)
        if ent.label_ == "ORG" and " " not in text and len(text) < 8:
            base_confidence *= 0.8

        return round(base_confidence, 2)
