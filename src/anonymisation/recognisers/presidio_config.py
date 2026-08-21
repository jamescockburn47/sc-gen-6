"""Presidio analyser configuration for UK legal PII detection.

Configures the Presidio AnalyzerEngine with:
  - Built-in recognisers for common PII types
  - Custom recognisers for UK-specific entities (NI numbers, postcodes, etc.)
  - Adjusted score thresholds for legal document context

NOTE: Presidio's import chain triggers a torch re-import that segfaults
on ROCm PyTorch builds. All Presidio imports are therefore LAZY — they
only happen when PresidioDetector.detect() is actually called, and the
segfault is caught by a subprocess probe on first use.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Optional

from loguru import logger

from ..models import PIICategory, PIIEntity, DetectionLayer


# Mapping from Presidio entity types to our PIICategory
PRESIDIO_TO_CATEGORY: dict[str, PIICategory] = {
    "PERSON": PIICategory.PERSON_NAME,
    "LOCATION": PIICategory.LOCATION,
    "PHONE_NUMBER": PIICategory.PHONE_NUMBER,
    "EMAIL_ADDRESS": PIICategory.EMAIL_ADDRESS,
    "DATE_TIME": PIICategory.DATE,
    "NRP": PIICategory.CUSTOM,
    "CREDIT_CARD": PIICategory.FINANCIAL_ACCOUNT,
    "IBAN_CODE": PIICategory.FINANCIAL_ACCOUNT,
    "IP_ADDRESS": PIICategory.IP_ADDRESS,
    "URL": PIICategory.URL,
    "UK_NHS": PIICategory.MEDICAL_IDENTIFIER,
    "UK_POSTCODE": PIICategory.POSTCODE,
    "UK_NI_NUMBER": PIICategory.NI_NUMBER,
    "ORGANIZATION": PIICategory.ORGANISATION,
}


def _probe_presidio_import() -> bool:
    """Test whether presidio_analyzer can be imported without segfaulting.

    Runs the import in a subprocess so a segfault doesn't kill the main process.

    Returns:
        True if Presidio imports successfully.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import presidio_analyzer; print('ok')"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return False


class PresidioDetector:
    """Wrapper around Presidio AnalyzerEngine configured for UK legal PII.

    All Presidio imports are deferred until first use to avoid ROCm segfaults.

    Args:
        language: Language code for NLP engine. Defaults to "en".
        score_threshold: Minimum Presidio score to accept. Defaults to 0.4.
    """

    def __init__(
        self,
        language: str = "en",
        score_threshold: float = 0.4,
    ) -> None:
        self._language = language
        self._score_threshold = score_threshold
        self._analyzer = None
        self._init_attempted = False
        self._available = False

    def _lazy_init(self) -> None:
        """Attempt to initialise Presidio on first use."""
        if self._init_attempted:
            return
        self._init_attempted = True

        # Probe in subprocess first
        if not _probe_presidio_import():
            logger.warning(
                "Presidio unavailable (import probe failed — likely ROCm PyTorch segfault). "
                "spaCy NER + rule patterns will provide equivalent coverage."
            )
            return

        try:
            from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            nlp_config = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
            provider = NlpEngineProvider(nlp_configuration=nlp_config)
            nlp_engine = provider.create_engine()

            self._analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine,
                supported_languages=["en"],
            )

            # Register custom UK recognisers
            ni_rec = PatternRecognizer(
                supported_entity="UK_NI_NUMBER",
                name="UK NI Number",
                patterns=[Pattern(
                    name="ni",
                    regex=r"\b(?!BG|GB|NK|KN|TN|NT|ZZ)"
                          r"[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]"
                          r"[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?[A-D]\b",
                    score=0.95,
                )],
                supported_language="en",
            )
            self._analyzer.registry.add_recognizer(ni_rec)

            pc_rec = PatternRecognizer(
                supported_entity="UK_POSTCODE",
                name="UK Postcode",
                patterns=[Pattern(
                    name="pc",
                    regex=r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b",
                    score=0.85,
                )],
                supported_language="en",
            )
            self._analyzer.registry.add_recognizer(pc_rec)

            self._available = True
            logger.info("Presidio AnalyzerEngine initialised with UK recognisers")

        except Exception as e:
            logger.warning(f"Presidio init failed: {e}")

    @property
    def is_available(self) -> bool:
        """Whether Presidio is ready for use."""
        if not self._init_attempted:
            self._lazy_init()
        return self._available

    def detect(self, text: str) -> list[PIIEntity]:
        """Detect PII entities in text using Presidio.

        Args:
            text: Text to analyse.

        Returns:
            List of PIIEntity instances.
        """
        if not self._init_attempted:
            self._lazy_init()

        if not self._analyzer:
            return []

        try:
            results = self._analyzer.analyze(
                text=text,
                language=self._language,
                score_threshold=self._score_threshold,
            )

            entities: list[PIIEntity] = []
            for result in results:
                category = PRESIDIO_TO_CATEGORY.get(
                    result.entity_type, PIICategory.CUSTOM
                )
                ctx_start = max(0, result.start - 50)
                ctx_end = min(len(text), result.end + 50)

                entities.append(PIIEntity(
                    category=category,
                    original_text=text[result.start:result.end],
                    start=result.start,
                    end=result.end,
                    confidence=result.score,
                    detection_layer=DetectionLayer.PRESIDIO,
                    context=text[ctx_start:ctx_end],
                    metadata={
                        "presidio_entity_type": result.entity_type,
                    },
                ))

            return entities

        except Exception as e:
            logger.error(f"Presidio detection failed: {e}")
            return []
