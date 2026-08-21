"""AnonymisationService — main orchestrator for the anonymisation system.

This is the primary interface for the rest of the application.
It coordinates PII detection, anonymisation, de-anonymisation,
validation, and audit logging through a single coherent API.

Usage:
    service = AnonymisationService(matter_id="abc123", passphrase="secret")

    # Anonymise a document for cloud export
    anon_doc = service.anonymise_document(text, filename="witness.pdf")

    # Anonymise Q&A output for cloud analysis
    payload = service.anonymise_qa(query, answer, chunks)

    # De-anonymise a cloud response
    real_text = service.deanonymise(cloud_response)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from loguru import logger

from .anonymiser import Anonymiser
from .deanonymiser import DeAnonymiser
from .detector import PIIDetector
from .models import (
    AnonymisedDocument,
    AnonymisedPayload,
    PIIEntity,
    ReviewStatus,
    RiskLevel,
)
from .registry import AnonymisationRegistry


# Default database path
_DEFAULT_DB_DIR = Path("data/anonymisation")


class AnonymisationService:
    """Main anonymisation service — the single entry point for all operations.

    Manages the lifecycle of:
      - PII detection (multi-layer)
      - Consistent pseudonymisation
      - De-anonymisation (local only)
      - Double-pass validation
      - Audit trail

    Args:
        matter_id: Matter/case identifier for token scoping.
        passphrase: Encryption passphrase for the registry.
        db_path: Path to the SQLite registry database.
        use_presidio: Enable Presidio detection layer.
        use_spacy: Enable spaCy NER detection layer.
        use_patterns: Enable rule-based pattern detection.
        use_llm: Enable local LLM contextual detection.
        llm_service: Optional LLM service instance for contextual detection.
        confidence_threshold: Minimum detection confidence.
        preserve_relationships: Maintain relationship structure in anonymised text.
        preserve_temporal_order: Maintain date ordering.
        location_granularity: How to handle locations (region/city/suppress).
        date_handling: How to handle dates (offset/generalise/suppress).
        age_handling: How to handle ages (band/suppress).
        require_review_for_critical: Require human review for critical risk entities.
        double_pass_validation: Run detection again on output to verify completeness.
    """

    def __init__(
        self,
        matter_id: str,
        passphrase: Optional[str] = None,
        db_path: Optional[str | Path] = None,
        use_presidio: bool = True,
        use_spacy: bool = True,
        use_patterns: bool = True,
        use_llm: bool = True,
        llm_service: Optional[Any] = None,
        confidence_threshold: float = 0.5,
        preserve_relationships: bool = True,
        preserve_temporal_order: bool = True,
        location_granularity: str = "region",
        date_handling: str = "offset",
        age_handling: str = "band",
        require_review_for_critical: bool = True,
        double_pass_validation: bool = True,
    ) -> None:
        self._matter_id = matter_id
        self._require_review = require_review_for_critical
        self._double_pass = double_pass_validation

        # Database path
        if db_path is None:
            db_path = _DEFAULT_DB_DIR / f"registry_{matter_id}.db"
        self._db_path = Path(db_path)

        # Initialise registry
        self._registry = AnonymisationRegistry(
            db_path=self._db_path,
            passphrase=passphrase,
        )

        # Initialise detector
        self._detector = PIIDetector(
            use_presidio=use_presidio,
            use_spacy=use_spacy,
            use_patterns=use_patterns,
            use_llm=use_llm,
            llm_service=llm_service,
            confidence_threshold=confidence_threshold,
        )

        # Initialise anonymiser
        self._anonymiser = Anonymiser(
            registry=self._registry,
            matter_id=matter_id,
            preserve_relationships=preserve_relationships,
            preserve_temporal_order=preserve_temporal_order,
            location_granularity=location_granularity,
            date_handling=date_handling,
            age_handling=age_handling,
        )

        # Initialise de-anonymiser
        self._deanonymiser = DeAnonymiser(registry=self._registry)

        logger.info(
            f"AnonymisationService initialised for matter '{matter_id}' "
            f"(encrypted={self._registry.is_encrypted}, "
            f"layers={self._detector.active_layers})"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def matter_id(self) -> str:
        """The matter this service is scoped to."""
        return self._matter_id

    @property
    def is_encrypted(self) -> bool:
        """Whether the registry uses encryption."""
        return self._registry.is_encrypted

    @property
    def token_count(self) -> int:
        """Number of tokens in the registry for this matter."""
        return self._registry.token_count(self._matter_id)

    @property
    def active_layers(self) -> list[str]:
        """Active detection layers."""
        return self._detector.active_layers

    @property
    def registry(self) -> AnonymisationRegistry:
        """Access the underlying registry (for audit log access etc.)."""
        return self._registry

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def anonymise_document(
        self,
        text: str,
        source_document_id: str = "",
        source_filename: str = "",
        enable_llm_detection: bool = False,
    ) -> AnonymisedDocument:
        """Anonymise a full document for cloud export.

        This is the primary method for document-level anonymisation.
        Runs multi-layer detection, applies pseudonymisation, and
        optionally validates the output with a second detection pass.

        Args:
            text: Full document text (parsed).
            source_document_id: Document ID in the catalog.
            source_filename: Original filename.
            enable_llm_detection: Use local LLM for contextual detection.

        Returns:
            AnonymisedDocument with anonymised text ready for export.
        """
        # Step 1: Detect PII
        entities = self._detector.detect(
            text,
            include_context=True,
            enable_llm_layer=enable_llm_detection,
        )

        # Step 2: Anonymise
        anon_doc = self._anonymiser.anonymise_text(
            text=text,
            entities=entities,
            source_document_id=source_document_id,
            source_filename=source_filename,
            require_review_for_critical=self._require_review,
        )

        # Step 3: Double-pass validation
        if self._double_pass and entities:
            anon_doc.validation_passed = self._validate_anonymisation(
                anon_doc.anonymised_text
            )
            if not anon_doc.validation_passed:
                logger.warning(
                    f"Double-pass validation FAILED for '{source_filename}' — "
                    "PII may remain in anonymised output. Flagging for review."
                )
                anon_doc.review_status = ReviewStatus.PENDING
        else:
            anon_doc.validation_passed = True

        # Step 4: Audit
        self._registry.log_audit(
            action="anonymise_document",
            matter_id=self._matter_id,
            document_id=source_document_id,
            entity_count=len(entities),
            details=(
                f"filename={source_filename}, "
                f"entities={len(entities)}, "
                f"validated={anon_doc.validation_passed}, "
                f"review={anon_doc.review_status.value}"
            ),
        )

        return anon_doc

    def anonymise_qa(
        self,
        query: str,
        answer: str,
        chunks: list[dict[str, Any]],
        enable_llm_detection: bool = False,
    ) -> AnonymisedPayload:
        """Anonymise a Q&A interaction for cloud export.

        Anonymises the query, answer, and all source chunks with
        consistent tokens so the cloud LLM can reason about them.

        Args:
            query: User's original query.
            answer: LLM-generated answer.
            chunks: Retrieved source chunks.
            enable_llm_detection: Use local LLM for detection.

        Returns:
            AnonymisedPayload ready for cloud export.
        """
        # Detect PII in query and answer
        query_entities = self._detector.detect(query, enable_llm_layer=enable_llm_detection)
        answer_entities = self._detector.detect(answer, enable_llm_layer=enable_llm_detection)

        # Detect PII in each chunk
        chunk_entities: dict[str, list[PIIEntity]] = {}
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            chunk_text = chunk.get("text", "")
            if chunk_text:
                chunk_entities[chunk_id] = self._detector.detect(chunk_text)

        # Anonymise everything
        result = self._anonymiser.anonymise_qa_output(
            query=query,
            answer=answer,
            chunks=chunks,
            query_entities=query_entities,
            answer_entities=answer_entities,
            chunk_entities=chunk_entities,
        )

        # Build payload
        token_legend = self._registry.get_token_legend(self._matter_id)

        total_entities = (
            len(query_entities)
            + len(answer_entities)
            + sum(len(v) for v in chunk_entities.values())
        )

        payload = AnonymisedPayload(
            matter_id=self._matter_id,
            mapping_id=f"{self._matter_id}_qa",
            anonymised_text=result["anonymised_answer"],
            anonymised_query=result["anonymised_query"],
            anonymised_chunks=result["anonymised_chunks"],
            token_legend=token_legend,
            validation_passed=True,  # Will be updated by validation
        )

        # Double-pass validation on the answer (most critical)
        if self._double_pass:
            payload.validation_passed = self._validate_anonymisation(
                payload.anonymised_text
            )

        # Audit
        self._registry.log_audit(
            action="anonymise_qa",
            matter_id=self._matter_id,
            entity_count=total_entities,
            details=(
                f"query_entities={len(query_entities)}, "
                f"answer_entities={len(answer_entities)}, "
                f"chunk_count={len(chunks)}, "
                f"validated={payload.validation_passed}"
            ),
        )

        return payload

    def deanonymise(self, text: str) -> str:
        """De-anonymise text by replacing tokens with original values.

        LOCAL-ONLY — never call this before sending data externally.

        Args:
            text: Anonymised text containing tokens.

        Returns:
            Text with original PII values restored.
        """
        return self._deanonymiser.deanonymise_text(text, self._matter_id)

    def deanonymise_cloud_response(self, response: str) -> str:
        """De-anonymise a cloud LLM response.

        Args:
            response: Cloud LLM response containing anonymisation tokens.

        Returns:
            Response with real names/values restored.
        """
        return self._deanonymiser.deanonymise_cloud_response(
            response, self._matter_id
        )

    # ------------------------------------------------------------------
    # Detection only (for preview/review)
    # ------------------------------------------------------------------

    def detect_pii(
        self,
        text: str,
        enable_llm: bool = False,
    ) -> list[PIIEntity]:
        """Detect PII entities in text without anonymising.

        Useful for preview and human review workflows.

        Args:
            text: Text to scan.
            enable_llm: Use local LLM detection.

        Returns:
            List of detected PIIEntity instances.
        """
        return self._detector.detect(text, enable_llm_layer=enable_llm)

    def get_detection_summary(self, entities: list[PIIEntity]) -> dict[str, Any]:
        """Get a summary of detected entities.

        Args:
            entities: Detected entities.

        Returns:
            Summary dict with counts by category, risk, and layer.
        """
        return self._detector.get_detection_summary(entities)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_anonymisation(self, anonymised_text: str) -> bool:
        """Run detection again on anonymised output to verify no PII remains.

        This is the defence-in-depth check. If any PII is detected in
        the already-anonymised text, anonymisation was incomplete.

        Args:
            anonymised_text: The output after anonymisation.

        Returns:
            True if no PII detected (validation passed).
        """
        residual_entities = self._detector.detect(
            anonymised_text, enable_llm_layer=False
        )

        # Filter out our own tokens (they look like entities but aren't)
        real_residual = [
            e for e in residual_entities
            if not e.original_text.startswith("[")
            or not e.original_text.endswith("]")
        ]

        if real_residual:
            logger.warning(
                f"Double-pass validation found {len(real_residual)} residual PII "
                f"entities in anonymised output: "
                f"{[e.original_text[:30] for e in real_residual[:5]]}"
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Review management
    # ------------------------------------------------------------------

    def approve_review(
        self,
        doc: AnonymisedDocument,
        reviewer: str = "user",
    ) -> AnonymisedDocument:
        """Approve an anonymised document after human review.

        Args:
            doc: AnonymisedDocument awaiting review.
            reviewer: Identifier of the reviewer.

        Returns:
            Updated AnonymisedDocument with approved status.
        """
        from datetime import datetime

        doc.review_status = ReviewStatus.APPROVED
        doc.reviewed_by = reviewer
        doc.reviewed_at = datetime.now()

        self._registry.log_audit(
            action="review_approved",
            matter_id=self._matter_id,
            document_id=doc.source_document_id,
            user=reviewer,
            details=f"Approved anonymisation of '{doc.source_filename}'",
        )

        return doc

    def reject_review(
        self,
        doc: AnonymisedDocument,
        reviewer: str = "user",
        reason: str = "",
    ) -> AnonymisedDocument:
        """Reject an anonymised document — blocks export.

        Args:
            doc: AnonymisedDocument to reject.
            reviewer: Identifier of the reviewer.
            reason: Rejection reason.

        Returns:
            Updated AnonymisedDocument with rejected status.
        """
        from datetime import datetime

        doc.review_status = ReviewStatus.REJECTED
        doc.reviewed_by = reviewer
        doc.reviewed_at = datetime.now()

        self._registry.log_audit(
            action="review_rejected",
            matter_id=self._matter_id,
            document_id=doc.source_document_id,
            user=reviewer,
            details=f"Rejected: {reason}",
            success=False,
        )

        return doc

    # ------------------------------------------------------------------
    # Registry access
    # ------------------------------------------------------------------

    def get_token_legend(self) -> dict[str, str]:
        """Get the token legend (token → category) for this matter."""
        return self._registry.get_token_legend(self._matter_id)

    def get_audit_log(self, limit: int = 100) -> list[dict]:
        """Get audit log entries for this matter."""
        return self._registry.get_audit_log(matter_id=self._matter_id, limit=limit)

    def preview_deanonymisation(self, text: str) -> list[dict[str, str]]:
        """Preview what de-anonymisation would produce.

        Args:
            text: Anonymised text.

        Returns:
            List of token → original mappings found in the text.
        """
        return self._deanonymiser.preview_deanonymisation(text, self._matter_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Clean shutdown — wipe encryption key from memory."""
        self._registry.close()
        logger.info(f"AnonymisationService closed for matter '{self._matter_id}'")
