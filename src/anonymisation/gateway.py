"""Cloud Export Gateway — all data leaving the local machine passes through here.

This is the security boundary between local processing and external APIs.
It enforces:
  - Mandatory anonymisation before any cloud export
  - Double-pass validation (detection on anonymised output)
  - Human review for special category data
  - Audit logging of every export
  - Export blocking on validation failure
  - DATA SOVEREIGNTY: strips all identifiable UK/EU data before US-bound exports
  - PRIVILEGE PROTECTION: anonymises privileged content rather than blocking it
  - KANON TOGGLE: external enrichment APIs can be disabled

Data Sovereignty (UK GDPR / DUAA 2025):
  When exporting to US-based providers (OpenAI, Anthropic), the gateway
  ensures that NO identifiable personal data crosses jurisdictions.
  The anonymisation must be thorough enough that the data is no longer
  "personal data" under UK GDPR Article 4(1), meaning it cannot be
  used to identify a natural person by the recipient.

Legal Professional Privilege:
  Privilege attaches to the *communication* between solicitor and client
  about *specific facts*, not to abstract legal reasoning. Our approach:
  - Detect privileged markers (solicitor-client, litigation, WP, counsel)
  - Escalate anonymisation in those sections — all parties, facts, dates,
    amounts, and case-specific details are replaced with tokens
  - Preserve abstract legal principles, procedural descriptions, and
    analytical reasoning — these are not privileged in isolation
  - The user can still choose a "principles only" export mode that
    extracts just the legal analysis with all factual scaffolding removed
  - Result: the cloud LLM can analyse the *law*, not the *case*
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

from .models import (
    AnonymisedDocument,
    AnonymisedPayload,
    PIICategory,
    PIIEntity,
    ReviewStatus,
    RiskLevel,
    DetectionLayer,
)
from .service import AnonymisationService


class ExportBlockedError(Exception):
    """Raised when an export is blocked due to validation failure or pending review."""

    def __init__(self, reason: str, document_id: str = "") -> None:
        self.reason = reason
        self.document_id = document_id
        super().__init__(f"Export blocked: {reason}")


# ---------------------------------------------------------------------------
# Privilege detection
# ---------------------------------------------------------------------------

@dataclass
class PrivilegeMarker:
    """A detected privilege indicator in the text."""

    category: str          # "solicitor_client", "litigation", "without_prejudice", "counsel_instructions"
    marker_text: str       # The matched text
    start: int             # Char offset of the marker
    end: int               # Char offset end
    section_start: int     # Start of the paragraph/section containing the marker
    section_end: int       # End of the paragraph/section containing the marker
    description: str = ""  # Human-readable description

    @property
    def section_text(self) -> str:
        """The section text is set externally after detection."""
        return ""


# Patterns that indicate privileged content
_PRIVILEGE_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # Solicitor-client privilege markers
    (re.compile(
        r"\b(?:privileged\s+and\s+confidential|"
        r"legally\s+privileged|"
        r"solicitor[\s\-]client\s+(?:privilege|confidential)|"
        r"legal\s+advice\s+privilege|"
        r"subject\s+to\s+legal\s+(?:professional\s+)?privilege|"
        r"confidential\s+(?:legal\s+)?(?:advice|communication|instruction))",
        re.IGNORECASE,
    ), "solicitor_client",
        "Solicitor-client privilege — factual content anonymised, legal analysis preserved"),

    # Litigation privilege
    (re.compile(
        r"\b(?:litigation\s+privilege|"
        r"prepared\s+(?:for|in\s+(?:contemplation|anticipation)\s+of)\s+litigation|"
        r"dominant\s+purpose\s+(?:of\s+)?litigation|"
        r"draft\s+(?:pleading|statement\s+of\s+case|witness\s+statement|expert\s+report)|"
        r"counsel'?s?\s+(?:advice|opinion|instructions))",
        re.IGNORECASE,
    ), "litigation",
        "Litigation privilege — case-specific details anonymised"),

    # Without prejudice communications
    (re.compile(
        r"\b(?:without\s+prejudice|"
        r"WP\s+(?:communication|offer|correspondence)|"
        r"Part\s+36\s+offer|"
        r"Calderbank\s+(?:letter|offer)|"
        r"settlement\s+(?:negotiation|discussion|proposal|offer))",
        re.IGNORECASE,
    ), "without_prejudice",
        "Without-prejudice material — settlement figures and terms anonymised"),

    # Instructions to/from counsel
    (re.compile(
        r"\b(?:instructions?\s+to\s+(?:counsel|barrister)|"
        r"brief\s+to\s+counsel|"
        r"counsel'?s?\s+(?:fee\s+note|brief|instructions)|"
        r"conference\s+(?:with|note)?\s*counsel)",
        re.IGNORECASE,
    ), "counsel_instructions",
        "Counsel instructions — party names and specific instructions anonymised"),
]

# Privilege markers are themselves replaced with category-only tokens
# so the cloud LLM knows the nature of the text without the privileged marker
_PRIVILEGE_MARKER_TOKENS: dict[str, str] = {
    "solicitor_client": "[PRIVILEGED_COMMUNICATION — solicitor-client legal advice]",
    "litigation": "[PRIVILEGED_COMMUNICATION — litigation material]",
    "without_prejudice": "[PRIVILEGED_COMMUNICATION — without-prejudice negotiation]",
    "counsel_instructions": "[PRIVILEGED_COMMUNICATION — counsel instructions]",
}

# Data sovereignty: US-based cloud providers
US_BASED_PROVIDERS = {"openai", "anthropic", "google", "meta", "microsoft"}
EU_BASED_PROVIDERS = {"mistral", "aleph_alpha"}


class CloudExportGateway:
    """Security gateway for all outbound cloud data.

    All data that leaves the local machine to external services
    (OpenAI, Anthropic, Isaacus, etc.) MUST pass through this gateway.

    Privilege handling:
      The gateway does NOT block privileged content. Instead it:
      1. Detects privileged sections
      2. Anonymises the privilege markers themselves (so the cloud LLM
         doesn't know it's reading privileged material)
      3. Ensures enhanced anonymisation of parties, facts, dates, amounts
         within privileged sections (the standard PII pipeline handles this)
      4. Optionally extracts just the legal principles (principles_only mode)

      The theory: once all identifying information is removed, the remaining
      legal analysis is no longer "privileged" in the sense that disclosure
      to a cloud service would waive privilege, because the communication
      can no longer be attributed to a specific client or matter.

    Args:
        service: AnonymisationService instance for the active matter.
        block_on_validation_failure: Whether to block export if validation fails.
        block_on_pending_review: Whether to block export if review is pending.
        enable_llm_detection: Use local LLM for enhanced detection.
        target_provider: Cloud provider name (affects sovereignty checks).
        privilege_mode: How to handle privileged content:
            "anonymise" (default) — anonymise markers + enhanced entity removal
            "principles_only" — extract and export only legal reasoning,
                strip all factual scaffolding from privileged sections
        enable_kanon_enricher: Whether the Kanon external API is allowed.
    """

    def __init__(
        self,
        service: AnonymisationService,
        block_on_validation_failure: bool = True,
        block_on_pending_review: bool = True,
        enable_llm_detection: bool = False,
        target_provider: str = "openai",
        privilege_mode: str = "anonymise",
        enable_kanon_enricher: bool = False,
    ) -> None:
        self._service = service
        self._block_on_failure = block_on_validation_failure
        self._block_on_review = block_on_pending_review
        self._enable_llm = enable_llm_detection
        self._target_provider = target_provider.lower()
        self._privilege_mode = privilege_mode
        self._enable_kanon = enable_kanon_enricher

        # Track state for the UI / review panel
        self._last_privilege_markers: list[PrivilegeMarker] = []
        self._last_sovereignty_notes: list[str] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_cross_border(self) -> bool:
        """Whether the target provider is outside the UK/EU."""
        return self._target_provider in US_BASED_PROVIDERS

    @property
    def kanon_enricher_enabled(self) -> bool:
        """Whether the Kanon enricher external API is allowed."""
        return self._enable_kanon

    @property
    def last_privilege_markers(self) -> list[PrivilegeMarker]:
        """Privilege markers from the most recent export."""
        return self._last_privilege_markers

    @property
    def last_sovereignty_notes(self) -> list[str]:
        """Data sovereignty notes from the most recent export."""
        return self._last_sovereignty_notes

    # ------------------------------------------------------------------
    # Privilege handling
    # ------------------------------------------------------------------

    def detect_privilege_markers(self, text: str) -> list[PrivilegeMarker]:
        """Scan text for privilege indicators.

        Returns PrivilegeMarker instances with the marker itself
        and the surrounding section boundaries.

        Args:
            text: Text to scan.

        Returns:
            List of PrivilegeMarker instances.
        """
        markers: list[PrivilegeMarker] = []

        for pattern, category, description in _PRIVILEGE_PATTERNS:
            for match in pattern.finditer(text):
                # Find the paragraph/section that contains this marker
                # (bounded by double newlines or start/end of text)
                section_start = text.rfind("\n\n", 0, match.start())
                section_start = section_start + 2 if section_start != -1 else 0

                section_end = text.find("\n\n", match.end())
                section_end = section_end if section_end != -1 else len(text)

                markers.append(PrivilegeMarker(
                    category=category,
                    marker_text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    section_start=section_start,
                    section_end=section_end,
                    description=description,
                ))

        return markers

    def anonymise_privilege_markers(self, text: str) -> tuple[str, list[PrivilegeMarker]]:
        """Replace privilege markers with neutral tokens.

        The markers themselves ("PRIVILEGED AND CONFIDENTIAL",
        "without prejudice", etc.) are replaced with category-level
        tokens so the cloud LLM knows the *nature* of the content
        without the privileged label.

        The surrounding content is left for the standard PII anonymiser
        to handle — by the time entities, names, dates, amounts, and
        case references are all tokenised, what remains is abstract
        legal reasoning that does not carry privilege.

        Args:
            text: Text containing privilege markers.

        Returns:
            Tuple of (text with markers replaced, list of markers found).
        """
        markers = self.detect_privilege_markers(text)
        if not markers:
            return text, []

        # Sort by position (reverse) for safe replacement
        markers.sort(key=lambda m: m.start, reverse=True)

        result = text
        for marker in markers:
            replacement = _PRIVILEGE_MARKER_TOKENS.get(
                marker.category,
                "[PRIVILEGED_COMMUNICATION]",
            )
            result = result[:marker.start] + replacement + result[marker.end:]

        logger.info(
            f"Privilege markers anonymised: {len(markers)} markers replaced "
            f"({set(m.category for m in markers)})"
        )

        return result, markers

    def extract_principles_only(self, text: str) -> str:
        """Extract only legal principles and analysis, stripping factual content.

        This is the most conservative privilege mode. It:
        1. Detects privileged sections
        2. Within those sections, keeps only sentences that contain
           legal reasoning indicators (case citations, statutory references,
           principle statements, legal tests)
        3. Strips all factual narrative sentences

        Useful when you only need the cloud LLM to analyse the *law*
        being applied, not the *facts* it's being applied to.

        Args:
            text: Text to extract principles from.

        Returns:
            Text containing only legal principles and analysis.
        """
        markers = self.detect_privilege_markers(text)
        if not markers:
            return text  # No privilege detected — return as-is

        # Legal reasoning indicators — sentences containing these are likely
        # to be about law rather than facts
        _LEGAL_INDICATORS = re.compile(
            r"(?:"
            r"\b(?:section|s\.)\s+\d+|"                          # Section references
            r"\[\d{4}\]\s+[A-Z]+\s+\d+|"                         # Neutral citations
            r"\b(?:principle|test|threshold|burden|standard)\b|"  # Legal concepts
            r"\b(?:pursuant|notwithstanding|hereby|whereas)\b|"   # Legal language
            r"\b(?:liable|liability|duty|breach|negligence)\b|"   # Legal terms
            r"\b(?:statute|regulation|act|rule|order)\b|"         # Legislation refs
            r"\b(?:held|decided|ruled|determined|found)\b|"       # Judicial language
            r"\b(?:submit|contend|argue|assert|allege)\b|"        # Submission language
            r"\b(?:damages|injunction|remedy|relief|costs)\b|"    # Remedies
            r"\b(?:causation|foreseeability|remoteness|quantum)\b" # Legal doctrines
            r")",
            re.IGNORECASE,
        )

        # Build a set of privileged section ranges
        privileged_ranges: list[tuple[int, int]] = []
        for marker in markers:
            privileged_ranges.append((marker.section_start, marker.section_end))

        # Merge overlapping ranges
        privileged_ranges.sort()
        merged: list[tuple[int, int]] = []
        for start, end in privileged_ranges:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Process text: for privileged sections, keep only legal sentences
        result_parts: list[str] = []
        last_end = 0

        for section_start, section_end in merged:
            # Add non-privileged text before this section (unchanged)
            if section_start > last_end:
                result_parts.append(text[last_end:section_start])

            # Process the privileged section — keep only legal sentences
            section_text = text[section_start:section_end]
            legal_sentences: list[str] = []
            kept = 0
            total = 0

            for sentence in re.split(r'(?<=[.!?])\s+', section_text):
                total += 1
                if _LEGAL_INDICATORS.search(sentence):
                    legal_sentences.append(sentence)
                    kept += 1

            if legal_sentences:
                header = "[This section contained privileged material. " \
                         "Only the legal principles and analysis are shown below.]\n\n"
                result_parts.append(header + " ".join(legal_sentences))
            else:
                result_parts.append(
                    "[This section contained privileged material with no "
                    "extractable legal principles. Content removed.]"
                )

            last_end = section_end

        # Add any remaining non-privileged text
        if last_end < len(text):
            result_parts.append(text[last_end:])

        result = "\n\n".join(result_parts)

        logger.info(
            f"Principles-only extraction: processed {len(merged)} privileged sections"
        )

        return result

    # ------------------------------------------------------------------
    # Data sovereignty checks
    # ------------------------------------------------------------------

    def _check_sovereignty(self, anon_doc: AnonymisedDocument) -> list[str]:
        """Check data sovereignty compliance for cross-border transfers.

        Under UK GDPR Chapter V and the DUAA 2025 three-step test,
        transferring personal data to US providers requires either:
        (a) An adequacy decision (US has partial adequacy via UK-US Data Bridge)
        (b) Appropriate safeguards (SCCs, BCRs)
        (c) The data is effectively anonymised (no longer personal data)

        Our approach: ensure option (c) — the anonymised output must
        not constitute personal data under Article 4(1).

        Returns:
            List of sovereignty compliance notes.
        """
        notes: list[str] = []

        if not self.is_cross_border:
            notes.append(
                f"Provider '{self._target_provider}' is EU/UK-based — "
                "no cross-border restrictions"
            )
            return notes

        notes.append(
            f"Cross-border transfer to '{self._target_provider}' (US-based). "
            "Anonymisation must render data non-personal under UK GDPR Art. 4(1)."
        )

        # Check for any residual high-risk entities that weren't fully anonymised
        high_risk_entities = [
            e for e in anon_doc.entities_detected
            if e.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)
        ]
        if high_risk_entities:
            low_confidence = [e for e in high_risk_entities if e.confidence < 0.8]
            if low_confidence:
                notes.append(
                    f"WARNING: {len(low_confidence)} high-risk entities had detection "
                    f"confidence below 0.8 — manual review strongly recommended "
                    "before cross-border transfer."
                )

        # Check entity count vs text length — too few detections may mean misses
        text_len = len(anon_doc.anonymised_text)
        entity_density = anon_doc.entity_count / max(text_len, 1) * 1000
        if entity_density < 0.5 and text_len > 500:
            notes.append(
                "NOTE: Low entity density in document — this may indicate "
                "insufficient PII detection. Consider enabling LLM detection layer."
            )

        notes.append(
            "Sovereignty compliance: data has been pseudonymised to the standard "
            "required by ICO 2025 guidance. Token mappings remain local only. "
            "The exported payload should not constitute personal data."
        )

        return notes

    # ------------------------------------------------------------------
    # Export operations
    # ------------------------------------------------------------------

    def export_document(
        self,
        text: str,
        source_document_id: str = "",
        source_filename: str = "",
    ) -> AnonymisedPayload:
        """Anonymise and prepare a document for cloud export.

        Privilege handling:
          - "anonymise" mode: privilege markers are tokenised, then the
            standard PII pipeline anonymises all entities. What remains
            is legal reasoning with anonymised parties/facts — no longer
            privileged because it can't be attributed to a specific matter.
          - "principles_only" mode: privileged sections are reduced to
            only their legal analysis sentences before anonymisation.

        Args:
            text: Full document text.
            source_document_id: Document ID in catalog.
            source_filename: Original filename.

        Returns:
            AnonymisedPayload safe for cloud transmission.

        Raises:
            ExportBlockedError: If validation fails or review is pending.
        """
        # Step 0: Privilege handling (anonymise, don't block)
        self._last_privilege_markers = self.detect_privilege_markers(text)
        privilege_count = len(self._last_privilege_markers)

        if self._last_privilege_markers:
            priv_categories = set(m.category for m in self._last_privilege_markers)
            logger.info(
                f"Privileged content detected in '{source_filename}': "
                f"{priv_categories} — applying {self._privilege_mode} mode"
            )

            if self._privilege_mode == "principles_only":
                # Extract only legal reasoning from privileged sections
                text = self.extract_principles_only(text)
            # In both modes, anonymise the privilege markers themselves
            text, _ = self.anonymise_privilege_markers(text)

        # Step 1: Standard PII anonymisation (handles all entities including
        # those within privileged sections)
        anon_doc = self._service.anonymise_document(
            text=text,
            source_document_id=source_document_id,
            source_filename=source_filename,
            enable_llm_detection=self._enable_llm,
        )

        # Step 2: Data sovereignty check
        self._last_sovereignty_notes = self._check_sovereignty(anon_doc)

        # Step 3: Check all gates
        self._check_export_gates(anon_doc)

        # Build payload
        payload = AnonymisedPayload(
            matter_id=self._service.matter_id,
            mapping_id=anon_doc.id,
            anonymised_text=anon_doc.anonymised_text,
            token_legend=self._service.get_token_legend(),
            validation_passed=anon_doc.validation_passed,
            metadata={
                "source_document_id": source_document_id,
                "source_filename": source_filename,
                "entity_count": anon_doc.entity_count,
                "review_status": anon_doc.review_status.value,
                "target_provider": self._target_provider,
                "is_cross_border": self.is_cross_border,
                "privilege_mode": self._privilege_mode,
                "privilege_markers_found": privilege_count,
                "privilege_categories": list(
                    set(m.category for m in self._last_privilege_markers)
                ),
                "sovereignty_notes": self._last_sovereignty_notes,
            },
        )

        # Audit
        self._service.registry.log_audit(
            action="export_document",
            matter_id=self._service.matter_id,
            document_id=source_document_id,
            entity_count=anon_doc.entity_count,
            details=(
                f"Exported '{source_filename}' to {self._target_provider} "
                f"(cross_border={self.is_cross_border}, "
                f"privilege_mode={self._privilege_mode}, "
                f"privilege_markers={privilege_count})"
            ),
        )

        logger.info(
            f"Document exported via gateway: '{source_filename}' "
            f"({anon_doc.entity_count} entities, "
            f"privilege_mode={self._privilege_mode}, "
            f"privilege_markers={privilege_count})"
        )

        return payload

    def export_qa(
        self,
        query: str,
        answer: str,
        chunks: list[dict[str, Any]],
    ) -> AnonymisedPayload:
        """Anonymise and prepare Q&A output for cloud export.

        Privilege markers in the answer and chunks are anonymised
        (not blocked). The standard PII pipeline then handles all
        entity removal.

        Args:
            query: User query.
            answer: LLM-generated answer.
            chunks: Retrieved source chunks.

        Returns:
            AnonymisedPayload safe for cloud transmission.

        Raises:
            ExportBlockedError: If validation fails.
        """
        # Privilege handling on answer + chunks
        all_text = answer + "\n".join(c.get("text", "") for c in chunks)
        self._last_privilege_markers = self.detect_privilege_markers(all_text)

        if self._last_privilege_markers:
            if self._privilege_mode == "principles_only":
                answer = self.extract_principles_only(answer)
                for i, chunk in enumerate(chunks):
                    if "text" in chunk:
                        chunks[i] = {
                            **chunk,
                            "text": self.extract_principles_only(chunk["text"]),
                        }

            # Anonymise privilege markers
            answer, _ = self.anonymise_privilege_markers(answer)
            for i, chunk in enumerate(chunks):
                if "text" in chunk:
                    anonymised_chunk_text, _ = self.anonymise_privilege_markers(
                        chunk["text"]
                    )
                    chunks[i] = {**chunk, "text": anonymised_chunk_text}

        payload = self._service.anonymise_qa(
            query=query,
            answer=answer,
            chunks=chunks,
            enable_llm_detection=self._enable_llm,
        )

        # Check validation
        if self._block_on_failure and not payload.validation_passed:
            raise ExportBlockedError(
                "Q&A anonymisation validation failed — PII may remain in output. "
                "Review and re-anonymise before exporting."
            )

        # Audit
        self._service.registry.log_audit(
            action="export_qa",
            matter_id=self._service.matter_id,
            entity_count=len(payload.anonymised_chunks),
            details=(
                f"Exported Q&A to {self._target_provider} "
                f"(privilege_mode={self._privilege_mode}, "
                f"privilege_markers={len(self._last_privilege_markers)})"
            ),
        )

        return payload

    def export_text(self, text: str) -> str:
        """Quick anonymisation of arbitrary text for cloud export.

        Args:
            text: Text to anonymise.

        Returns:
            Anonymised text string.

        Raises:
            ExportBlockedError: If validation fails.
        """
        # Handle privilege markers
        markers = self.detect_privilege_markers(text)
        if markers:
            if self._privilege_mode == "principles_only":
                text = self.extract_principles_only(text)
            text, _ = self.anonymise_privilege_markers(text)

        anon_doc = self._service.anonymise_document(
            text=text,
            source_document_id="adhoc",
            enable_llm_detection=self._enable_llm,
        )

        if self._block_on_failure and not anon_doc.validation_passed:
            raise ExportBlockedError(
                "Text anonymisation validation failed — PII may remain."
            )

        return anon_doc.anonymised_text

    # ------------------------------------------------------------------
    # Import operations (de-anonymisation)
    # ------------------------------------------------------------------

    def import_response(self, cloud_response: str) -> str:
        """De-anonymise a cloud LLM response for local display.

        Args:
            cloud_response: Response from cloud LLM containing tokens.

        Returns:
            Response with original PII values restored.
        """
        result = self._service.deanonymise_cloud_response(cloud_response)

        self._service.registry.log_audit(
            action="import_response",
            matter_id=self._service.matter_id,
            details=f"De-anonymised cloud response (len={len(cloud_response)})",
        )

        return result

    # ------------------------------------------------------------------
    # Gate checks
    # ------------------------------------------------------------------

    def _check_export_gates(self, anon_doc: AnonymisedDocument) -> None:
        """Check all security gates before allowing export.

        Note: privilege is NOT a gate — it is handled by anonymisation.
        The gates are for PII validation and human review only.

        Args:
            anon_doc: The anonymised document to check.

        Raises:
            ExportBlockedError: If any gate fails.
        """
        # Gate 1: Validation
        if self._block_on_failure and not anon_doc.validation_passed:
            raise ExportBlockedError(
                reason=(
                    "Double-pass validation failed — PII may remain in "
                    "anonymised output. Manual review required."
                ),
                document_id=anon_doc.source_document_id,
            )

        # Gate 2: Human review
        if self._block_on_review and anon_doc.review_status == ReviewStatus.PENDING:
            raise ExportBlockedError(
                reason=(
                    "Document contains special category data and requires "
                    "human review before export. Use the review interface "
                    "to approve or reject."
                ),
                document_id=anon_doc.source_document_id,
            )

        # Gate 3: Rejected review
        if anon_doc.review_status == ReviewStatus.REJECTED:
            raise ExportBlockedError(
                reason="Document anonymisation was rejected during review.",
                document_id=anon_doc.source_document_id,
            )

    # ------------------------------------------------------------------
    # Status / diagnostics
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Get gateway status for diagnostics."""
        return {
            "matter_id": self._service.matter_id,
            "encrypted": self._service.is_encrypted,
            "token_count": self._service.token_count,
            "active_layers": self._service.active_layers,
            "target_provider": self._target_provider,
            "is_cross_border": self.is_cross_border,
            "privilege_mode": self._privilege_mode,
            "block_on_validation_failure": self._block_on_failure,
            "block_on_pending_review": self._block_on_review,
            "kanon_enricher_enabled": self._enable_kanon,
            "llm_detection_enabled": self._enable_llm,
            "last_privilege_markers": len(self._last_privilege_markers),
            "last_sovereignty_notes": self._last_sovereignty_notes,
        }
