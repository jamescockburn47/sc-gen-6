"""Data models for the anonymisation system.

Defines PII entity types, risk levels, anonymisation tokens,
and the anonymised document wrapper used throughout the pipeline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PIICategory(str, Enum):
    """Categories of personally identifiable information."""

    PERSON_NAME = "person_name"
    DATE_OF_BIRTH = "date_of_birth"
    AGE = "age"
    ADDRESS = "address"
    POSTCODE = "postcode"
    PHONE_NUMBER = "phone_number"
    EMAIL_ADDRESS = "email_address"
    NI_NUMBER = "ni_number"            # National Insurance number
    FINANCIAL_ACCOUNT = "financial_account"
    CASE_REFERENCE = "case_reference"
    COURT_NAME = "court_name"
    JUDGE_NAME = "judge_name"
    SOLICITOR_NAME = "solicitor_name"
    BARRISTER_NAME = "barrister_name"
    WITNESS_NAME = "witness_name"
    VICTIM_IDENTIFIER = "victim_identifier"
    PERPETRATOR_IDENTIFIER = "perpetrator_identifier"
    LOCATION = "location"
    SCHOOL_NAME = "school_name"
    INSTITUTION_NAME = "institution_name"  # Care homes, hospitals, etc.
    RELATIONSHIP_DESCRIPTOR = "relationship_descriptor"
    MEDICAL_IDENTIFIER = "medical_identifier"
    MONETARY_AMOUNT = "monetary_amount"
    DATE = "date"
    ORGANISATION = "organisation"
    SRA_NUMBER = "sra_number"          # Solicitor Regulation Authority
    BAR_NUMBER = "bar_number"
    VEHICLE_REG = "vehicle_reg"
    PASSPORT_NUMBER = "passport_number"
    IP_ADDRESS = "ip_address"
    URL = "url"
    CUSTOM = "custom"


class RiskLevel(str, Enum):
    """Risk level for PII entities — drives review requirements."""

    CRITICAL = "critical"    # Victim/perpetrator identifiers in abuse cases
    HIGH = "high"            # Addresses, witness names, locations
    MEDIUM = "medium"        # Legal professionals, case refs, dates
    LOW = "low"              # Generic organisation names, URLs


class AnonymisationMethod(str, Enum):
    """How an entity was anonymised."""

    TOKENISATION = "tokenisation"        # Replaced with consistent token
    GENERALISATION = "generalisation"    # Reduced specificity
    SUPPRESSION = "suppression"          # Completely removed
    PERTURBATION = "perturbation"        # Value slightly altered


class DetectionLayer(str, Enum):
    """Which detection layer found the entity."""

    PRESIDIO = "presidio"
    SPACY_NER = "spacy_ner"
    RULE_PATTERN = "rule_pattern"
    LOCAL_LLM = "local_llm"
    HUMAN_REVIEW = "human_review"


class ReviewStatus(str, Enum):
    """Status of human review for an anonymisation decision."""

    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"


# ---------------------------------------------------------------------------
# Risk level mapping per PII category
# ---------------------------------------------------------------------------

CATEGORY_RISK_MAP: dict[PIICategory, RiskLevel] = {
    PIICategory.VICTIM_IDENTIFIER: RiskLevel.CRITICAL,
    PIICategory.PERPETRATOR_IDENTIFIER: RiskLevel.CRITICAL,
    PIICategory.PERSON_NAME: RiskLevel.HIGH,
    PIICategory.WITNESS_NAME: RiskLevel.HIGH,
    PIICategory.DATE_OF_BIRTH: RiskLevel.HIGH,
    PIICategory.AGE: RiskLevel.HIGH,
    PIICategory.ADDRESS: RiskLevel.HIGH,
    PIICategory.NI_NUMBER: RiskLevel.HIGH,
    PIICategory.SCHOOL_NAME: RiskLevel.HIGH,
    PIICategory.INSTITUTION_NAME: RiskLevel.HIGH,
    PIICategory.RELATIONSHIP_DESCRIPTOR: RiskLevel.HIGH,
    PIICategory.MEDICAL_IDENTIFIER: RiskLevel.HIGH,
    PIICategory.PHONE_NUMBER: RiskLevel.HIGH,
    PIICategory.EMAIL_ADDRESS: RiskLevel.HIGH,
    PIICategory.PASSPORT_NUMBER: RiskLevel.HIGH,
    PIICategory.POSTCODE: RiskLevel.MEDIUM,
    PIICategory.LOCATION: RiskLevel.MEDIUM,
    PIICategory.JUDGE_NAME: RiskLevel.MEDIUM,
    PIICategory.SOLICITOR_NAME: RiskLevel.MEDIUM,
    PIICategory.BARRISTER_NAME: RiskLevel.MEDIUM,
    PIICategory.CASE_REFERENCE: RiskLevel.MEDIUM,
    PIICategory.COURT_NAME: RiskLevel.MEDIUM,
    PIICategory.DATE: RiskLevel.MEDIUM,
    PIICategory.MONETARY_AMOUNT: RiskLevel.MEDIUM,
    PIICategory.FINANCIAL_ACCOUNT: RiskLevel.MEDIUM,
    PIICategory.SRA_NUMBER: RiskLevel.MEDIUM,
    PIICategory.BAR_NUMBER: RiskLevel.MEDIUM,
    PIICategory.VEHICLE_REG: RiskLevel.MEDIUM,
    PIICategory.ORGANISATION: RiskLevel.LOW,
    PIICategory.IP_ADDRESS: RiskLevel.LOW,
    PIICategory.URL: RiskLevel.LOW,
    PIICategory.CUSTOM: RiskLevel.MEDIUM,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PIIEntity:
    """A single detected PII entity in text."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    category: PIICategory = PIICategory.PERSON_NAME
    original_text: str = ""
    start: int = 0                     # Character offset in source text
    end: int = 0                       # Character offset end
    confidence: float = 0.0            # Detection confidence [0.0, 1.0]
    detection_layer: DetectionLayer = DetectionLayer.PRESIDIO
    risk_level: RiskLevel = RiskLevel.MEDIUM
    context: str = ""                  # Surrounding text snippet for review
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Derive risk level from category if not explicitly set."""
        if self.risk_level == RiskLevel.MEDIUM and self.category in CATEGORY_RISK_MAP:
            self.risk_level = CATEGORY_RISK_MAP[self.category]


@dataclass
class AnonymisationToken:
    """Mapping between an original value and its anonymised replacement."""

    token_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    matter_id: str = ""                # Isolation per matter
    category: PIICategory = PIICategory.PERSON_NAME
    original_value: str = ""           # The real PII (encrypted at rest)
    anonymised_value: str = ""         # e.g. [PERSON_001]
    method: AnonymisationMethod = AnonymisationMethod.TOKENISATION
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnonymisedDocument:
    """A document or text that has been anonymised, with full provenance."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    matter_id: str = ""
    source_document_id: str = ""       # Original document ID in catalog
    source_filename: str = ""          # Original filename
    original_text: str = ""            # Pre-anonymisation text (local only, never exported)
    anonymised_text: str = ""          # Post-anonymisation text (safe for cloud)
    entities_detected: list[PIIEntity] = field(default_factory=list)
    tokens_applied: list[AnonymisationToken] = field(default_factory=list)
    detection_summary: dict[str, int] = field(default_factory=dict)  # {category: count}
    review_status: ReviewStatus = ReviewStatus.NOT_REQUIRED
    reviewed_by: str = ""              # Reviewer identifier
    reviewed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    validation_passed: bool = False    # Double-pass validation result
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def entity_count(self) -> int:
        """Total detected PII entities."""
        return len(self.entities_detected)

    @property
    def has_critical_entities(self) -> bool:
        """Whether any CRITICAL risk entities were found."""
        return any(e.risk_level == RiskLevel.CRITICAL for e in self.entities_detected)

    @property
    def requires_review(self) -> bool:
        """Whether human review is required before export."""
        return self.review_status == ReviewStatus.PENDING

    def to_dict(self) -> dict[str, Any]:
        """Serialise for audit/storage (excludes original_text for safety)."""
        return {
            "id": self.id,
            "matter_id": self.matter_id,
            "source_document_id": self.source_document_id,
            "source_filename": self.source_filename,
            "entity_count": self.entity_count,
            "has_critical_entities": self.has_critical_entities,
            "detection_summary": self.detection_summary,
            "review_status": self.review_status.value,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "created_at": self.created_at.isoformat(),
            "validation_passed": self.validation_passed,
            "metadata": self.metadata,
        }


@dataclass
class AnonymisedPayload:
    """Payload ready for cloud export — contains ONLY anonymised data."""

    payload_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    matter_id: str = ""
    mapping_id: str = ""               # Registry key for de-anonymisation
    anonymised_text: str = ""          # The safe-to-export text
    anonymised_query: str = ""         # If Q&A export, the anonymised query
    anonymised_chunks: list[dict[str, Any]] = field(default_factory=list)
    token_legend: dict[str, str] = field(default_factory=dict)  # {[PERSON_001]: "Category: person_name"}
    export_timestamp: datetime = field(default_factory=datetime.now)
    validation_passed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for cloud export."""
        return {
            "payload_id": self.payload_id,
            "anonymised_text": self.anonymised_text,
            "anonymised_query": self.anonymised_query,
            "anonymised_chunks": self.anonymised_chunks,
            "token_legend": self.token_legend,
            "export_timestamp": self.export_timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class AuditEntry:
    """Single entry in the anonymisation audit trail."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = field(default_factory=datetime.now)
    action: str = ""                   # "anonymise", "deanonymise", "review", "export"
    matter_id: str = ""
    document_id: str = ""
    entity_count: int = 0
    user: str = "system"
    details: str = ""
    success: bool = True
