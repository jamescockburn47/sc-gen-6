"""SC Gen 6 Anonymisation System.

Provides ICO 2025 / UK GDPR compliant pseudonymisation for exporting
documents and LLM output to cloud services. Designed for litigation
support involving sensitive data including sexual abuse cases.

Architecture:
  - Multi-layer PII detection (Presidio, spaCy NER, rules, local LLM)
  - Consistent pseudonymisation with encrypted token registry
  - Reversible de-anonymisation (local only)
  - Double-pass validation (defence in depth)
  - Human-in-the-loop review for special category data
  - Full audit trail for compliance

Usage:
    from src.anonymisation import AnonymisationService, CloudExportGateway

    service = AnonymisationService(
        matter_id="abc123",
        passphrase="your-secure-passphrase",
    )

    # Anonymise a full document
    anon_doc = service.anonymise_document(text, source_filename="witness.pdf")

    # Export via gateway (enforces all security gates)
    gateway = CloudExportGateway(service)
    payload = gateway.export_document(text, source_filename="witness.pdf")

    # De-anonymise a cloud response
    real_text = service.deanonymise_cloud_response(cloud_response)

    # Clean up (wipes encryption key from memory)
    service.close()
"""

from .service import AnonymisationService
from .gateway import CloudExportGateway, ExportBlockedError
from .models import (
    AnonymisationMethod,
    AnonymisationToken,
    AnonymisedDocument,
    AnonymisedPayload,
    AuditEntry,
    DetectionLayer,
    PIICategory,
    PIIEntity,
    ReviewStatus,
    RiskLevel,
)

__all__ = [
    "AnonymisationService",
    "CloudExportGateway",
    "ExportBlockedError",
    "AnonymisationMethod",
    "AnonymisationToken",
    "AnonymisedDocument",
    "AnonymisedPayload",
    "AuditEntry",
    "DetectionLayer",
    "PIICategory",
    "PIIEntity",
    "ReviewStatus",
    "RiskLevel",
]
