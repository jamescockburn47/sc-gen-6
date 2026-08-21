"""Human review queue for anonymisation decisions.

Manages a queue of AnonymisedDocuments that require human review
before they can be exported to cloud services. This is mandatory
for special category data (abuse cases, medical records, etc.)
per ICO 2025 guidance.

The queue is persistent (SQLite-backed via the registry audit log)
so review state survives application restarts.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from datetime import datetime
from typing import Any, Callable, Optional

from loguru import logger

from ..models import (
    AnonymisedDocument,
    PIIEntity,
    ReviewStatus,
    RiskLevel,
)
from ..registry import AnonymisationRegistry


class ReviewItem:
    """A single item in the review queue."""

    def __init__(
        self,
        document: AnonymisedDocument,
        priority: int = 0,
    ) -> None:
        self.document = document
        self.priority = priority  # Higher = more urgent
        self.queued_at = datetime.now()

    @property
    def id(self) -> str:
        """Item ID (same as document ID)."""
        return self.document.id

    @property
    def requires_critical_review(self) -> bool:
        """Whether the document contains critical risk entities."""
        return self.document.has_critical_entities

    def to_dict(self) -> dict[str, Any]:
        """Serialise for UI display."""
        return {
            "id": self.id,
            "source_filename": self.document.source_filename,
            "entity_count": self.document.entity_count,
            "has_critical": self.document.has_critical_entities,
            "detection_summary": self.document.detection_summary,
            "review_status": self.document.review_status.value,
            "queued_at": self.queued_at.isoformat(),
            "priority": self.priority,
            "validation_passed": self.document.validation_passed,
        }


class HumanReviewQueue:
    """In-memory review queue with persistence through the audit log.

    Args:
        registry: AnonymisationRegistry for audit logging.
        on_review_complete: Optional callback when a review is completed.
    """

    def __init__(
        self,
        registry: AnonymisationRegistry,
        on_review_complete: Optional[Callable[[AnonymisedDocument, ReviewStatus], None]] = None,
    ) -> None:
        self._registry = registry
        self._on_complete = on_review_complete
        self._queue: OrderedDict[str, ReviewItem] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        """Number of items in the queue."""
        return len(self._queue)

    @property
    def pending_count(self) -> int:
        """Number of items awaiting review."""
        return sum(
            1
            for item in self._queue.values()
            if item.document.review_status == ReviewStatus.PENDING
        )

    def enqueue(self, document: AnonymisedDocument) -> ReviewItem:
        """Add a document to the review queue.

        Args:
            document: AnonymisedDocument requiring review.

        Returns:
            ReviewItem wrapping the document.
        """
        # Calculate priority
        priority = 0
        if document.has_critical_entities:
            priority += 10
        if not document.validation_passed:
            priority += 5
        priority += document.entity_count  # More entities = more urgent

        item = ReviewItem(document=document, priority=priority)

        with self._lock:
            self._queue[item.id] = item

        # Sort by priority (highest first)
        self._sort_queue()

        self._registry.log_audit(
            action="review_enqueued",
            matter_id=document.matter_id,
            document_id=document.source_document_id,
            entity_count=document.entity_count,
            details=f"Queued for review: priority={priority}, critical={document.has_critical_entities}",
        )

        logger.info(
            f"Review queued: '{document.source_filename}' "
            f"(priority={priority}, entities={document.entity_count})"
        )

        return item

    def get_next(self) -> Optional[ReviewItem]:
        """Get the next item to review (highest priority pending).

        Returns:
            Next ReviewItem, or None if queue is empty.
        """
        with self._lock:
            for item in self._queue.values():
                if item.document.review_status == ReviewStatus.PENDING:
                    return item
        return None

    def get_item(self, item_id: str) -> Optional[ReviewItem]:
        """Get a specific review item by ID.

        Args:
            item_id: The review item ID.

        Returns:
            ReviewItem or None.
        """
        return self._queue.get(item_id)

    def get_all(self) -> list[ReviewItem]:
        """Get all items in the queue.

        Returns:
            List of all ReviewItems, sorted by priority.
        """
        return list(self._queue.values())

    def approve(
        self,
        item_id: str,
        reviewer: str = "user",
        edits: Optional[list[dict[str, Any]]] = None,
    ) -> Optional[AnonymisedDocument]:
        """Approve an item after review.

        Args:
            item_id: Review item ID to approve.
            reviewer: Reviewer identifier.
            edits: Optional list of edits to entity detections.

        Returns:
            Updated AnonymisedDocument, or None if item not found.
        """
        item = self._queue.get(item_id)
        if not item:
            logger.warning(f"Review item {item_id} not found")
            return None

        doc = item.document
        doc.review_status = ReviewStatus.APPROVED
        doc.reviewed_by = reviewer
        doc.reviewed_at = datetime.now()

        self._registry.log_audit(
            action="review_approved",
            matter_id=doc.matter_id,
            document_id=doc.source_document_id,
            user=reviewer,
            entity_count=doc.entity_count,
            details=f"Approved: '{doc.source_filename}'",
        )

        if self._on_complete:
            self._on_complete(doc, ReviewStatus.APPROVED)

        logger.info(f"Review approved: '{doc.source_filename}' by {reviewer}")
        return doc

    def reject(
        self,
        item_id: str,
        reviewer: str = "user",
        reason: str = "",
    ) -> Optional[AnonymisedDocument]:
        """Reject an item — blocks export.

        Args:
            item_id: Review item ID to reject.
            reviewer: Reviewer identifier.
            reason: Rejection reason.

        Returns:
            Updated AnonymisedDocument, or None if item not found.
        """
        item = self._queue.get(item_id)
        if not item:
            logger.warning(f"Review item {item_id} not found")
            return None

        doc = item.document
        doc.review_status = ReviewStatus.REJECTED
        doc.reviewed_by = reviewer
        doc.reviewed_at = datetime.now()

        self._registry.log_audit(
            action="review_rejected",
            matter_id=doc.matter_id,
            document_id=doc.source_document_id,
            user=reviewer,
            details=f"Rejected: '{doc.source_filename}' — {reason}",
            success=False,
        )

        if self._on_complete:
            self._on_complete(doc, ReviewStatus.REJECTED)

        logger.info(f"Review rejected: '{doc.source_filename}' by {reviewer} — {reason}")
        return doc

    def add_entity(
        self,
        item_id: str,
        entity: PIIEntity,
        reviewer: str = "user",
    ) -> bool:
        """Add a manually identified entity during review.

        Args:
            item_id: Review item ID.
            entity: Manually identified PIIEntity.
            reviewer: Reviewer identifier.

        Returns:
            True if entity was added successfully.
        """
        item = self._queue.get(item_id)
        if not item:
            return False

        item.document.entities_detected.append(entity)

        self._registry.log_audit(
            action="review_add_entity",
            matter_id=item.document.matter_id,
            document_id=item.document.source_document_id,
            user=reviewer,
            details=f"Added entity: {entity.category.value} '{entity.original_text[:30]}'",
        )

        return True

    def remove_entity(
        self,
        item_id: str,
        entity_id: str,
        reviewer: str = "user",
    ) -> bool:
        """Remove a false-positive entity during review.

        Args:
            item_id: Review item ID.
            entity_id: ID of the entity to remove.
            reviewer: Reviewer identifier.

        Returns:
            True if entity was removed.
        """
        item = self._queue.get(item_id)
        if not item:
            return False

        original_count = len(item.document.entities_detected)
        item.document.entities_detected = [
            e for e in item.document.entities_detected if e.id != entity_id
        ]
        removed = original_count - len(item.document.entities_detected)

        if removed > 0:
            self._registry.log_audit(
                action="review_remove_entity",
                matter_id=item.document.matter_id,
                document_id=item.document.source_document_id,
                user=reviewer,
                details=f"Removed entity: {entity_id}",
            )

        return removed > 0

    def clear_completed(self) -> int:
        """Remove all completed (approved/rejected) items from the queue.

        Returns:
            Number of items removed.
        """
        with self._lock:
            to_remove = [
                item_id
                for item_id, item in self._queue.items()
                if item.document.review_status
                in (ReviewStatus.APPROVED, ReviewStatus.REJECTED)
            ]
            for item_id in to_remove:
                del self._queue[item_id]

        return len(to_remove)

    def _sort_queue(self) -> None:
        """Sort queue by priority (highest first)."""
        with self._lock:
            sorted_items = sorted(
                self._queue.items(),
                key=lambda x: x[1].priority,
                reverse=True,
            )
            self._queue = OrderedDict(sorted_items)
