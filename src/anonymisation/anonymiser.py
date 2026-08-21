"""Anonymiser — applies pseudonymisation to text using detected PII entities.

Replaces PII with consistent tokens from the AnonymisationRegistry,
maintaining referential consistency within a matter (the same person
always maps to the same token across all documents).

Tokens carry analytical context so cloud LLMs can still do useful analysis:
  - Person tokens include role:  [PERSON_001:director]
  - Amount tokens include magnitude:  [AMOUNT_001:six_figures]
  - Date tokens preserve the year:  [DATE_001:2023]
  - Age tokens include vulnerability band:  [AGE_001:primary_school_age — child_vulnerability]

The identifying data is fully removed. The context metadata is
non-identifying and preserves the analytical structure.

Supports:
  - Consistent tokenisation with contextual enrichment (primary method)
  - Generalisation (locations → region, ages → legal vulnerability bands)
  - Suppression (complete removal for high-risk identifiers)
"""

from __future__ import annotations

import re
from typing import Any, Optional

from loguru import logger

from .models import (
    AnonymisationMethod,
    AnonymisationToken,
    AnonymisedDocument,
    PIICategory,
    PIIEntity,
    ReviewStatus,
    RiskLevel,
)
from .registry import AnonymisationRegistry


# Categories that should be generalised rather than tokenised
GENERALISE_CATEGORIES = {
    PIICategory.LOCATION,
    PIICategory.POSTCODE,
}

# Categories that should be suppressed entirely
SUPPRESS_CATEGORIES = {
    PIICategory.NI_NUMBER,
    PIICategory.PASSPORT_NUMBER,
    PIICategory.IP_ADDRESS,
    PIICategory.FINANCIAL_ACCOUNT,
    PIICategory.VEHICLE_REG,
    PIICategory.SRA_NUMBER,
    PIICategory.BAR_NUMBER,
}

# ---------------------------------------------------------------------------
# Contextual enrichment — analytical metadata for tokens
# ---------------------------------------------------------------------------

# Role indicators for person names — detected from surrounding context
_ROLE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:director|directors)\b", re.I), "director"),
    (re.compile(r"\b(?:claimant|claimants|plaintiff|plaintiffs)\b", re.I), "claimant"),
    (re.compile(r"\b(?:defendant|defendants|respondent|respondents)\b", re.I), "defendant"),
    (re.compile(r"\b(?:applicant|applicants)\b", re.I), "applicant"),
    (re.compile(r"\b(?:witness|witnesses)\b", re.I), "witness"),
    (re.compile(r"\b(?:solicitor|solicitors|lawyer|lawyers)\b", re.I), "solicitor"),
    (re.compile(r"\b(?:barrister|barristers|counsel|QC|KC)\b", re.I), "counsel"),
    (re.compile(r"\b(?:judge|judges|His Honour|Her Honour|Mr Justice|Mrs Justice)\b", re.I), "judge"),
    (re.compile(r"\b(?:expert|experts)\b", re.I), "expert"),
    (re.compile(r"\b(?:trustee|trustees)\b", re.I), "trustee"),
    (re.compile(r"\b(?:beneficiary|beneficiaries)\b", re.I), "beneficiary"),
    (re.compile(r"\b(?:shareholder|shareholders|member|members)\b", re.I), "shareholder"),
    (re.compile(r"\b(?:employer|employers)\b", re.I), "employer"),
    (re.compile(r"\b(?:employee|employees|worker|workers)\b", re.I), "employee"),
    (re.compile(r"\b(?:tenant|tenants|lessee|lessees)\b", re.I), "tenant"),
    (re.compile(r"\b(?:landlord|landlords|lessor|lessors)\b", re.I), "landlord"),
    (re.compile(r"\b(?:victim|victims|complainant|complainants)\b", re.I), "victim"),
    (re.compile(r"\b(?:perpetrator|perpetrators|abuser|offender)\b", re.I), "perpetrator"),
    (re.compile(r"\b(?:insurer|insurers|underwriter)\b", re.I), "insurer"),
    (re.compile(r"\b(?:creditor|creditors)\b", re.I), "creditor"),
    (re.compile(r"\b(?:debtor|debtors)\b", re.I), "debtor"),
    (re.compile(r"\b(?:liquidator|administrator|receiver)\b", re.I), "insolvency_practitioner"),
    (re.compile(r"\b(?:guardian|parent|carer|foster)\b", re.I), "guardian"),
    (re.compile(r"\b(?:child|minor|infant)\b", re.I), "child"),
    (re.compile(r"\b(?:patient)\b", re.I), "patient"),
    (re.compile(r"\b(?:client|clients)\b", re.I), "client"),
]


def _detect_role(context: str) -> Optional[str]:
    """Detect the analytical role of a person from surrounding context.

    Args:
        context: Text surrounding the person name (typically ±100 chars).

    Returns:
        Role string (e.g. "director", "claimant") or None.
    """
    for pattern, role in _ROLE_PATTERNS:
        if pattern.search(context):
            return role
    return None


def _monetary_magnitude(amount_text: str) -> str:
    """Extract order-of-magnitude from a monetary amount string.

    Preserves analytical relevance for quantum analysis while
    removing the exact figure.

    Examples:
        "£450,000" → "six_figures"
        "GBP 2.5 million" → "seven_figures"
        "£50" → "two_figures"
    """
    # Normalise: remove currency symbols, commas, spaces
    cleaned = re.sub(r"[£$€,\s]", "", amount_text.upper())
    cleaned = re.sub(r"^GBP", "", cleaned)

    # Handle "million", "billion", "thousand" suffixes
    multiplier = 1
    if re.search(r"MILLION|M$", cleaned):
        multiplier = 1_000_000
        cleaned = re.sub(r"MILLION|M$", "", cleaned)
    elif re.search(r"BILLION|B$", cleaned):
        multiplier = 1_000_000_000
        cleaned = re.sub(r"BILLION|B$", "", cleaned)
    elif re.search(r"THOUSAND|K$", cleaned):
        multiplier = 1_000
        cleaned = re.sub(r"THOUSAND|K$", "", cleaned)

    # Parse the numeric part
    try:
        value = float(cleaned) * multiplier
    except (ValueError, OverflowError):
        return "unspecified"

    if value < 100:
        return "two_figures"
    elif value < 1_000:
        return "three_figures"
    elif value < 10_000:
        return "four_figures"
    elif value < 100_000:
        return "five_figures"
    elif value < 1_000_000:
        return "six_figures"
    elif value < 10_000_000:
        return "seven_figures"
    elif value < 100_000_000:
        return "eight_figures"
    else:
        return "nine_figures_plus"


def _extract_year(date_text: str) -> Optional[str]:
    """Extract the year from a date string.

    The specific day/month is PII but the year is typically needed
    for limitation period analysis.

    Returns:
        Year string (e.g. "2023") or None.
    """
    # Try 4-digit year
    m = re.search(r"\b((?:19|20)\d{2})\b", date_text)
    if m:
        return m.group(1)

    # Try 2-digit year (interpret as 2000s or 1900s)
    m = re.search(r"\b(\d{2})\b", date_text)
    if m:
        yr = int(m.group(1))
        return str(2000 + yr) if yr < 50 else str(1900 + yr)

    return None

# ---------------------------------------------------------------------------
# Legal vulnerability age categories
# ---------------------------------------------------------------------------
# These bands are designed for legal relevance: they distinguish between
# categories that affect legal analysis (capacity, vulnerability status,
# sentencing guidelines, limitation periods, etc.) while preventing
# identification by specific age.

AGE_CATEGORIES: list[tuple[int, int, str, str]] = [
    # (min, max, band_label, vulnerability_category)
    (0, 1, "infant (under 2)", "young_child_vulnerability"),
    (2, 4, "pre-school (2-4)", "young_child_vulnerability"),
    (5, 10, "primary school age (5-10)", "child_vulnerability"),
    (11, 13, "early secondary (11-13)", "child_vulnerability"),
    (14, 15, "mid-teens (14-15)", "adolescent_vulnerability"),
    (16, 17, "older minor (16-17)", "adolescent_vulnerability"),
    (18, 25, "young adult (18-25)", "none"),
    (26, 64, "adult (26-64)", "none"),
    (65, 74, "older adult (65-74)", "elderly_vulnerability"),
    (75, 84, "elderly (75-84)", "elderly_vulnerability"),
    (85, 200, "very elderly (85+)", "elderly_vulnerability"),
]


def _age_to_legal_category(age_text: str) -> tuple[str, str]:
    """Convert an age string to a legal vulnerability category.

    Returns:
        Tuple of (age_band_label, vulnerability_category).
        The vulnerability category indicates legal significance:
        - young_child_vulnerability: under 5, special protection
        - child_vulnerability: 5-13, child protection frameworks apply
        - adolescent_vulnerability: 14-17, Gillick competence considerations
        - elderly_vulnerability: 65+, vulnerable adult protections
        - none: no age-based vulnerability
    """
    import re

    numbers = re.findall(r"\d+", age_text)
    if not numbers:
        # Check for descriptive terms
        lower = age_text.lower()
        if any(w in lower for w in ("infant", "baby", "newborn", "toddler")):
            return "infant (under 2)", "young_child_vulnerability"
        if any(w in lower for w in ("teenager", "adolescent", "pre-teen")):
            return "adolescent (13-17)", "adolescent_vulnerability"
        if any(w in lower for w in ("elderly", "pensioner", "retired")):
            return "elderly (65+)", "elderly_vulnerability"
        return "unknown age", "unknown"

    age = int(numbers[0])
    for low, high, band, vuln in AGE_CATEGORIES:
        if low <= age <= high:
            return band, vuln
    return "unknown age", "unknown"


class Anonymiser:
    """Applies anonymisation to text based on detected PII entities.

    Args:
        registry: Token mapping registry for consistent pseudonymisation.
        matter_id: Matter/case identifier for token scoping.
        preserve_relationships: Keep relationship structure in anonymised form.
        preserve_temporal_order: Maintain date ordering in anonymised form.
        location_granularity: How much location detail to preserve (region/city/suppress).
        date_handling: How to handle dates (offset/generalise/suppress).
        age_handling: How to handle ages (band/suppress).
    """

    def __init__(
        self,
        registry: AnonymisationRegistry,
        matter_id: str,
        preserve_relationships: bool = True,
        preserve_temporal_order: bool = True,
        location_granularity: str = "region",
        date_handling: str = "offset",
        age_handling: str = "band",
    ) -> None:
        self._registry = registry
        self._matter_id = matter_id
        self._preserve_relationships = preserve_relationships
        self._preserve_temporal_order = preserve_temporal_order
        self._location_granularity = location_granularity
        self._date_handling = date_handling
        self._age_handling = age_handling

    def anonymise_text(
        self,
        text: str,
        entities: list[PIIEntity],
        source_document_id: str = "",
        source_filename: str = "",
        require_review_for_critical: bool = True,
    ) -> AnonymisedDocument:
        """Anonymise text by replacing all detected PII entities.

        Entities are replaced in reverse order (end→start) to preserve
        character offsets during replacement.

        Args:
            text: Original text to anonymise.
            entities: Detected PII entities (from PIIDetector.detect()).
            source_document_id: ID of the source document.
            source_filename: Original filename.
            require_review_for_critical: Flag documents with CRITICAL entities for review.

        Returns:
            AnonymisedDocument with both original and anonymised text.
        """
        if not entities:
            return AnonymisedDocument(
                matter_id=self._matter_id,
                source_document_id=source_document_id,
                source_filename=source_filename,
                original_text=text,
                anonymised_text=text,
                entities_detected=[],
                tokens_applied=[],
                review_status=ReviewStatus.NOT_REQUIRED,
                validation_passed=True,
            )

        # Sort entities by start position (descending) for safe replacement
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

        # Remove overlapping entities — when two entities overlap, keep the
        # one that starts first (or the longer one if they start at the same place).
        # This prevents mangled tokens like [ORG_001]OSTCODE_001].
        cleaned: list[PIIEntity] = []
        last_start = len(text) + 1  # Since we're iterating in reverse
        for entity in sorted_entities:
            if entity.end <= last_start:
                cleaned.append(entity)
                last_start = entity.start
            else:
                # This entity overlaps with the previously accepted one — skip it
                logger.debug(
                    f"Skipping overlapping entity '{entity.original_text}' "
                    f"({entity.category.value}) at {entity.start}-{entity.end}"
                )

        anonymised = text
        tokens_applied: list[AnonymisationToken] = []

        for entity in cleaned:
            token = self._get_replacement(entity)
            tokens_applied.append(token)

            # Replace in text, ensuring adjacent tokens don't mangle each other
            before = anonymised[:entity.start]
            after = anonymised[entity.end:]
            replacement = token.anonymised_value

            # If the character immediately after the replacement is '[' (another token)
            # or the character before is ']', insert a space to prevent mangling
            if after and after[0] == "[":
                replacement = replacement + " "
            if before and before[-1] == "]":
                replacement = " " + replacement

            anonymised = before + replacement + after

        # ------------------------------------------------------------------
        # Surname / location residual sweep
        # ------------------------------------------------------------------
        # After main replacement, check if any *component* of detected names
        # or locations survived elsewhere in the text. This catches patterns
        # like "Thompson v Diocese" where spaCy detected "Sarah Thompson" but
        # missed the standalone "Thompson" in the case title.
        #
        # Uses BOTH the current document's tokens AND the full registry for
        # the matter, so cross-document name references are caught.
        # ------------------------------------------------------------------
        import re as _re

        # Common words that should never be treated as name residuals
        _SKIP_WORDS = frozenset({
            # Titles
            "mr", "mrs", "ms", "dr", "miss", "rev", "reverend",
            "father", "canon", "bishop", "sir", "dame",
            "professor", "prof", "lord", "lady", "hon",
            # Common English words that spaCy may include in entity spans
            "complaint", "received", "previous", "complaints",
            "subject", "action", "taken", "outcome", "filed",
            "note", "schedule", "timeline", "report", "statement",
            "reference", "dear", "yours", "sincerely", "faithfully",
            "signed", "dated", "prepared", "for", "the", "and",
            "that", "this", "with", "from", "about", "between",
            # Legal terms
            "claimant", "defendant", "plaintiff", "respondent",
            "solicitor", "barrister", "counsel", "witness",
            "diocese", "parish", "church", "school", "hospital",
            "society", "trust", "foundation", "authority",
            # Address components
            "road", "street", "lane", "avenue", "close", "drive",
            "place", "court", "terrace", "gardens", "grove",
            "park", "rise", "row", "walk", "mews", "hill",
            "green", "square", "parade", "house",
        })

        _PERSON_CATEGORIES = frozenset({
            PIICategory.PERSON_NAME, PIICategory.WITNESS_NAME,
            PIICategory.SOLICITOR_NAME, PIICategory.BARRISTER_NAME,
            PIICategory.JUDGE_NAME, PIICategory.VICTIM_IDENTIFIER,
            PIICategory.PERPETRATOR_IDENTIFIER,
        })

        _LOCATION_CATEGORIES = frozenset({
            PIICategory.LOCATION, PIICategory.ADDRESS,
            PIICategory.INSTITUTION_NAME, PIICategory.SCHOOL_NAME,
        })

        def _extract_residual_words(
            token: AnonymisationToken,
        ) -> list[tuple[str, str]]:
            """Extract significant words from a token's original value.

            Returns list of (word, token_anonymised_value) pairs.
            """
            original = token.original_value
            if not original or len(original) < 3:
                return []

            results: list[tuple[str, str]] = []
            parts = original.split()
            for part in parts:
                cleaned_part = part.strip(".,;:()'\"")
                if len(cleaned_part) < 3:
                    continue
                if not cleaned_part[0].isupper():
                    continue
                if cleaned_part.lower() in _SKIP_WORDS:
                    continue
                # For person names, only keep words >= 3 chars
                # For locations, only keep words >= 4 chars
                min_len = 3 if token.category in _PERSON_CATEGORIES else 4
                if len(cleaned_part) >= min_len:
                    results.append((cleaned_part, token.anonymised_value))
            return results

        _residual_map: dict[str, str] = {}  # raw_word -> token_value

        # Phase 1: Current document's tokens
        for token in tokens_applied:
            if token.category in _PERSON_CATEGORIES or token.category in _LOCATION_CATEGORIES:
                for word, value in _extract_residual_words(token):
                    if word not in _residual_map:
                        _residual_map[word] = value

        # Phase 2: Full registry for the matter (cross-document)
        # This catches names that were detected in earlier documents
        try:
            all_matter_tokens = self._registry.get_all_tokens(self._matter_id)
            for token in all_matter_tokens:
                if token.category in _PERSON_CATEGORIES or token.category in _LOCATION_CATEGORIES:
                    for word, value in _extract_residual_words(token):
                        if word not in _residual_map:
                            _residual_map[word] = value
        except Exception:
            pass  # Registry query failed — rely on Phase 1 only

        # Apply residual replacements (case-insensitive to catch ALL-CAPS variants)
        residual_count = 0
        for raw_word, token_value in _residual_map.items():
            # Only replace if the word appears *outside* existing tokens
            # i.e. not already inside [TOKEN_NNN] brackets
            pattern = _re.compile(
                r"(?<!\[)"           # Not preceded by [
                r"\b" + _re.escape(raw_word) + r"\b"
                r"(?![^\[]*\])",     # Not followed by ] without intervening [
                _re.IGNORECASE,
            )
            new_anon, count = pattern.subn(token_value, anonymised)
            if count > 0:
                anonymised = new_anon
                residual_count += count

        if residual_count:
            logger.info(
                f"Surname/location residual sweep: {residual_count} additional "
                f"occurrences anonymised in '{source_filename}'"
            )

        # Build detection summary
        summary: dict[str, int] = {}
        for e in entities:
            key = e.category.value
            summary[key] = summary.get(key, 0) + 1

        # Determine review status
        has_critical = any(e.risk_level == RiskLevel.CRITICAL for e in entities)
        low_confidence = any(e.confidence < 0.7 for e in entities)

        if require_review_for_critical and has_critical:
            review_status = ReviewStatus.PENDING
        elif low_confidence:
            review_status = ReviewStatus.PENDING
        else:
            review_status = ReviewStatus.NOT_REQUIRED

        doc = AnonymisedDocument(
            matter_id=self._matter_id,
            source_document_id=source_document_id,
            source_filename=source_filename,
            original_text=text,
            anonymised_text=anonymised,
            entities_detected=list(entities),  # Keep original order
            tokens_applied=list(reversed(tokens_applied)),  # Restore order
            detection_summary=summary,
            review_status=review_status,
        )

        logger.info(
            f"Anonymised document '{source_filename}': "
            f"{len(entities)} entities, review={review_status.value}"
        )

        return doc

    def _get_replacement(self, entity: PIIEntity) -> AnonymisationToken:
        """Get the appropriate replacement for a PII entity.

        Routes to tokenisation, generalisation, or suppression based
        on the entity category. Enriches tokens with analytical context
        so cloud LLMs can still do useful analysis.
        """
        category = entity.category

        # Suppression: replace with category-only token
        if category in SUPPRESS_CATEGORIES:
            return self._registry.get_or_create_token(
                matter_id=self._matter_id,
                category=category,
                original_value=entity.original_text,
                method=AnonymisationMethod.SUPPRESSION,
            )

        # Age: generalise to legal vulnerability category
        if category == PIICategory.AGE and self._age_handling == "band":
            band_label, vuln_category = _age_to_legal_category(entity.original_text)
            token = self._registry.get_or_create_token(
                matter_id=self._matter_id,
                category=category,
                original_value=entity.original_text,
                method=AnonymisationMethod.GENERALISATION,
            )
            vuln_suffix = f" — {vuln_category}" if vuln_category != "none" else ""
            token.anonymised_value = (
                f"{token.anonymised_value.rstrip(']')}, {band_label}{vuln_suffix}]"
            )
            return token

        # Dates: preserve the year for limitation analysis
        if category == PIICategory.DATE:
            token = self._registry.get_or_create_token(
                matter_id=self._matter_id,
                category=category,
                original_value=entity.original_text,
                method=AnonymisationMethod.TOKENISATION,
            )
            year = _extract_year(entity.original_text)
            if year:
                base = token.anonymised_value.rstrip("]")
                token.anonymised_value = f"{base}:{year}]"
            return token

        # Monetary amounts: preserve magnitude for quantum analysis
        if category == PIICategory.MONETARY_AMOUNT:
            token = self._registry.get_or_create_token(
                matter_id=self._matter_id,
                category=category,
                original_value=entity.original_text,
                method=AnonymisationMethod.TOKENISATION,
            )
            magnitude = _monetary_magnitude(entity.original_text)
            if magnitude != "unspecified":
                base = token.anonymised_value.rstrip("]")
                token.anonymised_value = f"{base}:{magnitude}]"
            return token

        # Person names: detect and attach analytical role
        if category in (
            PIICategory.PERSON_NAME,
            PIICategory.WITNESS_NAME,
            PIICategory.JUDGE_NAME,
            PIICategory.SOLICITOR_NAME,
            PIICategory.BARRISTER_NAME,
            PIICategory.VICTIM_IDENTIFIER,
            PIICategory.PERPETRATOR_IDENTIFIER,
        ):
            token = self._registry.get_or_create_token(
                matter_id=self._matter_id,
                category=category,
                original_value=entity.original_text,
                method=AnonymisationMethod.TOKENISATION,
            )
            # Use the entity's context (surrounding text) to detect role
            context = entity.context or ""
            role = _detect_role(context)
            # Some categories already imply a role
            if not role:
                role = {
                    PIICategory.JUDGE_NAME: "judge",
                    PIICategory.SOLICITOR_NAME: "solicitor",
                    PIICategory.BARRISTER_NAME: "counsel",
                    PIICategory.WITNESS_NAME: "witness",
                    PIICategory.VICTIM_IDENTIFIER: "victim",
                    PIICategory.PERPETRATOR_IDENTIFIER: "perpetrator",
                }.get(category)
            if role:
                base = token.anonymised_value.rstrip("]")
                token.anonymised_value = f"{base}:{role}]"
            return token

        # Location: generalise based on granularity setting
        if category in GENERALISE_CATEGORIES and self._location_granularity == "suppress":
            return self._registry.get_or_create_token(
                matter_id=self._matter_id,
                category=category,
                original_value=entity.original_text,
                method=AnonymisationMethod.SUPPRESSION,
            )

        # Default: consistent tokenisation
        return self._registry.get_or_create_token(
            matter_id=self._matter_id,
            category=category,
            original_value=entity.original_text,
            method=AnonymisationMethod.TOKENISATION,
        )

    def anonymise_qa_output(
        self,
        query: str,
        answer: str,
        chunks: list[dict[str, Any]],
        query_entities: list[PIIEntity],
        answer_entities: list[PIIEntity],
        chunk_entities: Optional[dict[str, list[PIIEntity]]] = None,
    ) -> dict[str, Any]:
        """Anonymise a Q&A interaction (query + answer + source chunks).

        Args:
            query: Original user query.
            answer: Generated LLM answer.
            chunks: Retrieved source chunks (list of dicts with 'text', 'metadata').
            query_entities: PII entities detected in the query.
            answer_entities: PII entities detected in the answer.
            chunk_entities: Optional dict of chunk_id → entities for each chunk.

        Returns:
            Dict with anonymised versions of query, answer, and chunks.
        """
        # Anonymise query
        anon_query_doc = self.anonymise_text(
            query, query_entities, source_document_id="query"
        )

        # Anonymise answer
        anon_answer_doc = self.anonymise_text(
            answer, answer_entities, source_document_id="answer"
        )

        # Anonymise chunks
        anon_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id", "")

            # Use provided entities or empty list
            c_entities = (chunk_entities or {}).get(chunk_id, [])

            if c_entities:
                anon_chunk_doc = self.anonymise_text(
                    chunk_text, c_entities, source_document_id=chunk_id
                )
                anon_text = anon_chunk_doc.anonymised_text
            else:
                anon_text = chunk_text

            # Anonymise metadata file names
            metadata = dict(chunk.get("metadata", {}))
            if "file_name" in metadata:
                file_token = self._registry.get_or_create_token(
                    matter_id=self._matter_id,
                    category=PIICategory.CUSTOM,
                    original_value=metadata["file_name"],
                )
                metadata["file_name"] = file_token.anonymised_value

            anon_chunks.append({
                "text": anon_text,
                "score": chunk.get("score", 0.0),
                "metadata": metadata,
            })

        return {
            "anonymised_query": anon_query_doc.anonymised_text,
            "anonymised_answer": anon_answer_doc.anonymised_text,
            "anonymised_chunks": anon_chunks,
            "query_entity_count": len(query_entities),
            "answer_entity_count": len(answer_entities),
        }
