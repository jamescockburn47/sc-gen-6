"""Financial PII patterns — bank accounts, sort codes, card numbers.

Relevant for civil fraud litigation where financial documents
are part of disclosure bundles.
"""

from __future__ import annotations

import re

from .uk_legal import PatternMatch
from ..models import PIICategory


# ---------------------------------------------------------------------------
# Financial patterns
# ---------------------------------------------------------------------------

# UK bank sort code — XX-XX-XX
_SORT_CODE = re.compile(
    r"\b\d{2}[\s\-]\d{2}[\s\-]\d{2}\b"
)

# UK bank account number — 8 digits (context-gated)
_ACCOUNT_NUMBER = re.compile(
    r"(?:(?:account|acc|a/c)[\s\.]*(?:no|number|num|#)?[\s:\.]*)"
    r"(\d{8})\b",
    re.IGNORECASE,
)

# IBAN (any country, but UK starts GB)
_IBAN = re.compile(
    r"\b[A-Z]{2}\d{2}\s?[A-Z0-9]{4}\s?\d{4}\s?\d{4}\s?(?:\d{4}\s?){0,4}\d{0,4}\b"
)

# Credit/debit card numbers (13-19 digits, may have spaces/dashes)
_CARD_NUMBER = re.compile(
    r"\b(?:\d{4}[\s\-]?){3,4}\d{1,4}\b"
)

# BIC/SWIFT code
_SWIFT_BIC = re.compile(
    r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"
)

# Companies House number (8 digits or 2 letters + 6 digits)
_COMPANIES_HOUSE = re.compile(
    r"\b(?:(?:SC|NI|OC|SO|NC|NF|RS|IP|SP|IC|SI|NP|NO|RC|SR|NR|CE|CS|FC|GE|SF|GS|GN|LP|SL|NL|OE)\d{6}|\d{8})\b"
)

# VAT registration number — GB 123 4567 89
_VAT_NUMBER = re.compile(
    r"\b(?:GB\s?)?\d{3}\s?\d{4}\s?\d{2}\b"
)


FINANCIAL_PATTERNS: list[tuple[re.Pattern, PIICategory, str, float]] = [
    (_SORT_CODE, PIICategory.FINANCIAL_ACCOUNT, "sort_code", 0.75),
    (_ACCOUNT_NUMBER, PIICategory.FINANCIAL_ACCOUNT, "account_number", 0.88),
    (_IBAN, PIICategory.FINANCIAL_ACCOUNT, "iban", 0.92),
    (_CARD_NUMBER, PIICategory.FINANCIAL_ACCOUNT, "card_number", 0.70),
    (_SWIFT_BIC, PIICategory.FINANCIAL_ACCOUNT, "swift_bic", 0.65),
    (_COMPANIES_HOUSE, PIICategory.ORGANISATION, "companies_house", 0.60),
    (_VAT_NUMBER, PIICategory.FINANCIAL_ACCOUNT, "vat_number", 0.55),
]


def scan_financial_patterns(
    text: str,
    min_confidence: float = 0.0,
) -> list[PatternMatch]:
    """Scan text for financial PII patterns.

    Args:
        text: Text to scan.
        min_confidence: Minimum confidence threshold.

    Returns:
        List of PatternMatch instances.
    """
    matches: list[PatternMatch] = []

    for pattern, category, name, confidence in FINANCIAL_PATTERNS:
        if confidence < min_confidence:
            continue
        for m in pattern.finditer(text):
            # For account numbers, use group(1) if available (the actual number)
            match_text = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group()
            match_start = m.start(1) if m.lastindex and m.lastindex >= 1 else m.start()
            match_end = m.end(1) if m.lastindex and m.lastindex >= 1 else m.end()

            matches.append(PatternMatch(
                category=category,
                text=match_text,
                start=match_start,
                end=match_end,
                confidence=confidence,
                pattern_name=name,
            ))

    return matches
