"""UK legal domain PII patterns — postcodes, NI numbers, case references, etc.

These rule-based patterns complement Presidio and spaCy NER for
UK-specific identifiers that statistical models often miss.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..models import PIICategory


@dataclass
class PatternMatch:
    """A single regex pattern match."""

    category: PIICategory
    text: str
    start: int
    end: int
    confidence: float
    pattern_name: str


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# UK postcodes — covers all valid formats (A9 9AA through AA9A 9AA)
_UK_POSTCODE = re.compile(
    r"\b([Gg][Ii][Rr]\s*0[Aa]{2})|"
    r"(([A-Za-z]\d{1,2})|(([A-Za-z][A-Ha-hJ-Yj-y]\d{1,2})|(([A-Za-z]\d[A-Za-z])"
    r"|([A-Za-z][A-Ha-hJ-Yj-y]\d[A-Za-z]?))))\s*\d[A-Za-z]{2}\b",
    re.IGNORECASE,
)

# National Insurance numbers — AA 99 99 99 A (with optional spaces/dashes)
_NI_NUMBER = re.compile(
    r"\b(?!BG|GB|NK|KN|TN|NT|ZZ)"  # Invalid prefixes
    r"[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]"
    r"[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?[A-D]\b",
    re.IGNORECASE,
)

# UK phone numbers — mobile and landline
_UK_PHONE = re.compile(
    r"\b(?:(?:\+44\s?|0)(?:"
    r"7\d{3}[\s\-]?\d{6}|"           # Mobile: 07xxx xxxxxx
    r"1\d{3}[\s\-]?\d{5,6}|"         # Landline: 01xxx xxxxx(x)
    r"2\d[\s\-]?\d{4}[\s\-]?\d{4}|"  # London: 020 xxxx xxxx
    r"3\d{2}[\s\-]?\d{3}[\s\-]?\d{4}" # Non-geographic: 03xx
    r"))\b",
)

# Neutral citation — [2024] EWHC 1234 (Ch)
_NEUTRAL_CITATION = re.compile(
    r"\[\d{4}\]\s+"
    r"(?:UKSC|UKPC|EWCA\s+(?:Civ|Crim)|EWHC|UKUT|UKFTT|UKEAT|"
    r"EWCOP|EWFC|UKIPO|CSIH|CSOH|ScotCS)"
    r"\s+\d+(?:\s*\([A-Za-z]+\))?",
)

# Claim/case numbers — common formats
_CLAIM_NUMBER = re.compile(
    r"\b(?:CL|QB|QBD|CH|BL|CR|CO|IP|PT|HQ|TLC|CF|FL|FD)"
    r"[\-/]?\d{4}[\-/]\d{4,6}\b",
    re.IGNORECASE,
)

# SRA numbers (Solicitors Regulation Authority) — handles "SRA No. 612847"
_SRA_NUMBER = re.compile(
    r"\bSRA\s*(?:No\.?|Number|Ref\.?)?:?\s*\d{5,7}\b", re.IGNORECASE
)

# Bar number / BPTC / pupillage reference
_BAR_NUMBER = re.compile(
    r"\b(?:Bar\s*(?:No|Number|Ref)\.?\s*:?\s*\d{4,8})\b", re.IGNORECASE
)

# GMC numbers (General Medical Council) — "GMC No. 4578123"
_GMC_NUMBER = re.compile(
    r"\bGMC\s*(?:No\.?|Number|Ref\.?)?:?\s*\d{6,8}\b", re.IGNORECASE
)

# NHS numbers — 10 digits (with optional spaces: 483 291 8472)
_NHS_NUMBER = re.compile(
    r"\bNHS\s*(?:No\.?|Number)?:?\s*\d{3}\s?\d{3}\s?\d{4}\b", re.IGNORECASE
)

# Police crime reference numbers — CR/YYYY/NNNNNNN and similar
_CRIME_REF = re.compile(
    r"\b(?:CR|crime\s+ref(?:erence)?)\s*[:/\-]?\s*\d{4}[/\-]\d{5,10}\b",
    re.IGNORECASE,
)

# UK street addresses — number + street name patterns
# Catches: "47 Meadow Lane", "12 Church Lane", "14 Primrose Close"
_STREET_ADDRESS = re.compile(
    r"\b\d{1,4}\s+"
    r"(?:[A-Z][a-z]+\s+)?"                  # Optional first word (e.g. "Old")
    r"[A-Z][a-z]+\s+"                        # Street name word
    r"(?:Street|St|Road|Rd|Lane|Ln|Avenue|Ave|Drive|Dr|Close|Cl|"
    r"Crescent|Cres|Way|Place|Pl|Court|Ct|Terrace|Tce|"
    r"Gardens|Gdns|Grove|Park|Rise|Row|Walk|Mews|"
    r"Hill|Green|Square|Sq|Parade|Passage|Yard)\b",
)

# Email addresses
_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# UK vehicle registration (various formats)
_VEHICLE_REG = re.compile(
    r"\b(?:"
    r"[A-Z]{2}\d{2}\s?[A-Z]{3}|"     # New style: AB12 CDE
    r"[A-Z]\d{1,3}\s?[A-Z]{3}|"       # Prefix: A123 BCD
    r"[A-Z]{3}\s?\d{1,3}[A-Z]"        # Suffix: BCD 123A
    r")\b",
)

# UK passport number — 9 digits
_PASSPORT = re.compile(r"\b\d{9}\b")  # Very broad — context-gated below

# Dates in various UK formats
_UK_DATE = re.compile(
    r"\b(?:"
    r"\d{1,2}[\s/\-\.]\d{1,2}[\s/\-\.]\d{2,4}|"    # DD/MM/YYYY or DD-MM-YY
    r"\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"[\s,]*\d{2,4}|"                                 # 14th March 2024
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?[\s,]*\d{2,4}"       # March 14, 2024
    r")\b",
    re.IGNORECASE,
)

# Age patterns — "aged 7", "age 14", "X years old", "X-year-old"
_AGE_PATTERN = re.compile(
    r"\b(?:"
    r"aged?\s+\d{1,3}|"
    r"\d{1,3}[\s\-]years?[\s\-]old|"
    r"\d{1,3}\s*(?:yo|y/o|yrs)"
    r")\b",
    re.IGNORECASE,
)

# Monetary amounts (£ and GBP)
_MONETARY = re.compile(
    r"(?:£|GBP\s?)\s?\d[\d,]*(?:\.\d{1,2})?(?:\s*(?:million|m|billion|b|thousand|k))?",
    re.IGNORECASE,
)

# IP addresses
_IP_ADDRESS = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

UK_LEGAL_PATTERNS: list[tuple[re.Pattern, PIICategory, str, float]] = [
    (_UK_POSTCODE, PIICategory.POSTCODE, "uk_postcode", 0.95),
    (_NI_NUMBER, PIICategory.NI_NUMBER, "ni_number", 0.98),
    (_UK_PHONE, PIICategory.PHONE_NUMBER, "uk_phone", 0.90),
    (_NEUTRAL_CITATION, PIICategory.CASE_REFERENCE, "neutral_citation", 0.99),
    (_CLAIM_NUMBER, PIICategory.CASE_REFERENCE, "claim_number", 0.85),
    (_CRIME_REF, PIICategory.CASE_REFERENCE, "crime_reference", 0.92),
    (_SRA_NUMBER, PIICategory.SRA_NUMBER, "sra_number", 0.95),
    (_BAR_NUMBER, PIICategory.BAR_NUMBER, "bar_number", 0.80),
    (_GMC_NUMBER, PIICategory.MEDICAL_IDENTIFIER, "gmc_number", 0.95),
    (_NHS_NUMBER, PIICategory.MEDICAL_IDENTIFIER, "nhs_number", 0.95),
    (_EMAIL, PIICategory.EMAIL_ADDRESS, "email", 0.98),
    (_STREET_ADDRESS, PIICategory.ADDRESS, "street_address", 0.85),
    (_VEHICLE_REG, PIICategory.VEHICLE_REG, "vehicle_reg", 0.70),
    (_UK_DATE, PIICategory.DATE, "uk_date", 0.80),
    (_AGE_PATTERN, PIICategory.AGE, "age_pattern", 0.92),
    (_MONETARY, PIICategory.MONETARY_AMOUNT, "monetary_amount", 0.85),
    (_IP_ADDRESS, PIICategory.IP_ADDRESS, "ip_address", 0.75),
]


def scan_uk_legal_patterns(
    text: str,
    min_confidence: float = 0.0,
) -> list[PatternMatch]:
    """Scan text for UK legal PII patterns.

    Args:
        text: Text to scan.
        min_confidence: Minimum confidence threshold.

    Returns:
        List of PatternMatch instances.
    """
    matches: list[PatternMatch] = []

    for pattern, category, name, confidence in UK_LEGAL_PATTERNS:
        if confidence < min_confidence:
            continue
        for m in pattern.finditer(text):
            matches.append(PatternMatch(
                category=category,
                text=m.group(),
                start=m.start(),
                end=m.end(),
                confidence=confidence,
                pattern_name=name,
            ))

    # De-duplicate overlapping matches — keep highest confidence
    matches.sort(key=lambda m: (m.start, -m.confidence))
    deduped: list[PatternMatch] = []
    last_end = -1
    for match in matches:
        if match.start >= last_end:
            deduped.append(match)
            last_end = match.end
        elif match.confidence > deduped[-1].confidence:
            deduped[-1] = match
            last_end = match.end

    return deduped
