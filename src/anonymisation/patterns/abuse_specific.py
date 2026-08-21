"""Abuse-case-specific PII patterns for child sexual abuse and sensitive legal proceedings.

These patterns target identifiers that are uniquely dangerous in abuse cases:
  - Victim/perpetrator relationship chains
  - School and care institution names
  - Social worker / CAFCASS references
  - Unique factual combinations that could identify individuals
  - Minor age references

ICO 2025 guidance: even indirect identifiers (relationship + location + time)
can constitute personal data if they narrow the pool to a small group.
"""

from __future__ import annotations

import re
from typing import Optional

from .uk_legal import PatternMatch
from ..models import PIICategory


# ---------------------------------------------------------------------------
# Relationship descriptors that can narrow victim identification
# ---------------------------------------------------------------------------

_RELATIONSHIP_DESCRIPTORS = re.compile(
    r"\b(?:"
    r"(?:the\s+)?(?:complainant|victim|alleged\s+victim|survivor|"
    r"child|minor|young\s+person|foster\s+child|looked[\s\-]after\s+child|"
    r"vulnerable\s+(?:adult|person|child))'s?\s+"
    r"(?:mother|father|parent|guardian|step[\s\-]?(?:mother|father|parent)|"
    r"foster[\s\-]?(?:mother|father|parent|carer)|"
    r"brother|sister|sibling|half[\s\-]?(?:brother|sister)|"
    r"uncle|aunt|grandfather|grandmother|grandparent|cousin|"
    r"son|daughter|child|children|nephew|niece|"
    r"partner|husband|wife|spouse|ex[\s\-]?(?:partner|husband|wife)|"
    r"boyfriend|girlfriend|carer|"
    r"teacher|headteacher|head\s+teacher|tutor|"
    r"social\s+worker|key\s+worker|"
    r"doctor|GP|therapist|counsellor|psychiatrist)"
    r")\b",
    re.IGNORECASE,
)

# Reverse: "[role]'s [victim descriptor]"
_REVERSE_RELATIONSHIP = re.compile(
    r"\b(?:(?:the\s+)?(?:defendant|accused|respondent|perpetrator|"
    r"alleged\s+(?:perpetrator|abuser|offender))'s?\s+"
    r"(?:victim|child|foster\s+child|step[\s\-]?(?:child|daughter|son)|"
    r"daughter|son|niece|nephew|ward|pupil|student|patient)"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Institution names that appear in abuse contexts
# ---------------------------------------------------------------------------

# Care homes, children's homes, residential units
_CARE_INSTITUTION = re.compile(
    r"\b(?:"
    r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+)?"
    r"(?:Children'?s?\s+Home|Care\s+Home|Residential\s+(?:Home|Unit|Centre)|"
    r"Secure\s+(?:Unit|Centre)|Assessment\s+Centre|"
    r"Pupil\s+Referral\s+Unit|PRU|"
    r"Foster(?:ing)?\s+(?:Agency|Service)|"
    r"Adoption\s+(?:Agency|Service)|"
    r"CAMHS|Child\s+and\s+Adolescent\s+Mental\s+Health)"
    r")\b",
    re.IGNORECASE,
)

# School names (patterns like "X Primary School", "X Academy", etc.)
_SCHOOL_NAME = re.compile(
    r"\b(?:[A-Z][A-Za-z']+(?:\s+[A-Z][A-Za-z']+){0,4}\s+"
    r"(?:Primary|Secondary|Junior|Infant|Grammar|"
    r"Academy|School|College|Sixth\s+Form|"
    r"High\s+School|Preparatory|Prep\s+School|"
    r"Comprehensive|Community\s+School|"
    r"Free\s+School|Faith\s+School|Church\s+School|"
    r"Special\s+School|SEN\s+School))\b",
)

# Social services / CAFCASS references
_SOCIAL_SERVICES = re.compile(
    r"\b(?:"
    r"CAFCASS|Children\s+and\s+Family\s+Court\s+Advisory|"
    r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+)?"
    r"(?:Social\s+Services|Children'?s?\s+Services|"
    r"Safeguarding\s+(?:Board|Team|Panel)|"
    r"Local\s+Authority\s+Designated\s+Officer|LADO|"
    r"Multi[\s\-]?Agency\s+Safeguarding\s+Hub|MASH|"
    r"Child\s+Protection\s+(?:Team|Unit|Conference|Plan)|"
    r"Section\s+47\s+(?:enquiry|investigation|assessment))"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Age and minor indicators
# ---------------------------------------------------------------------------

# Minor age — specifically child ages that are more identifying
_MINOR_AGE = re.compile(
    r"\b(?:"
    r"aged?\s+(?:[1-9]|1[0-7])|"           # aged 1-17
    r"(?:[1-9]|1[0-7])[\s\-]years?[\s\-]old|"
    r"(?:a\s+)?(?:[1-9]|1[0-7])[\s\-]year[\s\-]old|"
    r"\b(?:toddler|infant|baby|teenager|adolescent|"
    r"pre[\s\-]?teen|pre[\s\-]?schooler)\b"
    r")\b",
    re.IGNORECASE,
)

# School year references (can narrow age)
_SCHOOL_YEAR = re.compile(
    r"\b(?:Year\s+(?:\d{1,2}|[Rr]eception)|"
    r"Key\s+Stage\s+[1-4]|KS[1-4]|"
    r"(?:GCSE|A[\s\-]?[Ll]evel)\s+(?:year|student))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Sexual abuse specific terminology that may contain identifiers
# ---------------------------------------------------------------------------

# ABE interview references (Achieving Best Evidence)
_ABE_REFERENCE = re.compile(
    r"\b(?:ABE\s+(?:interview|recording|transcript)|"
    r"Achieving\s+Best\s+Evidence|"
    r"video[\s\-]recorded\s+interview|VRI)\b",
    re.IGNORECASE,
)

# Specific offence descriptors that combined with other data increase risk
_OFFENCE_DESCRIPTOR = re.compile(
    r"\b(?:Count\s+\d+|"
    r"(?:first|second|third|fourth|fifth)\s+(?:count|charge|offence)|"
    r"(?:specimen|sample)\s+(?:count|charge))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

ABUSE_PATTERNS: list[tuple[re.Pattern, PIICategory, str, float]] = [
    (_RELATIONSHIP_DESCRIPTORS, PIICategory.RELATIONSHIP_DESCRIPTOR, "relationship_victim", 0.88),
    (_REVERSE_RELATIONSHIP, PIICategory.RELATIONSHIP_DESCRIPTOR, "relationship_perpetrator", 0.88),
    (_CARE_INSTITUTION, PIICategory.INSTITUTION_NAME, "care_institution", 0.82),
    (_SCHOOL_NAME, PIICategory.SCHOOL_NAME, "school_name", 0.78),
    (_SOCIAL_SERVICES, PIICategory.INSTITUTION_NAME, "social_services", 0.85),
    (_MINOR_AGE, PIICategory.AGE, "minor_age", 0.93),
    (_SCHOOL_YEAR, PIICategory.AGE, "school_year", 0.80),
    (_ABE_REFERENCE, PIICategory.CUSTOM, "abe_reference", 0.70),
    (_OFFENCE_DESCRIPTOR, PIICategory.CUSTOM, "offence_descriptor", 0.60),
]


def scan_abuse_patterns(
    text: str,
    min_confidence: float = 0.0,
) -> list[PatternMatch]:
    """Scan text for abuse-case-specific PII patterns.

    Args:
        text: Text to scan.
        min_confidence: Minimum confidence threshold.

    Returns:
        List of PatternMatch instances.
    """
    matches: list[PatternMatch] = []

    for pattern, category, name, confidence in ABUSE_PATTERNS:
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

    return matches
