"""Citation guard — protects published case law citations from anonymisation.

Ported from CaseKit's citationExtractor.ts (legalquant/casekit).

Published case law is public domain. Case names, neutral citations, and
law report references are THE LAW — they must survive anonymisation intact
so a cloud LLM can actually do legal analysis with the exported text.

This module:
  1. Extracts all case citations (neutral + traditional law reports)
  2. Extracts associated case names (walking backward from "v")
  3. Returns "protected spans" — character ranges the anonymiser must skip
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class ProtectedCitation:
    """A case citation span that must not be anonymised."""

    citation: str          # The neutral/traditional citation string
    case_name: Optional[str]  # e.g. "Smith v Brown" if detected
    start: int             # Character offset of the full span start
    end: int               # Character offset of the full span end
    is_neutral: bool       # True for neutral citations, False for traditional


# ---------------------------------------------------------------------------
# Neutral citation patterns (UK courts)
# ---------------------------------------------------------------------------

_NEUTRAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("UKSC", re.compile(r"\[(\d{4})\]\s+UKSC\s+(\d+)", re.IGNORECASE)),
    ("UKHL", re.compile(r"\[(\d{4})\]\s+UKHL\s+(\d+)", re.IGNORECASE)),
    ("UKPC", re.compile(r"\[(\d{4})\]\s+UKPC\s+(\d+)", re.IGNORECASE)),
    ("EWCA Civ", re.compile(r"\[(\d{4})\]\s+EWCA\s+Civ\s+(\d+)", re.IGNORECASE)),
    ("EWCA Crim", re.compile(r"\[(\d{4})\]\s+EWCA\s+Crim\s+(\d+)", re.IGNORECASE)),
    ("EWHC", re.compile(r"\[(\d{4})\]\s+EWHC\s+(\d+)(?:\s+\([A-Za-z]+\))?", re.IGNORECASE)),
    ("EWCOP", re.compile(r"\[(\d{4})\]\s+EWCOP\s+(\d+)", re.IGNORECASE)),
    ("EWFC", re.compile(r"\[(\d{4})\]\s+EWFC\s+(\d+)", re.IGNORECASE)),
    ("UKUT", re.compile(r"\[(\d{4})\]\s+UKUT\s+(\d+)(?:\s+\([A-Za-z]+\))?", re.IGNORECASE)),
    ("UKFTT", re.compile(r"\[(\d{4})\]\s+UKFTT\s+(\d+)(?:\s+\([A-Za-z]+\))?", re.IGNORECASE)),
    ("UKEAT", re.compile(r"\[(\d{4})\]\s+UKEAT\s+(\d+)", re.IGNORECASE)),
    # Scottish courts
    ("CSIH", re.compile(r"\[(\d{4})\]\s+CSIH\s+(\d+)", re.IGNORECASE)),
    ("CSOH", re.compile(r"\[(\d{4})\]\s+CSOH\s+(\d+)", re.IGNORECASE)),
    ("ScotCS", re.compile(r"\[(\d{4})\]\s+ScotCS\s+(\d+)", re.IGNORECASE)),
    # Northern Ireland
    ("NICA", re.compile(r"\[(\d{4})\]\s+NICA\s+(\d+)", re.IGNORECASE)),
    ("NIQB", re.compile(r"\[(\d{4})\]\s+NIQB\s+(\d+)", re.IGNORECASE)),
    ("UKIPO", re.compile(r"\[(\d{4})\]\s+UKIPO\s+(\d+)", re.IGNORECASE)),
]

# ---------------------------------------------------------------------------
# Traditional law report patterns
# ---------------------------------------------------------------------------

_TRADITIONAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"\[\d{4}\]\s+\d*\s*AC\s+\d+", re.IGNORECASE),
    re.compile(r"\[\d{4}\]\s+\d*\s*QB\s+\d+", re.IGNORECASE),
    re.compile(r"\[\d{4}\]\s+\d*\s*KB\s+\d+", re.IGNORECASE),
    re.compile(r"\[\d{4}\]\s+\d*\s*WLR\s+\d+", re.IGNORECASE),
    re.compile(r"\[\d{4}\]\s+\d*\s*All\s*ER\s+\d+", re.IGNORECASE),
    re.compile(r"\[\d{4}\]\s+\d*\s*Ch\s+\d+", re.IGNORECASE),
    re.compile(r"\[\d{4}\]\s+\d*\s*Fam\s+\d+", re.IGNORECASE),
    re.compile(r"\[\d{4}\]\s+\d*\s*ICR\s+\d+", re.IGNORECASE),
    re.compile(r"\[\d{4}\]\s+\d*\s*IRLR\s+\d+", re.IGNORECASE),
    re.compile(r"\[\d{4}\]\s+\d*\s*FLR\s+\d+", re.IGNORECASE),
    re.compile(r"\[\d{4}\]\s+\d*\s*BCLC\s+\d+", re.IGNORECASE),
    re.compile(r"\[\d{4}\]\s+\d*\s*BCC\s+\d+", re.IGNORECASE),
    re.compile(r"\[\d{4}\]\s+\d*\s*Lloyd['\u2019]?\s*s\s+Rep\s+\d+", re.IGNORECASE),
    re.compile(r"\[\d{4}\]\s+\d*\s*P\s*&\s*CR\s+\d+", re.IGNORECASE),
    re.compile(r"\[\d{4}\]\s+\d*\s*HLR\s+\d+", re.IGNORECASE),
    re.compile(r"\[\d{4}\]\s+\d*\s*CMLR\s+\d+", re.IGNORECASE),
    # Competition reports
    re.compile(r"\[\d{4}\]\s+\d*\s*Comp\s*AR\s+\d+", re.IGNORECASE),
    re.compile(r"\[\d{4}\]\s+\d*\s*UKCLR\s+\d+", re.IGNORECASE),
    # Older series with round brackets: (1942) Ch 304
    re.compile(r"\(\d{4}\)\s+\d*\s*Ch\s+\d+", re.IGNORECASE),
    re.compile(r"\(\d{4}\)\s+\d*\s*AC\s+\d+", re.IGNORECASE),
    re.compile(r"\(\d{4}\)\s+\d*\s*QB\s+\d+", re.IGNORECASE),
    re.compile(r"\(\d{4}\)\s+\d*\s*KB\s+\d+", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Case name extraction
# ---------------------------------------------------------------------------

# Legal name connectors — allowed between proper nouns in a party name
_NAME_CONNECTORS = frozenset({
    "of", "the", "for", "and", "&", "de", "van", "von", "du", "la", "le", "el",
})

# Legal entity suffixes
_LEGAL_SUFFIXES = re.compile(
    r"^(Ltd|Limited|Plc|PLC|LLP|Inc|Corp|LLC|Council|Borough|NHS|CIC|Ors|ORS)$",
    re.IGNORECASE,
)


def _extract_case_name(text: str, citation_start: int) -> tuple[Optional[str], int]:
    """Extract the case name preceding a citation.

    Ported from CaseKit's extractCaseName().
    Uses backward word-walk from "v" to find party boundaries.

    Args:
        text: Full document text.
        citation_start: Character offset where the citation begins.

    Returns:
        Tuple of (case_name, name_start_offset).
        name_start_offset is the character position where the case name begins.
    """
    # Grab text before the citation
    window_start = max(0, citation_start - 300)
    before = text[window_start:citation_start].rstrip(",;: \t\n")

    # Special case: "R v Name" or "R (Name) v Name"
    r_match = re.search(
        r"\bR\s*(?:\([^)]+\)\s*)?v\.?\s+[A-Z][A-Za-z'\'\-]+(?:\s+[A-Za-z'\'\-&()]+)*$",
        before,
    )
    if r_match:
        name = r_match.group().strip()
        abs_start = window_start + r_match.start()
        return name, abs_start

    # Special case: "In re Name" or "Re Name"
    re_match = re.search(
        r"\b(?:In\s+re|Re)\s+[A-Z][A-Za-z'\'\-]+(?:\s+[A-Za-z'\'\-&()]+)*$",
        before,
        re.IGNORECASE,
    )
    if re_match:
        name = re_match.group().strip()
        abs_start = window_start + re_match.start()
        return name, abs_start

    # General "Party A v Party B"
    v_regex = re.compile(r"\s+v\.?\s+")
    last_v_match = None
    for m in v_regex.finditer(before):
        last_v_match = m

    if not last_v_match:
        return None, citation_start

    v_start = last_v_match.start()
    v_end = last_v_match.end()

    # Party 2 = everything after "v" to end
    party2 = before[v_end:].strip()
    if not party2 or not party2[0].isupper():
        return None, citation_start

    # Party 1 = walk backwards from "v" word by word
    # We split on whitespace but also track each word's actual position
    # in the original text (to handle newlines, multi-space, etc.)
    party1_raw = before[:v_start]
    # Build word list with positions
    word_spans: list[tuple[str, int, int]] = []  # (word, start_in_before, end_in_before)
    for wm in re.finditer(r"\S+", party1_raw):
        word_spans.append((wm.group(), wm.start(), wm.end()))

    if not word_spans:
        return None, citation_start

    start_idx = len(word_spans)

    for i in range(len(word_spans) - 1, -1, -1):
        word = word_spans[i][0]
        bare = re.sub(r"[,;:()]", "", word)

        is_upper_start = bool(bare) and bare[0].isupper()
        is_connector = bare.lower() in _NAME_CONNECTORS
        is_legal_suffix = bool(_LEGAL_SUFFIXES.match(bare))
        is_ampersand = bare == "&"

        if is_upper_start or is_legal_suffix or is_ampersand:
            start_idx = i
        elif is_connector:
            if start_idx == i + 1:
                start_idx = i
            else:
                break
        else:
            break

    if start_idx >= len(word_spans):
        return None, citation_start

    party1 = " ".join(ws[0] for ws in word_spans[start_idx:])
    if not party1 or not party2:
        return None, citation_start

    if not any(c.isupper() for c in party1):
        return None, citation_start

    case_name = f"{party1} v {party2}"
    if len(case_name) < 5 or len(case_name) > 200:
        return None, citation_start

    # Calculate absolute start position using the actual word position
    # (not string search — avoids failure on newline/whitespace differences)
    first_word_pos_in_before = word_spans[start_idx][1]
    abs_start = window_start + first_word_pos_in_before

    return case_name, abs_start


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_protected_citations(text: str) -> list[ProtectedCitation]:
    """Extract all published case citations and mark them as protected spans.

    These spans must NOT be anonymised — they are public law.

    Args:
        text: Full document text.

    Returns:
        List of ProtectedCitation instances with character offsets.
    """
    citations: list[ProtectedCitation] = []
    seen_spans: set[tuple[int, int]] = set()

    def _add_citation(
        match: re.Match,
        is_neutral: bool,
    ) -> None:
        """Add a citation, extending the span to include the case name."""
        cit_text = match.group().strip()
        cit_start = match.start()
        cit_end = match.end()

        # Try to extract the case name before the citation
        case_name, name_start = _extract_case_name(text, cit_start)

        # The protected span covers both case name and citation
        span_start = name_start if case_name else cit_start
        span_end = cit_end

        # Skip if we've already seen an overlapping span
        for existing_start, existing_end in seen_spans:
            if span_start < existing_end and span_end > existing_start:
                # Extend existing span if this one is wider
                if span_start < existing_start or span_end > existing_end:
                    seen_spans.discard((existing_start, existing_end))
                    span_start = min(span_start, existing_start)
                    span_end = max(span_end, existing_end)
                else:
                    return  # Fully contained in existing span

        seen_spans.add((span_start, span_end))

        citations.append(ProtectedCitation(
            citation=cit_text,
            case_name=case_name,
            start=span_start,
            end=span_end,
            is_neutral=is_neutral,
        ))

    # Pass 1: Neutral citations (higher priority)
    for _code, pattern in _NEUTRAL_PATTERNS:
        for match in pattern.finditer(text):
            _add_citation(match, is_neutral=True)

    # Pass 2: Traditional law report citations
    for pattern in _TRADITIONAL_PATTERNS:
        for match in pattern.finditer(text):
            _add_citation(match, is_neutral=False)

    # Sort by position
    citations.sort(key=lambda c: c.start)

    if citations:
        logger.debug(
            f"Citation guard: {len(citations)} published citations protected "
            f"({', '.join(c.citation for c in citations[:5])})"
        )

    return citations


def is_in_protected_span(
    start: int,
    end: int,
    protected: list[ProtectedCitation],
) -> bool:
    """Check if a character range falls within any protected citation span.

    Used by the PII detector to skip entities that are part of
    published case law.

    Args:
        start: Entity start offset.
        end: Entity end offset.
        protected: List of protected citation spans.

    Returns:
        True if the entity overlaps with a protected citation.
    """
    for citation in protected:
        # Check overlap
        if start < citation.end and end > citation.start:
            return True
    return False
