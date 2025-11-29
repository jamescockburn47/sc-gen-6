"""Intelligent entity extraction for UK litigation documents."""

from __future__ import annotations

import re
import uuid
from collections import Counter
from datetime import datetime
from typing import Iterable

from src.graph.models import GraphNode, GraphEdge, GraphUpdate
from src.graph.storage import GraphStore
from src.schema import ParsedDocument


class GraphExtractionService:
    """Extracts entities from legal documents with UK litigation awareness."""

    # ============================================================
    # EXCLUSION LISTS - Things that look like entities but aren't
    # ============================================================
    
    # Legal/court terms that should NOT be parties
    LEGAL_NOISE = {
        # Judges and court roles
        "justice", "honour", "lord", "lady", "judge", "master", "registrar",
        "recorder", "tribunal", "bench", "court", "chamber", "division",
        
        # Court names and references
        "ewhc", "ewca", "uksc", "ukut", "ewfc", "ukeat", "qbd", "chd", "tcc",
        "chancery", "queens", "kings", "bench", "admiralty", "commercial",
        "mercantile", "patents", "enterprise", "family", "crown", "county",
        "magistrates", "supreme", "appeal", "high",
        
        # Legal terminology
        "claimant", "defendant", "applicant", "respondent", "appellant",
        "petitioner", "intervener", "plaintiff", "pursuer", "defender",
        "witness", "expert", "barrister", "solicitor", "counsel", "advocate",
        
        # Procedural terms
        "cpr", "practice", "direction", "order", "judgment", "ruling",
        "hearing", "trial", "application", "claim", "action", "proceedings",
        "disclosure", "discovery", "pleading", "statement", "schedule",
        "bundle", "skeleton", "submission", "evidence", "exhibit",
        
        # Document types
        "particulars", "defence", "reply", "counterclaim", "rejoinder",
        "affidavit", "declaration", "certificate", "notice", "request",
        
        # Time/date words
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "monday", "tuesday", "wednesday", "thursday", "friday",
        
        # Common abbreviations
        "ltd", "plc", "llp", "inc", "corp", "para", "paras", "page", "pages",
        "section", "sections", "clause", "clauses", "annex", "annexe",
        "appendix", "schedule", "part", "chapter", "article", "regulation",
        
        # Common words that get capitalized
        "the", "and", "for", "from", "with", "this", "that", "which", "where",
        "when", "what", "who", "how", "why", "between", "against", "before",
        "after", "during", "upon", "under", "over", "into", "onto", "about",
        
        # UK specific
        "hmrc", "fca", "pra", "cma", "ofcom", "ofgem", "nhs", "dwp",
        "parliament", "government", "ministry", "department", "secretary",
        "minister", "majesty", "crown", "royal", "british", "english",
        "united", "kingdom", "england", "wales", "scotland", "ireland",
    }
    
    # ============================================================
    # PARTY DETECTION PATTERNS
    # ============================================================
    
    # UK Company patterns - these ARE parties
    # STRICT: Must start with capital word(s), not lowercase/sentence fragments
    COMPANY_PATTERNS = [
        # "ABC Limited" or "XYZ Ltd" - proper noun(s) followed by company suffix
        # Requires: Capital word(s), then Ltd/Limited/PLC/LLP
        re.compile(r"\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*){0,4})\s+(?:Limited|Ltd\.?)\b"),
        re.compile(r"\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*){0,4})\s+(?:PLC|Plc|LLP|LP)\b"),
        # "ABC & Co" - proper noun then & Co
        re.compile(r"\b([A-Z][a-z]+)\s+(?:&|and)\s+Co\.?\b"),
        # "ABC Corporation" or "ABC Inc"
        re.compile(r"\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*){0,3})\s+(?:Corporation|Corp\.?|Inc\.?)\b"),
    ]
    
    # REMOVED: Person patterns too noisy - captures witnesses, counsel, etc.
    # Focus only on companies with explicit suffixes (Ltd, LLP, PLC)
    PERSON_PATTERNS = []
    
    # Judges - explicitly excluded
    JUDGE_PATTERNS = [
        re.compile(r"\b(?:Mr|Mrs|Lord|Lady|Dame|Sir)\s+Justice\s+[A-Z][a-z]+"),
        re.compile(r"\bJudge\s+[A-Z][a-z]+"),
        re.compile(r"\b(?:His|Her)\s+Honou?r\s+Judge\s+[A-Z][a-z]+"),
    ]
    
    # Litigation role patterns - extract the actual party name
    # These need to be very precise to avoid noise
    ROLE_PATTERNS = [
        # "ABC Limited (the 'Claimant')" - company before role designation
        re.compile(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:Limited|Ltd|PLC|LLP))\s*\(\s*(?:the\s+)?[\"']?(?:First|Second|Third|Fourth|Fifth)?\s*(?:Claimant|Defendant|Applicant|Respondent)[\"']?\s*\)", re.IGNORECASE),
    ]
    
    # REMOVED: "v" pattern causes too much noise from case citations
    # Every cited case (e.g., "Ruhan v Wilder") was being extracted as parties
    # VERSUS_PATTERN = None
    
    # Date patterns
    DATE_PATTERNS = [
        # "25 December 2024" or "25th December 2024"
        re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b", re.IGNORECASE),
        # "25/12/2024" or "25-12-2024"
        re.compile(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b"),
        # "2024-12-25" (ISO format)
        re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
    ]

    # Document types to extract from (authoritative for parties/relationships)
    EXTRACTABLE_DOC_TYPES = {
        "witness_statement",
        "pleading", 
        "court_filing",
        "skeleton_argument",
        "expert_report",
        # Judgments if you have that type
    }

    def __init__(self, store: GraphStore | None = None):
        self.store = store or GraphStore()

    def extract(self, document: ParsedDocument) -> GraphUpdate:
        """Produce a GraphUpdate proposal from a parsed document.
        
        Only extracts entities from authoritative document types:
        - Witness statements, pleadings, court filings, skeleton arguments
        
        Other document types (disclosure, emails, contracts) are linked
        as documents but don't contribute entity extraction.
        """
        document_node = GraphNode(
            node_id=document.file_path,
            label=document.file_name,
            node_type="document",
            metadata={
                "document_type": document.document_type,
                "parsed_at": document.parsed_at.isoformat(),
            },
        )

        # Only extract from authoritative document types
        doc_type = document.document_type or "unknown"
        if doc_type in self.EXTRACTABLE_DOC_TYPES:
            parties = self._extract_parties(document)
            events = self._extract_events(document)
        else:
            # Non-authoritative docs: just add document node, no entity extraction
            parties = []
            events = []

        # Build edges
        edges: list[GraphEdge] = []
        for party in parties:
            edges.append(
                GraphEdge(
                    source=party.node_id,
                    target=document_node.node_id,
                    relation="mentioned_in",
                    metadata={},
                )
            )

        for event in events:
            edges.append(
                GraphEdge(
                    source=document_node.node_id,
                    target=event.node_id,
                    relation="references_date",
                    metadata={"date": event.metadata.get("date")},
                )
            )

        return GraphUpdate(
            document_id=document_node.node_id,
            document_name=document.file_name,
            nodes=[document_node, *parties, *events],
            edges=edges,
            auto_generated=True,
            notes="Auto-extracted entities (companies, persons, dates). Review and edit as needed.",
        )

    def queue_update(self, document: ParsedDocument) -> None:
        """Run extraction and queue for review."""
        update = self.extract(document)
        self.store.queue_update(update)

    # ============================================================
    # EXTRACTION METHODS
    # ============================================================
    
    def _extract_parties(self, document: ParsedDocument) -> list[GraphNode]:
        """Extract parties using intelligent pattern matching."""
        text = document.text
        candidates: dict[str, dict] = {}  # normalized_name -> {name, type, count}
        
        # Build set of judge names to exclude
        judge_names = set()
        for pattern in self.JUDGE_PATTERNS:
            for match in pattern.finditer(text):
                judge_names.add(self._normalize_name(match.group(0)))
        
        # NOTE: "v" pattern removed - caused massive noise from case citations
        # Every "Smith v Jones" cited was being extracted as parties
        
        # 1. Extract companies with explicit suffixes (most reliable)
        for pattern in self.COMPANY_PATTERNS:
            for match in pattern.finditer(text):
                name = self._clean_party_name(match.group(0))  # Full match including suffix
                if name and self._is_valid_party(name) and self._normalize_name(name) not in judge_names:
                    self._add_candidate(candidates, name, "company", 2)
        
        # 2. Extract from role patterns (explicit "Claimant"/"Defendant" designations)
        for pattern in self.ROLE_PATTERNS:
            for match in pattern.finditer(text):
                name = self._clean_party_name(match.group(1))
                if name and self._is_valid_party(name) and self._normalize_name(name) not in judge_names:
                    self._add_candidate(candidates, name, "litigant", 3)
        
        # NOTE: Person patterns removed - too noisy (captures witnesses, counsel, etc.)
        
        # Filter by minimum confidence and create nodes
        # Require score >= 6 (appears 3+ times)
        # This filters out one-off mentions from case citations
        # Only companies mentioned repeatedly are likely parties to THIS case
        nodes: list[GraphNode] = []
        for normalized, info in candidates.items():
            if info["score"] >= 6:
                node_id = f"party_{uuid.uuid4().hex[:8]}"
                nodes.append(
                    GraphNode(
                        node_id=node_id,
                        label=info["display_name"],
                        node_type="party",
                        metadata={
                            "source_file": document.file_name,
                            "entity_type": info["type"],
                            "confidence": min(info["score"] / 5.0, 1.0),
                        },
                    )
                )
        
        return nodes
    
    def _add_candidate(self, candidates: dict, name: str, entity_type: str, score: int):
        """Add or update a candidate party."""
        normalized = self._normalize_name(name)
        if normalized in candidates:
            candidates[normalized]["score"] += score
            # Keep the longer/better formatted version
            if len(name) > len(candidates[normalized]["display_name"]):
                candidates[normalized]["display_name"] = name
        else:
            candidates[normalized] = {
                "display_name": name,
                "type": entity_type,
                "score": score,
            }
    
    def _clean_party_name(self, name: str) -> str:
        """Clean up extracted party name."""
        if not name:
            return ""
        # Remove extra whitespace
        name = " ".join(name.split())
        # Remove leading/trailing punctuation
        name = name.strip(".,;:()[]\"'")
        # Remove common prefixes that aren't part of the name
        for prefix in ["the ", "The ", "a ", "A ", "an ", "An "]:
            if name.startswith(prefix) and len(name) > len(prefix) + 3:
                name = name[len(prefix):]
        return name.strip()
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for deduplication."""
        # Lowercase, remove punctuation, collapse whitespace
        normalized = re.sub(r"[^\w\s]", "", name.lower())
        normalized = " ".join(normalized.split())
        return normalized
    
    def _is_valid_party(self, name: str) -> bool:
        """Check if name is a valid party (not noise)."""
        if not name or len(name) < 3:
            return False
        
        name_lower = name.lower()
        name_words = set(name_lower.split())
        
        # Reject if starts with common noise words
        first_word = name_lower.split()[0] if name_lower.split() else ""
        if first_word in {"the", "a", "an", "or", "and", "any", "all", "no", "each", "every", "seeks", "alleges", "claims"}:
            return False
        
        # Reject if contains obvious sentence fragments
        bad_phrases = {"or ", " or ", "and ", " and ", " a ", " the ", " is ", " was ", " are ", " were ", " be ", " been ", " being ", " has ", " have ", " had ", " will ", " would ", " could ", " should ", " may ", " might ", " must ", " shall ", " can "}
        if any(bp in name_lower for bp in bad_phrases):
            return False
        
        # Reject if too many words (probably a sentence fragment)
        if len(name_words) > 5:
            return False
        
        # If ALL words are noise, reject
        if name_words.issubset(self.LEGAL_NOISE):
            return False
        
        # If the entire name (normalized) is in noise list
        normalized = self._normalize_name(name)
        if normalized in self.LEGAL_NOISE:
            return False
        
        # Single word that's just an acronym (all caps, no spaces)
        if name.isupper() and " " not in name and len(name) < 6:
            if name_lower in self.LEGAL_NOISE:
                return False
        
        # Reject if it's just a title
        if name_lower in {"mr", "mrs", "ms", "miss", "dr", "prof", "sir", "dame", "company", "corporation", "firm", "entity", "person", "individual", "party", "defendant", "claimant"}:
            return False
        
        # Must contain at least one proper noun indicator (capital letter not at start of sentence)
        # or be a known company suffix
        has_company_suffix = any(suf in name_lower for suf in ["limited", "ltd", "plc", "llp", "inc", "corp"])
        has_proper_noun = bool(re.search(r"[A-Z][a-z]", name))
        
        if not has_company_suffix and not has_proper_noun:
            return False
        
        # Reject judges
        if "justice" in name_lower or "judge" in name_lower or "honour" in name_lower:
            return False
        
        # Reject common legal document fragments
        if any(frag in name_lower for frag in ["incorporated", "thereafter", "including", "pursuant", "notwithstanding", "hereinafter", "aforementioned"]):
            return False
        
        # Reject generic company name fragments (no proper noun before suffix)
        generic_starts = ["holdings", "investments", "properties", "services", "management", "realisations", "recoveries", "solicitors", "london"]
        first_word = name_lower.split()[0] if name_lower.split() else ""
        if first_word in generic_starts:
            return False
        
        return True

    def _extract_events(self, document: ParsedDocument) -> list[GraphNode]:
        """Extract significant dates as timeline events."""
        text = document.text
        nodes: list[GraphNode] = []
        seen_dates: set[str] = set()
        date_counts: Counter = Counter()
        
        # Count date occurrences to find significant ones
        for pattern in self.DATE_PATTERNS:
            for match in pattern.finditer(text):
                normalized = self._normalize_date_match(match)
                if normalized:
                    date_counts[normalized] += 1
        
        # Only include dates that appear 3+ times (significant)
        # One-off dates are usually citations or incidental mentions
        for date_str, count in date_counts.items():
            if count >= 3:
                if date_str not in seen_dates:
                    seen_dates.add(date_str)
                    node_id = f"event_{uuid.uuid4().hex[:8]}"
                    nodes.append(
                        GraphNode(
                            node_id=node_id,
                            label=self._format_date_label(date_str),
                            node_type="event",
                            metadata={
                                "date": date_str,
                                "source_file": document.file_name,
                                "occurrences": count,
                            },
                        )
                    )
        
        return nodes
    
    def _normalize_date_match(self, match: re.Match) -> str | None:
        """Normalize a date match to YYYY-MM-DD format."""
        groups = match.groups()
        
        # Handle different pattern formats
        if len(groups) == 3:
            g1, g2, g3 = groups
            
            # Check if it's ISO format (YYYY-MM-DD)
            if len(str(g1)) == 4 and str(g1).isdigit():
                return f"{g1}-{g2.zfill(2)}-{g3.zfill(2)}"
            
            # Month name format (day, month_name, year)
            if isinstance(g2, str) and g2.isalpha():
                months = {
                    "january": "01", "february": "02", "march": "03", "april": "04",
                    "may": "05", "june": "06", "july": "07", "august": "08",
                    "september": "09", "october": "10", "november": "11", "december": "12"
                }
                month = months.get(g2.lower())
                if month:
                    return f"{g3}-{month}-{str(g1).zfill(2)}"
            
            # Numeric format (DD/MM/YYYY)
            if str(g1).isdigit() and str(g2).isdigit() and str(g3).isdigit():
                day, month, year = g1, g2, g3
                if int(month) <= 12 and int(day) <= 31:
                    return f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
        
        return None
    
    def _is_significant_date_context(self, text: str, date_str: str) -> bool:
        """Check if date appears in significant context."""
        # Look for keywords near the date that indicate importance
        significant_keywords = [
            "agreement", "contract", "signed", "executed", "dated",
            "hearing", "trial", "judgment", "order", "filed",
            "incident", "accident", "breach", "termination", "notice",
        ]
        # Simple check - could be enhanced with proximity search
        text_lower = text.lower()
        return any(kw in text_lower for kw in significant_keywords)
    
    def _format_date_label(self, date_str: str) -> str:
        """Format date for display."""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.strftime("%d %B %Y")
        except ValueError:
            return date_str
