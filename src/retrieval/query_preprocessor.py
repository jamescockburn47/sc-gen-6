"""Query preprocessing for improved retrieval.

Strips conversational phrases and extracts substantive query content
before passing to embedding, keyword search, and reranking.
"""

import re
from typing import Optional, Tuple

# Conversational prefixes to strip (case-insensitive)
# These are meant for the LLM, not the retrieval system
CONVERSATIONAL_PREFIXES = [
    # Questions
    r"^tell me about\s+",
    r"^what is\s+",
    r"^what are\s+",
    r"^what was\s+",
    r"^what were\s+",
    r"^what do you know about\s+",
    r"^what can you tell me about\s+",
    r"^can you tell me about\s+",
    r"^can you explain\s+",
    r"^please explain\s+",
    r"^explain\s+",
    r"^describe\s+",
    r"^who is\s+",
    r"^who are\s+",
    r"^who was\s+",
    r"^when did\s+",
    r"^when was\s+",
    r"^where is\s+",
    r"^where was\s+",
    r"^how does\s+",
    r"^how did\s+",
    r"^why did\s+",
    r"^why does\s+",
    r"^why is\s+",
    # Requests
    r"^give me information about\s+",
    r"^give me details about\s+",
    r"^provide information about\s+",
    r"^provide details about\s+",
    r"^find information about\s+",
    r"^find\s+",
    r"^search for\s+",
    r"^look up\s+",
    r"^show me\s+",
    r"^i want to know about\s+",
    r"^i need to know about\s+",
    r"^i'm looking for\s+",
    r"^i am looking for\s+",
    # Polite prefixes
    r"^please\s+",
    r"^could you\s+",
    r"^would you\s+",
    r"^can you\s+",
    # Articles/filler at start after stripping
    r"^the\s+",
    r"^a\s+",
    r"^an\s+",
]

# Suffixes to strip
CONVERSATIONAL_SUFFIXES = [
    r"\s*\?$",  # Trailing question mark
    r"\s*please\s*$",
    r"\s*thanks\s*$",
    r"\s*thank you\s*$",
]

# Compile patterns for efficiency
_PREFIX_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CONVERSATIONAL_PREFIXES]
_SUFFIX_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CONVERSATIONAL_SUFFIXES]


def preprocess_query(query: str) -> Tuple[str, str]:
    """Preprocess a user query for retrieval.
    
    Strips conversational phrases that are meant for the LLM,
    extracting the substantive search content.
    
    Args:
        query: Raw user query (e.g., "tell me about the smith metadata issue")
        
    Returns:
        Tuple of (processed_query, original_query):
        - processed_query: Stripped query for retrieval ("smith metadata issue")
        - original_query: Preserved for LLM context
    """
    original = query
    processed = query.strip()
    
    # Strip prefixes (may need multiple passes for nested prefixes)
    changed = True
    max_iterations = 5  # Prevent infinite loops
    iterations = 0
    
    while changed and iterations < max_iterations:
        changed = False
        for pattern in _PREFIX_PATTERNS:
            new_processed = pattern.sub("", processed, count=1)
            if new_processed != processed:
                processed = new_processed.strip()
                changed = True
                break  # Restart from beginning after a match
        iterations += 1
    
    # Strip suffixes
    for pattern in _SUFFIX_PATTERNS:
        processed = pattern.sub("", processed).strip()
    
    # Ensure we have something left
    if not processed or len(processed) < 3:
        processed = original.strip()
    
    # Log if significant stripping occurred
    if processed != original.strip():
        print(f"[QUERY PREPROCESS] '{original}' → '{processed}'")
    
    return processed, original


def detect_entity_query(query: str) -> tuple[bool, list[str]]:
    """Detect if a query contains proper nouns/entities that need exact matching.
    
    Identifies queries that should use keyword-only search for precision.
    Proper nouns like company names, person names, etc. need exact matching,
    not semantic similarity (which may return phonetically similar but wrong results).
    
    Args:
        query: The search query
        
    Returns:
        Tuple of (is_entity_query, entity_terms):
        - is_entity_query: True if query contains proper nouns needing exact match
        - entity_terms: List of detected entity terms
    """
    entity_terms = []
    
    # Pattern 1: Quoted phrases (user explicitly wants exact match)
    quoted = re.findall(r'"([^"]+)"', query)
    entity_terms.extend(quoted)
    
    # Pattern 2: Capitalized words that aren't at sentence start
    # Split and check for capitalized words after the first word
    words = query.split()
    for i, word in enumerate(words):
        # Clean word of punctuation for checking
        clean_word = re.sub(r'[^\w]', '', word)
        
        if not clean_word:
            continue
            
        # Check if word is capitalized (not all caps, not all lower)
        if clean_word[0].isupper() and not clean_word.isupper():
            # Skip common sentence starters
            if i == 0 and clean_word.lower() in {'who', 'what', 'when', 'where', 'why', 'how', 'is', 'are', 'was', 'were', 'tell', 'show', 'find', 'the', 'a'}:
                continue
            # Skip common words that might be capitalized
            if clean_word.lower() in {'i', 'and', 'or', 'but', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}:
                continue
            # This looks like a proper noun
            entity_terms.append(clean_word)
    
    # Pattern 3: Single-word query with capital letter (likely a name)
    if len(words) == 1:
        clean = re.sub(r'[^\w]', '', words[0])
        if clean and clean[0].isupper() and len(clean) > 2:
            if clean not in entity_terms:
                entity_terms.append(clean)
    
    # Pattern 4: CamelCase or mixed case (likely proper noun)
    for word in words:
        clean = re.sub(r'[^\w]', '', word)
        # Check for mixed case like "McDonald" or "iOS"
        if re.search(r'[a-z][A-Z]', clean):
            if clean not in entity_terms:
                entity_terms.append(clean)
    
    is_entity_query = len(entity_terms) > 0
    
    if is_entity_query:
        print(f"[ENTITY DETECT] Query contains entities: {entity_terms} → forcing keyword search")
    
    return is_entity_query, entity_terms


def extract_key_terms(query: str) -> list[str]:
    """Extract key search terms from a query.
    
    Useful for keyword search where we want specific terms.
    
    Args:
        query: Preprocessed query
        
    Returns:
        List of key terms (nouns, proper nouns, significant words)
    """
    # Remove common stop words for search
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "there", "here", "when", "where", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such",
        "no", "not", "only", "same", "so", "than", "too", "very", "just",
        "also", "now", "any", "about", "into", "through", "during", "before",
        "after", "above", "below", "between", "under", "again", "further",
        "then", "once", "issue", "issues", "regarding", "concerning", "relating"
    }
    
    # Tokenize and filter
    words = re.findall(r'\b\w+\b', query.lower())
    key_terms = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    
    return key_terms


def enhance_query_for_legal(query: str) -> str:
    """Enhance query with legal-specific expansions.
    
    Adds common legal variations and synonyms.
    
    Args:
        query: Preprocessed query
        
    Returns:
        Enhanced query with expansions
    """
    # Common legal synonyms/expansions
    LEGAL_EXPANSIONS = {
        "contract": ["agreement", "deed"],
        "agreement": ["contract", "deed"],
        "claim": ["allegation", "assertion"],
        "evidence": ["proof", "exhibit", "document"],
        "witness": ["testimony", "deponent"],
        "court": ["tribunal", "judge"],
        "defendant": ["respondent"],
        "claimant": ["plaintiff", "applicant"],
        "breach": ["violation", "non-compliance"],
        "damages": ["compensation", "loss"],
        "liability": ["responsibility", "obligation"],
    }
    
    # For now, just return the query - expansion can be added later
    # This is a placeholder for future enhancement
    return query
