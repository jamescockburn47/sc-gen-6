"""Guardrail service for input validation and output safety."""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class GuardrailCheck:
    passed: bool
    reason: str
    severity: str  # "info", "warning", "block"

class GuardrailService:
    """Service for enforcing safety and relevance guardrails."""

    def __init__(self):
        # Simple patterns that might indicate non-litigation use or jailbreak attempts
        self.jailbreak_patterns = [
            re.compile(r"ignore previous instructions", re.IGNORECASE),
            re.compile(r"act as a", re.IGNORECASE),
            re.compile(r"you are now", re.IGNORECASE),
        ]
        
    def validate_input(self, query: str) -> GuardrailCheck:
        """Validate user input query.
        
        Args:
            query: User query string
            
        Returns:
            GuardrailCheck result
        """
        if not query or len(query.strip()) < 3:
            return GuardrailCheck(False, "Query too short", "block")
            
        for pattern in self.jailbreak_patterns:
            if pattern.search(query):
                return GuardrailCheck(False, "Potential jailbreak attempt detected", "warning")
                
        return GuardrailCheck(True, "Input valid", "info")

    def validate_retrieval_confidence(self, chunks: List[dict], threshold: float) -> GuardrailCheck:
        """Validate retrieved chunks against confidence threshold.
        
        Args:
            chunks: List of retrieved chunks
            threshold: Confidence threshold (0.0-1.0)
            
        Returns:
            GuardrailCheck result
        """
        if not chunks:
            return GuardrailCheck(False, "No documents found", "block")
            
        # Check if we have reranker scores (usually < 1.0) or RRF scores (usually small)
        # Assuming reranker scores here if available
        valid_chunks = [c for c in chunks if c.get("score", 0) >= threshold]
        
        if not valid_chunks:
            # Check if we are using RRF (max score is very small)
            max_score = max((c.get("score", 0) for c in chunks), default=0)
            if max_score < 0.1: 
                # Likely RRF scores, skip threshold check
                return GuardrailCheck(True, "RRF scores used, skipping threshold", "info")
                
            return GuardrailCheck(
                False, 
                f"No sources met confidence threshold ({threshold:.2f}). Best match: {max_score:.2f}", 
                "block"
            )
            
        return GuardrailCheck(True, f"Found {len(valid_chunks)} valid chunks", "info")

    def validate_output(self, response: str, verification_result: dict) -> GuardrailCheck:
        """Validate generated output.
        
        Args:
            response: LLM generated text
            verification_result: Result from CitationVerifier
            
        Returns:
            GuardrailCheck result
        """
        # Check for hallucination (low citation ratio)
        total_citations = verification_result.get("total_citations", 0)
        valid_citations = verification_result.get("valid_citations", 0)
        
        if total_citations > 0:
            ratio = valid_citations / total_citations
            if ratio < 0.5:
                return GuardrailCheck(False, "High hallucination risk: < 50% citations valid", "warning")
                
        # Check for uncited claims if required
        # (This logic is handled by the verifier mostly, but we can enforce stricter rules here)
        
        return GuardrailCheck(True, "Output valid", "info")






