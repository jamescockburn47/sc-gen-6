from typing import List, Dict, Any, Optional
from datetime import datetime
from .assessment_models import AssessmentPayload

class AssessmentCollector:
    """
    Collects metadata from the generation pipeline to build assessment payloads.
    """
    
    def __init__(self):
        self.current_payload: Optional[AssessmentPayload] = None
        
    def collect(self, 
                query: str,
                retrieved_chunks: List[Dict[str, Any]],
                generated_answer: str,
                model_used: str,
                system_prompt: str,
                generation_config: Dict[str, Any],
                diagnostics: Dict[str, Any]) -> AssessmentPayload:
        """
        Create an assessment payload from generation data.
        
        Args:
            query: The user's input query
            retrieved_chunks: List of chunks used for context
            generated_answer: The final answer from the LLM
            model_used: Name of the model used
            system_prompt: The system prompt used
            generation_config: Configuration dict (temp, top_p, etc.)
            diagnostics: Performance metrics (time, tokens, etc.)
            
        Returns:
            AssessmentPayload object ready for evaluation
        """
        # Sanitize chunks to ensure they are serializable
        sanitized_chunks = []
        for chunk in retrieved_chunks:
            sanitized_chunk = {
                "text": chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", str(chunk)),
                "score": chunk.get("score", 0.0) if isinstance(chunk, dict) else getattr(chunk, "score", 0.0),
                "metadata": chunk.get("metadata", {}) if isinstance(chunk, dict) else getattr(chunk, "metadata", {})
            }
            sanitized_chunks.append(sanitized_chunk)
            
        self.current_payload = AssessmentPayload(
            query=query,
            retrieved_chunks=sanitized_chunks,
            generated_answer=generated_answer,
            model_used=model_used,
            system_prompt=system_prompt,
            generation_config=generation_config,
            diagnostics=diagnostics,
            timestamp=datetime.now()
        )
        
        return self.current_payload

    def get_last_payload(self) -> Optional[AssessmentPayload]:
        """Retrieve the most recently collected payload."""
        return self.current_payload
