from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class AssessmentPayload:
    """Data collected from a generation event for quality assessment."""
    query: str
    retrieved_chunks: List[Dict[str, Any]]  # text, score, metadata
    generated_answer: str
    model_used: str
    system_prompt: str
    generation_config: Dict[str, Any]
    diagnostics: Dict[str, Any]  # tokens, time, temperature, etc.
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert payload to dictionary for serialization."""
        return {
            "query": self.query,
            "retrieved_chunks": self.retrieved_chunks,
            "generated_answer": self.generated_answer,
            "model_used": self.model_used,
            "system_prompt": self.system_prompt,
            "generation_config": self.generation_config,
            "diagnostics": self.diagnostics,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class EvaluationResult:
    """Result of a cloud-based quality evaluation."""
    provider: str
    timestamp: datetime
    scores: Dict[str, float]  # answer_quality, retrieval_quality, etc.
    suggestions: List[str]
    prompt_improvements: List[str]
    config_changes: Dict[str, Any]
    overall_rating: float
    raw_response: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """Create EvaluationResult from dictionary."""
        return cls(
            provider=data["provider"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            scores=data["scores"],
            suggestions=data["suggestions"],
            prompt_improvements=data["prompt_improvements"],
            config_changes=data["config_changes"],
            overall_rating=data["overall_rating"],
            raw_response=data["raw_response"]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "provider": self.provider,
            "timestamp": self.timestamp.isoformat(),
            "scores": self.scores,
            "suggestions": self.suggestions,
            "prompt_improvements": self.prompt_improvements,
            "config_changes": self.config_changes,
            "overall_rating": self.overall_rating,
            "raw_response": self.raw_response
        }
