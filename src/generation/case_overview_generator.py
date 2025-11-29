"""Case overview generator from document summaries.

Synthesizes a high-level case overview from all document summaries,
with source document tracking for all extracted information.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Optional

from src.generation.summary_based_generator import SummaryBasedGenerator
from src.retrieval.summary_store import DocumentSummary

logger = logging.getLogger(__name__)


@dataclass
class CaseOverview:
    """High-level case overview."""
    overview_id: str
    generated_at: datetime
    
    # Core information
    case_title: str
    case_summary: str  # 2-3 paragraphs
    key_parties: list[dict]  # [{name, role, description, source_documents: []}]
    key_dates: list[dict]  # [{date, event, significance, source_documents: []}]
    key_issues: list[str]  # Main legal issues
    
    # Document statistics
    document_count: int
    date_range: tuple[Optional[date], Optional[date]]
    document_types: dict[str, int]
    
    # Metadata
    model_used: str
    generation_time_seconds: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "overview_id": self.overview_id,
            "generated_at": self.generated_at.isoformat(),
            "case_title": self.case_title,
            "case_summary": self.case_summary,
            "key_parties": self.key_parties,
            "key_dates": self.key_dates,
            "key_issues": self.key_issues,
            "document_count": self.document_count,
            "date_range": [
                d.isoformat() if d else None
                for d in self.date_range
            ],
            "document_types": self.document_types,
            "model_used": self.model_used,
            "generation_time_seconds": self.generation_time_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CaseOverview":
        """Create from dictionary."""
        return cls(
            overview_id=data["overview_id"],
            generated_at=datetime.fromisoformat(data["generated_at"]),
            case_title=data["case_title"],
            case_summary=data["case_summary"],
            key_parties=data["key_parties"],
            key_dates=data["key_dates"],
            key_issues=data["key_issues"],
            document_count=data["document_count"],
            date_range=tuple(
                date.fromisoformat(d) if d else None
                for d in data["date_range"]
            ),
            document_types=data["document_types"],
            model_used=data["model_used"],
            generation_time_seconds=data["generation_time_seconds"],
        )


CASE_OVERVIEW_PROMPT = """You are analyzing a legal case based on document summaries. Generate a comprehensive case overview.

Documents ({count} total):
{summaries}

Provide a JSON response with:
{{
  "case_title": "Brief descriptive title of the case",
  "case_summary": "2-3 paragraph summary of the case",
  "key_parties": [
    {{
      "name": "Party name",
      "role": "Role in case (e.g., Plaintiff, Defendant, Witness)",
      "description": "Brief description",
      "source_documents": ["doc1.pdf", "doc2.pdf"]
    }}
  ],
  "key_dates": [
    {{
      "date": "YYYY-MM-DD or null",
      "event": "Event description",
      "significance": "Why this date matters",
      "source_documents": ["doc1.pdf"]
    }}
  ],
  "key_issues": ["Issue 1", "Issue 2", "Issue 3"]
}}

Focus on accuracy and include source_documents for all extracted information.
"""


from pydantic import BaseModel, Field, ValidationError

class KeyParty(BaseModel):
    name: str
    role: str
    description: str
    source_documents: list[str] = Field(default_factory=list)

class KeyDate(BaseModel):
    date: Optional[str]
    event: str
    significance: str
    source_documents: list[str] = Field(default_factory=list)

class CaseOverviewSchema(BaseModel):
    case_title: str
    case_summary: str
    key_parties: list[KeyParty]
    key_dates: list[KeyDate]
    key_issues: list[str]

class CaseOverviewGenerator(SummaryBasedGenerator):
    """Generate case overview from document summaries."""
    
    def generate_overview(
        self,
        model: Optional[str] = "qwen2.5:72b",  # Upgraded to 72B for better reasoning
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        incremental: bool = False,
    ) -> CaseOverview:
        """Generate comprehensive case overview from all summaries.
        
        Includes source document references for all extracted information.
        
        Args:
            model: Model to use for generation
            progress_callback: Optional progress callback
            incremental: If True, merge with existing overview
            
        Returns:
            Complete case overview
        """
        import time
        import uuid
        
        start_time = time.time()
        
        # Get summaries
        summaries = self.get_all_summaries(summary_type="overview")
        
        if not summaries:
            logger.warning("No summaries found for case overview generation")
            # Return empty overview
            return CaseOverview(
                overview_id=str(uuid.uuid4()),
                generated_at=datetime.now(),
                case_title="No Case Data",
                case_summary="No document summaries available.",
                key_parties=[],
                key_dates=[],
                key_issues=[],
                document_count=0,
                date_range=(None, None),
                document_types={},
                model_used=model or "unknown",
                generation_time_seconds=0.0,
            )
        
        if progress_callback:
            progress_callback("Generating case overview...", 0, 1)
        
        # Format summaries for prompt
        summaries_text = "\n\n".join([
            f"Document: {s.file_name}\n{s.content[:500]}..."
            for s in summaries[:20]  # Limit to first 20 for context
        ])
        
        # Generate overview using LLM with retries
        llm_client = self.get_llm_client(model)
        
        prompt = CASE_OVERVIEW_PROMPT.format(
            count=len(summaries),
            summaries=summaries_text
        )
        
        max_retries = 3
        overview_data = None
        
        for attempt in range(max_retries):
            try:
                response = llm_client.generate(prompt, temperature=0.5)
                
                # Clean response if it contains markdown code blocks
                cleaned_response = response.strip()
                if "```json" in cleaned_response:
                    cleaned_response = cleaned_response.split("```json")[1].split("```")[0].strip()
                elif "```" in cleaned_response:
                    cleaned_response = cleaned_response.split("```")[1].split("```")[0].strip()
                
                # Parse and validate with Pydantic
                try:
                    # First parse as dict to handle potential loose JSON
                    data_dict = json.loads(cleaned_response)
                    # Then validate structure
                    validated_data = CaseOverviewSchema(**data_dict)
                    overview_data = validated_data.model_dump()
                    break  # Success!
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.warning(f"Attempt {attempt+1}/{max_retries} failed validation: {e}")
                    if attempt == max_retries - 1:
                        raise  # Re-raise on last attempt
                    
            except Exception as e:
                logger.error(f"Error generating case overview (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    # Fallback to safe empty structure
                    overview_data = {
                        "case_title": "Error Parsing Case Data",
                        "case_summary": f"Failed to generate structured overview after {max_retries} attempts. Raw response:\n\n{response if 'response' in locals() else str(e)}",
                        "key_parties": [],
                        "key_dates": [],
                        "key_issues": []
                    }
        
        # Calculate document statistics
        doc_types: dict[str, int] = {}
        for summary in summaries:
            doc_type = summary.metadata.get("doc_type", "unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # Create overview object
        overview = CaseOverview(
            overview_id=str(uuid.uuid4()),
            generated_at=datetime.now(),
            case_title=overview_data.get("case_title", "Untitled Case"),
            case_summary=overview_data.get("case_summary", ""),
            key_parties=overview_data.get("key_parties", []),
            key_dates=overview_data.get("key_dates", []),
            key_issues=overview_data.get("key_issues", []),
            document_count=len(summaries),
            date_range=(None, None),  # TODO: Extract from summaries
            document_types=doc_types,
            model_used=model or self.settings.models.llm.default,
            generation_time_seconds=time.time() - start_time,
        )
        
        # Save overview
        self._save_overview(overview)
        
        logger.info(f"Case overview generated in {overview.generation_time_seconds:.2f}s")
        
        return overview
    
    def _save_overview(self, overview: CaseOverview):
        """Save overview to file."""
        overview_file = Path("data/case_overview.json")
        overview_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(overview_file, "w") as f:
            json.dump(overview.to_dict(), f, indent=2)
        
        logger.info(f"Case overview saved to {overview_file}")
    
    def load_overview(self) -> Optional[CaseOverview]:
        """Load existing overview from file."""
        overview_file = Path("data/case_overview.json")
        
        if not overview_file.exists():
            return None
        
        try:
            with open(overview_file) as f:
                data = json.load(f)
            return CaseOverview.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading case overview: {e}")
            return None
