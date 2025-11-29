"""Document renamer from summaries.

Generates precise, succinct document names from summaries while
preserving original filenames in brackets.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from src.generation.summary_based_generator import SummaryBasedGenerator
from src.retrieval.summary_store import DocumentSummary

logger = logging.getLogger(__name__)


@dataclass
class DocumentName:
    """Document naming information."""
    document_id: str
    original_name: str
    suggested_name: str
    display_name: str  # "Suggested Name (original_name.pdf)"
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "original_name": self.original_name,
            "suggested_name": self.suggested_name,
            "display_name": self.display_name,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DocumentName":
        """Create from dictionary."""
        return cls(
            document_id=data["document_id"],
            original_name=data["original_name"],
            suggested_name=data["suggested_name"],
            display_name=data["display_name"],
            confidence=data.get("confidence", 0.8),
            metadata=data.get("metadata", {}),
        )


DOCUMENT_NAMING_PROMPT = """Based on this document summary, generate a precise, succinct document name.

Format: [Document Type] - [Key Subject/Party]

Examples:
- "Witness Statement - Jane Doe"
- "Employment Contract - TechCorp Inc"
- "Court Filing - Motion for Summary Judgment"
- "Email Chain - Project Alpha Discussion"
- "Meeting Minutes - Board Meeting 2024-03-15"

Summary:
{summary_content}

Original filename: {original_name}

Return ONLY a JSON object:
{{
  "suggested_name": "Your suggested name here (max 60 characters)",
  "confidence": 0.95
}}

Be specific and descriptive. Focus on document type and key parties/subjects.
"""


class DocumentRenamer(SummaryBasedGenerator):
    """Generate improved document names from summaries."""
    
    def generate_document_name(
        self,
        summary: DocumentSummary,
        model: Optional[str] = None
    ) -> DocumentName:
        """Generate a precise, succinct name for a document.
        
        Uses the document summary to create a descriptive name that
        captures the document type, key parties, and purpose.
        
        Examples:
        - "Witness Statement - John Smith (2024-03-15_witness_stmt_v2_final.pdf)"
        - "Employment Contract - ABC Corp (contract_signed_2023.pdf)"
        - "Court Filing - Motion to Dismiss (MTD_draft_v3_FINAL.pdf)"
        
        Args:
            summary: Document summary
            model: Model to use for generation
            
        Returns:
            DocumentName with suggested name and display name
        """
        llm_client = self.get_llm_client(model)
        
        # Format prompt
        prompt = DOCUMENT_NAMING_PROMPT.format(
            summary_content=summary.content[:800],  # Limit for context
            original_name=summary.file_name
        )
        
        try:
            # Call LLM
            response = llm_client.generate(prompt, temperature=0.3)
            result = json.loads(response)
            
            suggested_name = result.get("suggested_name", "").strip()
            confidence = result.get("confidence", 0.8)
            
            # Truncate if too long
            if len(suggested_name) > 60:
                suggested_name = suggested_name[:57] + "..."
            
            # Create display name
            display_name = f"{suggested_name} ({summary.file_name})"
            
            return DocumentName(
                document_id=summary.document_id,
                original_name=summary.file_name,
                suggested_name=suggested_name,
                display_name=display_name,
                confidence=confidence,
                metadata={
                    "generated_at": datetime.now().isoformat(),
                    "model_used": model or self.settings.models.llm.default,
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating name for {summary.file_name}: {e}")
            # Fallback to original name
            return DocumentName(
                document_id=summary.document_id,
                original_name=summary.file_name,
                suggested_name=summary.file_name,
                display_name=summary.file_name,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def rename_all_documents(
        self,
        model: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        incremental: bool = False,
    ) -> list[DocumentName]:
        """Generate names for all documents with summaries.
        
        Args:
            model: Model to use for generation
            progress_callback: Optional progress callback
            incremental: If True, only process new documents
            
        Returns:
            List of document names
        """
        # Get summaries
        if incremental:
            # Load existing names to get last update time
            existing_names = self._load_document_names()
            last_update = None
            
            if existing_names:
                # Get most recent generation time
                timestamps = [
                    n.metadata.get("generated_at")
                    for n in existing_names.values()
                    if n.metadata.get("generated_at")
                ]
                if timestamps:
                    last_update = max(timestamps)
            
            summaries = self.get_new_summaries(
                last_processed_time=last_update,
                summary_type="overview"
            )
            logger.info(f"Incremental update: processing {len(summaries)} new summaries")
        else:
            summaries = self.get_all_summaries(summary_type="overview")
            logger.info(f"Full generation: processing {len(summaries)} summaries")
        
        if not summaries:
            logger.warning("No summaries found for document renaming")
            return []
        
        # Generate names
        document_names: list[DocumentName] = []
        total = len(summaries)
        
        for idx, summary in enumerate(summaries):
            if progress_callback:
                progress_callback(f"Renaming {summary.file_name}", idx, total)
            
            doc_name = self.generate_document_name(summary, model)
            document_names.append(doc_name)
        
        # Save names
        self._save_document_names(document_names, incremental)
        
        logger.info(f"Document renaming complete: {len(document_names)} documents")
        
        return document_names
    
    def _save_document_names(self, document_names: list[DocumentName], incremental: bool = False):
        """Save document names to file."""
        names_file = Path("data/document_names.json")
        names_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing if incremental
        if incremental and names_file.exists():
            try:
                with open(names_file) as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = {}
        else:
            existing_data = {}
        
        # Update with new names
        for doc_name in document_names:
            existing_data[doc_name.document_id] = doc_name.to_dict()
        
        # Save
        with open(names_file, "w") as f:
            json.dump(existing_data, f, indent=2)
        
        logger.info(f"Document names saved to {names_file}")
    
    def _load_document_names(self) -> dict[str, DocumentName]:
        """Load existing document names."""
        names_file = Path("data/document_names.json")
        
        if not names_file.exists():
            return {}
        
        try:
            with open(names_file) as f:
                data = json.load(f)
            
            return {
                doc_id: DocumentName.from_dict(name_data)
                for doc_id, name_data in data.items()
            }
        except Exception as e:
            logger.error(f"Error loading document names: {e}")
            return {}
    
    def get_display_name(self, document_id: str, original_name: str) -> str:
        """Get display name for a document.
        
        Args:
            document_id: Document ID
            original_name: Original filename
            
        Returns:
            Display name (suggested or original)
        """
        names = self._load_document_names()
        
        if document_id in names:
            return names[document_id].display_name
        
        return original_name
