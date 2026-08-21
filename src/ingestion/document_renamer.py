"""Document Renamer Service.

Uses LLM to analyze document content and generate a descriptive, standardized
title (e.g. "Witness Statement of John Doe dated 2023-10-12").
"""

import re
from typing import Optional
from src.config_loader import Settings, get_settings
from src.schema import ParsedDocument
from src.generation.llm_service import LLMService

class DocumentRenamer:
    """Intelligent document renamer using LLM."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.llm_service = LLMService(settings=self.settings)

    def rename_document(self, document: ParsedDocument) -> str:
        """Generate a descriptive name for the document.
        
        Args:
            document: ParsedDocument to rename
            
        Returns:
            New descriptive name (or original filename if generation fails)
        """
        if not self.settings.ingestion.auto_rename:
            return document.file_name

        # Extract first 3000 chars (enough for header/intro)
        context_text = document.text[:3000]
        
        prompt = f"""Analyze the following legal document start and extract:
1. Document Type (e.g. Witness Statement, Contract, Email, Pleading)
2. Main Entity/Author (e.g. Person Name, Company Name)
3. Date (YYYY-MM-DD or relevant date)

Based on this, generate a concise title using the format:
"{self.settings.ingestion.rename_format}"

If you cannot find specific details, use "Unknown". 
Output ONLY the new title string. Do not add quotes or markdown.

Document Context:
{context_text}
"""

        try:
            # Use smaller/faster model for renaming if available, or default
            # We use complete() from LLM service
            new_name = self.llm_service.complete(
                 prompt=prompt,
                 max_tokens=64, # Short output
                 temperature=0.1 # Deterministic
            ).strip()
            
            # Basic validation: If output is too long or empty, fallback
            if not new_name or len(new_name) > 100 or "\n" in new_name:
                return document.file_name
                
            # Remove any trailing periods or quotes
            new_name = new_name.strip('".')
            
            # Ensure safe filename chars (optional, mainly for display but good practice)
            # new_name = re.sub(r'[<>:"/\\|?*]', '_', new_name)
            
            return new_name

        except Exception as e:
            print(f"Error renaming document {document.file_name}: {e}")
            return document.file_name
