"""Agentic Chunker - LLM-guided document splitting.

Uses LLM to intelligently determine chunk boundaries based on document structure.
Falls back to semantic splitting for long sections or when LLM guidance fails.
"""

from typing import Optional, List, Dict, Any
import hashlib
import re
import json

from src.config_loader import Settings, get_settings
from src.schema import Chunk, ParsedDocument
from src.llm.client import LLMClient


class AgenticChunker:
    """LLM-guided chunker for intelligent document splitting."""

    # Qwen2.5 is best for structured JSON output
    CHUNKING_MODEL = "qwen2.5-7b"  # Fast, good at JSON
    
    def __init__(self, settings: Optional[Settings] = None, llm_client: Optional[LLMClient] = None):
        self.settings = settings or get_settings()
        self.llm_client = llm_client
        # Import semantic chunker as fallback
        from src.ingestion.chunkers.semantic_chunker import SemanticHybridChunker
        self.semantic_chunker = SemanticHybridChunker(settings)

    def chunk_document(self, document: ParsedDocument) -> List[Chunk]:
        """Chunk document using LLM-guided multi-pass strategy."""
        
        text = document.text
        
        # Pass 1: Structure Detection
        # Check for clear legal headers first (fast, no LLM needed)
        sections = self._detect_structure(text)
        
        if not sections:
            # No clear headers, use LLM to understand format
            sections = self._llm_analyze_structure(text, document.file_name)
        
        if not sections:
            # LLM failed, fall back to semantic chunker
            return self.semantic_chunker.chunk_document(document)
        
        # Pass 2: Process each section
        all_chunks = []
        chunk_index = 0
        
        for section in sections:
            section_text = section["text"]
            section_header = section["header"]
            parent_id = hashlib.md5(f"{document.file_name}:{section_header}".encode()).hexdigest()[:12]
            
            # Estimate tokens (4 chars/token)
            section_tokens = len(section_text) / 4
            
            if section_tokens > 1000:
                # Long section: use semantic splitting to create children
                child_texts = self._semantic_split_section(section_text)
            else:
                # Short section: keep as single chunk
                child_texts = [section_text]
            
            for child_text in child_texts:
                # Enrich with context header (no summary)
                doc_name = document.metadata.get("display_name", document.file_name) if document.metadata else document.file_name
                enriched_text = f"[Doc: {doc_name} | Type: {document.document_type} | Section: {section_header}]\n{child_text}"
                
                chunk_id = hashlib.md5(f"{document.file_name}:{chunk_index}".encode()).hexdigest()[:16]
                
                metadata = document.metadata.copy() if document.metadata else {}
                metadata.update({
                    "section_header": section_header,
                    "parent_id": parent_id,
                    "chunk_index": chunk_index,
                    "strategy": "agentic"
                })
                
                all_chunks.append(Chunk(
                    chunk_id=chunk_id,
                    document_id=hashlib.md5(document.file_path.encode()).hexdigest()[:8],
                    file_name=document.file_name,
                    document_type=document.document_type,
                    text=enriched_text,
                    char_start=0,
                    char_end=len(enriched_text),
                    metadata=metadata
                ))
                chunk_index += 1
        
        return all_chunks

    def _detect_structure(self, text: str) -> List[Dict[str, str]]:
        """Detect structure using regex patterns for legal headers."""
        header_pattern = r"(^|\n)((?:ARTICLE|SECTION|PARAGRAPH|Clause|PART)\s+\d+(?:\.\d+)*|[A-Z][A-Z\s]{5,}:)"
        
        splits = re.split(header_pattern, text)
        sections = []
        
        current_header = "Introduction"
        current_text = ""
        
        for i, s in enumerate(splits):
            if re.match(header_pattern, s):
                if current_text.strip():
                    sections.append({"header": current_header, "text": current_text.strip()})
                current_header = s.strip().replace(":", "")
                current_text = ""
            else:
                current_text += s
        
        if current_text.strip():
            sections.append({"header": current_header, "text": current_text.strip()})
        
        # Only use regex if we found meaningful structure
        if len(sections) <= 1:
            return []
        
        return sections

    def _llm_analyze_structure(self, text: str, filename: str) -> List[Dict[str, str]]:
        """Use LLM to analyze document structure (for docs without clear headers)."""
        if not self.llm_client:
            return []
        
        # Only analyze first 3000 chars to understand format
        sample = text[:3000]
        
        prompt = f"""Analyze this legal document excerpt and identify logical section boundaries.

Document: {filename}

---
{sample}
---

Output a JSON array of sections found, with "header" and "start_position" (character offset).
Example: [{{"header": "Definitions", "start_position": 0}}, {{"header": "Obligations", "start_position": 500}}]

Only output the JSON array, no explanation."""

        try:
            # Use generate_chat_completion with system message
            messages = [
                {"role": "system", "content": "You are a legal document structure analyzer. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ]
            response = self.llm_client.generate_chat_completion(
                messages=messages,
                model=self.CHUNKING_MODEL,
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                boundaries = json.loads(json_match.group())
                
                # Convert boundaries to sections with text
                sections = []
                for i, boundary in enumerate(boundaries):
                    start = boundary.get("start_position", 0)
                    end = boundaries[i+1]["start_position"] if i+1 < len(boundaries) else len(text)
                    sections.append({
                        "header": boundary.get("header", f"Section {i+1}"),
                        "text": text[start:end].strip()
                    })
                return sections
                
        except Exception as e:
            print(f"AgenticChunker: LLM analysis failed: {e}")
        
        return []

    def _semantic_split_section(self, text: str) -> List[str]:
        """Split a long section using LegalBERT semantic similarity."""
        # Delegate to semantic chunker's internal method
        return self.semantic_chunker._semantic_split(text, max_tokens=512, header="")
