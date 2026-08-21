"""Semantic Hybrid Chunker with Contextual Enrichment.

Uses ONNX-accelerated LegalBERT to split text based on semantic similarity drops
while respecting document structure. Enriches each chunk with context.

Performance: Uses GPU-accelerated ONNX Runtime with batched encoding for 10-50x speedup.
"""

from typing import Optional, List, Dict, Any
import hashlib
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config_loader import Settings, get_settings
from src.schema import Chunk, DocumentType, ParsedDocument
from src.ingestion.chunkers.legalbert_onnx import get_legalbert_encoder

class SemanticHybridChunker:
    """Semantic chunker respecting structure and enriching context.
    
    Now uses ONNX-accelerated LegalBERT with batched encoding for GPU acceleration.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        # Get singleton ONNX LegalBERT encoder
        self._encoder = get_legalbert_encoder()
        print(f"[SemanticChunker] LegalBERT ready")
        
    def chunk_document(self, document: ParsedDocument) -> List[Chunk]:
        """Chunk document using semantic hybrid strategy."""
        import time
        start_time = time.time()
        
        # Log GPU/CPU stats before chunking (lightweight, non-blocking)
        try:
            from src.system.gpu_monitor import log_performance
            log_performance(f"[Chunking] BEFORE {document.file_name}")
        except ImportError:
            pass
        
        # 1. Structural Split (Legal Headers)
        sections = self._split_by_structure(document.text)
        
        chunks = []
        chunk_index = 0
        
        for section in sections:
            # 2. Semantic Split (Sub-chunking long sections)
            section_chunks = self._semantic_split(
                text=section["text"],
                max_tokens=self._get_max_tokens(document.document_type),
                header=section["header"]
            )
            
            for sub_text in section_chunks:
                # 3. Contextual Enrichment
                enriched_text = sub_text
                if self.settings.chunking.enable_contextual_header:
                    enriched_text = self._enrich_context(document, section["header"], sub_text)
                
                # Create Chunk Object
                chunk_id = hashlib.md5(f"{document.file_name}:{chunk_index}:{enriched_text}".encode()).hexdigest()[:16]
                
                metadata = document.metadata.copy() if document.metadata else {}
                metadata.update({
                    "section_header": section["header"],
                    "chunk_index": chunk_index,
                    "strategy": "semantic_hybrid"
                })

                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    document_id=self._generate_doc_id(document),
                    file_name=document.file_name,
                    document_type=document.document_type,
                    text=enriched_text,
                    # Char offsets are approximate in semantic mode due to enrichment/splitting
                    char_start=0,
                    char_end=len(enriched_text),
                    metadata=metadata,
                ))
                chunk_index += 1
        
        # Log GPU/CPU stats after chunking with timing
        elapsed = time.time() - start_time
        try:
            from src.system.gpu_monitor import log_performance
            log_performance(f"[Chunking] AFTER {document.file_name} ({elapsed:.1f}s, {len(chunks)} chunks)")
        except ImportError:
            pass
                
        return chunks

    # Minimum section body size in chars — sections smaller than this get
    # merged into the next section to avoid micro-chunks.
    MIN_SECTION_CHARS = 200

    def _split_by_structure(self, text: str) -> List[Dict[str, str]]:
        """Split text by major legal document headers.

        Conservative approach: only split on clearly structural headers to
        avoid fragmenting numbered paragraphs (common in legal forms, witness
        statements, and pleadings) into micro-chunks.

        Detects:
        - Explicit keywords: ``ARTICLE 1``, ``Section 3.2``, ``Clause 7``,
          ``PART II``, ``Schedule 1``
        - ALL-CAPS lines on their own line (≥5 chars, ≤60 chars):
          ``BACKGROUND AND QUALIFICATIONS``, ``STATEMENT OF TRUTH``
        """
        # Only match clear structural markers — NOT numbered paragraphs
        header_re = re.compile(
            r"(?:^|\n)[ \t]*"  # Start of text or newline + optional indent
            r"("
            # Explicit legal section keywords with numbers
            r"(?:ARTICLE|SECTION|PARAGRAPH|PART|Clause|Schedule|ANNEX|APPENDIX)"
            r"\s+(?:\d+(?:\.\d+)*|[IVXLC]+)"
            r"|"
            # ALL-CAPS standalone line: ≥5 chars, ≤60 chars, letters/spaces/hyphens only
            r"[A-Z][A-Z\s\-&]{4,59}"
            r")"
            r"\s*:?\s*(?=\n)",  # Optional colon, then must be followed by newline
            re.MULTILINE,
        )

        # Find all headers and their positions
        headers: List[tuple[int, int, str]] = []
        for m in header_re.finditer(text):
            hdr_text = m.group(1).strip().rstrip(":")
            # Reject false positives: must be mostly uppercase and short
            if len(hdr_text) > 60:
                continue
            # Must be at least 60% uppercase letters (filters out mixed-case content)
            alpha = [c for c in hdr_text if c.isalpha()]
            if alpha and sum(1 for c in alpha if c.isupper()) / len(alpha) < 0.6:
                continue
            headers.append((m.start(), m.end(), hdr_text))

        if not headers:
            return [{"header": "General", "text": text}]

        # Build sections from header positions
        raw_sections: List[Dict[str, str]] = []

        # Content before first header
        pre = text[: headers[0][0]].strip()
        if pre:
            raw_sections.append({"header": "Intro", "text": pre})

        for i, (_, hdr_end, hdr_text) in enumerate(headers):
            next_start = headers[i + 1][0] if i + 1 < len(headers) else len(text)
            body = text[hdr_end:next_start].strip()
            if body:
                raw_sections.append({"header": hdr_text, "text": body})

        if not raw_sections:
            return [{"header": "General", "text": text}]

        # Merge small sections into the following section to prevent micro-chunks
        merged: List[Dict[str, str]] = []
        carry = ""
        carry_header = ""

        for sec in raw_sections:
            body = (carry + "\n\n" + sec["text"]).strip() if carry else sec["text"]
            header = carry_header or sec["header"]

            if len(body) < self.MIN_SECTION_CHARS:
                # Too small — carry into next section
                carry = body
                carry_header = header
            else:
                merged.append({"header": header, "text": body})
                carry = ""
                carry_header = ""

        # Flush any remaining carry
        if carry:
            if merged:
                merged[-1]["text"] += "\n\n" + carry
            else:
                merged.append({"header": carry_header or "General", "text": carry})

        return merged

    def _semantic_split(self, text: str, max_tokens: int, header: str) -> List[str]:
        """Split text semantically if it exceeds max_tokens."""
        # Estimate token count (4 chars/token)
        if len(text) / 4 <= max_tokens:
            return [text]
            
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 1:
            return [text]
            
        # Helper manual split if no model (fallback)
        # But we initialized model above.
        
        embeddings = self._encode_sentences(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_len = len(sentences[0]) / 4
        
        for i in range(1, len(sentences)):
            sent = sentences[i]
            sent_len = len(sent) / 4
            
            # Check similarity with previous sentence
            similarity = cosine_similarity(
                [embeddings[i-1]], 
                [embeddings[i]]
            )[0][0]
            
            # Split if:
            # 1. Similarity drops below threshold AND current chunk is big enough (Safety Guard)
            # 2. OR current chunk exceeds max size
            min_size = self.settings.chunking.min_chunk_size
            
            if (current_len + sent_len > max_tokens) or \
               (similarity < self.settings.chunking.semantic_threshold and current_len >= min_size):
                
                # Check if we should split
                chunks.append(" ".join(current_chunk))
                
                # OVERLAP LOGIC: Carry over last few sentences to next chunk context
                # Standard overlap is ~2-3 sentences or 10% of window
                overlap_window = 4 # Increased to 4 sentences for robust context continuity
                overlap_buffer = current_chunk[-overlap_window:] if len(current_chunk) > overlap_window else current_chunk
                
                current_chunk = overlap_buffer + [sent]
                current_len = sum(len(s)/4 for s in current_chunk)
            else:
                current_chunk.append(sent)
                current_len += sent_len
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def _encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """Encode sentences using ONNX-accelerated LegalBERT with batched processing."""
        return self._encoder.encode_sentences(sentences)

    def _enrich_context(self, doc: ParsedDocument, header: str, text: str) -> str:
        """Prepend context to chunk text (no summary injection to avoid error propagation)."""
        # Use intelligent display name if available, else filename
        doc_name = doc.metadata.get("display_name", doc.file_name) if doc.metadata else doc.file_name
        context_header = f"[Doc: {doc_name} | Type: {doc.document_type} | Section: {header}]\n"
        # Summary injection removed per user request - prevents error propagation
        return context_header + text

    def _get_max_tokens(self, doc_type: str) -> int:
        return getattr(self.settings.chunking.sizes, doc_type, 512)

    def _generate_doc_id(self, doc: ParsedDocument) -> str:
        return hashlib.md5(doc.file_path.encode()).hexdigest()[:8]
