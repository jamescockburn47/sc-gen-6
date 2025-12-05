"""Summary-Enhanced Retrieval with 2025 Best Practices.

Implements multi-level retrieval strategies using document summaries:
1. Summary-First Retrieval: Search summaries → expand to chunks
2. Hybrid Summary+Chunk: Search both, merge results
3. Hierarchical Context: Add summary context to retrieved chunks

Based on 2025 RAG best practices:
- RAPTOR-style recursive summarization
- Parent document retriever pattern
- Small-to-big context expansion
"""

from typing import Any, Optional
from dataclasses import dataclass

from src.config_loader import Settings, get_settings
from src.retrieval.summary_store import SummaryStore, DocumentSummary
from src.retrieval.fts5_index import FTS5Index
from src.retrieval.vector_store import VectorStore


@dataclass
class EnhancedRetrievalResult:
    """Result from summary-enhanced retrieval.
    
    Attributes:
        chunk_id: Retrieved chunk ID
        chunk_text: Chunk content
        score: Relevance score
        document_summary: Optional document-level summary for context
        section_summary: Optional section-level summary for context
        metadata: Additional metadata
    """
    chunk_id: str
    chunk_text: str
    score: float
    document_summary: Optional[str] = None
    section_summary: Optional[str] = None
    metadata: dict[str, Any] = None


class SummaryEnhancedRetriever:
    """Retriever that uses document summaries to improve search quality.
    
    Implements three strategies based on 2025 RAG best practices:
    
    1. **Summary-First (RAPTOR-style)**:
       - Search document summaries first
       - Identify relevant documents
       - Then retrieve specific chunks from those documents
       - Good for: broad queries, finding the right document
    
    2. **Hybrid Summary+Chunk**:
       - Search both summaries and chunks in parallel
       - Merge and re-rank results
       - Good for: balanced precision and recall
    
    3. **Chunk-with-Context (Parent Document)**:
       - Standard chunk retrieval
       - Augment each chunk with its document summary
       - Good for: providing LLM with document-level context
    """
    
    def __init__(
        self,
        summary_store: Optional[SummaryStore] = None,
        fts5_index: Optional[FTS5Index] = None,
        vector_store: Optional[VectorStore] = None,
        settings: Optional[Settings] = None,
    ):
        """Initialize summary-enhanced retriever.
        
        Args:
            summary_store: Summary store instance
            fts5_index: FTS5 index for keyword search
            vector_store: Vector store for semantic search
            settings: Settings instance
        """
        self.settings = settings or get_settings()
        self.summary_store = summary_store or SummaryStore(settings=self.settings)
        self.fts5_index = fts5_index
        self.vector_store = vector_store
    
    def retrieve_summary_first(
        self,
        query: str,
        top_k_summaries: int = 5,
        chunks_per_doc: int = 3,
    ) -> list[EnhancedRetrievalResult]:
        """Summary-first retrieval (RAPTOR-style).
        
        1. Search document summaries to find relevant documents
        2. For each relevant document, retrieve top chunks
        3. Return chunks with their document summaries attached
        
        Args:
            query: Search query
            top_k_summaries: Number of document summaries to retrieve
            chunks_per_doc: Number of chunks to retrieve per document
            
        Returns:
            List of enhanced retrieval results
        """
        results = []
        
        # Step 1: Search summaries to find relevant documents
        summary_results = self.summary_store.search(
            query=query,
            top_k=top_k_summaries,
            summary_level="document",
        )
        
        if not summary_results:
            return results
        
        # Step 2: For each relevant document, get chunks
        seen_chunks = set()
        for summary, summary_score in summary_results:
            # Get chunks from this document using FTS5
            if self.fts5_index:
                chunk_results = self.fts5_index.search(
                    query=query,
                    top_k=chunks_per_doc,
                    selected_documents=[summary.file_name],
                )
                
                for chunk_id, chunk_score in chunk_results:
                    if chunk_id in seen_chunks:
                        continue
                    seen_chunks.add(chunk_id)
                    
                    # Get chunk data
                    chunk_data = self.fts5_index.get_chunk(chunk_id)
                    if chunk_data:
                        results.append(EnhancedRetrievalResult(
                            chunk_id=chunk_id,
                            chunk_text=chunk_data["chunk_text"],
                            score=chunk_score * summary_score,  # Combined score
                            document_summary=summary.content,
                            metadata={
                                "file_name": summary.file_name,
                                "summary_score": summary_score,
                                "chunk_score": chunk_score,
                            },
                        ))
        
        # Sort by combined score
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def retrieve_with_context(
        self,
        chunks: list[dict[str, Any]],
        include_document_summary: bool = True,
        include_section_summary: bool = False,
    ) -> list[EnhancedRetrievalResult]:
        """Augment retrieved chunks with summary context.
        
        Takes already-retrieved chunks and adds document/section summaries
        to provide the LLM with broader context.
        
        Args:
            chunks: List of chunk dicts from standard retrieval
            include_document_summary: Include document-level summary
            include_section_summary: Include section-level summary
            
        Returns:
            List of enhanced retrieval results
        """
        results = []
        
        # Cache summaries to avoid repeated lookups
        doc_summary_cache: dict[str, str] = {}
        
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            document_id = chunk.get("document_id", "")
            file_name = chunk.get("metadata", {}).get("file_name", "")
            
            # Get document summary
            doc_summary = None
            if include_document_summary and document_id:
                if document_id not in doc_summary_cache:
                    summaries = self.summary_store.get_document_summaries(
                        document_id=document_id,
                        summary_type="overview",
                        summary_level="document",
                    )
                    doc_summary_cache[document_id] = summaries[0].content if summaries else None
                doc_summary = doc_summary_cache.get(document_id)
            
            results.append(EnhancedRetrievalResult(
                chunk_id=chunk_id,
                chunk_text=chunk.get("text", ""),
                score=chunk.get("score", 0.0),
                document_summary=doc_summary,
                metadata=chunk.get("metadata", {}),
            ))
        
        return results
    
    def build_context_with_summaries(
        self,
        chunks: list[dict[str, Any]],
        max_chunks: int = 10,
        include_doc_summary: bool = True,
    ) -> str:
        """Build LLM context that includes document summaries.
        
        Creates a structured context string that includes:
        1. Document summaries for broader context
        2. Specific chunks for detailed evidence
        
        This is the recommended approach for 2025 - giving the LLM
        both the "big picture" and the "evidence".
        
        Args:
            chunks: Retrieved chunks
            max_chunks: Maximum chunks to include
            include_doc_summary: Whether to include document summaries
            
        Returns:
            Formatted context string for LLM
        """
        context_parts = []
        
        # Group chunks by document
        docs: dict[str, list[dict]] = {}
        for chunk in chunks[:max_chunks]:
            doc_id = chunk.get("document_id", "unknown")
            if doc_id not in docs:
                docs[doc_id] = []
            docs[doc_id].append(chunk)
        
        # Build context for each document
        for doc_id, doc_chunks in docs.items():
            file_name = doc_chunks[0].get("metadata", {}).get("file_name", "Unknown")
            
            # Add document summary if available
            if include_doc_summary:
                summaries = self.summary_store.get_document_summaries(
                    document_id=doc_id,
                    summary_type="overview",
                    summary_level="document",
                )
                if summaries:
                    context_parts.append(f"## Document: {file_name}")
                    context_parts.append(f"**Summary:** {summaries[0].content}")
                    context_parts.append("")
            
            # Add specific chunks
            for chunk in doc_chunks:
                page = chunk.get("metadata", {}).get("page_number", "")
                para = chunk.get("metadata", {}).get("paragraph_number", "")
                location = f"Page {page}" if page else ""
                if para:
                    location += f", Para {para}"
                
                context_parts.append(f"### Source: {file_name} ({location})")
                context_parts.append(chunk.get("text", ""))
                context_parts.append("")
        
        return "\n".join(context_parts)


def estimate_summary_time(
    num_documents: int,
    avg_pages_per_doc: int = 10,
    model_tokens_per_sec: float = 50,
) -> dict[str, float]:
    """Estimate time to generate summaries for a document set.
    
    Args:
        num_documents: Number of documents
        avg_pages_per_doc: Average pages per document
        model_tokens_per_sec: LLM generation speed
        
    Returns:
        Dict with time estimates
    """
    # Estimates based on 2025 benchmarks
    # Input processing: ~500 tokens/page
    # Summary output: ~200 tokens per document
    
    avg_input_tokens = avg_pages_per_doc * 500
    output_tokens = 200
    
    # Time per document (seconds)
    # Input processing is fast, output generation is the bottleneck
    time_per_doc = output_tokens / model_tokens_per_sec
    
    # Add overhead for loading, API calls, etc.
    time_per_doc *= 1.5
    
    total_time = num_documents * time_per_doc
    
    return {
        "total_seconds": total_time,
        "total_minutes": total_time / 60,
        "per_document_seconds": time_per_doc,
        "num_documents": num_documents,
        "avg_pages": avg_pages_per_doc,
    }



