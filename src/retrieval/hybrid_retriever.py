"""Hybrid retriever combining semantic search, FTS5, RRF, and reranking.

Implements the full retrieval pipeline:
Query → Dense (Chroma) [Top N] + FTS5 [Top N] → RRF → Cross-Encoder Rerank → Top M

Optimized with parallel retrieval (semantic + keyword run concurrently).
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Optional

try:
    from PySide6.QtCore import QObject, Signal
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

from src.config_loader import Settings, get_settings
from src.retrieval.fts5_index import FTS5IndexCompat
from src.retrieval.rrf import ScoredChunk, fuse
from src.retrieval.vector_store import VectorStore
from src.schema import DocumentType


if QT_AVAILABLE:
    class RetrieverProgressSignals(QObject):
        """Progress signals for hybrid retriever."""

        # Stage progress: (stage_name, status, details_dict)
        stage_started = Signal(str)
        stage_completed = Signal(str, dict)

        # Semantic search: (current, total, top_score, time_ms)
        semantic_progress = Signal(int, int, float, float)

        # Keyword search: (current, total, top_score, time_ms)
        keyword_progress = Signal(int, int, float, float)

        # Fusion: (merged_count, time_ms)
        fusion_complete = Signal(int, float)

        # Reranking: (current, total, time_ms)
        rerank_progress = Signal(int, int, float)

        # Filtering: (passed_count, avg_confidence, min_confidence, max_confidence, time_ms)
        filtering_complete = Signal(int, float, float, float, float)

        # Final retrieval complete: (results_count, total_time_ms, stats_dict)
        retrieval_complete = Signal(int, float, dict)


class HybridRetriever:
    """Hybrid retriever combining semantic and keyword search with reranking.

    Pipeline:
    1. Semantic search (Chroma) → Top N
    2. Keyword search (FTS5) → Top N
    3. RRF fusion → merge results
    4. Cross-encoder rerank → Top K
    5. Filter by confidence threshold → Top M
    """

    def __init__(
        self,
        embedding_service=None,
        vector_store: Optional[VectorStore] = None,
        keyword_index: Optional[FTS5IndexCompat] = None,
        reranker_service=None,
        settings: Optional[Settings] = None,
        progress_signals: Optional['RetrieverProgressSignals'] = None,
    ):
        """Initialize hybrid retriever.

        Args:
            embedding_service: Embedding service. If None, creates new instance using factory.
            vector_store: Vector store. If None, creates new instance.
            keyword_index: FTS5 keyword index. If None, creates new instance.
            reranker_service: Reranker service. If None, creates new instance.
            settings: Settings instance. If None, loads from config.
            progress_signals: Optional progress signals for UI updates.
        """
        self.settings = settings or get_settings()
        self.progress_signals = progress_signals

        # Use factory function to get best embedding service (ONNX GPU or CPU)
        if embedding_service is None:
            from src.retrieval import get_embedding_service
            self.embedding_service = get_embedding_service(settings=self.settings)
        else:
            self.embedding_service = embedding_service
            
        self.vector_store = vector_store or VectorStore(settings=self.settings)
        
        # Always use FTS5 keyword index
        if keyword_index is not None:
            self.keyword_index = keyword_index
        else:
            print("[Retrieval] Using FTS5 keyword index (SQLite)")
            self.keyword_index = FTS5IndexCompat(settings=self.settings)
        
        # Use factory function to get best reranker service (ONNX GPU or CPU)
        if reranker_service is None:
            from src.models.reranker import get_reranker_service
            self.reranker_service = get_reranker_service(settings=self.settings)
        else:
            self.reranker_service = reranker_service
            
        # Initialize summary store if enabled
        self.summary_store = None
        if self.settings.retrieval.use_summaries:
            from src.retrieval.summary_store import SummaryStore
            self.summary_store = SummaryStore(settings=self.settings)

    def retrieve(
        self,
        query: str,
        semantic_top_n: Optional[int] = None,
        keyword_top_n: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        context_to_llm: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        rrf_k: Optional[int] = None,
        doc_type_filter: Optional[DocumentType] = None,
        date_range: Optional[tuple[datetime, datetime]] = None,
        selected_documents: Optional[list[str]] = None,
        cancel_event: Optional[threading.Event] = None,
        skip_reranking: Optional[bool] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve and rank chunks for a query.

        Args:
            query: Query text
            semantic_top_n: Number of semantic results (default from config)
            keyword_top_n: Number of BM25 results (default from config)
            rerank_top_k: Number of candidates to rerank (default from config)
            context_to_llm: Number of chunks to return (default from config)
            confidence_threshold: Minimum reranker score (default from config)
            rrf_k: RRF fusion parameter (default from config)
            doc_type_filter: Optional document type filter
            date_range: Optional date range filter (start, end)
            selected_documents: Optional list of document file names to search (None = all)
            skip_reranking: Skip cross-encoder reranking (default from config)

        Returns:
            List of result dicts with keys:
            - chunk_id: Chunk identifier
            - text: Chunk text
            - score: Reranker score
            - metadata: Chunk metadata
            - rrf_score: RRF fusion score (before reranking)
        """

        def _check_cancel(stage: str) -> None:
            if cancel_event and cancel_event.is_set():
                raise RuntimeError(f"Retrieval cancelled during {stage}")

        # Use config defaults if not provided
        semantic_top_n = semantic_top_n or self.settings.retrieval.semantic_top_n
        keyword_top_n = keyword_top_n or self.settings.retrieval.keyword_top_n
        rerank_top_k = rerank_top_k or self.settings.retrieval.rerank_top_k
        context_to_llm = context_to_llm or self.settings.retrieval.context_to_llm
        confidence_threshold = (
            confidence_threshold or self.settings.retrieval.confidence_threshold
        )
        rrf_k = rrf_k or self.settings.retrieval.rrf_k
        
        # Handle skip_reranking - use config default if not provided
        if skip_reranking is None:
            skip_reranking = self.settings.retrieval.skip_reranking

        # Track overall timing
        start_time = time.time()

        _check_cancel("start")

        # Pre-compute query embedding (needed for semantic search)
        embed_start = time.time()
        query_embedding = self.embedding_service.embed_query(query)
        embed_time_ms = (time.time() - embed_start) * 1000
        
        # Log embedding time for performance monitoring
        if embed_time_ms > 100:
            print(f"[PERF WARNING] Query embedding took {embed_time_ms:.1f}ms (expected 10-50ms)")
        else:
            print(f"[TIMING] Query embedding: {embed_time_ms:.1f}ms")
        
        where_clause = self._build_where_clause(doc_type_filter, date_range, selected_documents)

        # Step 1 & 2: Run semantic and keyword search IN PARALLEL
        if self.progress_signals:
            self.progress_signals.stage_started.emit("semantic_search")
            self.progress_signals.stage_started.emit("keyword_search")

        semantic_rrf = []
        bm25_rrf = []
        semantic_time = 0.0
        bm25_time = 0.0
        top_semantic_score = 0.0
        top_bm25_score = 0.0

        def _do_semantic_search():
            """Run semantic (vector) search."""
            nonlocal semantic_rrf, semantic_time, top_semantic_score
            stage_start = time.time()
            semantic_results = self.vector_store.query(
                [query_embedding], n_results=semantic_top_n, where=where_clause
            )
            semantic_time = (time.time() - stage_start) * 1000
            
            if semantic_results.get("ids") and semantic_results["ids"][0]:
                for i, chunk_id in enumerate(semantic_results["ids"][0]):
                    distance = semantic_results["distances"][0][i] if semantic_results.get("distances") else 0.0
                    score = 1.0 / (1.0 + distance)
                    semantic_rrf.append({"id": chunk_id, "score": score})
                    if i == 0:
                        top_semantic_score = score

        def _do_keyword_search():
            """Run keyword (FTS5) search."""
            nonlocal bm25_rrf, bm25_time, top_bm25_score
            stage_start = time.time()
            try:
                bm25_results = self.keyword_index.search(
                    query, top_k=keyword_top_n, selected_documents=selected_documents
                )
                bm25_rrf = [{"id": chunk_id, "score": score} for chunk_id, score in bm25_results]
                if bm25_rrf:
                    top_bm25_score = bm25_rrf[0]["score"]
            except RuntimeError as e:
                if "bm25_index.pkl" not in str(e) and "No such file" not in str(e) and "Index not built" not in str(e):
                    print(f"Warning: BM25 search skipped - {str(e)}")
            bm25_time = (time.time() - stage_start) * 1000

        # Execute both searches in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(_do_semantic_search),
                executor.submit(_do_keyword_search),
            ]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        _check_cancel("parallel search")

        # Emit progress for both searches
        if self.progress_signals:
            self.progress_signals.semantic_progress.emit(
                len(semantic_rrf), semantic_top_n, top_semantic_score, semantic_time
            )
            self.progress_signals.stage_completed.emit("semantic_search", {
                "count": len(semantic_rrf),
                "time_ms": semantic_time,
                "top_score": top_semantic_score
            })
            self.progress_signals.keyword_progress.emit(
                len(bm25_rrf), keyword_top_n, top_bm25_score, bm25_time
            )
            self.progress_signals.stage_completed.emit("keyword_search", {
                "count": len(bm25_rrf),
                "time_ms": bm25_time,
                "top_score": top_bm25_score
            })

        # Step 3: RRF fusion
        if self.progress_signals:
            self.progress_signals.stage_started.emit("fusion")

        stage_start = time.time()
        fused_results = fuse(semantic_rrf, bm25_rrf, k=rrf_k)
        fusion_time = (time.time() - stage_start) * 1000
        _check_cancel("fusion")

        # Emit fusion progress
        if self.progress_signals:
            self.progress_signals.fusion_complete.emit(len(fused_results), fusion_time)
            self.progress_signals.stage_completed.emit("fusion", {
                "count": len(fused_results),
                "time_ms": fusion_time
            })

        # Check if we should skip reranking (for large context models)
        # Note: skip_reranking parameter is already resolved to config default above

        if skip_reranking:
            # Skip reranking - use RRF scores directly (much faster!)
            print(f"Skipping reranking - using RRF scores for top {context_to_llm} chunks")
            print(f"  Total fused results available: {len(fused_results)}")

            # Get top chunks directly from RRF fusion
            top_chunks = fused_results[:rerank_top_k] # Use rerank_top_k as the pool for advanced filtering
            print(f"  Selected {len(top_chunks)} chunks for LLM")

            top_k_chunk_ids = [chunk.chunk_id for chunk in top_chunks]

            # Retrieve chunk texts and metadata
            chunk_data = self._get_chunk_data(top_k_chunk_ids)
            print(f"  Retrieved data for {len(chunk_data)} chunks")

            # Emit reranking skipped signal (0ms time)
            if self.progress_signals:
                self.progress_signals.stage_started.emit("reranking")
                self.progress_signals.rerank_progress.emit(0, 0, 0.0)
                self.progress_signals.stage_completed.emit("reranking", {
                    "count": 0,
                    "time_ms": 0.0,
                    "skipped": True
                })

            rerank_time = 0.0  # No reranking performed

            _check_cancel("chunk fetch before filtering")

            # Build results from RRF scores (no confidence filtering needed at this stage)
            if self.progress_signals:
                self.progress_signals.stage_started.emit("filtering")

            stage_start = time.time()
            results = []
            all_scores = [chunk.score for chunk in top_chunks]

            for chunk in top_chunks:
                _check_cancel("filtering (skip rerank)")
                if chunk.chunk_id in chunk_data:
                    chunk_info = chunk_data[chunk.chunk_id]
                    results.append({
                        "chunk_id": chunk.chunk_id,
                        "text": chunk_info["text"],
                        "score": float(chunk.score),  # RRF score
                        "rrf_score": float(chunk.score),
                        "metadata": chunk_info["metadata"],
                    })
                else:
                    print(f"  WARNING: Chunk {chunk.chunk_id} not found in chunk_data!")

            filtering_time = (time.time() - stage_start) * 1000
            print(f"  Initial results after skip rerank: {len(results)} chunks")

            # For stats - create fake "reranked" list
            reranked = [(chunk.chunk_id, chunk.score) for chunk in top_chunks]
        else:
            # Traditional path with reranking
            # Get top K for reranking
            top_k_chunk_ids = [chunk.chunk_id for chunk in fused_results[:rerank_top_k]]

            # Retrieve chunk texts and metadata for reranking
            chunk_data = self._get_chunk_data(top_k_chunk_ids)
            _check_cancel("chunk fetch before rerank")

            # Step 4: Cross-encoder rerank
            if self.progress_signals:
                self.progress_signals.stage_started.emit("reranking")

            stage_start = time.time()
            
            # OPTIMIZATION: Truncate chunk text for reranking (cross-encoders are O(n²) in sequence length)
            # First ~512 chars usually contain the most relevant content for ranking
            max_rerank_chars = self.settings.retrieval.rerank_max_chars
            rerank_pairs = [
                (chunk_id, chunk_data[chunk_id]["text"][:max_rerank_chars]) 
                for chunk_id in top_k_chunk_ids
                if chunk_id in chunk_data
            ]
            reranked = self.reranker_service.rerank(query, rerank_pairs, top_k=rerank_top_k)
            rerank_time = (time.time() - stage_start) * 1000
            _check_cancel("reranking")

            # Emit rerank progress
            if self.progress_signals:
                self.progress_signals.rerank_progress.emit(
                    len(reranked), rerank_top_k, rerank_time
                )
                self.progress_signals.stage_completed.emit("reranking", {
                    "count": len(reranked),
                    "time_ms": rerank_time
                })

            # Step 5: Filter by confidence threshold and return top M
            if self.progress_signals:
                self.progress_signals.stage_started.emit("filtering")

            stage_start = time.time()
            results = []
            all_scores = []
            
            # Guarantee minimal context if available, regardless of threshold
            # This prevents "empty answers" when scores are generally low but valid
            MIN_GUARANTEED = min(3, context_to_llm) 
            
            for i, (chunk_id, rerank_score) in enumerate(reranked):
                _check_cancel("filtering")
                all_scores.append(rerank_score)
                
                # Keep if above threshold OR if we haven't met the minimum guarantee yet
                if rerank_score >= confidence_threshold or len(results) < MIN_GUARANTEED:
                    chunk_info = chunk_data[chunk_id]
                    # Find RRF score for this chunk
                    rrf_score = next(
                        (chunk.score for chunk in fused_results if chunk.chunk_id == chunk_id), 0.0
                    )

                    results.append(
                        {
                            "chunk_id": chunk_id,
                            "text": chunk_info["text"],
                            "score": float(rerank_score),
                            "rrf_score": float(rrf_score),
                            "metadata": chunk_info["metadata"],
                        }
                    )

                    # Do not break here, as advanced filtering might reorder/remove
                    # We will apply context_to_llm limit later
                    # if len(results) >= context_to_llm:
                    #     break

            filtering_time = (time.time() - stage_start) * 1000

        # Calculate confidence stats (based on scores *before* advanced filtering)
        avg_confidence = sum(all_scores) / len(all_scores) if all_scores else 0.0
        min_confidence = min(all_scores) if all_scores else 0.0
        max_confidence = max(all_scores) if all_scores else 0.0

        # Emit filtering progress (before advanced filtering)
        if self.progress_signals:
            self.progress_signals.filtering_complete.emit(
                len(results), avg_confidence, min_confidence, max_confidence, filtering_time
            )
            self.progress_signals.stage_completed.emit("filtering", {
                "passed_count": len(results),
                "avg_confidence": avg_confidence,
                "min_confidence": min_confidence,
                "max_confidence": max_confidence,
                "time_ms": filtering_time
            })

        # Apply confidence threshold (if not already applied by the MIN_GUARANTEED logic)
        # This ensures that if MIN_GUARANTEED was met but scores were low, they are now filtered
        results = [r for r in results if r["score"] >= self.settings.retrieval.confidence_threshold]
        
        # --- Advanced Retrieval Pipeline ---
        if self.settings.advanced_retrieval.enabled:
            # A. MMR Diversity Filtering
            if self.settings.advanced_retrieval.enable_mmr and len(results) > 1:
                results = self._apply_mmr(
                    query=query,
                    results=results,
                    top_k=self.settings.retrieval.context_to_llm,
                    lambda_mult=self.settings.advanced_retrieval.mmr_lambda
                )
            
            # B. LLM Relevance Grading
            if self.settings.advanced_retrieval.enable_llm_grading and results:
                results = self._llm_grade_results(query, results)
        
        # Limit to context_to_llm (if not already done by MMR)
        results = results[: self.settings.retrieval.context_to_llm]

        # Apply date range filter (post-retrieval)
        if date_range:
            results = self._filter_by_date(results, date_range)

        # Enhance with document summaries if enabled
        if self.settings.retrieval.use_summaries:
            results = self.enhance_with_summaries(results, include_doc_summary=True)

        # Emit overall retrieval complete
        total_time = (time.time() - start_time) * 1000
        if self.progress_signals:
            # Count unique documents
            unique_docs = set()
            for result in results:
                if "metadata" in result and "file_name" in result["metadata"]:
                    unique_docs.add(result["metadata"]["file_name"])

            self.progress_signals.retrieval_complete.emit(
                len(results),
                total_time,
                {
                    "semantic_count": len(semantic_rrf),
                    "bm25_count": len(bm25_rrf),
                    "fused_count": len(fused_results),
                    "reranked_count": len(reranked),
                    "final_count": len(results),
                    "unique_documents": len(unique_docs),
                    "semantic_time_ms": semantic_time,
                    "bm25_time_ms": bm25_time,
                    "fusion_time_ms": fusion_time,
                    "rerank_time_ms": rerank_time,
                    "filtering_time_ms": filtering_time,
                    "total_time_ms": total_time,
                    "avg_confidence": avg_confidence,
                }
            )

        return results

    def _build_where_clause(
        self,
        doc_type_filter: Optional[DocumentType],
        date_range: Optional[tuple[datetime, datetime]],
        selected_documents: Optional[list[str]],
    ) -> Optional[dict[str, Any]]:
        """Build ChromaDB where clause from filters.

        Args:
            doc_type_filter: Optional document type filter
            date_range: Optional date range filter
            selected_documents: Optional list of document file names to filter by

        Returns:
            Where clause dict or None
        """
        where_clause = {}

        if doc_type_filter:
            where_clause["document_type"] = doc_type_filter

        if selected_documents:
            # Filter by file names using $in operator
            where_clause["file_name"] = {"$in": selected_documents}

        # Note: Date filtering is done as post-retrieval filter in _filter_by_date
        # since dates are stored in various string formats

        return where_clause if where_clause else None

    def _filter_by_date(
        self,
        results: list[dict[str, Any]],
        date_range: tuple[datetime, datetime],
    ) -> list[dict[str, Any]]:
        """Filter results by date range (post-retrieval).
        
        Parses dates from various formats stored in chunk metadata.
        
        Args:
            results: List of result dicts with metadata
            date_range: (start_date, end_date) tuple
            
        Returns:
            Filtered list of results
        """
        import re
        from datetime import date
        
        start_date, end_date = date_range
        
        # Convert to date if datetime
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()
        
        def parse_date(date_str: str) -> Optional[date]:
            """Try to parse date from various formats."""
            if not date_str or not isinstance(date_str, str):
                return None
                
            # Try common patterns
            patterns = [
                # ISO format
                (r'(\d{4})-(\d{2})-(\d{2})', lambda m: date(int(m[1]), int(m[2]), int(m[3]))),
                # UK format DD/MM/YYYY
                (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: date(int(m[3]), int(m[2]), int(m[1]))),
                # UK format DD.MM.YYYY
                (r'(\d{1,2})\.(\d{1,2})\.(\d{4})', lambda m: date(int(m[3]), int(m[2]), int(m[1]))),
                # UK format DD-MM-YYYY
                (r'(\d{1,2})-(\d{1,2})-(\d{4})', lambda m: date(int(m[3]), int(m[2]), int(m[1]))),
                # Month name formats
                (r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
                 lambda m: date(int(m[3]), 
                              ['January','February','March','April','May','June','July','August','September','October','November','December'].index(m[2]) + 1,
                              int(m[1]))),
            ]
            
            for pattern, converter in patterns:
                match = re.search(pattern, date_str, re.IGNORECASE)
                if match:
                    try:
                        return converter(match)
                    except (ValueError, IndexError):
                        continue
            return None
        
        filtered = []
        for result in results:
            metadata = result.get("metadata", {})
            
            # Try to get date from various metadata fields
            doc_date = None
            for date_field in ["document_date", "date", "dates_mentioned"]:
                date_str = metadata.get(date_field)
                if date_str:
                    doc_date = parse_date(date_str)
                    if doc_date:
                        break
            
            # If no date found, include the result (don't filter out undated docs)
            if doc_date is None:
                filtered.append(result)
                continue
            
            # Check if date is within range
            if start_date <= doc_date <= end_date:
                filtered.append(result)
        
        return filtered

    def _get_chunk_data(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Get chunk text and metadata for given chunk IDs.

        Args:
            chunk_ids: List of chunk IDs

        Returns:
            Dictionary mapping chunk_id to {text, metadata}
        """
        chunk_data = {}

        # Query FTS5 index directly by IDs (fast SQL lookup)
        import json
        conn = self.keyword_index._get_conn()
        placeholders = ",".join("?" * len(chunk_ids))
        cursor = conn.execute(f"""
            SELECT chunk_id, chunk_text, document_id, file_name, 
                   document_type, page, paragraph, metadata_json
            FROM chunks WHERE chunk_id IN ({placeholders})
        """, chunk_ids)
        
        for row in cursor.fetchall():
            metadata = json.loads(row[7]) if row[7] else {}
            chunk_data[row[0]] = {
                "text": row[1],
                "metadata": {
                    "document_id": row[2],
                    "file_name": row[3],
                    "document_type": row[4],
                    "page_number": row[5],
                    "paragraph_number": row[6],
                    "section_header": metadata.get("section_header", ""),
                    **metadata,
                },
            }

        # If not found in keyword index, try vector store
        missing_ids = [cid for cid in chunk_ids if cid not in chunk_data]
        if missing_ids:
            # Query vector store for missing chunks
            try:
                vector_results = self.vector_store.collection.get(ids=missing_ids)
                if vector_results and vector_results.get("documents"):
                    for i, chunk_id in enumerate(missing_ids):
                        if i < len(vector_results["documents"]):
                            chunk_data[chunk_id] = {
                                "text": vector_results["documents"][i],
                                "metadata": (
                                    vector_results["metadatas"][i]
                                    if vector_results.get("metadatas")
                                    else {}
                                ),
                            }
            except Exception:
                # If vector store query fails, continue with what we have
                pass

        return chunk_data

    def enhance_with_summaries(
        self,
        results: list[dict[str, Any]],
        include_doc_summary: bool = True,
    ) -> list[dict[str, Any]]:
        """Enhance retrieval results with document summaries.
        
        Adds document-level summaries to each result for richer context.
        This helps the LLM understand the broader document context.
        
        Args:
            results: List of retrieval results from retrieve()
            include_doc_summary: Whether to include document summaries
            
        Returns:
            Results with added 'document_summary' field
        """
        if not include_doc_summary or not results:
            return results
        
        try:
            # Cache summaries by document_id to avoid repeated lookups
            summary_cache: dict[str, str] = {}
            
            for result in results:
                # document_id is in metadata, not at top level
                doc_id = result.get("metadata", {}).get("document_id", "")
                
                if doc_id and doc_id not in summary_cache:
                    # Try to get document summary
                    if self.summary_store:
                        summaries = self.summary_store.get_document_summaries(
                            document_id=doc_id,
                            summary_type="overview",
                            summary_level="document",
                        )
                        summary_cache[doc_id] = summaries[0].content if summaries else None
                
                # Add summary to result
                if doc_id and summary_cache.get(doc_id):
                    result["document_summary"] = summary_cache[doc_id]
            
            return results
            
        except Exception as e:
            # If summary store fails, return results without summaries
            print(f"Warning: Could not enhance with summaries: {e}")
            return results

    def _apply_mmr(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int,
        lambda_mult: float = 0.5
    ) -> list[dict[str, Any]]:
        """Apply Maximal Marginal Relevance (MMR) to diversify results.
        
        MMR = argmax [ lambda * Sim(D, Q) - (1-lambda) * max(Sim(D, Di)) ]
        """
        try:
            if not results:
                return []
                
            # Get embeddings for all result chunks
            chunk_ids = [r["chunk_id"] for r in results]
            embeddings_map = self.vector_store.get_embeddings(chunk_ids)
            
            # Filter out chunks without embeddings
            valid_results = []
            valid_embeddings = []
            for r in results:
                if r["chunk_id"] in embeddings_map:
                    valid_results.append(r)
                    valid_embeddings.append(embeddings_map[r["chunk_id"]])
            
            if not valid_results:
                return results[:top_k]
                
            # Need query embedding
            query_emb = self.embedding_service.embed_query(query)
            
            # Compute similarities
            import numpy as np
            
            # Convert to numpy
            doc_embs = np.array(valid_embeddings)
            query_vec = np.array(query_emb)
            
            # Normalize
            doc_norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
            doc_embs = doc_embs / (doc_norms + 1e-9)
            
            query_norm = np.linalg.norm(query_vec)
            query_vec = query_vec / (query_norm + 1e-9)
            
            # Sim(D, Q)
            sim_query = np.dot(doc_embs, query_vec)
            
            # Sim(D, D)
            sim_docs = np.dot(doc_embs, doc_embs.T)
            
            # MMR Selection
            selected_indices = []
            candidate_indices = list(range(len(valid_results)))
            
            while len(selected_indices) < min(top_k, len(valid_results)):
                best_score = -float("inf")
                best_idx = -1
                
                for idx in candidate_indices:
                    # Relevance term
                    relevance = sim_query[idx]
                    
                    # Diversity term (max sim to already selected)
                    if not selected_indices:
                        diversity = 0
                    else:
                        diversity = np.max([sim_docs[idx][sel_idx] for sel_idx in selected_indices])
                    
                    mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx
                
                if best_idx != -1:
                    selected_indices.append(best_idx)
                    candidate_indices.remove(best_idx)
                else:
                    break
            
            return [valid_results[i] for i in selected_indices]
            
        except Exception as e:
            print(f"MMR failed: {e}")
            return results[:top_k]

    def _llm_grade_results(self, query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Grade results using LLM to filter irrelevant chunks."""
        try:
            from src.generation.llm_service import LLMService
            llm = LLMService(settings=self.settings)
            
            graded_results = []
            model = self.settings.advanced_retrieval.grading_model
            
            # Prepare batch prompt (grading individually is too slow, batching 5 at a time)
            batch_size = 5
            for i in range(0, len(results), batch_size):
                batch = results[i : i + batch_size]
                
                prompt = (
                    f"Query: {query}\n\n"
                    "Evaluate the relevance of the following text chunks to the query. "
                    "Output ONLY a JSON list of booleans [true, false, ...] corresponding to the chunks.\n\n"
                )
                
                for j, res in enumerate(batch):
                    # Use 'text' field for content, not 'content'
                    prompt += f"Chunk {j+1}:\n{res['text'][:400]}...\n\n"
                
                response = llm.generate(
                    prompt=prompt,
                    model=model,
                    temperature=0.0,
                    max_tokens=100
                )
                
                # Parse JSON response
                import json
                import re
                try:
                    # Find JSON list in response
                    match = re.search(r'\[.*?\]', response, re.DOTALL)
                    if match:
                        grades = json.loads(match.group(0))
                        if len(grades) == len(batch):
                            for k, is_relevant in enumerate(grades):
                                if is_relevant:
                                    graded_results.append(batch[k])
                        else:
                            # Fallback: keep all if count mismatch
                            graded_results.extend(batch)
                    else:
                        graded_results.extend(batch)
                except Exception:
                    graded_results.extend(batch)
            
            return graded_results
            
        except Exception as e:
            print(f"LLM grading failed: {e}")
            return results

    def build_context_with_summaries(
        self,
        results: list[dict[str, Any]],
        max_chunks: int = 10,
    ) -> str:
        """Build LLM context that includes document summaries.
        
        Creates a structured context string with:
        1. Document summaries for broader context
        2. Specific chunks for detailed evidence
        
        Args:
            results: Retrieved chunks (enhanced with summaries)
            max_chunks: Maximum chunks to include
            
        Returns:
            Formatted context string for LLM
        """
        context_parts = []
        
        # Group chunks by document
        docs: dict[str, list[dict]] = {}
        for result in results[:max_chunks]:
            doc_id = result.get("document_id", "unknown")
            if doc_id not in docs:
                docs[doc_id] = []
            docs[doc_id].append(result)
        
        # Build context for each document
        for doc_id, doc_chunks in docs.items():
            file_name = doc_chunks[0].get("metadata", {}).get("file_name", "Unknown")
            doc_summary = doc_chunks[0].get("document_summary")
            
            # Add document header and summary if available
            context_parts.append(f"## Document: {file_name}")
            if doc_summary:
                context_parts.append(f"**Document Summary:** {doc_summary}")
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

