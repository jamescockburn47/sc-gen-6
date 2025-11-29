"""Base class for generators that work from document summaries.

This module provides the foundation for generating insights (case graph,
timeline, case overview) from document summaries rather than raw documents.
"""

from __future__ import annotations

from typing import Any, Optional

from src.config_loader import Settings, get_settings
from src.llm.client import LLMClient
from src.retrieval.summary_store import DocumentSummary, SummaryStore


class SummaryBasedGenerator:
    """Base class for generating insights from document summaries.
    
    Provides common functionality for:
    - Loading summaries from the summary store
    - Creating LLM clients with appropriate models
    - Progress tracking
    """
    
    def __init__(
        self,
        summary_store: Optional[SummaryStore] = None,
        llm_client: Optional[LLMClient] = None,
        settings: Optional[Settings] = None,
    ):
        """Initialize summary-based generator.
        
        Args:
            summary_store: Summary store instance. If None, creates new one.
            llm_client: LLM client instance. If None, creates new one.
            settings: Settings instance. If None, loads from config.
        """
        self.settings = settings or get_settings()
        self.summary_store = summary_store or SummaryStore(settings=self.settings)
        self._llm_client = llm_client
        self._current_model: Optional[str] = None
    
    def get_llm_client(self, model: Optional[str] = None) -> LLMClient:
        """Get or create LLM client for the specified model.
        
        Args:
            model: Model name. If None, uses default model.
            
        Returns:
            LLM client configured for the model
        """
        target_model = model or self.settings.models.llm.default
        
        # Reuse client if same model
        if self._llm_client and self._current_model == target_model:
            return self._llm_client
        
        from src.config.llm_config import load_llm_config
        
        # Load base config and override model
        config = load_llm_config()
        config.model_name = target_model
        
        self._llm_client = LLMClient(config)
        self._current_model = target_model
        
        return self._llm_client
    
    def get_all_summaries(
        self,
        summary_type: Optional[str] = None,
        summary_level: str = "document"
    ) -> list[DocumentSummary]:
        """Retrieve all document summaries from the store.
        
        Args:
            summary_type: Optional filter by summary type
            summary_level: Summary level (default: "document")
            
        Returns:
            List of document summaries
        """
        # Get all summaries from the store
        # This is a simplified implementation - in production you'd want
        # to query the database more efficiently
        conn = self.summary_store._get_conn()
        
        query = "SELECT * FROM document_summaries WHERE summary_level = ?"
        params = [summary_level]
        
        if summary_type:
            query += " AND summary_type = ?"
            params.append(summary_type)
        
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        
        return [self.summary_store._row_to_summary(row) for row in rows]
    
    def get_new_summaries(
        self,
        last_processed_time: Optional[str] = None,
        summary_type: Optional[str] = None,
        summary_level: str = "document"
    ) -> list[DocumentSummary]:
        """Get summaries created after a specific time.
        
        Args:
            last_processed_time: ISO timestamp of last processing
            summary_type: Optional filter by summary type
            summary_level: Summary level (default: "document")
            
        Returns:
            List of new summaries
        """
        if not last_processed_time:
            return self.get_all_summaries(summary_type, summary_level)
        
        conn = self.summary_store._get_conn()
        
        query = "SELECT * FROM document_summaries WHERE summary_level = ? AND created_at > ?"
        params = [summary_level, last_processed_time]
        
        if summary_type:
            query += " AND summary_type = ?"
            params.append(summary_type)
        
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        
        return [self.summary_store._row_to_summary(row) for row in rows]
    
    def generate_from_summaries(
        self,
        summaries: list[DocumentSummary],
        model: Optional[str] = None,
        incremental: bool = False,
        **kwargs
    ) -> Any:
        """Generate output from summaries.
        
        This method should be implemented by subclasses.
        
        Args:
            summaries: List of document summaries
            model: Model to use for generation
            incremental: If True, only process new items
            **kwargs: Additional arguments
            
        Returns:
            Generated output (type depends on subclass)
        """
        raise NotImplementedError("Subclasses must implement generate_from_summaries")
    
    def close(self):
        """Clean up resources."""
        if self.summary_store:
            self.summary_store.close()
