"""Document Summarization Service.

Generates document summaries using a dedicated (smaller, faster) model.
Supports background generation with progress tracking and cancellation.
"""

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from PySide6.QtCore import QObject, Signal
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

from src.config_loader import Settings, get_settings
from src.retrieval.summary_store import SummaryStore, DocumentSummary, SummaryType


# Summarization prompts optimized for different document types
SUMMARY_PROMPTS = {
    "overview": """Provide a concise overview of this legal document.
Structure your response as follows:
1. **Parties**: List the key parties and their roles (e.g., Claimant, Defendant, Witness).
2. **Summary**: A 2-3 paragraph summary of the document's content, purpose, and key facts.
3. **Key Dates**: Important dates mentioned.

Be factual and precise.

Document: {file_name}
Type: {doc_type}

---
{text}
---

Overview:""",

    "key_points": """Extract the 5-10 most important facts from this legal document as bullet points.
Focus on: parties, dates, amounts, legal claims, and key evidence.
Format as a bulleted list.

Document: {file_name}
Type: {doc_type}

---
{text}
---

Key Points:""",

    "entities": """Extract all named entities from this legal document.
Categories:
- PARTIES: Legal parties (Claimants, Defendants, Appellants, Respondents)
- PEOPLE: Other individuals (Witnesses, Experts, etc.)
- ORGANIZATIONS: Companies, Courts, Bodies
- DATES: Important dates
- AMOUNTS: Monetary values
- REFERENCES: Case citations, statutes

Document: {file_name}

---
{text}
---

Entities:""",

    "timeline": """Create a chronological timeline of events from this document.
Format each entry as: [DATE] - Event description
Include all dates mentioned with their context.

Document: {file_name}

---
{text}
---

Timeline:""",
}


if QT_AVAILABLE:
    class SummarizerProgressSignals(QObject):
        """Progress signals for background summarization."""
        
        # Overall progress: (completed, total, current_file)
        progress = Signal(int, int, str)
        
        # Batch progress: (batch_num, total_batches)
        batch_progress = Signal(int, int)
        
        # Individual document: (file_name, status, time_seconds)
        document_completed = Signal(str, str, float)
        
        # Error: (file_name, error_message)
        error = Signal(str, str)
        
        # Finished: (total_completed, total_time_seconds)
        finished = Signal(int, float)
        
        # Model status: (model_name, status)
        model_status = Signal(str, str)


class SummarizerService:
    """Service for generating document summaries.
    
    Uses a dedicated (smaller, faster) model for efficient bulk summarization.
    Supports parallel processing and progress tracking.
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        progress_signals: Optional['SummarizerProgressSignals'] = None,
    ):
        """Initialize summarizer service.
        
        Args:
            settings: Settings instance
            progress_signals: Optional Qt signals for progress tracking
        """
        self.settings = settings or get_settings()
        self.progress_signals = progress_signals
        self.summary_store = SummaryStore(settings=self.settings)
        self._cancel_event = threading.Event()
        
        # Lazy-load LLM client for summarization model
        self._llm_client = None
        self._current_model = None
    
    def _get_llm_client(self, model: Optional[str] = None):
        """Get or create LLM client for summarization.
        
        Args:
            model: Model name. If None, uses configured summarization model.
            
        Returns:
            LLM client configured for the summarization model
        """
        target_model = model or self.settings.summary.summarization_model
        
        # Reuse client if same model
        if self._llm_client and self._current_model == target_model:
            return self._llm_client
        
        from src.llm.client import LLMClient
        from src.config.llm_config import load_llm_config
        
        # Load base config and override model
        config = load_llm_config()
        config.model_name = target_model
        
        self._llm_client = LLMClient(config)
        self._current_model = target_model
        
        if self.progress_signals:
            self.progress_signals.model_status.emit(target_model, "loaded")
        
        return self._llm_client
    
    def ensure_model_available(self, model: Optional[str] = None) -> tuple[bool, str]:
        """Ensure the summarization model is available (pull if needed).
        
        Args:
            model: Model name. If None, uses configured summarization model.
            
        Returns:
            Tuple of (success, message)
        """
        target_model = model or self.settings.summary.summarization_model
        
        if self.progress_signals:
            self.progress_signals.model_status.emit(target_model, "checking")
        
        # Check if model exists via Ollama
        try:
            import requests
            
            ollama_host = self.settings.models.ollama.host
            response = requests.get(f"{ollama_host}/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                # Check if our model is available
                if any(target_model in name for name in model_names):
                    if self.progress_signals:
                        self.progress_signals.model_status.emit(target_model, "available")
                    return True, f"Model {target_model} is available"
                
                # Model not found - try to pull it
                if self.progress_signals:
                    self.progress_signals.model_status.emit(target_model, "pulling")
                
                print(f"Pulling summarization model: {target_model}")
                pull_response = requests.post(
                    f"{ollama_host}/api/pull",
                    json={"name": target_model},
                    timeout=600,  # 10 minute timeout for pulling
                    stream=True,
                )
                
                # Stream the pull progress
                for line in pull_response.iter_lines():
                    if self._cancel_event.is_set():
                        return False, "Model pull cancelled"
                    if line:
                        import json
                        try:
                            data = json.loads(line)
                            status = data.get("status", "")
                            if "pulling" in status:
                                print(f"  {status}")
                        except json.JSONDecodeError:
                            pass
                
                if self.progress_signals:
                    self.progress_signals.model_status.emit(target_model, "available")
                return True, f"Model {target_model} pulled successfully"
            
            return False, f"Ollama API error: {response.status_code}"
            
        except requests.exceptions.ConnectionError:
            return False, "Ollama is not running. Start it with: ollama serve"
        except Exception as e:
            return False, f"Error checking model: {str(e)}"
    
    def generate_summary(
        self,
        document_id: str,
        document_text: str,
        file_name: str,
        doc_type: str = "unknown",
        summary_type: SummaryType = "overview",
        model: Optional[str] = None,
    ) -> Optional[DocumentSummary]:
        """Generate a single document summary.
        
        Args:
            document_id: Document identifier
            document_text: Full document text
            file_name: Source file name
            doc_type: Document type for context
            summary_type: Type of summary to generate
            model: Model to use (None = configured summarization model)
            
        Returns:
            Generated DocumentSummary or None on error
        """
        if self._cancel_event.is_set():
            return None
        
        client = self._get_llm_client(model)
        
        # Truncate text if too long (~32K chars = ~8K tokens)
        max_chars = 32000
        text_to_summarize = document_text[:max_chars]
        if len(document_text) > max_chars:
            text_to_summarize += "\n\n[Document truncated for summarization...]"
        
        # Build prompt
        prompt_template = SUMMARY_PROMPTS.get(summary_type, SUMMARY_PROMPTS["overview"])
        prompt = prompt_template.format(
            file_name=file_name,
            doc_type=doc_type,
            text=text_to_summarize,
        )
        
        try:
            # Generate summary
            response = client.generate_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a legal document summarization assistant. Be accurate, concise, and cite specific details."},
                    {"role": "user", "content": prompt},
                ],
                model=self._current_model,
                temperature=0.3,  # Lower temperature for factual summarization
                max_tokens=self.settings.summary.max_summary_length * 2,
            )
            
            # Create summary object
            summary_id = hashlib.sha256(
                f"{document_id}:{summary_type}:document".encode()
            ).hexdigest()[:16]
            
            return DocumentSummary(
                summary_id=summary_id,
                document_id=document_id,
                file_name=file_name,
                summary_type=summary_type,
                summary_level="document",
                content=response.strip(),
                metadata={
                    "model": self._current_model,
                    "doc_type": doc_type,
                    "input_length": len(document_text),
                    "generated_at": datetime.now().isoformat(),
                },
            )
            
        except Exception as e:
            if self.progress_signals:
                self.progress_signals.error.emit(file_name, str(e))
            return None
    
    def summarize_documents(
        self,
        documents: list[dict[str, Any]],
        summary_types: Optional[list[SummaryType]] = None,
        model: Optional[str] = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[DocumentSummary]:
        """Summarize multiple documents.
        
        Args:
            documents: List of document dicts with keys:
                - document_id: str
                - text: str
                - file_name: str
                - doc_type: str (optional)
            summary_types: Types of summaries to generate
            model: Model to use
            on_progress: Optional callback(completed, total, current_file)
            
        Returns:
            List of generated summaries
        """
        self._cancel_event.clear()
        
        summary_types = summary_types or self.settings.summary.summary_types
        total = len(documents) * len(summary_types)
        completed = 0
        summaries = []
        
        start_time = time.time()
        
        # Ensure model is available
        success, message = self.ensure_model_available(model)
        if not success:
            if self.progress_signals:
                self.progress_signals.error.emit("", message)
            return summaries
        
        # Process documents
        for doc in documents:
            if self._cancel_event.is_set():
                break
            
            doc_start = time.time()
            file_name = doc.get("file_name", "Unknown")
            
            for summary_type in summary_types:
                if self._cancel_event.is_set():
                    break
                
                if self.progress_signals:
                    self.progress_signals.progress.emit(completed, total, file_name)
                if on_progress:
                    on_progress(completed, total, file_name)
                
                summary = self.generate_summary(
                    document_id=doc.get("document_id", ""),
                    document_text=doc.get("text", ""),
                    file_name=file_name,
                    doc_type=doc.get("doc_type", "unknown"),
                    summary_type=summary_type,
                    model=model,
                )
                
                if summary:
                    summaries.append(summary)
                    self.summary_store.add_summary(summary)
                
                completed += 1
            
            doc_time = time.time() - doc_start
            if self.progress_signals:
                self.progress_signals.document_completed.emit(file_name, "completed", doc_time)
        
        total_time = time.time() - start_time
        if self.progress_signals:
            self.progress_signals.finished.emit(len(summaries), total_time)
        
        return summaries
    
    def summarize_documents_parallel(
        self,
        documents: list[dict[str, Any]],
        summary_types: Optional[list[SummaryType]] = None,
        model: Optional[str] = None,
        max_workers: Optional[int] = None,
    ) -> list[DocumentSummary]:
        """Summarize documents in parallel.
        
        Args:
            documents: List of document dicts
            summary_types: Types of summaries to generate
            model: Model to use
            max_workers: Max parallel workers (default from config)
            
        Returns:
            List of generated summaries
        """
        self._cancel_event.clear()
        
        summary_types = summary_types or self.settings.summary.summary_types
        max_workers = max_workers or self.settings.summary.parallel_summaries
        
        # Ensure model is available first
        success, message = self.ensure_model_available(model)
        if not success:
            if self.progress_signals:
                self.progress_signals.error.emit("", message)
            return []
        
        summaries = []
        total = len(documents)
        completed = [0]  # Use list for mutable in closure
        start_time = time.time()
        
        def process_document(doc: dict) -> list[DocumentSummary]:
            """Process a single document (thread-safe)."""
            if self._cancel_event.is_set():
                return []
            
            # Create thread-local summary store to avoid SQLite threading issues
            thread_summary_store = SummaryStore(settings=self.settings)
            
            doc_summaries = []
            file_name = doc.get("file_name", "Unknown")
            doc_start = time.time()
            
            try:
                for summary_type in summary_types:
                    if self._cancel_event.is_set():
                        break
                    
                    summary = self.generate_summary(
                        document_id=doc.get("document_id", ""),
                        document_text=doc.get("text", ""),
                        file_name=file_name,
                        doc_type=doc.get("doc_type", "unknown"),
                        summary_type=summary_type,
                        model=model,
                    )
                    
                    if summary:
                        doc_summaries.append(summary)
                        thread_summary_store.add_summary(summary)
                
                completed[0] += 1
                doc_time = time.time() - doc_start
                
                if self.progress_signals:
                    self.progress_signals.progress.emit(completed[0], total, file_name)
                    self.progress_signals.document_completed.emit(file_name, "completed", doc_time)
            finally:
                thread_summary_store.close()
            
            return doc_summaries
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_document, doc): doc for doc in documents}
            
            for future in as_completed(futures):
                if self._cancel_event.is_set():
                    break
                
                try:
                    doc_summaries = future.result()
                    summaries.extend(doc_summaries)
                except Exception as e:
                    doc = futures[future]
                    if self.progress_signals:
                        self.progress_signals.error.emit(doc.get("file_name", ""), str(e))
        
        total_time = time.time() - start_time
        if self.progress_signals:
            self.progress_signals.finished.emit(len(summaries), total_time)
        
        return summaries
    
    def cancel(self):
        """Cancel ongoing summarization."""
        self._cancel_event.set()
    
    def get_stats(self) -> dict[str, Any]:
        """Get summarization statistics.
        
        Returns:
            Dict with summary store stats and model info
        """
        stats = self.summary_store.stats()
        stats["summarization_model"] = self.settings.summary.summarization_model
        stats["model_loaded"] = self._current_model
        return stats


def estimate_summarization_time(
    num_documents: int,
    model: str = "qwen2.5:7b",
    summary_types: int = 1,
) -> dict[str, Any]:
    """Estimate time to summarize documents.
    
    Args:
        num_documents: Number of documents
        model: Model to use
        summary_types: Number of summary types per document
        
    Returns:
        Time estimates
    """
    # Estimated tokens/second for different models
    model_speeds = {
        "qwen2.5:3b": 80,
        "qwen2.5:7b": 40,
        "llama3.2:3b": 70,
        "mistral:7b": 35,
        "phi3:mini": 60,
        "deepseek-r1:14b": 25,
    }
    
    # Get speed for model (default to 40 tok/s)
    speed = 40
    for model_name, model_speed in model_speeds.items():
        if model_name in model.lower():
            speed = model_speed
            break
    
    # Estimate: ~200 output tokens per summary
    tokens_per_summary = 200
    seconds_per_summary = tokens_per_summary / speed
    
    # Add overhead (loading, API, etc.)
    seconds_per_summary *= 1.3
    
    total_summaries = num_documents * summary_types
    total_seconds = total_summaries * seconds_per_summary
    
    return {
        "num_documents": num_documents,
        "summary_types": summary_types,
        "total_summaries": total_summaries,
        "model": model,
        "estimated_speed_tok_s": speed,
        "seconds_per_summary": seconds_per_summary,
        "total_seconds": total_seconds,
        "total_minutes": total_seconds / 60,
        "formatted_time": f"{int(total_seconds // 60)}m {int(total_seconds % 60)}s",
    }

