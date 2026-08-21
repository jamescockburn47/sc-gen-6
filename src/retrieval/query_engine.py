import logging
import queue
import threading
from typing import Generator, Dict, Any, Optional

from src.config_loader import Settings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_service import LLMService

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Facade for retrieval and generation logic, designed for the Desktop UI.
    Supports multi-turn conversation context and summary-guided retrieval.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.retriever = HybridRetriever(settings=settings)
        self.llm_service = LLMService(settings=settings)

        # Lazy-init summary store for summary-first retrieval
        self._summary_store = None

    @property
    def summary_store(self):
        """Lazy-loaded SummaryStore instance."""
        if self._summary_store is None:
            try:
                from src.retrieval.summary_store import SummaryStore
                self._summary_store = SummaryStore(settings=self.settings)
            except Exception:
                self._summary_store = False  # sentinel: unavailable
        return self._summary_store if self._summary_store is not False else None

    # -------------------------------------------------------------- #
    #  Summary-guided pre-filter
    # -------------------------------------------------------------- #
    def _summary_guided_documents(
        self, query: str, top_k: int = 8
    ) -> Optional[list[str]]:
        """Search document summaries to identify the most relevant documents.

        This implements the "summary-first" retrieval strategy: before
        retrieving chunks, consult the summary index to narrow the
        document scope.  Returns a list of file-names for the top-k
        matching documents, or *None* if summaries are unavailable
        (which tells the retriever to search everything).

        Args:
            query: User query.
            top_k: Max documents to shortlist.

        Returns:
            List of file names or None.
        """
        store = self.summary_store
        if not store:
            return None

        try:
            results = store.search(
                query=query,
                top_k=top_k,
                summary_type="overview",
                summary_level="document",
            )
            if not results:
                return None  # no summaries yet — search everything

            # Return unique file names (summaries may overlap on doc)
            seen: set[str] = set()
            file_names: list[str] = []
            for summary, _score in results:
                if summary.file_name not in seen:
                    seen.add(summary.file_name)
                    file_names.append(summary.file_name)
            logger.info(
                "Summary-guided pre-filter: shortlisted %d documents for query",
                len(file_names),
            )
            return file_names if file_names else None
        except Exception as exc:
            logger.warning("Summary pre-filter failed, searching all docs: %s", exc)
            return None

    def query(
        self,
        text: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
        thinking_callback=None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Execute a query: Summary pre-filter -> Retrieve -> Generate.

        Yields events:
            - {"source": {...}}
            - {"thinking": "..."}  — reasoning/chain-of-thought token
            - {"token": "..."}    — final answer token
            - {"error": "..."}
        """
        try:
            # 0. Summary-guided pre-filter (narrows retrieval scope)
            shortlisted_docs: Optional[list[str]] = None
            if self.settings.retrieval.search_summaries:
                shortlisted_docs = self._summary_guided_documents(text)

            # 1. Retrieval (scoped to shortlisted docs if available)
            results = self.retriever.retrieve(
                query=text,
                selected_documents=shortlisted_docs,
            )

            # Yield sources immediately
            for res in results:
                yield {
                    "source": {
                        "file_name": res.get("metadata", {}).get("file_name"),
                        "page": res.get("metadata", {}).get("page_number"),
                        "text": res.get("text")[:200] + "..."
                    }
                }

            if not results:
                yield {"token": "I couldn't find any relevant information in your documents."}
                return

            # 2. Build conversation preamble for multi-turn context
            system_prompt = None
            if conversation_history:
                preamble_parts: list[str] = []
                # Include last 5 turns max to stay within context window
                for turn in conversation_history[-5:]:
                    q = turn.get("query", "")
                    r = turn.get("response", "")
                    # Truncate long responses to ~500 chars for context
                    if len(r) > 500:
                        r = r[:500] + " [truncated]"
                    preamble_parts.append(f"User: {q}\nAssistant: {r}")
                conversation_block = "\n\n".join(preamble_parts)
                system_prompt = (
                    "You are a litigation support assistant. Answer questions "
                    "based strictly on the provided evidence with full citations.\n\n"
                    "PREVIOUS CONVERSATION:\n"
                    f"{conversation_block}\n\n"
                    "Continue the conversation. The user's new question follows."
                )

            # 3. Generation (streaming)
            token_queue: queue.Queue = queue.Queue()
            generation_finished = threading.Event()

            def callback(token):
                token_queue.put(token)

            def run_generation():
                try:
                    kwargs: dict[str, Any] = {
                        "query": text,
                        "chunks": results,
                        "stream": True,
                        "callback": callback,
                    }
                    if thinking_callback:
                        kwargs["thinking_callback"] = thinking_callback
                    if system_prompt:
                        kwargs["system_prompt"] = system_prompt
                    if max_tokens is not None:
                        kwargs["max_tokens"] = max_tokens
                    self.llm_service.generate_with_context(**kwargs)
                except Exception as e:
                    token_queue.put({"error": str(e)})
                finally:
                    generation_finished.set()

            gen_thread = threading.Thread(target=run_generation)
            gen_thread.start()

            while not generation_finished.is_set() or not token_queue.empty():
                try:
                    item = token_queue.get(timeout=0.1)
                    if isinstance(item, dict) and "error" in item:
                        yield item
                    else:
                        yield {"token": item}
                except queue.Empty:
                    continue

        except Exception as e:
            yield {"error": str(e)}
