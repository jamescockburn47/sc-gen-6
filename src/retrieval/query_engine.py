import queue
import threading
from typing import Generator, Dict, Any

from src.config_loader import Settings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_service import LLMService

class QueryEngine:
    """
    Facade for retrieval and generation logic, designed for the Desktop UI.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.retriever = HybridRetriever(settings)
        self.llm_service = LLMService(settings)

    def query(self, text: str) -> Generator[Dict[str, Any], None, None]:
        """
        Execute a query: Retrieve -> Generate.
        Yields events:
        - {"source": {...}}
        - {"token": "..."}
        - {"error": "..."}
        """
        try:
            # 1. Retrieval
            results = self.retriever.retrieve(query=text)
            
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

            # 2. Generation
            # We use a queue to bridge the callback-based LLM service to this generator
            token_queue = queue.Queue()
            generation_finished = threading.Event()

            def callback(token):
                token_queue.put(token)

            def run_generation():
                try:
                    self.llm_service.generate_with_context(
                        query=text,
                        chunks=results,
                        stream=True,
                        callback=callback
                    )
                except Exception as e:
                    token_queue.put({"error": str(e)})
                finally:
                    generation_finished.set()

            # Start generation in a separate thread so we can yield from the queue
            gen_thread = threading.Thread(target=run_generation)
            gen_thread.start()

            # Consume queue
            while not generation_finished.is_set() or not token_queue.empty():
                try:
                    # Wait for token with timeout to check finished flag
                    item = token_queue.get(timeout=0.1)
                    if isinstance(item, dict) and "error" in item:
                        yield item
                    else:
                        yield {"token": item}
                except queue.Empty:
                    continue

        except Exception as e:
            yield {"error": str(e)}
