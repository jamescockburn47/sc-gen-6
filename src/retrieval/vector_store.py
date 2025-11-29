"""Vector store using ChromaDB v1.0+ with persistent storage.

Manages the litigation_docs collection with cosine similarity search.
"""

from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config_loader import Settings, get_settings
from src.schema import Chunk


class VectorStore:
    """Vector store using ChromaDB for persistent storage.

    Manages the litigation_docs collection with cosine similarity.
    Supports batch insertion, querying with metadata filters, and deletion.
    """

    COLLECTION_NAME = "litigation_docs"

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        settings: Optional[Settings] = None,
    ):
        """Initialize vector store.

        Args:
            db_path: Path to ChromaDB storage. If None, uses path from config.
            settings: Settings instance. If None, loads from config.
        """
        self.settings = settings or get_settings()
        self.db_path = Path(db_path or self.settings.paths.vector_db)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(
                anonymized_telemetry=False,  # Disable telemetry for privacy
                allow_reset=True,
            ),
        )

        # Get or create collection
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get or create the litigation_docs collection.

        Returns:
            ChromaDB Collection instance
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.COLLECTION_NAME)
            return collection
        except Exception:
            # Collection doesn't exist, create it
            collection = self.client.create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "Litigation documents collection"},
                # Cosine similarity is default in Chroma v1.0+
            )
            return collection

    # ChromaDB batch size limit (leave margin below 5461)
    CHROMA_BATCH_SIZE = 5000

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks to the vector store in batches.

        ChromaDB has a batch size limit (~5461 embeddings). This method
        automatically batches large insertions to avoid the limit.

        Args:
            chunks: List of Chunk objects to add
            embeddings: List of embedding vectors (must match chunks length)

        Raises:
            ValueError: If chunks and embeddings lengths don't match
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )

        if not chunks:
            return

        # Prepare all data first
        all_ids = [chunk.chunk_id for chunk in chunks]
        all_documents = [chunk.text for chunk in chunks]

        # Prepare metadata (ChromaDB requires string values for metadata)
        all_metadatas = []
        for chunk in chunks:
            metadata: dict[str, Any] = {
                "document_id": chunk.document_id,
                "file_name": chunk.file_name,
                "document_type": chunk.document_type,
            }

            # Add optional fields if present
            if chunk.page_number is not None:
                metadata["page_number"] = str(chunk.page_number)
            if chunk.paragraph_number is not None:
                metadata["paragraph_number"] = str(chunk.paragraph_number)
            if chunk.section_header:
                metadata["section_header"] = chunk.section_header
            if chunk.char_start is not None:
                metadata["char_start"] = str(chunk.char_start)
            if chunk.char_end is not None:
                metadata["char_end"] = str(chunk.char_end)

            # Add any additional metadata (convert to strings)
            for key, value in chunk.metadata.items():
                if value is not None:
                    metadata[key] = str(value)

            all_metadatas.append(metadata)

        # Add in batches to respect ChromaDB limit
        total = len(chunks)
        for start in range(0, total, self.CHROMA_BATCH_SIZE):
            end = min(start + self.CHROMA_BATCH_SIZE, total)
            batch_num = (start // self.CHROMA_BATCH_SIZE) + 1
            total_batches = (total + self.CHROMA_BATCH_SIZE - 1) // self.CHROMA_BATCH_SIZE
            
            print(f"VectorStore: Adding batch {batch_num}/{total_batches} ({end - start} chunks)")
            
            self._add_batch(
                ids=all_ids[start:end],
                embeddings=embeddings[start:end],
                documents=all_documents[start:end],
                metadatas=all_metadatas[start:end],
            )
        
        print(f"VectorStore: Successfully added {total} chunks in {total_batches} batch(es)")

    def _add_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add a single batch to the collection.

        Args:
            ids: Chunk IDs
            embeddings: Embedding vectors
            documents: Document texts
            metadatas: Metadata dictionaries
        """
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as e:
            # Check for collection not found error (reset by another component)
            if "does not exist" in str(e):
                print("VectorStore: Collection invalid, refreshing reference...")
                self.collection = self._get_or_create_collection()
                try:
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas,
                    )
                    return
                except Exception as retry_error:
                    raise RuntimeError(f"Failed to add chunks after refresh: {str(retry_error)}") from retry_error
            
            raise RuntimeError(f"Failed to add chunks to vector store: {str(e)}") from e

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 10,
        where: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Query the vector store.

        Args:
            query_embeddings: List of query embedding vectors
            n_results: Number of results to return per query
            where: Optional metadata filter dictionary.
                   Example: {"document_type": "witness_statement", "file_name": "doc.pdf"}

        Returns:
            Dictionary with keys: 'ids', 'distances', 'metadatas', 'documents'
            Each value is a list of lists (one per query)
        """
        if not query_embeddings:
            return {
                "ids": [],
                "distances": [],
                "metadatas": [],
                "documents": [],
            }

        # Convert where clause values to strings (ChromaDB requirement)
        # But preserve operators like $in
        where_clause = None
        if where:
            where_clause = {}
            for k, v in where.items():
                if isinstance(v, dict):
                    # Handle operators like {"$in": ["doc1", "doc2"]}
                    where_clause[k] = v
                else:
                    # Convert simple values to strings
                    where_clause[k] = str(v)

        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where_clause,
            )
            return results
        except Exception as e:
            # Check for collection not found error (reset by another component)
            if "does not exist" in str(e):
                print("VectorStore: Collection invalid, refreshing reference...")
                self.collection = self._get_or_create_collection()
                try:
                    results = self.collection.query(
                        query_embeddings=query_embeddings,
                        n_results=n_results,
                        where=where_clause,
                    )
                    return results
                except Exception as retry_error:
                    raise RuntimeError(f"Failed to query vector store after refresh: {str(retry_error)}") from retry_error

            raise RuntimeError(f"Failed to query vector store: {str(e)}") from e

    def get_embeddings(self, chunk_ids: list[str]) -> dict[str, list[float]]:
        """Get embeddings for specific chunk IDs.

        Args:
            chunk_ids: List of chunk IDs

        Returns:
            Dictionary mapping chunk_id -> embedding vector
        """
        if not chunk_ids:
            return {}

        try:
            results = self.collection.get(
                ids=chunk_ids,
                include=["embeddings"]
            )
            
            # Map ids to embeddings
            id_to_emb = {}
            if results and results.get("ids") and results.get("embeddings"):
                for i, chunk_id in enumerate(results["ids"]):
                    if i < len(results["embeddings"]):
                        id_to_emb[chunk_id] = results["embeddings"][i]
            
            return id_to_emb
        except Exception as e:
            print(f"VectorStore: Failed to get embeddings: {e}")
            return {}

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a document.

        Args:
            document_id: Document ID to delete

        Raises:
            RuntimeError: If deletion fails
        """
        try:
            # Delete chunks where document_id matches
            self.collection.delete(
                where={"document_id": document_id},
            )
        except Exception as e:
            if "does not exist" in str(e):
                print("VectorStore: Collection invalid, refreshing reference...")
                self.collection = self._get_or_create_collection()
                try:
                    self.collection.delete(
                        where={"document_id": document_id},
                    )
                    return
                except Exception:
                    pass # Ignore if still fails, maybe already empty

            raise RuntimeError(
                f"Failed to delete document {document_id} from vector store: {str(e)}"
            ) from e

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Delete specific chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to delete

        Raises:
            RuntimeError: If deletion fails
        """
        if not chunk_ids:
            return

        try:
            self.collection.delete(ids=chunk_ids)
        except Exception as e:
            if "does not exist" in str(e):
                print("VectorStore: Collection invalid, refreshing reference...")
                self.collection = self._get_or_create_collection()
                try:
                    self.collection.delete(ids=chunk_ids)
                    return
                except Exception:
                    pass

            raise RuntimeError(
                f"Failed to delete chunks from vector store: {str(e)}"
            ) from e

    def stats(self, include_documents: bool = True) -> dict[str, Any]:
        """Get statistics about the vector store.

        Uses pagination to avoid memory issues with large datasets.

        Args:
            include_documents: Whether to include list of unique documents.
                              Set False for faster basic stats.

        Returns:
            Dictionary with collection statistics including list of unique documents
        """
        try:
            count = self.collection.count()
            
            # For large collections, use pagination to avoid memory issues
            BATCH_SIZE = 1000
            document_types: dict[str, int] = {}
            file_names: set[str] = set()

            if count > 0 and include_documents:
                # Paginate through metadata to avoid loading all at once
                offset = 0
                while offset < count:
                    try:
                        batch_results = self.collection.get(
                            limit=BATCH_SIZE,
                            offset=offset,
                            include=["metadatas"]  # Only fetch metadata, not documents
                        )
                        
                        if not batch_results or not batch_results.get("metadatas"):
                            break
                        
                        for metadata in batch_results["metadatas"]:
                            doc_type = metadata.get("document_type", "unknown")
                            document_types[doc_type] = document_types.get(doc_type, 0) + 1
                            file_name = metadata.get("file_name")
                            if file_name:
                                file_names.add(file_name)
                        
                        offset += BATCH_SIZE
                        
                        # Safety: if we've collected enough unique files, stop early
                        # (prevents infinite loop on very large datasets)
                        if offset > 100000:
                            break
                            
                    except Exception as batch_error:
                        print(f"VectorStore: Error fetching batch at offset {offset}: {batch_error}")
                        break

            return {
                "total_chunks": count,
                "document_types": document_types,
                "unique_files": len(file_names),
                "unique_documents": sorted(list(file_names)),
                "collection_name": self.COLLECTION_NAME,
                "db_path": str(self.db_path),
            }
        except Exception as e:
            return {
                "error": str(e),
                "total_chunks": 0,
                "collection_name": self.COLLECTION_NAME,
                "db_path": str(self.db_path),
            }
    
    def get_unique_documents(self) -> list[str]:
        """Get list of unique document names efficiently.
        
        Uses pagination to handle large collections without memory issues.
        
        Returns:
            Sorted list of unique document file names
        """
        try:
            count = self.collection.count()
            if count == 0:
                return []
            
            BATCH_SIZE = 1000
            file_names: set[str] = set()
            offset = 0
            
            while offset < count:
                try:
                    batch_results = self.collection.get(
                        limit=BATCH_SIZE,
                        offset=offset,
                        include=["metadatas"]
                    )
                    
                    if not batch_results or not batch_results.get("metadatas"):
                        break
                    
                    for metadata in batch_results["metadatas"]:
                        file_name = metadata.get("file_name")
                        if file_name:
                            file_names.add(file_name)
                    
                    offset += BATCH_SIZE
                    
                    # Safety limit
                    if offset > 100000:
                        break
                        
                except Exception:
                    break
            
            return sorted(list(file_names))
            
        except Exception:
            return []

    def reset(self) -> None:
        """Reset the collection (delete all data).

        Use with caution! This permanently deletes all stored chunks.
        """
        try:
            self.client.delete_collection(name=self.COLLECTION_NAME)
            self.collection = self._get_or_create_collection()
        except Exception as e:
            raise RuntimeError(f"Failed to reset vector store: {str(e)}") from e




