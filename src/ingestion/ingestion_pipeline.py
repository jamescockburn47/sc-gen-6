"""Document ingestion pipeline: auto-select parser, batch ingest, logging."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from src.ingestion.parsers.base_parser import BaseParser
from src.ingestion.parsers.docx_parser import DOCXParser
from src.ingestion.parsers.email_parser import EmailParser
from src.ingestion.parsers.ocr_parser import OCRParser
from src.ingestion.parsers.pdf_parser import PDFParser
from src.ingestion.parsers.spreadsheet_parser import SpreadsheetParser
from src.schema import DocumentType, ParsedDocument, Chunk


# Configure logging
log_file = Path("logs/ingestion.log")
log_file.parent.mkdir(parents=True, exist_ok=True)

logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)
logger.add(
    log_file,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
)


class IngestionPipeline:
    """Pipeline for ingesting documents with automatic parser selection.

    Supports batch processing of folders, error handling, and logging.
    """

    def __init__(self):
        """Initialize pipeline with available parsers."""
        self.parsers: list[BaseParser] = [
            PDFParser(),
            DOCXParser(),
            EmailParser(),
            SpreadsheetParser(),
            OCRParser(),  # Should be last as it can delegate to PDFParser
        ]

    def get_parser(self, file_path: str | Path) -> Optional[BaseParser]:
        """Get appropriate parser for a file.

        Args:
            file_path: Path to file

        Returns:
            Parser instance or None if no parser can handle the file
        """
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        return None

    def parse_document(
        self, file_path: str | Path, document_type: Optional[DocumentType] = None
    ) -> Optional[ParsedDocument]:
        """Parse a single document.

        Args:
            file_path: Path to document
            document_type: Optional document type override

        Returns:
            ParsedDocument or None if parsing fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        parser = self.get_parser(file_path)
        if parser is None:
            logger.warning(f"No parser available for file: {file_path}")
            return None

        try:
            logger.info(f"Parsing {file_path.name} with {parser.__class__.__name__}")
            parsed = parser.parse(file_path, document_type)
            logger.success(
                f"Successfully parsed {file_path.name}: "
                f"{parsed.document_type}, {len(parsed.text)} chars, "
                f"{len(parsed.paragraphs)} paragraphs"
            )
            return parsed
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {str(e)}", exc_info=True)
            return None

    def ingest_folder(
        self,
        folder_path: str | Path,
        document_type: Optional[DocumentType] = None,
        recursive: bool = True,
    ) -> list[ParsedDocument]:
        """Batch ingest all documents in a folder.

        Args:
            folder_path: Path to folder containing documents
            document_type: Optional document type override for all files
            recursive: If True, process subdirectories recursively

        Returns:
            List of successfully parsed documents
        """
        folder_path = Path(folder_path)
        if not folder_path.exists() or not folder_path.is_dir():
            logger.error(f"Folder not found or not a directory: {folder_path}")
            return []

        logger.info(f"Starting batch ingestion from {folder_path}")

        # Find all files
        if recursive:
            files = list(folder_path.rglob("*"))
        else:
            files = list(folder_path.glob("*"))

        # Filter to files only
        files = [f for f in files if f.is_file()]

        logger.info(f"Found {len(files)} files to process")

        parsed_documents = []
        failed_count = 0

        for file_path in files:
            parsed = self.parse_document(file_path, document_type)
            if parsed:
                parsed_documents.append(parsed)
            else:
                failed_count += 1

        logger.info(
            f"Batch ingestion complete: {len(parsed_documents)} succeeded, "
            f"{failed_count} failed"
        )

        return parsed_documents

    def ingest_files(
        self,
        file_paths: list[str | Path],
        document_type: Optional[DocumentType] = None,
        max_workers: Optional[int] = None,
    ) -> list[ParsedDocument]:
        """Ingest multiple files, optionally in parallel.

        Args:
            file_paths: List of file paths
            document_type: Optional document type override for all files
            max_workers: Number of worker processes. If None, runs sequentially.

        Returns:
            List of successfully parsed documents
        """
        logger.info(f"Processing {len(file_paths)} files")

        parsed_documents = []
        
        if max_workers is not None and max_workers > 1:
            # Parallel processing
            import concurrent.futures
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_path = {
                    executor.submit(self.parse_document, path, document_type): path 
                    for path in file_paths
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        parsed = future.result()
                        if parsed:
                            parsed_documents.append(parsed)
                    except Exception as e:
                        logger.error(f"Error processing {path}: {e}")
        else:
            # Sequential processing
            for file_path in file_paths:
                parsed = self.parse_document(file_path, document_type)
                if parsed:
                    parsed_documents.append(parsed)

        logger.info(f"Processed {len(parsed_documents)}/{len(file_paths)} files successfully")
        return parsed_documents
    
    def process_document(
        self,
        file_path: str | Path,
        document_type: Optional[DocumentType] = None,
        generate_summary: bool = True,
    ) -> bool:
        """Process a document through the complete RAG pipeline.
        
        This method orchestrates the full ingestion workflow:
        1. Parse document → ParsedDocument
        2. Chunk text using configured chunk sizes
        3. Generate embeddings (GPU-accelerated ONNX)
        4. Store in ChromaDB vector store
        5. Store in BM25/FTS5 keyword index
        6. Update document catalog
        7. Generate summaries (optional, parallel, GPU model)
        
        Args:
            file_path: Path to document file
            document_type: Optional document type override
            generate_summary: Whether to generate document summaries
            
        Returns:
            True if successful, False otherwise
        """
        from datetime import datetime
        import hashlib
        
        # Lazy imports to avoid circular dependencies
        from src.config_loader import get_settings
        from src.retrieval.vector_store import VectorStore
        from src.retrieval.fts5_index import FTS5Index
        from src.documents.catalog import DocumentCatalog
        from src.generation.summarizer import SummarizerService
        from src.schema import Chunk
        
        settings = get_settings()
        file_path = Path(file_path)
        
        try:
            # Step 1: Parse document
            logger.info(f"[1/7] Parsing {file_path.name}...")
            parsed_doc = self.parse_document(file_path, document_type)
            if not parsed_doc:
                logger.error(f"Failed to parse {file_path}")
                return False
            
            # Generate document ID
            document_id = hashlib.sha256(
                f"{parsed_doc.file_path}:{parsed_doc.file_name}".encode()
            ).hexdigest()[:16]
            
            # Step 2: Chunk text
            logger.info(f"[2/7] Chunking text for {file_path.name}...")
            chunks = self._chunk_document(parsed_doc, document_id, settings)
            if not chunks:
                logger.warning(f"No chunks generated for {file_path.name}")
                return False
            
            logger.info(f"Generated {len(chunks)} chunks")
            
            # Step 3: Generate embeddings (GPU-accelerated)
            logger.info(f"[3/7] Generating embeddings (GPU) for {len(chunks)} chunks...")
            from src.retrieval.embedding_service_onnx import ONNXEmbeddingService
            embedding_service = ONNXEmbeddingService(settings=settings)
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = embedding_service.embed_batch(chunk_texts)
            
            if len(embeddings) != len(chunks):
                logger.error(f"Embedding count mismatch: {len(embeddings)} vs {len(chunks)}")
                return False
            
            # Step 4: Store in vector database
            logger.info(f"[4/7] Storing {len(chunks)} chunks in ChromaDB...")
            vector_store = VectorStore(settings=settings)
            vector_store.add_chunks(chunks, embeddings)
            
            # Step 5: Store in keyword index (FTS5)
            logger.info(f"[5/7] Indexing in FTS5...")
            fts5_index = FTS5Index(settings=settings)
            fts5_index.add_chunks(chunks)
            
            # Step 6: Update document catalog
            logger.info(f"[6/7] Updating document catalog...")
            catalog = DocumentCatalog()
            record = catalog.ensure_record(parsed_doc)
            catalog.update_record(
                record,
                indexed=True,
                chunk_count=len(chunks),
                ingested_at=datetime.now().isoformat(),
            )
            
            # Step 7: Generate summaries (optional)
            if generate_summary and settings.summary.enabled:
                logger.info(f"[7/7] Generating summaries...")
                summarizer = SummarizerService(settings=settings)
                
                # Prepare document for summarization
                doc_dict = {
                    "document_id": document_id,
                    "text": parsed_doc.text,
                    "file_name": parsed_doc.file_name,
                    "doc_type": parsed_doc.document_type,
                }
                
                # Generate summaries (uses configured model)
                summaries = summarizer.summarize_documents(
                    documents=[doc_dict],
                    summary_types=settings.summary.summary_types,
                )
                logger.info(f"Generated {len(summaries)} summaries")
            else:
                logger.info(f"[7/7] Skipping summary generation")
            
            logger.success(f"Successfully processed {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}", exc_info=True)
            
            # Update catalog with error
            try:
                catalog = DocumentCatalog()
                if parsed_doc:
                    record = catalog.ensure_record(parsed_doc)
                    catalog.update_record(record, error=str(e), indexed=False)
            except Exception:
                pass
            
            return False
    
    def _chunk_document(
        self,
        parsed_doc: ParsedDocument,
        document_id: str,
        settings,
    ) -> list[Chunk]:
        """Chunk a parsed document using configured chunk sizes.
        
        Args:
            parsed_doc: Parsed document
            document_id: Document identifier
            settings: Settings instance
            
        Returns:
            List of Chunk objects
        """
        import hashlib
        from src.schema import Chunk
        
        # Get chunk size and overlap for document type
        doc_type = parsed_doc.document_type
        chunk_size = getattr(settings.chunking.sizes, doc_type, 512)
        chunk_overlap = getattr(settings.chunking.overlaps, doc_type, 200)
        
        # Simple character-based chunking
        text = parsed_doc.text
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Generate chunk ID
            chunk_id = hashlib.sha256(
                f"{document_id}:chunk:{chunk_index}".encode()
            ).hexdigest()[:16]
            
            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                file_name=parsed_doc.file_name,
                document_type=parsed_doc.document_type,
                text=chunk_text,
                char_start=start,
                char_end=end,
                metadata={
                    "chunk_index": chunk_index,
                    "chunk_size": len(chunk_text),
                },
            )
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start += (chunk_size - chunk_overlap)
            chunk_index += 1
        
        return chunks




def main():
    """CLI entry point for ingestion pipeline.

    Usage:
        python -m src.ingestion.ingestion_pipeline <folder_path>
    """
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.ingestion_pipeline <folder_path>")
        sys.exit(1)

    folder_path = Path(sys.argv[1])
    pipeline = IngestionPipeline()
    documents = pipeline.ingest_folder(folder_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Ingestion Summary")
    print(f"{'='*60}")
    print(f"Total documents parsed: {len(documents)}")

    if documents:
        print(f"\nSample metadata:")
        for i, doc in enumerate(documents[:3], 1):
            print(f"\n{i}. {doc.file_name}")
            print(f"   Type: {doc.document_type}")
            print(f"   Text length: {len(doc.text)} chars")
            print(f"   Paragraphs: {len(doc.paragraphs)}")
            if doc.metadata:
                print(f"   Metadata keys: {list(doc.metadata.keys())}")

    print(f"\nLogs written to: logs/ingestion.log")


if __name__ == "__main__":
    main()




