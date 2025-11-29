"""CLI utility for document ingestion with progress and chunking."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.config_loader import get_settings
from src.ingestion.chunkers.adaptive_chunker import AdaptiveChunker
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.retrieval.bm25_index import BM25Index
from src.retrieval.embedding_service import EmbeddingService
from src.retrieval.vector_store import VectorStore
from src.schema import DocumentType


def main():
    """Main entry point for ingest CLI."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into SC Gen 6 RAG system"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to file or folder to ingest",
    )
    parser.add_argument(
        "--doc-type",
        type=str,
        choices=[
            "witness_statement",
            "pleading",
            "statute",
            "contract",
            "disclosure",
            "email",
            "court_filing",
            "scanned_pdf",
        ],
        help="Override document type detection",
    )
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Chunk documents and add to indexes",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild indexes after ingestion",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output",
    )

    args = parser.parse_args()

    # Parse path
    input_path = Path(args.path)
    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Initialize pipeline
    print("Initializing ingestion pipeline...")
    pipeline = IngestionPipeline()

    # Ingest documents
    print(f"\nIngesting: {input_path}")
    if input_path.is_file():
        print("Processing single file...")
        documents = pipeline.ingest_files(
            [input_path], document_type=args.doc_type  # type: ignore
        )
    else:
        print("Processing folder...")
        documents = pipeline.ingest_folder(
            input_path, document_type=args.doc_type  # type: ignore
        )

    if not documents:
        print("\nNo documents were successfully parsed.")
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 60)
    print("Ingestion Summary")
    print("=" * 60)
    print(f"Total documents parsed: {len(documents)}")

    if args.verbose:
        print("\nDocuments:")
        for i, doc in enumerate(documents, 1):
            print(f"\n{i}. {doc.file_name}")
            print(f"   Type: {doc.document_type}")
            print(f"   Text length: {len(doc.text)} chars")
            print(f"   Paragraphs: {len(doc.paragraphs)}")
            if doc.metadata:
                print(f"   Metadata: {list(doc.metadata.keys())}")
    else:
        print("\nSample documents:")
        for i, doc in enumerate(documents[:3], 1):
            print(f"  {i}. {doc.file_name} ({doc.document_type})")

    # Chunk and index if requested
    if args.chunk:
        print("\n" + "=" * 60)
        print("Chunking and Indexing")
        print("=" * 60)

        settings = get_settings()
        chunker = AdaptiveChunker(settings=settings)
        embedding_service = EmbeddingService(settings=settings)
        vector_store = VectorStore(settings=settings)
        bm25_index = BM25Index(settings=settings)

        all_chunks = []
        for i, doc in enumerate(documents, 1):
            if args.verbose:
                print(f"\nChunking document {i}/{len(documents)}: {doc.file_name}")

            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)

            if args.verbose:
                print(f"  Created {len(chunks)} chunks")

        print(f"\nTotal chunks created: {len(all_chunks)}")

        # Generate embeddings
        print("\nGenerating embeddings...")
        chunk_texts = [chunk.text for chunk in all_chunks]
        embeddings = embedding_service.embed_batch(chunk_texts)
        print(f"✓ Generated {len(embeddings)} embeddings")

        # Add to vector store
        print("Adding to vector store...")
        vector_store.add_chunks(all_chunks, embeddings)
        print("✓ Added to vector store")

        # Build BM25 index
        print("Building BM25 index...")
        bm25_index.build_index(all_chunks)
        bm25_index.save()
        print("✓ BM25 index built and saved")

        # Show stats
        print("\n" + "=" * 60)
        print("Index Statistics")
        print("=" * 60)
        vs_stats = vector_store.stats()
        bm25_stats = bm25_index.stats()

        print(f"Vector Store:")
        print(f"  Documents: {vs_stats.get('num_documents', 0)}")
        print(f"  Chunks: {vs_stats.get('num_chunks', 0)}")

        print(f"\nBM25 Index:")
        print(f"  Documents: {bm25_stats.get('num_documents', 0)}")
        print(f"  Chunks: {bm25_stats.get('num_chunks', 0)}")

    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()




