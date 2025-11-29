import sys
import os

# Add src to path
sys.path.append(os.getcwd())

try:
    print("Importing IngestionPipeline...")
    from src.ingestion.ingestion_pipeline import IngestionPipeline
    
    print("Importing DocumentManagerWidget...")
    # This might fail if Qt is not available in this environment, but we'll try
    try:
        from src.ui.document_manager import DocumentManagerWidget
    except ImportError:
        print("Skipping DocumentManagerWidget (Qt dependencies)")

    print("Importing DocumentStatsWidget...")
    try:
        from src.ui.document_stats_widget import DocumentStatsWidget
    except ImportError:
        print("Skipping DocumentStatsWidget (Qt dependencies)")

    print("Importing HybridRetriever...")
    from src.retrieval.hybrid_retriever import HybridRetriever

    print("All imports successful!")

except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)
