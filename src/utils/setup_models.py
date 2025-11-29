"""Setup script to download and verify all required models."""

import sys
from pathlib import Path

try:
    import ollama
except ImportError:
    ollama = None
    print("Warning: ollama module not found. Install with: pip install ollama")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    print("Warning: sentence-transformers not found. Install with: pip install sentence-transformers")

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoTokenizer = None
    print("Warning: transformers not found. Install with: pip install transformers")


def check_ollama_models(settings):
    """Check and pull Ollama models.

    Args:
        settings: Settings instance

    Returns:
        Dictionary of model status
    """
    print("\n" + "=" * 60)
    print("Ollama Models")
    print("=" * 60)

    if ollama is None:
        print("\n✗ Ollama Python client not installed")
        print("  Install with: pip install ollama")
        return {}

    try:
        client = ollama.Client(host=settings.models.ollama.host)
        models_response = client.list()
        available_names = {m.get("name", ""): True for m in models_response.get("models", [])}

        model_status = {}
        default_model = settings.models.llm.default

        print(f"\nDefault model: {default_model}")
        print(f"Available models: {', '.join(settings.models.llm.available)}")
        print()

        for model in settings.models.llm.available:
            is_available = model in available_names
            model_status[model] = is_available

            if is_available:
                print(f"✓ {model} - Available")
            else:
                print(f"✗ {model} - Not found")
                print(f"  Run: ollama pull {model}")

        # Check default model specifically
        if default_model not in available_names:
            print(f"\n⚠ WARNING: Default model '{default_model}' not found!")
            print(f"  Pull it now with: ollama pull {default_model}")

        return model_status

    except Exception as e:
        print(f"✗ Error connecting to Ollama: {str(e)}")
        print("  Ensure Ollama is running: ollama serve")
        return {}


def check_embedding_model(settings):
    """Check and download embedding model.

    Args:
        settings: Settings instance

    Returns:
        True if model is available
    """
    print("\n" + "=" * 60)
    print("Embedding Model")
    print("=" * 60)

    if SentenceTransformer is None:
        print("\n✗ sentence-transformers not installed")
        print("  Install with: pip install sentence-transformers")
        return False

    model_name = settings.models.embedding.default
    print(f"\nModel: {model_name}")

    try:
        print("Loading model (this may take a while on first run)...")
        model = SentenceTransformer(model_name)
        print(f"✓ Model loaded successfully")
        print(f"  Dimension: {model.get_sentence_embedding_dimension()}")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        print("  The model will be downloaded automatically on first use.")
        return False


def check_reranker_model(settings):
    """Check and download reranker model.

    Args:
        settings: Settings instance

    Returns:
        True if model is available
    """
    print("\n" + "=" * 60)
    print("Reranker Model")
    print("=" * 60)

    if AutoTokenizer is None:
        print("\n✗ transformers not installed")
        print("  Install with: pip install transformers")
        return False

    model_name = settings.models.reranker.default
    print(f"\nModel: {model_name}")

    try:
        print("Checking model availability...")
        # Try to load tokenizer to check if model exists
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Model found")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print("  Note: Full model will be downloaded on first use")
        return True
    except Exception as e:
        print(f"✗ Error checking model: {str(e)}")
        print("  The model will be downloaded automatically on first use.")
        return False


def pull_ollama_model(model_name: str, settings):
    """Pull an Ollama model.

    Args:
        model_name: Name of model to pull
        settings: Settings instance

    Returns:
        True if successful
    """
    if ollama is None:
        print("✗ Ollama Python client not installed")
        return False

    try:
        client = ollama.Client(host=settings.models.ollama.host)
        print(f"\nPulling {model_name}...")
        print("This may take a while depending on model size...")

        # Use the pull method
        response = client.pull(model_name, stream=True)
        for chunk in response:
            if "status" in chunk:
                print(f"  {chunk['status']}")

        print(f"✓ Successfully pulled {model_name}")
        return True
    except Exception as e:
        print(f"✗ Error pulling model: {str(e)}")
        return False


def main():
    """Main entry point for model setup."""
    from src.config_loader import get_settings

    print("=" * 60)
    print("SC Gen 6 - Model Setup")
    print("=" * 60)

    settings = get_settings()

    # Check Ollama models
    ollama_status = check_ollama_models(settings)

    # Check embedding model
    embedding_ok = check_embedding_model(settings)

    # Check reranker model
    reranker_ok = check_reranker_model(settings)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    default_model = settings.models.llm.default
    default_available = ollama_status.get(default_model, False)

    if default_available and embedding_ok and reranker_ok:
        print("\n✓ All models are ready!")
        print("  You can now run the application.")
    else:
        print("\n⚠ Some models need attention:")

        if not default_available:
            print(f"  - Default LLM model '{default_model}' not found")
            print(f"    Run: ollama pull {default_model}")

        if not embedding_ok:
            print("  - Embedding model will be downloaded on first use")

        if not reranker_ok:
            print("  - Reranker model will be downloaded on first use")

    # Interactive pull
    if not default_available:
        print("\n" + "=" * 60)
        response = input(f"\nPull default model '{default_model}' now? (y/n): ")
        if response.lower() == 'y':
            pull_ollama_model(default_model, settings)


if __name__ == "__main__":
    main()

