"""Cross-platform launcher for SC Gen 6 desktop application."""

import sys
import subprocess
from pathlib import Path


def should_check_ollama() -> bool:
    """Return True if the active provider uses Ollama."""
    try:
        from src.config.llm_config import load_llm_config

        cfg = load_llm_config()
        return cfg.provider == "ollama"
    except Exception:
        return False


def check_python():
    """Check if Python is available."""
    try:
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 11):
            print("[ERROR] Python 3.11+ required")
            print(f"Current version: {version.major}.{version.minor}.{version.micro}")
            return False
        print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
        return True
    except Exception:
        print("[ERROR] Python not found")
        return False


def check_ollama():
    """Check if Ollama is running."""
    try:
        import requests
        response = requests.get("http://localhost:11434", timeout=2)
        print("[OK] Ollama is running")
        return True
    except ImportError:
        print("[WARNING] requests module not found (optional check)")
        return True  # Don't fail if requests not installed
    except Exception:
        print("[WARNING] Ollama may not be running")
        print("  Start Ollama with: ollama serve")
        return True  # Don't fail, just warn


def check_dependencies():
    """Check if required dependencies are installed."""
    required_modules = [
        ("PySide6", "PySide6"),
        ("pydantic", "pydantic"),
        ("yaml", "yaml"),
        ("chromadb", "chromadb"),
        ("sentence_transformers", "sentence_transformers"),
    ]

    missing = []
    for display_name, import_name in required_modules:
        try:
            __import__(import_name.replace("-", "_"))
        except ImportError:
            missing.append(display_name)

    if missing:
        print(f"[WARNING] Missing dependencies: {', '.join(missing)}")
        print("  Install with: pip install -r requirements.txt")
        return False

    print("[OK] Dependencies installed")
    return True


def main():
    """Main launcher entry point."""
    print("=" * 60)
    print("SC Gen 6 - Litigation Support RAG")
    print("=" * 60)
    print()

    # Check Python
    if not check_python():
        sys.exit(1)

    # Check dependencies
    deps_ok = check_dependencies()
    if not deps_ok:
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Check Ollama only if provider requires it
    if should_check_ollama():
        check_ollama()
    else:
        print("[INFO] Ollama check skipped (provider is not Ollama)")
    print()

    # Launch application
    print("Launching SC Gen 6...")
    print()

    try:
        # Import and run the main UI
        from src.ui.main import main as ui_main

        ui_main()
    except KeyboardInterrupt:
        print("\n\nApplication closed by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Failed to launch application: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

