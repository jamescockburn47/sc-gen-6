"""First-run wizard for checking Ollama, models, and initializing data folders."""

import sys
from pathlib import Path
from typing import Optional

try:
    import ollama
except ImportError:
    ollama = None

try:
    import requests
except ImportError:
    requests = None

from src.config.llm_config import LLMConfig, load_llm_config
from src.config_loader import Settings, get_settings


class FirstRunWizard:
    """Wizard for first-run setup and checks."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize wizard.

        Args:
            settings: Settings instance. If None, loads from config.
        """
        self.settings = settings or get_settings()
        self.llm_config: LLMConfig = load_llm_config()
        self.issues: list[str] = []
        self.warnings: list[str] = []

    def check_lmstudio(self) -> bool:
        """Check if LM Studio is running and accessible.

        Returns:
            True if LM Studio is accessible
        """
        return self._check_openai_endpoint("LM Studio")

    def check_llama_cpp(self) -> bool:
        """Check if llama.cpp llama-server is running."""
        return self._check_openai_endpoint("llama.cpp")

    def _check_openai_endpoint(self, label: str) -> bool:
        """Generic checker for OpenAI-compatible providers."""
        if requests is None:
            self.issues.append("requests package not installed. Install with: pip install requests")
            return False

        try:
            response = requests.get(self._provider_models_url(), timeout=2)
            if response.status_code == 200:
                return True
            self.warnings.append(f"{label} returned status {response.status_code}")
            return False
        except Exception as exc:
            self.warnings.append(f"{label} not accessible: {exc}")
            return False

    def _provider_models_url(self) -> str:
        base = self.llm_config.base_url.rstrip("/")
        return f"{base}/models"

    def check_ollama(self) -> bool:
        """Check if Ollama is running and accessible.

        Returns:
            True if Ollama is accessible
        """
        if ollama is None:
            self.issues.append("Ollama Python client not installed. Install with: pip install ollama")
            return False

        try:
            client = ollama.Client(host=self.settings.models.ollama.host)
            client.list()
            return True
        except Exception as e:
            self.issues.append(f"Ollama not accessible: {str(e)}")
            return False

    def check_models(self) -> dict[str, bool]:
        """Check which models are available in Ollama.

        Returns:
            Dictionary mapping model name to availability status
        """
        if ollama is None:
            self.issues.append("Ollama Python client not installed")
            return {model: False for model in self.settings.models.llm.available}

        try:
            client = ollama.Client(host=self.settings.models.ollama.host)
            models_response = client.list()
            available_names = {
                m.get("name", ""): True for m in models_response.get("models", [])
            }

            model_status = {}
            for model in self.settings.models.llm.available:
                model_status[model] = model in available_names
                if not model_status[model]:
                    self.warnings.append(f"Model not found: {model}")

            return model_status
        except Exception as e:
            self.issues.append(f"Cannot check models: {str(e)}")
            return {model: False for model in self.settings.models.llm.available}

    def check_default_model(self) -> bool:
        """Check if default model is available.

        Returns:
            True if default model is available
        """
        default_model = self.settings.models.llm.default
        model_status = self.check_models()
        return model_status.get(default_model, False)

    def create_data_folders(self) -> bool:
        """Create required data folders if they don't exist.

        Returns:
            True if all folders created successfully
        """
        paths = self.settings.paths
        folders_to_create = [
            paths.documents,
            paths.vector_db,
            paths.keyword_index,
            paths.logs,
        ]

        success = True
        for folder_path in folders_to_create:
            path = Path(folder_path)
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.issues.append(f"Cannot create folder {folder_path}: {str(e)}")
                success = False

        return success

    def check_config_file(self) -> bool:
        """Check if config file exists.

        Returns:
            True if config file exists
        """
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            self.issues.append(f"Config file not found: {config_path}")
            return False
        return True

    def run_all_checks(self) -> bool:
        """Run all first-run checks.

        Returns:
            True if all critical checks pass
        """
        print("Running first-run checks...")
        print()

        # Check config file
        print("Checking config file...")
        if not self.check_config_file():
            print("  ERROR: Config file missing")
        else:
            print("  OK: Config file found")

        # Create data folders
        print("\nCreating data folders...")
        if self.create_data_folders():
            print("  OK: Data folders created/verified")
        else:
            print("  ERROR: Failed to create some folders")

        # Check LLM backend (LM Studio or Ollama)
        backend = self.llm_config.provider
        print(f"\nChecking LLM provider ({backend})...")

        if backend == "lmstudio":
            if self.check_lmstudio():
                print(f"  OK: LM Studio is running at {self.llm_config.base_url}")
            else:
                print("  WARNING: LM Studio not accessible")
                print("  Please ensure:")
                print("    - LM Studio is running")
                print("    - Local Server is started (green indicator)")
                print("    - Model is loaded in LM Studio")
        elif backend == "llama_cpp":
            if self.check_llama_cpp():
                print(f"  OK: llama-server is running at {self.llm_config.base_url}")
            else:
                print("  WARNING: llama.cpp server not accessible")
                print("  Please ensure:")
                print("    - You've built llama-server with Vulkan support")
                print("    - LLAMA_MODEL_PATH is set")
                print("    - scripts/start_llama_server.bat is running")
        else:
            # Legacy path for Ollama or other backends defined in YAML
            yaml_backend = self.settings.models.llm.backend
            if yaml_backend == "ollama":
                if self.check_ollama():
                    print("  OK: Ollama is running")
                else:
                    print("  ERROR: Ollama not accessible")
                    print("  Please ensure Ollama is installed and running:")
                    print("    - Download from https://ollama.ai")
                    print("    - Run: ollama serve")

                print("\nChecking models...")
                model_status = self.check_models()
                for model, available in model_status.items():
                    status = "OK" if available else "MISSING"
                    print(f"  {status}: {model}")

                print("\nChecking default model...")
                if self.check_default_model():
                    print(f"  OK: Default model '{self.settings.models.llm.default}' is available")
                else:
                    print(f"  WARNING: Default model '{self.settings.models.llm.default}' not found")
                    print(f"  Run: ollama pull {self.settings.models.llm.default}")

        # Summary
        print("\n" + "=" * 60)
        if self.issues:
            print("CRITICAL ISSUES FOUND:")
            for issue in self.issues:
                print(f"  - {issue}")
            print()
            return False
        else:
            print("All critical checks passed!")
            if self.warnings:
                print("\nWarnings:")
                for warning in self.warnings:
                    print(f"  - {warning}")
            return True

    def print_setup_instructions(self):
        """Print setup instructions."""
        print("\n" + "=" * 60)
        print("SETUP INSTRUCTIONS")
        print("=" * 60)
        print("\n1. Install Ollama:")
        print("   - Download from https://ollama.ai")
        print("   - Run: ollama serve")
        print("\n2. Pull required models:")
        default_model = self.settings.models.llm.default
        print(f"   - ollama pull {default_model}")
        print("\n3. Verify installation:")
        print("   - python -m src.utils.first_run")
        print("\n4. Start the application:")
        print("   - python -m src.ui.main")


def main():
    """Main entry point for first-run wizard."""
    wizard = FirstRunWizard()

    if wizard.run_all_checks():
        print("\nSetup complete! You can now run the application.")
        sys.exit(0)
    else:
        wizard.print_setup_instructions()
        sys.exit(1)


if __name__ == "__main__":
    main()

