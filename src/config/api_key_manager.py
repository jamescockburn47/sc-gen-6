import json
import os
from typing import Dict, Optional
from pathlib import Path

class APIKeyManager:
    """
    Manages secure storage and retrieval of API keys for cloud providers.
    Keys are stored in a local JSON file that is excluded from version control.
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.keys_file = self.config_dir / "api_keys.json"
        self._ensure_config_dir()
        self._keys: Dict[str, str] = self._load_keys()

    def _ensure_config_dir(self):
        """Ensure the configuration directory exists."""
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)

    def _load_keys(self) -> Dict[str, str]:
        """Load keys from the JSON file."""
        if not self.keys_file.exists():
            return {}
        
        try:
            with open(self.keys_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
        except Exception as e:
            print(f"Error loading API keys: {e}")
            return {}

    def _save_keys(self):
        """Save current keys to the JSON file."""
        try:
            with open(self.keys_file, 'w') as f:
                json.dump(self._keys, f, indent=2)
        except Exception as e:
            print(f"Error saving API keys: {e}")

    def get_key(self, provider: str) -> Optional[str]:
        """
        Get an API key for a specific provider.
        
        Args:
            provider: The provider name (e.g., 'openai', 'anthropic', 'google')
            
        Returns:
            The API key if found, None otherwise.
        """
        return self._keys.get(provider.lower())

    def set_key(self, provider: str, key: str):
        """
        Set an API key for a specific provider.
        
        Args:
            provider: The provider name
            key: The API key string
        """
        self._keys[provider.lower()] = key
        self._save_keys()

    def has_key(self, provider: str) -> bool:
        """Check if a key exists for the provider."""
        return bool(self.get_key(provider))

    def get_all_providers(self) -> list[str]:
        """Get list of providers with configured keys."""
        return [k for k, v in self._keys.items() if v]
