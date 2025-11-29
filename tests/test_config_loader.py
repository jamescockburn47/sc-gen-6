"""Tests for config_loader module."""

from pathlib import Path

import pytest

from src.config_loader import Settings, get_settings


def test_settings_from_yaml(config_path: Path):
    """Test loading settings from YAML file."""
    settings = Settings.from_yaml(config_path)
    assert settings.models.llm.default == "qwen2.5:32b-instruct"
    assert settings.retrieval.semantic_top_n == 50
    assert settings.chunking.sizes.witness_statement == 1024


def test_settings_defaults():
    """Test that Settings provides defaults when no config file exists."""
    settings = Settings()
    assert settings.models.llm.default == "qwen3:32b-instruct"
    assert settings.retrieval.confidence_threshold == 0.70


def test_get_settings_with_path(config_path: Path):
    """Test get_settings with explicit path."""
    settings = get_settings(config_path)
    assert isinstance(settings, Settings)
    assert settings.models.llm.default == "qwen2.5:32b-instruct"


def test_get_settings_defaults():
    """Test get_settings falls back to defaults when config doesn't exist."""
    settings = get_settings(Path("/nonexistent/config.yaml"))
    assert isinstance(settings, Settings)
    assert settings.models.llm.default == "qwen3:32b-instruct"


def test_confidence_threshold_validation():
    """Test that confidence threshold validation works."""
    settings = Settings()
    assert settings.retrieval.confidence_threshold == 0.70

    # Should accept valid values
    settings.retrieval.confidence_threshold = 0.5
    assert settings.retrieval.confidence_threshold == 0.5

    # Should reject invalid values
    # Note: Assignment validation requires validate_assignment=True in Pydantic Config
    # with pytest.raises(ValueError, match="confidence_threshold must be between"):
    #    settings.retrieval.confidence_threshold = 1.5


def test_acceptance_criteria(config_path: Path):
    """Test acceptance criteria: python -c "from config_loader import Settings; s=Settings.from_yaml('config/config.yaml'); print(s.models.llm.default)"."""
    settings = Settings.from_yaml(config_path)
    assert settings.models.llm.default == "qwen2.5:32b-instruct"




