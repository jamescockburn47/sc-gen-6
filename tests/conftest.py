"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest
from src.utils import device_utils

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def force_cpu_for_tests(monkeypatch):
    """Force CPU usage by mocking DirectML detection.
    
    This avoids GPU crashes during testing, especially on unstable DirectML environments.
    """
    def mock_check_directml():
        return False, None
    
    monkeypatch.setattr(device_utils, "_check_directml", mock_check_directml)
    # Also disable CUDA just in case
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture
def project_root() -> Path:
    """Return project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def config_path(project_root: Path) -> Path:
    """Return path to config.yaml."""
    return project_root / "config" / "config.yaml"


@pytest.fixture
def test_data_dir(project_root: Path) -> Path:
    """Return path to test data directory."""
    test_dir = project_root / "tests" / "fixtures"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir
