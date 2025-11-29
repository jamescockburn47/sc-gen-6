"""Centralized warning configuration for cleaner output.

Import this module early to suppress noisy warnings during model loading.
"""

import warnings
import os


def configure_warnings():
    """Configure warning filters for cleaner output."""
    # Suppress RuntimeWarning about module imports
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")

    # Suppress HuggingFace/Transformers warnings
    # Note: Xet Storage warning removed - hf_xet is now installed for faster downloads!
    warnings.filterwarnings("ignore", message=".*were not initialized from the model checkpoint.*")
    warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")

    # Suppress sentence-transformers warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sentence_transformers")

    # Suppress torch UserWarnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    # Keep HuggingFace progress bars (set to '1' to disable them entirely)
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '0')


# Auto-configure on import
configure_warnings()
