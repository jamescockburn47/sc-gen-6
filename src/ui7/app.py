"""SCGen7 application entry point."""

from __future__ import annotations

import sys
import os
from pathlib import Path


def main() -> None:
    """Launch the SCGen7 desktop application."""
    # Ensure working directory is project root
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)

    # Load .env if present
    env_path = project_root / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont
    from PySide6.QtCore import Qt

    # High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("SCGen7")
    app.setApplicationVersion("7.0.0")
    app.setOrganizationName("LegalQuant")

    # Default font
    font = QFont("Inter", 10)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)

    # Import and show main window
    from src.ui7.main_window import SCGen7Window

    window = SCGen7Window()
    window.show()

    # Initial status check
    window.status_bar.refresh_status()

    # Check Kanon 2 availability
    try:
        from src.graph.enricher import KanonEnricher
        enricher = KanonEnricher()
        window.status_bar.set_enricher_status(enricher.is_available)
    except Exception:
        window.status_bar.set_enricher_status(False)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
