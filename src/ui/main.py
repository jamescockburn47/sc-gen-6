"""Main PySide6 application entry point."""

import sys
from pathlib import Path

# Configure warnings early for cleaner output
from src.utils.warnings_config import configure_warnings
configure_warnings()

from PySide6.QtWidgets import QApplication, QMessageBox

from src.ui.modern_main_window import ModernMainWindow as MainWindow
from src.utils.first_run import FirstRunWizard


def main():
    """Main entry point for desktop application."""
    # Check for first-run flag or run wizard if needed
    first_run_flag = Path(".first_run_complete")
    if not first_run_flag.exists():
        # Run first-run checks
        wizard = FirstRunWizard()
        if not wizard.run_all_checks():
            # Show error dialog
            app = QApplication(sys.argv)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Setup Required")
            msg.setText("First-run checks failed.")
            msg.setDetailedText("\n".join(wizard.issues + wizard.warnings))
            msg.exec()
            sys.exit(1)
        else:
            # Mark first run as complete
            first_run_flag.touch()

    app = QApplication(sys.argv)
    app.setApplicationName("SC Gen 6 - Litigation Support RAG")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

