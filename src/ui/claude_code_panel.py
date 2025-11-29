"""Claude Code Integration Panel for live debugging and development."""

import subprocess
import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal, QObject, QThread, QFileSystemWatcher, QTimer
from PySide6.QtGui import QTextCursor, QFont
from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QLineEdit,
    QWidget,
    QSplitter,
    QCheckBox,
    QProgressBar,
    QGroupBox,
    QComboBox,
    QFileDialog,
)

from src.utils.test_runner import TestRunner, TestResult


class TestRunnerWorker(QObject):
    """Worker thread for running tests."""

    output_received = Signal(str)
    finished = Signal(TestResult)  # Emits test result when done

    def __init__(self, test_runner: TestRunner, test_path: str, verbose: bool,
                 coverage: bool, markers: Optional[str]):
        """Initialize test runner worker."""
        super().__init__()
        self.test_runner = test_runner
        self.test_path = test_path
        self.verbose = verbose
        self.coverage = coverage
        self.markers = markers

    def run(self):
        """Run tests and emit results."""
        try:
            self.output_received.emit("Starting test run...\n")
            result = self.test_runner.run_tests(
                test_path=self.test_path,
                verbose=self.verbose,
                coverage=self.coverage,
                markers=self.markers,
            )
            self.finished.emit(result)
        except Exception as e:
            # Create error result
            error_result = TestResult(errors=1, output=f"Test execution failed: {str(e)}")
            self.finished.emit(error_result)


class ClaudeCodeWorker(QObject):
    """Worker thread for running Claude Code subprocess."""

    output_received = Signal(str)  # Emitted when output is received
    error_received = Signal(str)  # Emitted when error is received
    finished = Signal()  # Emitted when process finishes

    def __init__(self, message: str, working_dir: str):
        """Initialize worker.

        Args:
            message: Message to send to Claude Code
            working_dir: Working directory for Claude Code
        """
        super().__init__()
        self.message = message
        self.working_dir = working_dir
        self.process: Optional[subprocess.Popen] = None
        self._running = True

    def run(self):
        """Run Claude Code and stream output."""
        try:
            # Start Claude Code process
            self.process = subprocess.Popen(
                ["claude", "code", "--message", self.message],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.working_dir,
                bufsize=1,
            )

            # Stream stdout
            if self.process.stdout:
                for line in iter(self.process.stdout.readline, ''):
                    if not self._running:
                        break
                    if line:
                        self.output_received.emit(line.rstrip())

            # Wait for process to finish
            if self._running:
                self.process.wait()

                # Capture any stderr
                if self.process.stderr:
                    stderr = self.process.stderr.read()
                    if stderr:
                        self.error_received.emit(stderr)

        except FileNotFoundError:
            self.error_received.emit(
                "ERROR: Claude Code CLI not found. Install from: https://claude.com/claude-code"
            )
        except Exception as e:
            self.error_received.emit(f"ERROR: {str(e)}")
        finally:
            self.finished.emit()

    def stop(self):
        """Stop the running process."""
        self._running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


class ClaudeCodePanel(QWidget):
    """Panel for integrating Claude Code into the UI for live debugging."""

    def __init__(self, parent=None):
        """Initialize Claude Code panel."""
        super().__init__(parent)
        self.working_dir = str(Path.cwd())
        self.worker: Optional[ClaudeCodeWorker] = None
        self.worker_thread: Optional[QThread] = None
        self.is_running = False

        # Test runner components
        self.test_runner = TestRunner(self.working_dir)
        self.test_worker: Optional[TestRunnerWorker] = None
        self.test_thread: Optional[QThread] = None
        self.is_testing = False
        self.last_test_result: Optional[TestResult] = None

        # File watcher for auto-testing
        self.file_watcher: Optional[QFileSystemWatcher] = None
        self.auto_test_enabled = False
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self._run_auto_test)

        self._setup_ui()
        self._setup_file_watcher()

    def _setup_ui(self):
        """Set up UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title_layout = QHBoxLayout()
        title = QLabel("Claude Code - Live Debugging")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        title_layout.addWidget(title)

        # Status indicator
        self.status_label = QLabel("● Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        title_layout.addWidget(self.status_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)

        # Info label
        info = QLabel(
            "Chat with Claude Code for live debugging, code generation, and fixes. "
            "Claude Code has full access to this project."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: gray; font-size: 10pt; margin-bottom: 10px;")
        layout.addWidget(info)

        # Splitter for output and input
        splitter = QSplitter(Qt.Vertical)

        # Output area (conversation history)
        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)

        output_label = QLabel("Conversation:")
        output_layout.addWidget(output_label)

        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10pt;
                border: 1px solid #333;
                padding: 5px;
            }
        """)
        output_layout.addWidget(self.output_area)
        splitter.addWidget(output_widget)

        # Input area
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_layout.setContentsMargins(0, 0, 0, 0)

        # Input box
        input_label = QLabel("Your message:")
        input_layout.addWidget(input_label)

        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText(
            "Ask Claude Code to debug, fix bugs, add features, or explain code...\n"
            "Examples:\n"
            "- 'Fix the AttributeError in query_panel.py'\n"
            "- 'Add error handling to the embedding service'\n"
            "- 'Explain how the hybrid retrieval works'"
        )
        self.input_box.setMaximumHeight(120)
        self.input_box.setStyleSheet("font-size: 10pt;")
        input_layout.addWidget(self.input_box)

        # Button row
        button_layout = QHBoxLayout()

        # Auto-apply checkbox
        self.auto_apply_check = QCheckBox("Auto-apply changes")
        self.auto_apply_check.setChecked(True)
        self.auto_apply_check.setToolTip(
            "Automatically apply code changes suggested by Claude Code"
        )
        button_layout.addWidget(self.auto_apply_check)

        button_layout.addStretch()

        # Clear button
        self.clear_btn = QPushButton("Clear History")
        self.clear_btn.clicked.connect(self._clear_output)
        button_layout.addWidget(self.clear_btn)

        # Stop button
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_worker)
        self.stop_btn.setStyleSheet("background-color: #d32f2f; color: white;")
        button_layout.addWidget(self.stop_btn)

        # Send button
        self.send_btn = QPushButton("Send to Claude Code")
        self.send_btn.setDefault(True)
        self.send_btn.clicked.connect(self._send_message)
        self.send_btn.setStyleSheet(
            "background-color: #2196f3; color: white; font-weight: bold; padding: 8px;"
        )
        button_layout.addWidget(self.send_btn)

        input_layout.addLayout(button_layout)
        splitter.addWidget(input_widget)

        # Set splitter sizes (70% output, 30% input)
        splitter.setSizes([700, 300])
        layout.addWidget(splitter)

        # Test Controls Section
        test_group = QGroupBox("🧪 Automated Testing")
        test_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2196f3;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        test_layout = QVBoxLayout()

        # Test controls row
        test_controls_layout = QHBoxLayout()

        # Test path input
        test_path_label = QLabel("Test Path:")
        test_controls_layout.addWidget(test_path_label)

        self.test_path_input = QLineEdit("tests")
        self.test_path_input.setPlaceholderText("tests")
        self.test_path_input.setMaximumWidth(150)
        test_controls_layout.addWidget(self.test_path_input)

        # Markers input
        markers_label = QLabel("Markers:")
        test_controls_layout.addWidget(markers_label)

        self.markers_input = QLineEdit()
        self.markers_input.setPlaceholderText("e.g., 'not slow'")
        self.markers_input.setMaximumWidth(150)
        self.markers_input.setToolTip("Pytest markers to filter tests")
        test_controls_layout.addWidget(self.markers_input)

        # Verbose checkbox
        self.verbose_check = QCheckBox("Verbose")
        self.verbose_check.setChecked(True)
        test_controls_layout.addWidget(self.verbose_check)

        # Coverage checkbox
        self.coverage_check = QCheckBox("Coverage")
        self.coverage_check.setChecked(True)
        test_controls_layout.addWidget(self.coverage_check)

        # Auto-test checkbox
        self.auto_test_check = QCheckBox("Auto-test on save")
        self.auto_test_check.setToolTip("Automatically run tests when Python files are saved")
        self.auto_test_check.toggled.connect(self._toggle_auto_test)
        test_controls_layout.addWidget(self.auto_test_check)

        test_controls_layout.addStretch()

        # Test buttons
        self.run_tests_btn = QPushButton("▶ Run Tests")
        self.run_tests_btn.clicked.connect(self._run_tests)
        self.run_tests_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #666;
            }
        """)
        test_controls_layout.addWidget(self.run_tests_btn)

        self.export_btn = QPushButton("📄 Export Report")
        self.export_btn.clicked.connect(self._export_report)
        self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196f3;
                color: white;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #666;
            }
        """)
        test_controls_layout.addWidget(self.export_btn)

        test_layout.addLayout(test_controls_layout)

        # Test results display
        results_label = QLabel("Test Results:")
        test_layout.addWidget(results_label)

        self.test_output = QTextEdit()
        self.test_output.setReadOnly(True)
        self.test_output.setMaximumHeight(200)
        self.test_output.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
                border: 1px solid #ccc;
                padding: 5px;
            }
        """)
        self.test_output.setPlaceholderText("Test results will appear here...")
        test_layout.addWidget(self.test_output)

        # Progress bars row
        progress_layout = QHBoxLayout()

        # Success rate progress
        success_layout = QVBoxLayout()
        success_label = QLabel("Success Rate:")
        success_layout.addWidget(success_label)

        self.success_progress = QProgressBar()
        self.success_progress.setMaximum(100)
        self.success_progress.setValue(0)
        self.success_progress.setFormat("%p%")
        self.success_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 3px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
            }
        """)
        success_layout.addWidget(self.success_progress)
        progress_layout.addLayout(success_layout)

        # Coverage progress
        coverage_layout = QVBoxLayout()
        coverage_label = QLabel("Code Coverage:")
        coverage_layout.addWidget(coverage_label)

        self.coverage_progress = QProgressBar()
        self.coverage_progress.setMaximum(100)
        self.coverage_progress.setValue(0)
        self.coverage_progress.setFormat("%p%")
        self.coverage_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 3px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #2196f3;
            }
        """)
        coverage_layout.addWidget(self.coverage_progress)
        progress_layout.addLayout(coverage_layout)

        test_layout.addLayout(progress_layout)

        # Test stats label
        self.test_stats_label = QLabel("No tests run yet")
        self.test_stats_label.setStyleSheet("color: gray; font-size: 9pt; margin-top: 5px;")
        test_layout.addWidget(self.test_stats_label)

        test_group.setLayout(test_layout)
        layout.addWidget(test_group)

        # Keyboard shortcut (Ctrl+Enter to send)
        self.input_box.installEventFilter(self)

    def eventFilter(self, obj, event):
        """Handle keyboard shortcuts."""
        if obj == self.input_box and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key_Return and event.modifiers() == Qt.ControlModifier:
                self._send_message()
                return True
        return super().eventFilter(obj, event)

    def _send_message(self):
        """Send message to Claude Code."""
        message = self.input_box.toPlainText().strip()
        if not message:
            return

        if self.is_running:
            self._append_output("[ERROR] Claude Code is already running. Please wait or stop it first.\n")
            return

        # Display user message
        self._append_output(f"\n{'='*80}\n", color="#888")
        self._append_output(f"YOU: {message}\n\n", color="#4fc3f7", bold=True)

        # Clear input
        self.input_box.clear()

        # Update UI state
        self.is_running = True
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("● Running")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")

        # Start worker thread
        self.worker = ClaudeCodeWorker(message, self.working_dir)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.output_received.connect(self._handle_output)
        self.worker.error_received.connect(self._handle_error)
        self.worker.finished.connect(self._worker_finished)

        # Start thread
        self.worker_thread.start()

    def _handle_output(self, text: str):
        """Handle output from Claude Code."""
        self._append_output(f"{text}\n", color="#d4d4d4")

    def _handle_error(self, text: str):
        """Handle error from Claude Code."""
        self._append_output(f"\n{text}\n", color="#f44336", bold=True)

    def _worker_finished(self):
        """Handle worker thread finishing."""
        # Clean up thread
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
        self.worker = None

        # Update UI state
        self.is_running = False
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("● Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")

        self._append_output(f"\n{'='*80}\n", color="#888")
        self._append_output("[DONE]\n\n", color="#4caf50", bold=True)

    def _stop_worker(self):
        """Stop the running worker."""
        if self.worker:
            self._append_output("\n[Stopping Claude Code...]\n", color="#ff9800")
            self.worker.stop()

    def _clear_output(self):
        """Clear the output area."""
        self.output_area.clear()
        self._append_output("Claude Code - Ready\n\n", color="#2196f3", bold=True)

    def _append_output(self, text: str, color: str = "#d4d4d4", bold: bool = False):
        """Append text to output area with color.

        Args:
            text: Text to append
            color: Text color in hex format
            bold: Whether to make text bold
        """
        cursor = self.output_area.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Set format
        fmt = cursor.charFormat()
        fmt.setForeground(Qt.GlobalColor(Qt.white) if color == "#d4d4d4" else Qt.GlobalColor.fromString(color))
        if bold:
            font = QFont()
            font.setBold(True)
            fmt.setFont(font)

        cursor.setCharFormat(fmt)
        cursor.insertText(text)

        # Auto-scroll to bottom
        self.output_area.setTextCursor(cursor)
        self.output_area.ensureCursorVisible()

    # Test runner methods

    def _run_tests(self):
        """Run tests with current configuration."""
        if self.is_testing:
            self.test_output.append("[ERROR] Tests are already running.\n")
            return

        # Get configuration
        test_path = self.test_path_input.text().strip() or "tests"
        markers = self.markers_input.text().strip() or None
        verbose = self.verbose_check.isChecked()
        coverage = self.coverage_check.isChecked()

        # Clear previous results
        self.test_output.clear()
        self.test_output.append(f"Running tests from: {test_path}\n")
        if markers:
            self.test_output.append(f"Markers: {markers}\n")
        self.test_output.append(f"Verbose: {verbose}, Coverage: {coverage}\n")
        self.test_output.append("-" * 80 + "\n\n")

        # Update UI state
        self.is_testing = True
        self.run_tests_btn.setEnabled(False)
        self.run_tests_btn.setText("⏳ Running...")
        self.export_btn.setEnabled(False)

        # Start test worker thread
        self.test_worker = TestRunnerWorker(
            self.test_runner, test_path, verbose, coverage, markers
        )
        self.test_thread = QThread()
        self.test_worker.moveToThread(self.test_thread)

        # Connect signals
        self.test_thread.started.connect(self.test_worker.run)
        self.test_worker.output_received.connect(self._handle_test_output)
        self.test_worker.finished.connect(self._handle_test_finished)

        # Start thread
        self.test_thread.start()

    def _handle_test_output(self, text: str):
        """Handle output from test worker."""
        self.test_output.append(text)

    def _handle_test_finished(self, result: TestResult):
        """Handle test completion and display results."""
        # Clean up thread
        if self.test_thread:
            self.test_thread.quit()
            self.test_thread.wait()
            self.test_thread = None
        self.test_worker = None

        # Store result
        self.last_test_result = result
        self.test_runner.last_result = result

        # Update UI state
        self.is_testing = False
        self.run_tests_btn.setEnabled(True)
        self.run_tests_btn.setText("▶ Run Tests")
        self.export_btn.setEnabled(True)

        # Display full output
        self.test_output.append("\n" + "=" * 80 + "\n")
        self.test_output.append(result.output)
        self.test_output.append("\n" + "=" * 80 + "\n")

        # Update progress bars
        self.success_progress.setValue(int(result.success_rate))
        self.coverage_progress.setValue(int(result.coverage))

        # Update stats label
        status = "✅ PASSED" if result.is_success else "❌ FAILED"
        stats_text = (
            f"{status} | "
            f"Total: {result.total} | "
            f"Passed: {result.passed} | "
            f"Failed: {result.failed} | "
            f"Skipped: {result.skipped} | "
            f"Errors: {result.errors} | "
            f"Duration: {result.duration:.2f}s"
        )
        self.test_stats_label.setText(stats_text)

        # Color code the stats label
        if result.is_success:
            self.test_stats_label.setStyleSheet("color: green; font-weight: bold; font-size: 9pt;")
        else:
            self.test_stats_label.setStyleSheet("color: red; font-weight: bold; font-size: 9pt;")

        # Show failure details if any
        if result.failures:
            self.test_output.append("\nFailed Tests:\n")
            for failure in result.failures:
                self.test_output.append(f"  - {failure['test']}: {failure['error']}\n")

    def _export_report(self):
        """Export test report to file."""
        if not self.last_test_result:
            self.test_output.append("[ERROR] No test results to export.\n")
            return

        # Ask user for format
        from PySide6.QtWidgets import QMessageBox, QInputDialog

        format_choice, ok = QInputDialog.getItem(
            self,
            "Export Format",
            "Select export format:",
            ["HTML", "JSON"],
            0,
            False
        )

        if not ok:
            return

        format_str = format_choice.lower()

        # Ask for file location
        default_name = f"test_report_{self.last_test_result.timestamp.strftime('%Y%m%d_%H%M%S')}.{format_str}"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Test Report",
            default_name,
            f"{format_choice} Files (*.{format_str})"
        )

        if not file_path:
            return

        # Export report
        try:
            export_path = self.test_runner.export_report(format_str, file_path)
            if export_path:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("Export Successful")
                msg.setText(f"Test report exported successfully!")
                msg.setDetailedText(f"Location: {export_path}")
                msg.exec()

                self.test_output.append(f"\n✅ Report exported to: {export_path}\n")
            else:
                raise Exception("Export failed")
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Export Failed")
            msg.setText(f"Failed to export report: {str(e)}")
            msg.exec()

            self.test_output.append(f"\n❌ Export failed: {str(e)}\n")

    # File watcher methods

    def _setup_file_watcher(self):
        """Set up file system watcher for auto-testing."""
        self.file_watcher = QFileSystemWatcher()

        # Get all Python files in src/ and tests/ directories
        src_dir = Path(self.working_dir) / "src"
        tests_dir = Path(self.working_dir) / "tests"

        watched_files = []

        # Add src directory files
        if src_dir.exists():
            for py_file in src_dir.rglob("*.py"):
                watched_files.append(str(py_file))

        # Add tests directory files
        if tests_dir.exists():
            for py_file in tests_dir.rglob("*.py"):
                watched_files.append(str(py_file))

        # Add files to watcher
        if watched_files:
            self.file_watcher.addPaths(watched_files)
            # Also watch directories to detect new files
            self.file_watcher.addPath(str(src_dir))
            self.file_watcher.addPath(str(tests_dir))

        # Connect signal
        self.file_watcher.fileChanged.connect(self._on_file_changed)
        self.file_watcher.directoryChanged.connect(self._on_directory_changed)

    def _toggle_auto_test(self, enabled: bool):
        """Toggle auto-testing on/off."""
        self.auto_test_enabled = enabled
        if enabled:
            self.test_output.append("\n🔍 Auto-testing enabled - tests will run when files are saved\n")
        else:
            self.test_output.append("\n⏸️ Auto-testing disabled\n")

    def _on_file_changed(self, path: str):
        """Handle file change event."""
        if not self.auto_test_enabled:
            return

        # Only react to .py files
        if not path.endswith(".py"):
            return

        # Ignore if already testing
        if self.is_testing:
            return

        # Debounce: restart timer on each file change
        # This prevents multiple test runs when saving multiple files
        self.debounce_timer.stop()
        self.debounce_timer.start(1500)  # Wait 1.5 seconds after last change

    def _on_directory_changed(self, path: str):
        """Handle directory change event (new files added)."""
        if not self.auto_test_enabled:
            return

        # Refresh watched files list when directory changes
        self._refresh_watched_files()

        # Debounce test run
        if not self.is_testing:
            self.debounce_timer.stop()
            self.debounce_timer.start(1500)

    def _refresh_watched_files(self):
        """Refresh the list of watched files."""
        if not self.file_watcher:
            return

        # Get current watched files
        current_files = set(self.file_watcher.files())

        # Get all Python files in src/ and tests/
        src_dir = Path(self.working_dir) / "src"
        tests_dir = Path(self.working_dir) / "tests"

        new_files = set()

        if src_dir.exists():
            for py_file in src_dir.rglob("*.py"):
                new_files.add(str(py_file))

        if tests_dir.exists():
            for py_file in tests_dir.rglob("*.py"):
                new_files.add(str(py_file))

        # Add any new files
        files_to_add = new_files - current_files
        if files_to_add:
            self.file_watcher.addPaths(list(files_to_add))

    def _run_auto_test(self):
        """Run tests automatically (called by debounce timer)."""
        if not self.auto_test_enabled or self.is_testing:
            return

        self.test_output.append("\n🔄 Auto-test triggered by file change...\n")
        self._run_tests()
