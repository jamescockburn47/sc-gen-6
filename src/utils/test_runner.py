"""Test runner service for integrated testing within the UI."""

import json
import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TestResult:
    """Container for test execution results."""

    def __init__(
        self,
        passed: int = 0,
        failed: int = 0,
        skipped: int = 0,
        errors: int = 0,
        duration: float = 0.0,
        coverage: float = 0.0,
        output: str = "",
        failures: List[Dict[str, str]] = None,
    ):
        """Initialize test result.

        Args:
            passed: Number of passed tests
            failed: Number of failed tests
            skipped: Number of skipped tests
            errors: Number of tests with errors
            duration: Total test duration in seconds
            coverage: Code coverage percentage (0-100)
            output: Raw test output
            failures: List of failure details
        """
        self.passed = passed
        self.failed = failed
        self.skipped = skipped
        self.errors = errors
        self.duration = duration
        self.coverage = coverage
        self.output = output
        self.failures = failures or []
        self.timestamp = datetime.now()

    @property
    def total(self) -> int:
        """Total number of tests run."""
        return self.passed + self.failed + self.skipped + self.errors

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100

    @property
    def is_success(self) -> bool:
        """Whether all tests passed."""
        return self.failed == 0 and self.errors == 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "total": self.total,
            "duration": self.duration,
            "coverage": self.coverage,
            "success_rate": self.success_rate,
            "is_success": self.is_success,
            "failures": self.failures,
            "output": self.output,
        }


class TestRunner:
    """Test runner for executing pytest and generating reports."""

    def __init__(self, working_dir: str):
        """Initialize test runner.

        Args:
            working_dir: Working directory for test execution
        """
        self.working_dir = Path(working_dir)
        self.last_result: Optional[TestResult] = None

    def run_tests(
        self,
        test_path: str = "tests",
        verbose: bool = True,
        coverage: bool = True,
        markers: Optional[str] = None,
    ) -> TestResult:
        """Run pytest tests and return results.

        Args:
            test_path: Path to tests (relative to working_dir)
            verbose: Enable verbose output
            coverage: Enable coverage reporting
            markers: Pytest markers to filter tests (e.g., "not slow")

        Returns:
            TestResult object with execution results
        """
        # Build pytest command
        cmd = ["pytest", test_path]

        if verbose:
            cmd.append("-v")

        if coverage:
            cmd.extend(["--cov=src", "--cov-report=term", "--cov-report=json"])

        if markers:
            cmd.extend(["-m", markers])

        # Add JSON report output
        cmd.extend(["--json-report", "--json-report-file=.test_report.json"])

        try:
            # Run pytest
            process = subprocess.run(
                cmd,
                cwd=str(self.working_dir),
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            # Parse output
            result = self._parse_output(process.stdout, process.stderr, process.returncode)

            # Parse coverage if enabled
            if coverage:
                coverage_data = self._parse_coverage()
                result.coverage = coverage_data

            self.last_result = result
            return result

        except subprocess.TimeoutExpired:
            return TestResult(
                errors=1,
                output="Test execution timed out after 5 minutes.",
            )
        except Exception as e:
            return TestResult(
                errors=1,
                output=f"Test execution failed: {str(e)}",
            )

    def _parse_output(self, stdout: str, stderr: str, returncode: int) -> TestResult:
        """Parse pytest output to extract test results.

        Args:
            stdout: Standard output from pytest
            stderr: Standard error from pytest
            returncode: Process return code

        Returns:
            TestResult object
        """
        output = stdout + "\n" + stderr

        # Parse test counts from summary line
        # Example: "===== 10 passed, 2 failed, 1 skipped in 5.23s ====="
        passed = 0
        failed = 0
        skipped = 0
        errors = 0
        duration = 0.0

        # Extract counts
        if match := re.search(r"(\d+) passed", output):
            passed = int(match.group(1))
        if match := re.search(r"(\d+) failed", output):
            failed = int(match.group(1))
        if match := re.search(r"(\d+) skipped", output):
            skipped = int(match.group(1))
        if match := re.search(r"(\d+) error", output):
            errors = int(match.group(1))
        if match := re.search(r"in ([\d.]+)s", output):
            duration = float(match.group(1))

        # Extract failure details
        failures = self._extract_failures(output)

        return TestResult(
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration=duration,
            output=output,
            failures=failures,
        )

    def _extract_failures(self, output: str) -> List[Dict[str, str]]:
        """Extract failure details from pytest output.

        Args:
            output: Pytest output text

        Returns:
            List of failure dictionaries with test name and error message
        """
        failures = []

        # Look for FAILED test lines
        # Example: "tests/test_foo.py::test_bar FAILED"
        failed_pattern = r"(tests/[^\s]+::[^\s]+)\s+FAILED"
        for match in re.finditer(failed_pattern, output):
            test_name = match.group(1)

            # Try to find the associated error message
            # This is a simplified approach - pytest-json-report gives better data
            error_msg = "See test output for details"
            failures.append({"test": test_name, "error": error_msg})

        return failures

    def _parse_coverage(self) -> float:
        """Parse coverage data from coverage.json.

        Returns:
            Coverage percentage (0-100)
        """
        coverage_file = self.working_dir / "coverage.json"

        if not coverage_file.exists():
            return 0.0

        try:
            with open(coverage_file, "r") as f:
                data = json.load(f)
                # Extract total coverage percentage
                total_coverage = data.get("totals", {}).get("percent_covered", 0.0)
                return round(total_coverage, 2)
        except Exception:
            return 0.0

    def export_report(
        self, format: str = "json", output_path: Optional[str] = None
    ) -> Optional[str]:
        """Export last test results to file.

        Args:
            format: Export format ('json' or 'html')
            output_path: Output file path (auto-generated if None)

        Returns:
            Path to exported file, or None if no results available
        """
        if not self.last_result:
            return None

        # Generate filename if not provided
        if output_path is None:
            timestamp = self.last_result.timestamp.strftime("%Y%m%d_%H%M%S")
            output_path = f"test_report_{timestamp}.{format}"

        output_file = self.working_dir / output_path

        try:
            if format == "json":
                with open(output_file, "w") as f:
                    json.dump(self.last_result.to_dict(), f, indent=2)
            elif format == "html":
                html_content = self._generate_html_report()
                with open(output_file, "w") as f:
                    f.write(html_content)
            else:
                return None

            return str(output_file)

        except Exception as e:
            print(f"Failed to export report: {e}")
            return None

    def _generate_html_report(self) -> str:
        """Generate HTML test report.

        Returns:
            HTML report as string
        """
        if not self.last_result:
            return "<html><body><h1>No test results available</h1></body></html>"

        result = self.last_result
        status_color = "green" if result.is_success else "red"
        status_text = "PASSED" if result.is_success else "FAILED"

        failures_html = ""
        if result.failures:
            failures_html = "<h2>Failures</h2><ul>"
            for failure in result.failures:
                failures_html += f"<li><strong>{failure['test']}</strong>: {failure['error']}</li>"
            failures_html += "</ul>"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Report - {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: {status_color}; color: white; padding: 20px; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .stat {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .stat-value {{ font-size: 2em; font-weight: bold; }}
        .progress {{ background: #e0e0e0; height: 30px; border-radius: 5px; overflow: hidden; }}
        .progress-bar {{ background: {status_color}; height: 100%; line-height: 30px; text-align: center; color: white; }}
        pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Test Report: {status_text}</h1>
        <p>Generated: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="stats">
        <div class="stat">
            <div>Passed</div>
            <div class="stat-value">{result.passed}</div>
        </div>
        <div class="stat">
            <div>Failed</div>
            <div class="stat-value">{result.failed}</div>
        </div>
        <div class="stat">
            <div>Skipped</div>
            <div class="stat-value">{result.skipped}</div>
        </div>
        <div class="stat">
            <div>Duration</div>
            <div class="stat-value">{result.duration:.2f}s</div>
        </div>
    </div>

    <h2>Success Rate</h2>
    <div class="progress">
        <div class="progress-bar" style="width: {result.success_rate}%">
            {result.success_rate:.1f}%
        </div>
    </div>

    <h2>Coverage</h2>
    <div class="progress">
        <div class="progress-bar" style="width: {result.coverage}%; background: #2196f3">
            {result.coverage:.1f}%
        </div>
    </div>

    {failures_html}

    <h2>Full Output</h2>
    <pre>{result.output}</pre>
</body>
</html>
"""
        return html

    def get_test_files(self) -> List[str]:
        """Get list of test files in the project.

        Returns:
            List of test file paths
        """
        test_dir = self.working_dir / "tests"
        if not test_dir.exists():
            return []

        return [str(f.relative_to(self.working_dir)) for f in test_dir.rglob("test_*.py")]
