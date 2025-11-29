"""MCP server exposing local development helpers for SC Gen 6."""

from __future__ import annotations

import shlex
import subprocess
from collections import deque
from pathlib import Path
from typing import Iterable, Sequence

from mcp.server import FastMCP


server = FastMCP("local-tools")

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MAX_OUTPUT_CHARS = 50_000
ALLOWED_EXECUTABLES = {
    "python",
    "py",
    "python3",
    "git",
    "pytest",
    "uv",
    "uvx",
    "uvicorn",
    "pip",
    "pip3",
    "ruff",
    "mypy",
    "cmake",
    "cmake64",
    "ninja",
    "rg",
    "ripgrep",
    "node",
    "npm",
    "pnpm",
    "yarn",
    "npx",
    "poetry",
    "pipenv",
    "curl",
}


def _normalize_executable(value: str) -> str:
    name = Path(value).name.lower()
    if name.endswith(".exe"):
        name = name[:-4]
    return name


def _ensure_allowed(executable: str) -> None:
    normalized = _normalize_executable(executable)
    if normalized not in ALLOWED_EXECUTABLES:
        raise ValueError(f"Command '{normalized}' not allowed. Update ALLOWED_EXECUTABLES if needed.")


def _truncate_output(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return f"{text[:MAX_OUTPUT_CHARS]}\n\n... output truncated ({len(text) - MAX_OUTPUT_CHARS} chars omitted) ..."


def _resolve_path(path_str: str | None, *, expect_dir: bool = False, expect_file: bool = False) -> Path:
    if path_str:
        candidate = Path(path_str)
        if not candidate.is_absolute():
            candidate = WORKSPACE_ROOT / candidate
    else:
        candidate = WORKSPACE_ROOT

    candidate = candidate.expanduser().resolve()
    try:
        candidate.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:  # pragma: no cover - guardrail
        raise ValueError(f"Path '{candidate}' is outside workspace root '{WORKSPACE_ROOT}'.") from exc

    if expect_dir and not candidate.is_dir():
        raise ValueError(f"Directory not found: {candidate}")
    if expect_file and not candidate.is_file():
        raise ValueError(f"File not found: {candidate}")

    return candidate


def _run_cli(command: Sequence[str], cwd: str | Path | None = None) -> str:
    if not command:
        raise ValueError("Command is required")

    _ensure_allowed(command[0])
    cwd_path = _resolve_path(str(cwd) if cwd else None, expect_dir=True)

    result = subprocess.run(
        list(command),
        cwd=str(cwd_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    return _truncate_output(result.stdout)


def _split_args(args: str | None) -> list[str]:
    if not args:
        return []
    return shlex.split(args, posix=False)


@server.tool()
async def run_program(command: str, cwd: str | None = None) -> str:
    """Execute an allowlisted CLI command (e.g., git, python, pytest)."""

    parts = _split_args(command)
    return _run_cli(parts, cwd=cwd)


@server.tool()
async def read_text(path: str) -> str:
    """Return plain-text contents of a UTF-8 file inside the workspace."""

    file_path = _resolve_path(path, expect_file=True)
    return file_path.read_text(encoding="utf-8")


@server.tool()
async def list_directory(path: str | None = None, recursive: bool = False, max_entries: int = 200) -> str:
    """List files/folders relative to the workspace root."""

    target = _resolve_path(path, expect_dir=True)
    max_entries = max(1, min(max_entries, 5000))

    entries: list[Path] = []
    has_more = False

    iterator: Iterable[Path] = target.rglob("*") if recursive else target.iterdir()
    for idx, entry in enumerate(iterator):
        if idx < max_entries:
            entries.append(entry)
        else:
            has_more = True
            break

    lines = [f"Directory: {target} (showing up to {max_entries} entries)"]
    if not entries:
        lines.append("<empty>")
        return "\n".join(lines)

    for entry in entries:
        rel_path = entry.relative_to(WORKSPACE_ROOT)
        marker = "[DIR]" if entry.is_dir() else "[FILE]"
        lines.append(f"{marker:7} {rel_path}")

    if has_more:
        lines.append("... additional entries omitted ...")

    return "\n".join(lines)


@server.tool()
async def tail_file(path: str, lines: int = 200) -> str:
    """Return the last N lines from a file."""

    file_path = _resolve_path(path, expect_file=True)
    lines = max(1, min(lines, 2000))

    buffer: deque[str] = deque(maxlen=lines)
    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            buffer.append(line.rstrip("\n"))

    joined = "\n".join(buffer)
    return f"File: {file_path}\n--- Last {len(buffer)} lines ---\n{joined}"


@server.tool()
async def run_git(args: str = "status -sb", cwd: str | None = None) -> str:
    """Run a git command relative to the workspace."""

    return _run_cli(["git", *_split_args(args)], cwd=cwd)


@server.tool()
async def run_pytest(targets: str = "", extra_args: str = "", cwd: str | None = None) -> str:
    """Execute pytest for the project."""

    command = ["python", "-m", "pytest"]
    command += _split_args(targets)
    command += _split_args(extra_args)
    return _run_cli(command, cwd=cwd)


@server.tool()
async def run_python_module(module: str, module_args: str | None = None, cwd: str | None = None) -> str:
    """Execute `python -m <module>` with optional arguments."""

    if not module:
        raise ValueError("Module name is required")

    command = ["python", "-m", module]
    command += _split_args(module_args)
    return _run_cli(command, cwd=cwd)


@server.tool()
async def run_package_manager(tool: str, args: str = "", cwd: str | None = None) -> str:
    """Run npm/pnpm/yarn/uv/pip style commands."""

    if not tool:
        raise ValueError("Tool name is required")

    command = [tool] + _split_args(args)
    return _run_cli(command, cwd=cwd)


@server.tool()
async def ripgrep_search(pattern: str, path: str | None = None, extra_args: str = "") -> str:
    """Run ripgrep against the workspace."""

    if not pattern:
        raise ValueError("Search pattern is required")

    target = _resolve_path(path or ".", expect_dir=True)
    command = ["rg", "--line-number"] + _split_args(extra_args) + [pattern, str(target)]
    return _run_cli(command, cwd=str(target))


# =============================================================================
# DIAGNOSTIC TOOLS
# =============================================================================

@server.tool()
async def run_diagnostics() -> str:
    """Run full system diagnostics for SC Gen 6 (GPU, config, models, indexes)."""
    
    try:
        result = subprocess.run(
            ["python", "-c", """
from src.system.diagnostics import run_diagnostics, format_diagnostics
diag = run_diagnostics()
print(format_diagnostics(diag))
"""],
            cwd=str(WORKSPACE_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=60,
        )
        return _truncate_output(result.stdout)
    except Exception as e:
        return f"Diagnostics failed: {e}"


@server.tool()
async def check_llm_config() -> str:
    """Check LLM configuration (both YAML config and runtime JSON)."""
    
    try:
        result = subprocess.run(
            ["python", "-c", """
from src.config_loader import get_settings
from src.config.llm_config import load_llm_config

settings = get_settings()
llm_cfg = load_llm_config()

print("=== YAML Config (config/config.yaml) ===")
print(f"Backend: {settings.models.llm.backend}")
print(f"Default Model: {settings.models.llm.default}")

print()
print("=== Runtime Config (config/llm_runtime.json) ===")
print(f"Provider: {llm_cfg.provider}")
print(f"Model Name: {llm_cfg.model_name}")
print(f"Base URL: {llm_cfg.base_url}")

if hasattr(llm_cfg, 'llama_server') and llm_cfg.llama_server:
    ls = llm_cfg.llama_server
    print()
    print("=== llama.cpp Server Config ===")
    print(f"Executable: {getattr(ls, 'executable', 'N/A')}")
    print(f"Model Path: {getattr(ls, 'model_path', 'N/A')}")
    print(f"Context: {getattr(ls, 'context', 'N/A')}")
    print(f"GPU Layers: {getattr(ls, 'gpu_layers', 'N/A')}")
"""],
            cwd=str(WORKSPACE_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=30,
        )
        return _truncate_output(result.stdout)
    except Exception as e:
        return f"Config check failed: {e}"


@server.tool()
async def check_log_errors(log_file: str = "all") -> str:
    """Scan log files for ERROR, Exception, and Traceback entries.
    
    Args:
        log_file: Specific log file name (e.g., 'sc-gen-6.log') or 'all' for all logs
    """
    
    logs_dir = WORKSPACE_ROOT / "logs"
    if not logs_dir.exists():
        return "No logs directory found"
    
    results = []
    
    if log_file == "all":
        log_files = list(logs_dir.glob("*.log"))
    else:
        specific_log = logs_dir / log_file
        log_files = [specific_log] if specific_log.exists() else []
    
    if not log_files:
        return f"No log files found matching '{log_file}'"
    
    for log_path in log_files:
        try:
            content = log_path.read_text(encoding="utf-8", errors="replace")
            error_lines = []
            lines = content.split("\n")
            
            for i, line in enumerate(lines):
                if any(keyword in line for keyword in ["ERROR", "Exception", "Traceback", "CRITICAL"]):
                    # Get context: this line + up to 3 following lines
                    context = lines[i:i+4]
                    error_lines.append("\n".join(context))
            
            if error_lines:
                results.append(f"=== {log_path.name} ({len(error_lines)} errors) ===")
                results.extend(error_lines[-10:])  # Last 10 errors
                if len(error_lines) > 10:
                    results.append(f"... and {len(error_lines) - 10} more errors")
        except Exception as e:
            results.append(f"Error reading {log_path.name}: {e}")
    
    if not results:
        return "No errors found in log files"
    
    return "\n\n".join(results)


@server.tool()
async def db_stats() -> str:
    """Get vector database (Chroma) and BM25 index statistics."""
    
    try:
        result = subprocess.run(
            ["python", "-c", """
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index

print("=== Chroma Vector Store ===")
vs = VectorStore()
stats = vs.stats()
print(f"Total Chunks: {stats.get('total_chunks', 'N/A')}")
print(f"Unique Files: {stats.get('unique_files', 'N/A')}")
print(f"Document Types: {stats.get('document_types', {})}")
print(f"DB Path: {stats.get('db_path', 'N/A')}")

if stats.get('unique_documents'):
    print(f"\\nIndexed Documents:")
    for doc in stats['unique_documents'][:20]:
        print(f"  - {doc}")
    if len(stats['unique_documents']) > 20:
        print(f"  ... and {len(stats['unique_documents']) - 20} more")

print()
print("=== BM25 Index ===")
try:
    bm25 = BM25Index()
    if hasattr(bm25, 'stats'):
        bm25_stats = bm25.stats()
        print(f"Stats: {bm25_stats}")
    else:
        print("BM25 index loaded (no detailed stats available)")
except Exception as e:
    print(f"BM25 error: {e}")
"""],
            cwd=str(WORKSPACE_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=30,
        )
        return _truncate_output(result.stdout)
    except Exception as e:
        return f"DB stats failed: {e}"


@server.tool()
async def check_ollama() -> str:
    """Check Ollama status and list available models."""
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            return f"Ollama is running.\n\nAvailable models:\n{result.stdout}"
        else:
            return f"Ollama returned error:\n{result.stdout}"
    except FileNotFoundError:
        return "Ollama not installed or not in PATH"
    except subprocess.TimeoutExpired:
        return "Ollama command timed out - server may not be running. Start with: ollama serve"
    except Exception as e:
        return f"Ollama check failed: {e}\n\nTo start Ollama: ollama serve"


@server.tool()
async def check_llama_server() -> str:
    """Check llama.cpp server health and list loaded models."""
    
    import urllib.request
    import json
    
    base_url = "http://127.0.0.1:8000"
    results = []
    
    # Check health
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
            results.append(f"✓ Server healthy at {base_url}")
    except Exception as e:
        return f"✗ llama.cpp server not responding at {base_url}\n\nError: {e}\n\nTo start: scripts\\start_llama_server.bat"
    
    # Get models
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as response:
            data = json.loads(response.read().decode())
            results.append("\nLoaded models:")
            for model in data.get("data", []):
                results.append(f"  - {model.get('id', 'unknown')}")
    except Exception as e:
        results.append(f"\nCould not fetch models: {e}")
    
    # Get server props if available
    try:
        with urllib.request.urlopen(f"{base_url}/props", timeout=5) as response:
            data = json.loads(response.read().decode())
            results.append(f"\nServer properties:")
            results.append(f"  Context size: {data.get('default_generation_settings', {}).get('n_ctx', 'N/A')}")
    except:
        pass
    
    return "\n".join(results)


@server.tool()
async def list_document_types() -> str:
    """List all supported document types for classification."""
    
    try:
        result = subprocess.run(
            ["python", "-c", """
from src.schema import DocumentType
import typing

types = typing.get_args(DocumentType)
print("Supported Document Types (19 total):")
print()

categories = {
    "Core Pleadings": ["witness_statement", "court_filing", "pleading", "skeleton_argument"],
    "Expert & Technical": ["expert_report", "schedule_of_loss", "medical_report"],
    "Legal Sources": ["statute", "case_law", "contract"],
    "Correspondence": ["email", "letter", "disclosure", "disclosure_list"],
    "Court Forms": ["court_form", "case_management", "chronology"],
    "Specialist": ["tribunal_document", "regulatory_document"],
    "Fallback": ["scanned_pdf", "unknown"],
}

for category, doc_types in categories.items():
    print(f"{category}:")
    for dt in doc_types:
        if dt in types:
            print(f"  ✓ {dt}")
    print()
"""],
            cwd=str(WORKSPACE_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=15,
        )
        return _truncate_output(result.stdout)
    except Exception as e:
        return f"Failed to list document types: {e}"


@server.tool()
async def verify_config() -> str:
    """Run first-run configuration verification."""
    
    return _run_cli(["python", "-m", "src.utils.first_run"])


if __name__ == "__main__":
    server.run()
