"""Cross-platform launcher for SC Gen 6 desktop application."""

import atexit
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_FILE = PROJECT_ROOT / "config" / "llm_runtime.json"
LLAMA_SWAP_BIN = PROJECT_ROOT / "llama-swap" / "llama-swap"
LLAMA_SWAP_CONFIG = PROJECT_ROOT / "llama-swap" / "config.yaml"
LLAMA_SWAP_LOG = PROJECT_ROOT / "logs" / "llama-swap.log"
LLAMA_SWAP_PID_FILE = PROJECT_ROOT / "logs" / "llama-swap.pid"

_llama_swap_proc: subprocess.Popen | None = None


# ── Runtime config helpers ────────────────────────────────────────────────────

def _load_runtime() -> dict:
    try:
        if RUNTIME_FILE.exists():
            return json.loads(RUNTIME_FILE.read_text())
    except Exception:
        pass
    return {"provider": "llama_cpp", "model_name": "nemotron-3-nano",
            "base_url": "http://127.0.0.1:8000/v1"}


def _get_provider() -> str:
    return _load_runtime().get("provider", "llama_cpp")


# ── Backend startup ───────────────────────────────────────────────────────────

def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _llama_swap_port() -> int:
    """Read the listen port from llama-swap config.yaml."""
    try:
        import yaml  # type: ignore
        cfg = yaml.safe_load(LLAMA_SWAP_CONFIG.read_text())
        listen = cfg.get("listen", ":8000")
        return int(listen.split(":")[-1])
    except Exception:
        return 8000


def _kill_port(port: int) -> None:
    """Kill any process currently bound to the given TCP port."""
    try:
        # fuser -k <port>/tcp sends SIGKILL to whatever owns the port
        result = subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True, timeout=5
        )
        if result.returncode == 0:
            print(f"[INFO] Killed existing process on :{port}")
            time.sleep(0.5)  # give the OS time to release the port
    except FileNotFoundError:
        # fuser not available — try lsof fallback
        try:
            out = subprocess.check_output(
                ["lsof", "-ti", f"tcp:{port}"], timeout=5
            ).decode().strip()
            if out:
                pids = out.split()
                subprocess.run(["kill", "-9"] + pids, capture_output=True)
                print(f"[INFO] Killed PID(s) {pids} on :{port}")
                time.sleep(0.5)
        except Exception:
            pass
    except Exception:
        pass


def start_llama_swap() -> bool:
    """Start llama-swap if not already running. Returns True if ready."""
    global _llama_swap_proc

    if not LLAMA_SWAP_BIN.exists():
        print(f"[ERROR] llama-swap binary not found: {LLAMA_SWAP_BIN}")
        return False

    port = _llama_swap_port()

    # Already running?
    if _is_port_open("127.0.0.1", port):
        print(f"[OK] llama-swap already running on :{port}")
        return True

    # Kill anything on the port (stale llama-swap, rogue process, etc.)
    _kill_port(port)

    # Also kill by PID file if present
    if LLAMA_SWAP_PID_FILE.exists():
        try:
            old_pid = int(LLAMA_SWAP_PID_FILE.read_text().strip())
            subprocess.run(["kill", str(old_pid)], capture_output=True)
            time.sleep(0.3)
        except Exception:
            pass

    print(f"[INFO] Starting llama-swap on :{port}...")
    LLAMA_SWAP_LOG.parent.mkdir(parents=True, exist_ok=True)

    log_fh = LLAMA_SWAP_LOG.open("a")
    _llama_swap_proc = subprocess.Popen(
        [
            str(LLAMA_SWAP_BIN),
            "--config", str(LLAMA_SWAP_CONFIG),
            "--listen", f":{port}",  # explicit flag — config 'listen' key ignored by this build
        ],
        stdout=log_fh,
        stderr=log_fh,
        cwd=str(PROJECT_ROOT),
    )
    LLAMA_SWAP_PID_FILE.write_text(str(_llama_swap_proc.pid))
    atexit.register(_stop_llama_swap)

    # Wait up to 10s for llama-swap to bind
    for i in range(20):
        time.sleep(0.5)
        if _llama_swap_proc.poll() is not None:
            print(f"[ERROR] llama-swap exited immediately — check {LLAMA_SWAP_LOG}")
            return False
        if _is_port_open("127.0.0.1", port):
            print(f"[OK] llama-swap ready on :{port} (PID {_llama_swap_proc.pid})")
            return True
        if i % 4 == 3:
            print(f"[INFO] Waiting for llama-swap... ({(i+1)//2}s)")

    print(f"[WARNING] llama-swap started but not yet bound on :{port} — model loading in background")
    print(f"          Log: {LLAMA_SWAP_LOG}")
    return True  # Don't block app startup — model loads lazily


def _stop_llama_swap() -> None:
    """Gracefully stop llama-swap on app exit."""
    global _llama_swap_proc
    if _llama_swap_proc and _llama_swap_proc.poll() is None:
        print("[INFO] Stopping llama-swap...")
        _llama_swap_proc.terminate()
        try:
            _llama_swap_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _llama_swap_proc.kill()
    if LLAMA_SWAP_PID_FILE.exists():
        LLAMA_SWAP_PID_FILE.unlink(missing_ok=True)


def check_ollama() -> bool:
    """Check if Ollama is running (kept for backwards compat, not used by default)."""
    try:
        import requests
        requests.get("http://localhost:11434", timeout=2)
        print("[OK] Ollama is running")
        return True
    except Exception:
        print("[WARNING] Ollama not responding (not needed — using llama-swap)")
        return True


# ── Dependency check ──────────────────────────────────────────────────────────

def check_dependencies() -> bool:
    required = [
        ("PySide6", "PySide6"),
        ("pydantic", "pydantic"),
        ("yaml", "yaml"),
        ("chromadb", "chromadb"),
        ("sentence_transformers", "sentence_transformers"),
    ]
    missing = []
    for display, import_name in required:
        try:
            __import__(import_name.replace("-", "_"))
        except ImportError:
            missing.append(display)

    if missing:
        print(f"[WARNING] Missing dependencies: {', '.join(missing)}")
        print("  Install with: pip install -r requirements-linux.txt")
        return False

    print("[OK] Dependencies installed")
    return True


# ── Python version check ──────────────────────────────────────────────────────

def check_python() -> bool:
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 11):
        print(f"[ERROR] Python 3.11+ required (got {v.major}.{v.minor}.{v.micro})")
        return False
    print(f"[OK] Python {v.major}.{v.minor}.{v.micro}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("SCGen7 — Litigation Support RAG")
    print("=" * 60)
    print()

    if not check_python():
        sys.exit(1)

    deps_ok = check_dependencies()
    if not deps_ok:
        try:
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != "y":
                sys.exit(1)
        except EOFError:
            # Non-interactive (launched from desktop file) — continue
            pass

    provider = _get_provider()
    print(f"[INFO] Provider: {provider} — starting llama-swap")
    start_llama_swap()

    print()
    print("Launching SCGen7...")
    print()

    try:
        from src.ui7.app import main as ui_main
        ui_main()
    except KeyboardInterrupt:
        print("\n\nApplication closed by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Failed to launch: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
