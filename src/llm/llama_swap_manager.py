"""Unified llama-swap manager for SC Gen 6.

Manages the llama-swap proxy which handles both generation and embedding models.
llama-swap automatically starts/stops backend llama-server instances based on requests.
"""

from __future__ import annotations

import subprocess
import sys
import time
import requests
from pathlib import Path
from typing import Optional

from src.config_loader import get_settings


class LlamaSwapManager:
    """Manager for llama-swap proxy process.
    
    llama-swap is a transparent proxy that manages multiple llama-server instances.
    It handles on-demand model loading, health checks, and VRAM management.
    """
    
    # Default configuration
    DEFAULT_PORT = 8000  # llama-swap listens here, routes to backends
    CONFIG_FILE = Path("llama-cpp/config.yaml")
    EXECUTABLE = Path("llama-cpp/llama-swap.exe")
    
    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._log_file = None
        self.settings = get_settings()
        self._port = self.DEFAULT_PORT
    
    @property
    def base_url(self) -> str:
        """Get the base URL for llama-swap proxy."""
        return f"http://127.0.0.1:{self._port}"
    
    def start(
        self,
        port: int = 8000,
        config_path: Optional[Path] = None,
        timeout: float = 60.0,
    ) -> None:
        """Start llama-swap proxy.
        
        Args:
            port: Port for llama-swap to listen on
            config_path: Path to llama-swap config.yaml
            timeout: Seconds to wait for proxy to be ready
        """
        if self.is_running():
            print("[LlamaSwap] Already running")
            return
        
        self._port = port
        config = config_path or self.CONFIG_FILE
        exe = self.EXECUTABLE
        
        if not exe.exists():
            raise FileNotFoundError(
                f"llama-swap executable not found at {exe}. "
                f"Download from https://github.com/mostlygeek/llama-swap/releases"
            )
        
        if not config.exists():
            raise FileNotFoundError(
                f"llama-swap config not found at {config}. "
                f"Create config.yaml with model definitions."
            )
        
        # Resolve to absolute paths to avoid cwd-relative path issues
        exe_abs = exe.resolve()
        config_abs = config.resolve()
        
        # Build command with absolute paths
        cmd = [
            str(exe_abs),
            "--config", str(config_abs),
            "--listen", f":{port}",
        ]
        
        print(f"[LlamaSwap] Starting proxy on port {port}...")
        print(f"[LlamaSwap] Config: {config}")
        
        # Start process with output logging for debugging
        # Log stdout/stderr to file for live monitoring
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / "llama_swap.log"
        self._log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        
        print(f"[LlamaSwap] Output logging to: {log_path.resolve()}")
        print(f"[LlamaSwap] Monitor with: Get-Content '{log_path.resolve()}' -Wait")
        
        # Show visible console window for debugging - user can see llama-swap output
        creationflags = 0
        if sys.platform == "win32":
            # CREATE_NEW_CONSOLE for visible debugging window
            creationflags = subprocess.CREATE_NEW_CONSOLE
        
        self._process = subprocess.Popen(
            cmd,
            cwd=str(exe_abs.parent),  # Run from llama-cpp directory (absolute path)
            stdout=None,  # Console window handles output
            stderr=None,
            creationflags=creationflags,
        )
        
        # Give the process a moment to crash if it's going to
        time.sleep(1)
        
        # Wait for proxy to be ready
        self._wait_for_ready(timeout)
    
    def _wait_for_ready(self, timeout: float = 60.0) -> None:
        """Wait for llama-swap to respond to health check."""
        start = time.time()
        url = f"{self.base_url}/health"
        
        while time.time() - start < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    print(f"[LlamaSwap] Proxy ready on port {self._port}")
                    return
            except requests.exceptions.ConnectionError:
                pass
            except Exception as e:
                print(f"[LlamaSwap] Health check error: {e}")
            time.sleep(0.5)
        
        print(f"[LlamaSwap] Warning: Health check timeout after {timeout}s")
    
    def stop(self) -> None:
        """Stop llama-swap and all managed backends."""
        if self._process and self._process.poll() is None:
            print("[LlamaSwap] Stopping proxy...")
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
        
        # Close log file if open
        if hasattr(self, '_log_file') and self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
        
        # Clean up any remaining llama-server processes
        if sys.platform == "win32":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", "llama-server.exe", "/T"],
                    capture_output=True,
                    check=False
                )
            except Exception:
                pass
        
        self._process = None
        print("[LlamaSwap] Stopped")
    
    def is_running(self) -> bool:
        """Check if llama-swap process is running."""
        return self._process is not None and self._process.poll() is None
    
    def is_ready(self) -> bool:
        """Check if llama-swap is responding to health checks."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_loaded_models(self) -> list[str]:
        """Get list of currently loaded models from llama-swap."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m.get("id", "") for m in data.get("data", [])]
        except Exception:
            pass
        return []
    
    def preload_model(self, model_name: str) -> bool:
        """Send a minimal request to trigger model loading.
        
        llama-swap loads models on first request. This sends a minimal
        request to pre-load a model before it's actually needed.
        """
        try:
            # For embedding model, use /embedding endpoint
            if "embed" in model_name.lower():
                response = requests.post(
                    f"{self.base_url}/embedding",
                    json={"model": model_name, "content": ["test"]},
                    timeout=300,  # Model loading can take a while
                )
            else:
                # For generation models, use /v1/chat/completions
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                    },
                    timeout=300,
                )
            return response.status_code == 200
        except Exception as e:
            print(f"[LlamaSwap] Failed to preload {model_name}: {e}")
            return False
    
    def ensure_running(self, port: int = 8000) -> bool:
        """Ensure llama-swap is running, start if not.
        
        Returns:
            True if proxy is ready, False otherwise
        """
        if self.is_ready():
            return True
        
        try:
            self.start(port=port)
            return self.is_ready()
        except Exception as e:
            print(f"[LlamaSwap] Failed to start: {e}")
            return False


# Singleton instance
llama_swap_manager = LlamaSwapManager()


# Backward compatibility - keep old manager as alias
class LlamaServerManager:
    """Legacy manager - now wraps llama-swap."""
    
    def __init__(self) -> None:
        self._manager = llama_swap_manager
    
    def start(self, **kwargs) -> None:
        """Start llama-swap (ignores legacy kwargs)."""
        self._manager.ensure_running()
    
    def stop(self) -> None:
        """Stop llama-swap."""
        self._manager.stop()
    
    def is_running(self) -> bool:
        """Check if llama-swap is running."""
        return self._manager.is_running()


# Singleton for backward compatibility
manager = LlamaServerManager()
