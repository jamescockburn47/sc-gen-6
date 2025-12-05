"""Manage lifecycle of a local llama.cpp server process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional


class LlamaServerManager:
    """Simple manager to start/stop llama-server."""

    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen[str]] = None
        self._log_handle = None

    def start(
        self,
        executable: str,
        model_path: str,
        host: str,
        port: int,
        api_key: str,
        context: int,
        gpu_layers: int,
        parallel: int,
        batch: int,
        timeout: int,
        extra_args: Optional[list[str]] = None,
        detached: bool = False,
        log_path: Optional[Path] = None,
    ) -> None:
        """Start llama-server with the provided parameters."""
        if self.is_running():
            raise RuntimeError("llama-server is already running")

        exe_path = Path(executable)
        if not exe_path.exists():
            raise FileNotFoundError(f"llama-server executable not found at {executable}")

        model = Path(model_path)
        if not model.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        cmd = [
            str(exe_path),
            "-m",
            str(model),
            "-c",
            str(context),
            "-ngl",
            str(gpu_layers),
            "--host",
            host,
            "--port",
            str(port),
            "--api-key",
            api_key,
            "--parallel",
            str(parallel),
            "-b",
            str(batch),
            "--timeout",
            str(timeout),
        ]

        if extra_args:
            cmd.extend(extra_args)

        creationflags = 0
        stdout = None
        stderr = None

        if sys.platform == "win32":
            if detached:
                creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
            else:
                creationflags = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]

        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = log_path.open("a", encoding="utf-8")
            stdout = self._log_handle
            stderr = self._log_handle
        elif detached:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL
        else:
            stdout = None
            stderr = None

        text_mode = bool(log_path) or not detached

        self._process = subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            universal_newlines=text_mode,
            creationflags=creationflags,
        )

    def stop(self) -> None:
        """Stop the running llama-server if present."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

    def is_running(self) -> bool:
        """Return True if the server process is alive."""
        return self._process is not None and self._process.poll() is None


manager = LlamaServerManager()


