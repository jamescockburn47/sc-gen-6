"""Manage lifecycle of llama.cpp embedding server process.

Auto-downloads model and starts llama.cpp with --embedding flag for GPU-accelerated embeddings.
Mirrors LlamaServerManager pattern for consistency.
"""

from __future__ import annotations

import subprocess
import sys
import time
import requests
from pathlib import Path
from typing import Optional

from src.config_loader import get_settings


class EmbeddingServerManager:
    """Manager to auto-start/stop llama.cpp embedding server."""
    
    # Model download config
    MODEL_REPO = "sabafallah/llama-embed-nemotron-8b-GGUF"
    MODEL_FILE = "llama-embed-nemotron-8b-Q4_K_M.gguf"
    MODEL_DIR = Path("models/embeddings")
    
    # VRAM requirements (approximate, in GB)
    EMBED_MODEL_VRAM_GB = 5  # 8B Q4_K_M model
    MIN_FREE_VRAM_GB = 8    # Minimum free VRAM to leave for safety

    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self.settings = get_settings()
    
    def _check_vram_available(self) -> tuple[bool, float, float]:
        """Check if there's enough VRAM for embedding model.
        
        Returns:
            Tuple of (is_safe, total_vram_gb, free_vram_gb)
        """
        try:
            # Try DirectML/torch to get VRAM info
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(device) / (1024**3)
                free = total - allocated
                
                # Check if we have enough for embedding model + safety buffer
                required = self.EMBED_MODEL_VRAM_GB + self.MIN_FREE_VRAM_GB
                is_safe = free >= required
                
                print(f"[Embedding Server] VRAM check: {free:.1f}GB free / {total:.1f}GB total, need {required}GB")
                return is_safe, total, free
        except Exception:
            pass
        
        # Fallback: assume we have enough on 96GB system
        print("[Embedding Server] VRAM check unavailable, assuming safe (96GB system)")
        return True, 96.0, 50.0
        
    def _get_model_path(self) -> Path:
        """Get path to embedding model, downloading if needed."""
        model_path = self.MODEL_DIR / self.MODEL_FILE
        
        if not model_path.exists():
            print(f"[Embedding Server] Model not found, downloading {self.MODEL_FILE}...")
            self._download_model()
        
        return model_path
    
    def _download_model(self) -> None:
        """Download embedding model from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download
            
            self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            
            print(f"[Embedding Server] Downloading from {self.MODEL_REPO}...")
            hf_hub_download(
                repo_id=self.MODEL_REPO,
                filename=self.MODEL_FILE,
                local_dir=str(self.MODEL_DIR),
                local_dir_use_symlinks=False,
            )
            print(f"[Embedding Server] Download complete: {self.MODEL_DIR / self.MODEL_FILE}")
            
        except ImportError:
            raise RuntimeError(
                "huggingface_hub not installed. Run: pip install huggingface_hub"
            )
    
    def start(
        self,
        executable: str = "llama-server.exe",
        port: int = 8090,
        gpu_layers: int = 99,
        context: int = 8192,
        timeout: int = 300,
    ) -> None:
        """Start embedding server with --embedding flag.
        
        Args:
            executable: Path to llama-server executable
            port: Port for embedding server (default 8090, separate from LLM)
            gpu_layers: GPU layers to offload (99 = all)
            context: Context window size
            timeout: Server timeout
        """
        if self.is_running():
            print("[Embedding Server] Already running")
            return
        
        # Get/download model
        model_path = self._get_model_path()
        
        exe_path = Path(executable)
        if not exe_path.exists():
            # Try common locations
            for try_path in [
                Path("llama-cpp/llama-server.exe"),  # Project's llama-cpp folder
                Path("llama-server.exe"),
                Path("./llama-server.exe"),
                Path("C:/llama.cpp/build/bin/Release/llama-server.exe"),
            ]:
                if try_path.exists():
                    exe_path = try_path
                    break
        
        if not exe_path.exists():
            raise FileNotFoundError(
                f"llama-server executable not found. "
                f"Tried: {executable}"
            )
        
        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-c", str(context),
            "-ngl", str(gpu_layers),
            "--port", str(port),
            "--embedding",  # Enable embedding mode
            "--timeout", str(timeout),
        ]
        
        print(f"[Embedding Server] Starting on port {port} with GPU ({gpu_layers} layers)...")
        
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_CONSOLE
        
        self._process = subprocess.Popen(
            cmd,
            stdout=None,
            stderr=None,
            creationflags=creationflags,
        )
        
        # Wait for server to be ready
        self._wait_for_ready(port)
    
    def _wait_for_ready(self, port: int, timeout: float = 60) -> None:
        """Wait for server to respond to health check."""
        start = time.time()
        url = f"http://localhost:{port}/health"
        
        while time.time() - start < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    print(f"[Embedding Server] Ready on port {port}")
                    return
            except Exception:
                pass
            time.sleep(1)
        
        print(f"[Embedding Server] Warning: Health check timeout after {timeout}s")
    
    def stop(self) -> None:
        """Stop the embedding server."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        
        self._process = None
    
    def is_running(self) -> bool:
        """Check if server process is running."""
        return self._process is not None and self._process.poll() is None
    
    def ensure_running(self, port: int = 8090) -> bool:
        """Ensure embedding server is running, start if not.
        
        Checks VRAM availability before starting to avoid conflicts with LLM.
        
        Returns:
            True if server is ready, False otherwise
        """
        # Check if already responding
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        
        # Check VRAM before starting
        is_safe, total, free = self._check_vram_available()
        if not is_safe:
            print(f"[Embedding Server] Not enough VRAM ({free:.1f}GB free, need {self.EMBED_MODEL_VRAM_GB + self.MIN_FREE_VRAM_GB}GB)")
            print("[Embedding Server] Falling back to ONNX embeddings")
            return False
        
        # Start if not running
        try:
            self.start(port=port)
            return True
        except Exception as e:
            print(f"[Embedding Server] Failed to start: {e}")
            return False


# Singleton instance
embedding_server_manager = EmbeddingServerManager()
