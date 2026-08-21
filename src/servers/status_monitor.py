"""Server status monitor for generation and embedding servers.

Provides real-time status tracking with:
- Server up/down detection
- Model loading detection (503 vs 200)
- Log-based progress parsing for layer offloading
"""

import logging
import re
import threading
import time
from pathlib import Path
from typing import Optional, Callable

import requests
from PySide6.QtCore import QObject, Signal, QTimer

logger = logging.getLogger(__name__)


class ServerStatus:
    """Status constants for servers."""
    DOWN = "down"           # Can't connect
    PROXY_ONLY = "proxy"    # llama-swap up, no model
    LOADING = "loading"     # Model being loaded
    READY = "ready"         # Model loaded and responding


class ServerStatusMonitor(QObject):
    """Monitors both generation and embedding servers.
    
    Polls servers periodically and emits signals on state changes.
    Optionally parses logs for loading progress.
    """
    
    # Signals for status changes
    gen_status_changed = Signal(str, str, int)   # (status, model_name, progress_pct)
    embed_status_changed = Signal(str, str, int) # (status, model_name, progress_pct)
    
    # Combined status for simple checks
    all_ready = Signal()  # Emitted when both servers ready
    
    def __init__(
        self,
        gen_url: str = "http://127.0.0.1:8000",
        embed_url: str = "http://127.0.0.1:8001",
        gen_log_path: Optional[Path] = None,
        embed_log_path: Optional[Path] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.gen_url = gen_url
        self.embed_url = embed_url
        self.gen_log_path = gen_log_path
        self.embed_log_path = embed_log_path
        
        # Current state
        self._gen_status = ServerStatus.DOWN
        self._gen_model = ""
        self._gen_progress = 0
        
        self._embed_status = ServerStatus.DOWN
        self._embed_model = ""
        self._embed_progress = 0
        
        # Polling timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        
        # Log file positions for incremental reading
        self._gen_log_pos = 0
        self._embed_log_pos = 0
        
    def start(self, interval_ms: int = 2000):
        """Start polling servers.
        
        First poll happens immediately to detect model loading state.
        """
        self._poll()  # Check immediately to catch loading state (503)
        self._timer.start(interval_ms)
        
    def stop(self):
        """Stop polling."""
        self._timer.stop()
        
    def get_status(self) -> dict:
        """Get current status of both servers."""
        return {
            "generation": {
                "status": self._gen_status,
                "model": self._gen_model,
                "progress": self._gen_progress,
            },
            "embedding": {
                "status": self._embed_status,
                "model": self._embed_model,
                "progress": self._embed_progress,
            },
        }
    
    def is_ready(self) -> bool:
        """Check if both servers are ready."""
        return (
            self._gen_status == ServerStatus.READY and
            self._embed_status == ServerStatus.READY
        )
    
    def is_gen_ready(self) -> bool:
        """Check if generation server is ready."""
        return self._gen_status == ServerStatus.READY
    
    def is_embed_ready(self) -> bool:
        """Check if embedding server is ready."""
        return self._embed_status == ServerStatus.READY
        
    def _poll(self):
        """Poll both servers for status."""
        # Check generation server (llama-swap)
        new_gen_status, new_gen_model = self._check_llama_swap(self.gen_url)
        gen_progress = self._parse_log_progress(self.gen_log_path, "gen") if self.gen_log_path else 0
        
        # DEBUG: Log what we detect
        logger.debug("[StatusMonitor] Gen: %s (%s) | Old: %s", new_gen_status, new_gen_model, self._gen_status)
        
        if (new_gen_status != self._gen_status or 
            new_gen_model != self._gen_model or
            gen_progress != self._gen_progress):
            self._gen_status = new_gen_status
            self._gen_model = new_gen_model
            self._gen_progress = gen_progress
            self.gen_status_changed.emit(new_gen_status, new_gen_model, gen_progress)
        
        # Check embedding server (direct llama-server)
        new_embed_status, new_embed_model = self._check_direct_server(self.embed_url)
        embed_progress = self._parse_log_progress(self.embed_log_path, "embed") if self.embed_log_path else 0
        
        # DEBUG: Log what we detect
        logger.debug("[StatusMonitor] Embed: %s (%s) | Old: %s", new_embed_status, new_embed_model, self._embed_status)
        
        if (new_embed_status != self._embed_status or
            new_embed_model != self._embed_model or
            embed_progress != self._embed_progress):
            self._embed_status = new_embed_status
            self._embed_model = new_embed_model
            self._embed_progress = embed_progress
            self.embed_status_changed.emit(new_embed_status, new_embed_model, embed_progress)
        
        # Check if both ready
        if self.is_ready():
            self.all_ready.emit()
    
    def _check_llama_swap(self, url: str) -> tuple[str, str]:
        """Check llama-swap server status.
        
        /v1/models returns models from CONFIG (always populated).
        /upstream returns actually RUNNING models (empty until loaded).
        
        Returns (status, current_model_name)
        """
        try:
            # First check health
            health_resp = requests.get(f"{url}/health", timeout=2)
            
            if health_resp.status_code == 503:
                # Model is loading
                return (ServerStatus.LOADING, "")
            
            if health_resp.status_code != 200:
                return (ServerStatus.DOWN, "")
            
            # Health OK - check /upstream for ACTUALLY loaded models
            try:
                upstream_resp = requests.get(f"{url}/upstream", timeout=2)
                if upstream_resp.status_code == 200:
                    data = upstream_resp.json()
                    running = data.get("running", [])
                    if running:
                        # Model is actually loaded and running
                        model_id = running[0].get("model", "unknown") if isinstance(running[0], dict) else str(running[0])
                        return (ServerStatus.READY, model_id)
                    else:
                        # Proxy up, but no model loaded yet
                        # Model will load on first query
                        return (ServerStatus.PROXY_ONLY, "")
            except Exception:
                pass
            
            # Fallback: check /v1/models for config (but this is unreliable)
            try:
                models_resp = requests.get(f"{url}/v1/models", timeout=2)
                if models_resp.status_code == 502:
                    return (ServerStatus.LOADING, "")
            except Exception:
                pass
            
            # Proxy up but unclear
            return (ServerStatus.PROXY_ONLY, "")
            
        except requests.exceptions.ConnectionError:
            return (ServerStatus.DOWN, "")
        except Exception:
            return (ServerStatus.DOWN, "")
    
    def _check_direct_server(self, url: str) -> tuple[str, str]:
        """Check direct llama-server status (no proxy).
        
        llama-server returns /health 200 immediately, even before model loads.
        Must check /v1/models - empty list means still loading.
        """
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            
            if resp.status_code == 503:
                # Explicitly loading
                return (ServerStatus.LOADING, "")
            
            if resp.status_code == 200:
                # Health OK, but need to check if model is actually loaded
                try:
                    models_resp = requests.get(f"{url}/v1/models", timeout=2)
                    if models_resp.status_code == 200:
                        data = models_resp.json()
                        models = data.get("data", [])
                        if models:
                            # Model loaded - READY
                            return (ServerStatus.READY, models[0].get("id", "embedding"))
                        else:
                            # Empty model list - still LOADING
                            return (ServerStatus.LOADING, "")
                except Exception:
                    pass
                # Couldn't get models - assume still loading
                return (ServerStatus.LOADING, "")
            
            return (ServerStatus.DOWN, "")
            
        except requests.exceptions.ConnectionError:
            return (ServerStatus.DOWN, "")
        except Exception:
            return (ServerStatus.DOWN, "")
    
    def _parse_log_progress(self, log_path: Path, server_type: str) -> int:
        """Parse log file for loading progress.
        
        Looks for patterns like "offloading X/Y layers" or "loaded X/Y tensors"
        
        Returns percentage 0-100
        """
        if not log_path or not log_path.exists():
            return 0
            
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            # Look for layer offloading progress: "offloaded 33/33 layers"
            layer_match = re.search(r"offload(?:ing|ed)\s+(\d+)/(\d+)\s+(?:repeating\s+)?layers", content, re.IGNORECASE)
            if layer_match:
                current = int(layer_match.group(1))
                total = int(layer_match.group(2))
                if total > 0:
                    return min(100, int((current / total) * 100))
            
            # Alternative: tensor loading "loaded X tensors"
            if "model loaded" in content.lower():
                return 100
            
            if "loading model" in content.lower():
                return 10  # Started but no progress yet
                
        except Exception:
            pass
        
        return 0


# Singleton instance
_monitor: Optional[ServerStatusMonitor] = None


def get_status_monitor(
    gen_url: str = "http://127.0.0.1:8000",
    embed_url: str = "http://127.0.0.1:8001",
) -> ServerStatusMonitor:
    """Get or create the global status monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = ServerStatusMonitor(gen_url=gen_url, embed_url=embed_url)
    return _monitor
