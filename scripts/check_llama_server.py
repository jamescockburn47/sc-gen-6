"""Check if llama.cpp server is running and ready."""
import sys
import time
import requests
from pathlib import Path

# Add parent directory to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.runtime_store import load_runtime_state


def check_server_health(base_url: str, timeout: int = 60) -> bool:
    """
    Check if llama.cpp server is healthy and model is loaded.
    
    Args:
        base_url: Base URL of the server (e.g., http://127.0.0.1:8000/v1)
        timeout: Maximum seconds to wait for server
        
    Returns:
        True if server is healthy, False otherwise
    """
    # Strip /v1 suffix if present
    base = base_url.replace("/v1", "").rstrip("/")
    
    # Try both /health and /v1/models endpoints
    endpoints = [
        f"{base}/health",
        f"{base}/v1/models",
    ]
    
    start_time = time.time()
    attempt = 0
    
    print(f"[INFO] Checking llama.cpp server at {base}...")
    
    while time.time() - start_time < timeout:
        attempt += 1
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=5)
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    print(f"[OK] Server is ready! (took {elapsed:.1f}s, {attempt} attempts)")
                    return True
            except requests.exceptions.RequestException:
                pass  # Server not ready yet
        
        # Show progress every 5 seconds
        if attempt % 5 == 0:
            elapsed = time.time() - start_time
            print(f"[WAIT] Still waiting for server... ({elapsed:.0f}s elapsed)")
        
        time.sleep(1)
    
    print(f"[ERROR] Server did not become ready within {timeout} seconds")
    return False


def main() -> int:
    """Main entry point."""
    # Load runtime configuration
    state = load_runtime_state()
    base_url = state.get("base_url", "http://127.0.0.1:8000/v1")
    
    # Check if server is ready
    if check_server_health(base_url):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
