"""
Download Nemotron 3 Nano 30B BF16 (unquantized) GGUF with THROTTLED speed.

Uses requests with chunked download and speed limiting to prevent
connection issues on some networks.

Total size: ~63.2 GB (split into 2 files)
- Part 1: 49.9 GB  
- Part 2: 13.3 GB

Speed limit: ~50 MB/s (configurable below)
"""
import os
import sys
import time
import requests
from pathlib import Path

# Configuration
SPEED_LIMIT_MBPS = 50  # MB/s - adjust this if needed (lower = more stable)
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB chunks
RETRY_DELAY = 5  # seconds between retries
MAX_RETRIES = 10

# Files to download
FILES = [
    {
        "url": "https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF/resolve/main/BF16/Nemotron-3-Nano-30B-A3B-BF16-00001-of-00002.gguf",
        "filename": "BF16/Nemotron-3-Nano-30B-A3B-BF16-00001-of-00002.gguf",
        "size_gb": 49.9,
    },
    {
        "url": "https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF/resolve/main/BF16/Nemotron-3-Nano-30B-A3B-BF16-00002-of-00002.gguf",
        "filename": "BF16/Nemotron-3-Nano-30B-A3B-BF16-00002-of-00002.gguf",
        "size_gb": 13.3,
    },
]

LOCAL_DIR = Path("models/nemotron3-nano-30b-bf16")


def format_size(bytes_val):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} TB"


def format_time(seconds):
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m"


def download_file_throttled(url: str, dest_path: Path, expected_size_gb: float):
    """Download a file with speed throttling and resume support."""
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".downloading")
    
    # Check for existing partial download
    initial_pos = 0
    if temp_path.exists():
        initial_pos = temp_path.stat().st_size
        print(f"  Resuming from {format_size(initial_pos)}")
    
    expected_size = int(expected_size_gb * 1024 * 1024 * 1024)
    
    # Check if already complete
    if dest_path.exists():
        existing_size = dest_path.stat().st_size
        if existing_size >= expected_size * 0.99:  # Allow 1% tolerance
            print(f"  ✓ Already downloaded: {format_size(existing_size)}")
            return True
    
    headers = {}
    if initial_pos > 0:
        headers["Range"] = f"bytes={initial_pos}-"
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Connecting... (attempt {attempt + 1}/{MAX_RETRIES})")
            
            response = session.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get total size from response
            content_length = response.headers.get('content-length')
            if content_length:
                total_size = initial_pos + int(content_length)
            else:
                total_size = expected_size
            
            # Speed limiting setup
            bytes_per_second = SPEED_LIMIT_MBPS * 1024 * 1024
            chunk_delay = CHUNK_SIZE / bytes_per_second
            
            downloaded = initial_pos
            start_time = time.time()
            last_print = start_time
            
            mode = "ab" if initial_pos > 0 else "wb"
            with open(temp_path, mode) as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Speed limiting
                        elapsed_chunk = time.time() - start_time
                        expected_time = (downloaded - initial_pos) / bytes_per_second
                        if elapsed_chunk < expected_time:
                            time.sleep(expected_time - elapsed_chunk)
                        
                        # Progress update every 2 seconds
                        now = time.time()
                        if now - last_print >= 2:
                            elapsed = now - start_time
                            speed = (downloaded - initial_pos) / elapsed / 1024 / 1024
                            progress = downloaded / total_size * 100
                            remaining = (total_size - downloaded) / (bytes_per_second) if speed > 0 else 0
                            
                            print(f"\r  Progress: {format_size(downloaded)} / {format_size(total_size)} "
                                  f"({progress:.1f}%) | {speed:.1f} MB/s | ETA: {format_time(remaining)}    ", 
                                  end="", flush=True)
                            last_print = now
            
            print()  # New line after progress
            
            # Rename to final path
            if temp_path.exists():
                if dest_path.exists():
                    dest_path.unlink()
                temp_path.rename(dest_path)
            
            print(f"  ✓ Download complete: {format_size(downloaded)}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"\n  ⚠ Connection error: {e}")
            print(f"  Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
            
            # Update initial_pos for resume
            if temp_path.exists():
                initial_pos = temp_path.stat().st_size
                headers["Range"] = f"bytes={initial_pos}-"
            
        except KeyboardInterrupt:
            print("\n\n  ⚠ Download interrupted. Run again to resume.")
            return False
    
    print(f"  ✗ Failed after {MAX_RETRIES} attempts")
    return False


def main():
    print("=" * 70)
    print("  Nemotron 3 Nano 30B BF16 (Unquantized) - THROTTLED Download")
    print("=" * 70)
    print(f"\n  Speed limit: {SPEED_LIMIT_MBPS} MB/s")
    print(f"  Target: {LOCAL_DIR.absolute()}")
    print(f"  Total size: ~63.2 GB\n")
    
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    
    success = True
    for i, file_info in enumerate(FILES, 1):
        print(f"\n{'='*70}")
        print(f"  [{i}/{len(FILES)}] {file_info['filename']}")
        print(f"  Size: {file_info['size_gb']} GB")
        print("=" * 70)
        
        dest_path = LOCAL_DIR / file_info["filename"]
        if not download_file_throttled(file_info["url"], dest_path, file_info["size_gb"]):
            success = False
            break
    
    if success:
        print("\n" + "=" * 70)
        print("  ✓ All downloads complete!")
        print("=" * 70)
        
        model_path = LOCAL_DIR.absolute() / "BF16" / "Nemotron-3-Nano-30B-A3B-BF16-00001-of-00002.gguf"
        print(f"\n  Model path for llama.cpp:")
        print(f"    {model_path}")
        print("\n  To switch to this model:")
        print("    copy config\\llm_runtime_bf16.json config\\llm_runtime.json")
    else:
        print("\n  ⚠ Download incomplete. Run again to resume.")
        sys.exit(1)


if __name__ == "__main__":
    main()
