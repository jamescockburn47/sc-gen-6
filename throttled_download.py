import urllib.request
import time
import os
import sys

url = "https://huggingface.co/bartowski/Qwen_Qwen3-Next-80B-A3B-Instruct-GGUF/resolve/main/Qwen_Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf"
filename = "Qwen_Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf"
limit_mb_s = 45
chunk_size = 1024 * 1024  # 1 MB
target_time_per_chunk = 1.0 / limit_mb_s

print(f"Downloading {filename} with limit {limit_mb_s} MB/s...")

# Check for existing file to resume
req = urllib.request.Request(url)
mode = 'wb'
existing_size = 0

if os.path.exists(filename):
    existing_size = os.path.getsize(filename)
    # Check if file is already complete (optional, but good)
    # For now, just assume we resume if it exists.
    req.add_header('Range', f'bytes={existing_size}-')
    mode = 'ab'
    print(f"Resuming from {existing_size / (1024*1024):.2f} MB")

try:
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get('content-length', 0))
        if mode == 'ab':
            total_size += existing_size
            
        start_time = time.time()
        downloaded = 0
        if mode == 'ab':
            downloaded = existing_size

        with open(filename, mode) as f:
            while True:
                chunk_start = time.time()
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                
                f.write(chunk)
                downloaded += len(chunk)
                
                # Throttle
                chunk_end = time.time()
                duration = chunk_end - chunk_start
                if duration < target_time_per_chunk:
                    time.sleep(target_time_per_chunk - duration)
                
                # Progress (every ~100MB)
                if downloaded % (100 * 1024 * 1024) < chunk_size:
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    percent = (downloaded / total_size) * 100 if total_size else 0
                    elapsed = time.time() - start_time
                    # Avoid div by zero
                    if elapsed > 0:
                        speed = (downloaded - (existing_size if mode=='ab' else 0)) / elapsed / (1024 * 1024)
                    else:
                        speed = 0
                    print(f"\rProgress: {percent:.2f}% ({mb_downloaded:.0f}/{mb_total:.0f} MB) Speed: {speed:.2f} MB/s", end="")

    print("\nDownload complete!")

except urllib.error.HTTPError as e:
    if e.code == 416: # Range Not Satisfiable (likely already complete)
        print("\nFile likely already complete (Server returned 416).")
    else:
        print(f"\nHTTP Error: {e}")
except Exception as e:
    print(f"\nError: {e}")
