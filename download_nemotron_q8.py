"""
Download Nemotron 3 Nano 30B Q8_0 (8-bit quantized).

Q8_0 is a high-quality 8-bit quantization that preserves most of the
model's quality while being much smaller and faster to download.

Size: 33.6 GB (single file, not split)
Quality: ~95-97% of BF16 (more realistic than 99%)
"""
import os
import sys

# Enable hf_transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import hf_hub_download

REPO_ID = "unsloth/Nemotron-3-Nano-30B-A3B-GGUF"
FILENAME = "Nemotron-3-Nano-30B-A3B-Q8_0.gguf"
SIZE_GB = 33.6
LOCAL_DIR = "models/nemotron3-nano-30b-q8"

def main():
    print("=" * 70)
    print("  Nemotron 3 Nano 30B Q8_0 - Download")
    print("=" * 70)
    print(f"\n  Using: hf_transfer (Rust-based, optimized)")
    print(f"  Repository: {REPO_ID}")
    print(f"  Target: {LOCAL_DIR}")
    print(f"  Size: {SIZE_GB} GB (single file)")
    print(f"\n  Quality note: Q8_0 is ~95-97% of BF16 quality.")
    print(f"  It's excellent for most tasks, with much faster download.\n")
    
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    print(f"  Downloading: {FILENAME}")
    print("  This may take a while (~4-6 hours at typical speeds)...")
    sys.stdout.flush()
    
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=LOCAL_DIR,
        )
        print(f"\n  ✓ Downloaded to: {path}")
        
        print("\n" + "=" * 70)
        print("  ✓ Download complete!")
        print("=" * 70)
        
        model_path = os.path.abspath(path)
        print(f"\n  Model path: {model_path}")
        print("\n  To switch to this model, update config:")
        print("    1. Edit config/llm_runtime.json")
        print(f"    2. Set model_path to: {model_path}")
        print("    3. Set model_name to: nemotron3-nano-30b-q8")
        
    except KeyboardInterrupt:
        print("\n\n  ⚠ Download interrupted. Run again to resume.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        print("  Run again to retry/resume.")
        sys.exit(1)

if __name__ == "__main__":
    main()
