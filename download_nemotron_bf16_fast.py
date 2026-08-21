"""
Download Nemotron 3 Nano 30B BF16 using hf_transfer (MUCH FASTER).

hf_transfer is a Rust-based library that can be 5-10x faster than
standard downloads, especially for large files.

Total size: ~63.2 GB (split into 2 files)
"""
import os
import sys

# Enable hf_transfer BEFORE importing huggingface_hub
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import hf_hub_download

REPO_ID = "unsloth/Nemotron-3-Nano-30B-A3B-GGUF"
FILES = [
    ("BF16/Nemotron-3-Nano-30B-A3B-BF16-00001-of-00002.gguf", "49.9 GB"),
    ("BF16/Nemotron-3-Nano-30B-A3B-BF16-00002-of-00002.gguf", "13.3 GB"),
]
LOCAL_DIR = "models/nemotron3-nano-30b-bf16"

def main():
    print("=" * 70)
    print("  Nemotron 3 Nano 30B BF16 - FAST Download (hf_transfer)")
    print("=" * 70)
    print(f"\n  Using: hf_transfer (Rust-based, optimized for Hugging Face)")
    print(f"  Repository: {REPO_ID}")
    print(f"  Target: {LOCAL_DIR}")
    print(f"  Total size: ~63.2 GB\n")
    
    # Verify hf_transfer is enabled
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") != "1":
        print("  WARNING: HF_HUB_ENABLE_HF_TRANSFER not set!")
        print("  hf_transfer may not be used.")
    
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    for i, (filename, size) in enumerate(FILES, 1):
        print(f"\n{'='*70}")
        print(f"  [{i}/{len(FILES)}] {filename}")
        print(f"  Size: {size}")
        print("="*70)
        print("\n  Downloading with hf_transfer (should be much faster)...")
        sys.stdout.flush()
        
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=LOCAL_DIR,
            )
            print(f"\n  ✓ Downloaded to: {path}")
        except KeyboardInterrupt:
            print("\n\n  ⚠ Download interrupted. Run again to resume.")
            sys.exit(1)
        except Exception as e:
            print(f"\n  ✗ Error: {e}")
            print("  Run again to retry/resume.")
            sys.exit(1)
    
    print("\n" + "=" * 70)
    print("  ✓ All downloads complete!")
    print("=" * 70)
    
    model_path = os.path.join(
        os.path.abspath(LOCAL_DIR), 
        "BF16", 
        "Nemotron-3-Nano-30B-A3B-BF16-00001-of-00002.gguf"
    )
    print(f"\n  Model path: {model_path}")
    print("\n  To switch to this model:")
    print("    copy config\\llm_runtime_bf16.json config\\llm_runtime.json")

if __name__ == "__main__":
    main()
