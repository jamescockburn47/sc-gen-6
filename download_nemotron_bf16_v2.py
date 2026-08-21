"""
Download Nemotron 3 Nano 30B BF16 (unquantized) GGUF files.

Total size: ~63.2 GB (split into 2 files)
- Part 1: 49.9 GB
- Part 2: 13.3 GB

Run this script directly: python download_nemotron_bf16_v2.py
"""
from huggingface_hub import hf_hub_download
import os
import sys

REPO_ID = "unsloth/Nemotron-3-Nano-30B-A3B-GGUF"
FILES = [
    ("BF16/Nemotron-3-Nano-30B-A3B-BF16-00001-of-00002.gguf", "49.9 GB"),
    ("BF16/Nemotron-3-Nano-30B-A3B-BF16-00002-of-00002.gguf", "13.3 GB"),
]
LOCAL_DIR = "models/nemotron3-nano-30b-bf16"

def main():
    print("=" * 60)
    print("Downloading Nemotron 3 Nano 30B BF16 (Unquantized)")
    print("=" * 60)
    print(f"\nRepository: {REPO_ID}")
    print(f"Target: {LOCAL_DIR}")
    print(f"Total size: ~63.2 GB\n")
    
    # Create target directory
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    for i, (filename, size) in enumerate(FILES, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(FILES)}] Downloading: {filename}")
        print(f"Size: {size}")
        print("="*60)
        print("\nDownloading (this may take a while)...")
        sys.stdout.flush()  # Force output
        
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=LOCAL_DIR,
            )
            print(f"\n✓ Successfully downloaded to: {path}")
        except KeyboardInterrupt:
            print("\n\n⚠ Download interrupted. Run again to resume.")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ Error downloading {filename}: {e}")
            print("Run the script again to retry/resume.")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ Download complete!")
    print("=" * 60)
    
    # Show the expected path for llama.cpp
    model_path = os.path.join(
        os.path.abspath(LOCAL_DIR), 
        "BF16", 
        "Nemotron-3-Nano-30B-A3B-BF16-00001-of-00002.gguf"
    )
    print(f"\nModel path for llama.cpp:")
    print(f"  {model_path}")
    print("\nllama.cpp will automatically load both split files.")
    
    print("\nTo switch to this model, copy the runtime config:")
    print("  copy config\\llm_runtime_bf16.json config\\llm_runtime.json")

if __name__ == "__main__":
    main()
