"""
Download Nemotron 3 Nano 30B BF16 (unquantized) GGUF files from Unsloth.

Total size: ~63.2 GB (split into 2 files)
- Part 1: 49.9 GB
- Part 2: 13.3 GB

This downloads from: unsloth/Nemotron-3-Nano-30B-A3B-GGUF/BF16/
"""
from huggingface_hub import hf_hub_download
import os
import sys

REPO_ID = "unsloth/Nemotron-3-Nano-30B-A3B-GGUF"
FILES = [
    "BF16/Nemotron-3-Nano-30B-A3B-BF16-00001-of-00002.gguf",
    "BF16/Nemotron-3-Nano-30B-A3B-BF16-00002-of-00002.gguf",
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
    
    for i, filename in enumerate(FILES, 1):
        print(f"\n[{i}/{len(FILES)}] Downloading: {filename}")
        print("This may take a while depending on your connection...")
        
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"✓ Saved to: {path}")
        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    
    # Show the expected path for llama.cpp
    model_path = os.path.join(
        os.path.abspath(LOCAL_DIR), 
        "BF16", 
        "Nemotron-3-Nano-30B-A3B-BF16-00001-of-00002.gguf"
    )
    print(f"\nTo use with llama.cpp, point to the first split file:")
    print(f"  {model_path}")
    print("\nllama.cpp will automatically load the second part.")
    
    print("\nNOTE: BF16 uses ~63 GB VRAM. You may need to:")
    print("  - Reduce context size (e.g., 32768 instead of 131072)")
    print("  - Use aggressive KV cache quantization (q4_0)")
    print("  - Or accept slower inference if offloading to RAM")

if __name__ == "__main__":
    main()
