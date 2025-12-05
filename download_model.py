from huggingface_hub import hf_hub_download
import sys

repo_id = "bartowski/Qwen_Qwen3-Next-80B-A3B-Instruct-GGUF"
filename = "Qwen_Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf"

print(f"Downloading {filename} from {repo_id}...")
print("This may take a while (48GB)...")

try:
    path = hf_hub_download(
        repo_id=repo_id, 
        filename=filename, 
        local_dir=".", 
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"Successfully downloaded to {path}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
