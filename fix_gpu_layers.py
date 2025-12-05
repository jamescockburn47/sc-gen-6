from src.config.runtime_store import load_runtime_state, save_runtime_state, RUNTIME_FILE
import json

def fix_gpu_layers():
    print(f"Target file: {RUNTIME_FILE.absolute()}")
    
    if RUNTIME_FILE.exists():
        with open(RUNTIME_FILE, "r") as f:
            data = json.load(f)
    else:
        print("Config file not found!")
        return
        
    # Reduce GPU layers to a safe starting point
    # 80B model has around 80 layers. 999 is usually fine as "all", 
    # but maybe it's causing issues with the specific backend or VRAM fragmentation.
    # Let's try 80 explicitly, or maybe 60 to be safe and see if it starts.
    # Actually, the error might be "out of memory" or "failed to allocate".
    # Let's set it to 60 for now to verify startup.
    
    current_layers = data.get("llama_server", {}).get("gpu_layers")
    print(f"Current gpu_layers: {current_layers}")
    
    if "llama_server" not in data:
        data["llama_server"] = {}
        
    data["llama_server"]["gpu_layers"] = 40
    
    with open(RUNTIME_FILE, "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"Updated gpu_layers to 40.")

if __name__ == "__main__":
    fix_gpu_layers()
