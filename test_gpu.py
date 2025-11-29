"""Quick GPU verification test."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

print("=" * 80)
print("GPU VERIFICATION TEST")
print("=" * 80)

# Test 1: ONNX Runtime DirectML
print("\n[1/3] Checking ONNX Runtime DirectML...")
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"✓ Available providers: {providers}")
    if "DmlExecutionProvider" in providers:
        print("✓ DirectML GPU support ENABLED")
    else:
        print("✗ DirectML GPU support NOT AVAILABLE")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Reranker GPU
print("\n[2/3] Testing Reranker GPU...")
try:
    from src.models.reranker import get_reranker_service
    from src.config_loader import get_settings
    
    settings = get_settings()
    reranker = get_reranker_service(settings=settings)
    print(f"✓ Reranker loaded")
    print(f"  - Device: {getattr(reranker, 'device_label', 'Unknown')}")
    print(f"  - GPU Available: {getattr(reranker, 'is_gpu_available', lambda: False)()}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Embedding GPU
print("\n[3/3] Testing Embedding GPU...")
try:
    from src.retrieval import get_embedding_service
    from src.config_loader import get_settings
    
    settings = get_settings()
    embedder = get_embedding_service(settings=settings)
    print(f"✓ Embedder loaded")
    print(f"  - Device: {getattr(embedder, 'device_label', 'Unknown')}")
    print(f"  - GPU Available: {getattr(embedder, 'is_gpu_available', lambda: False)()}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("GPU VERIFICATION COMPLETE")
print("=" * 80)
