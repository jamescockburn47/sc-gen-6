import sys
import onnxruntime as ort
import torch

print(f"Python: {sys.version}")
print("-" * 20)

print("ONNX Runtime Providers:")
try:
    if hasattr(ort, 'get_available_providers'):
        providers = ort.get_available_providers()
        print(providers)
        if 'DmlExecutionProvider' in providers:
            print("✅ DirectML (GPU) is available for ONNX.")
        elif 'CUDAExecutionProvider' in providers:
            print("✅ CUDA (GPU) is available for ONNX.")
        else:
            print("⚠️  No GPU provider found for ONNX. Using CPU.")
    else:
        print("⚠️  ort.get_available_providers() not found. ONNX Runtime might be too old or minimal.")
except Exception as e:
    print(f"Error checking ONNX: {e}")

print("-" * 20)

print("PyTorch Device:")
try:
    if torch.cuda.is_available():
        print(f"✅ CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        # Check for DirectML if torch-directml was installed (it failed earlier, but let's check import)
        try:
            import torch_directml
            device = torch_directml.device()
            print(f"✅ DirectML is available: {device}")
        except ImportError:
            print("⚠️  No GPU found for PyTorch (CUDA or DirectML). Using CPU.")
except Exception as e:
    print(f"Error checking PyTorch: {e}")
