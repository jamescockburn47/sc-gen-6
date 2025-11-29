import sys
import importlib

required_packages = [
    "fastapi",
    "uvicorn",
    "python_multipart", # import name for python-multipart
    "yaml", # PyYAML
    "chromadb",
    "sentence_transformers",
    "torch",
    "numpy",
    "requests",
    "onnxruntime"
]

print(f"Checking dependencies with Python: {sys.executable}")
missing = []
for package in required_packages:
    try:
        importlib.import_module(package)
        print(f"✅ {package}")
    except ImportError:
        print(f"❌ {package} NOT FOUND")
        missing.append(package)

if missing:
    print(f"\nMissing packages: {', '.join(missing)}")
    sys.exit(1)
else:
    print("\nAll core dependencies are installed!")
    sys.exit(0)
