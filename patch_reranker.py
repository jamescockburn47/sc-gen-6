from pathlib import Path

src = Path("temp_reranker.py")
dst = Path("src/models/reranker/reranker_service_onnx.py")

content = src.read_text(encoding="utf-8")

# 1. Remove immediate load
content = content.replace('        self.session = None\n        self._load_model()', '        self.session = None\n        # Lazy load: self._load_model() is now called on first use')

# 2. Add ensure_model_loaded method
ensure_method = '''    def _ensure_model_loaded(self) -> None:
        """Ensure model and tokenizer are loaded."""
        if self.session is None or self.tokenizer is None:
            self._load_model()

    def rerank('''
content = content.replace('    def rerank(', ensure_method)

# 3. Call ensure_model_loaded in rerank
content = content.replace('        if not chunks:\n            return []', '        if not chunks:\n            return []\n\n        self._ensure_model_loaded()')

dst.write_text(content, encoding="utf-8")
print("Patched reranker service.")
