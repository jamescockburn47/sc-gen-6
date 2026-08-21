"""Settings slide-over panel — LLM config, API keys, matter selection.

Opens as an overlay from the right edge. Ctrl+, to toggle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.ui7.theme import C, S, T

PRESETS_FILE = Path("config/model_presets.json")
RUNTIME_FILE = Path("config/llm_runtime.json")

PROVIDER_DEFAULTS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "default_model": "glm-4.7-flash",
    },
    "llama_cpp": {
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "local-llama",
        "default_model": "nemotron-3-nano",
    },
}


def _load_presets() -> List[Dict[str, Any]]:
    try:
        if PRESETS_FILE.exists():
            with PRESETS_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _load_runtime() -> Dict[str, Any]:
    try:
        if RUNTIME_FILE.exists():
            with RUNTIME_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"provider": "llama_cpp", "model_name": "nemotron-3-nano"}


def _save_runtime(data: Dict[str, Any]) -> None:
    RUNTIME_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Preserve existing keys (like _comments)
    existing = _load_runtime()
    existing.update(data)
    with RUNTIME_FILE.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


class SettingsPanel(QWidget):
    """Slide-over settings panel (right edge)."""

    PANEL_WIDTH = 400

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(self.PANEL_WIDTH)
        self.setStyleSheet(f"""
            QWidget#settingsPanel {{
                background: {C.BG_SURFACE};
                border-left: 1px solid {C.BORDER};
            }}
        """)
        self.setObjectName("settingsPanel")
        self._presets = _load_presets()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet(f"""
            background: {C.BG_SURFACE};
            border-bottom: 1px solid {C.BORDER};
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(S.LG, S.MD, S.LG, S.MD)

        title = QLabel("Settings")
        title.setProperty("class", "heading")
        h_layout.addWidget(title)
        h_layout.addStretch(1)

        close_btn = QPushButton("\u2715")  # ✕
        close_btn.setProperty("class", "ghost")
        close_btn.setFixedSize(28, 28)
        close_btn.clicked.connect(self.hide)
        h_layout.addWidget(close_btn)

        layout.addWidget(header)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background: {C.BG_SURFACE}; }}")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(S.LG, S.LG, S.LG, S.LG)
        content_layout.setSpacing(S.LG)

        # ----- Quick Model Switch -----
        preset_group = QGroupBox("Quick Model Switch")
        preset_layout = QVBoxLayout(preset_group)
        preset_layout.setSpacing(S.SM)

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("— select preset —", None)
        for preset in self._presets:
            self.preset_combo.addItem(preset["label"], preset)
        preset_layout.addWidget(self.preset_combo)

        apply_btn = QPushButton("Apply & Save Preset")
        apply_btn.setObjectName("apply_preset_btn")
        apply_btn.setProperty("class", "primary")
        apply_btn.clicked.connect(self._apply_preset)
        preset_layout.addWidget(apply_btn)

        self.preset_status = QLabel("")
        self.preset_status.setWordWrap(True)
        self.preset_status.setStyleSheet(f"color: {C.ACCENT}; font-size: 11px; background: transparent;")
        preset_layout.addWidget(self.preset_status)

        content_layout.addWidget(preset_group)

        # ----- LLM Configuration -----
        llm_group = QGroupBox("LLM Configuration")
        llm_layout = QVBoxLayout(llm_group)
        llm_layout.setSpacing(S.SM)

        # Provider
        llm_layout.addWidget(QLabel("Provider"))
        self.backend_combo = QComboBox()
        self.backend_combo.setObjectName("provider_combo")
        self.backend_combo.addItems(["llama_cpp (llama-swap)", "ollama"])
        self.backend_combo.currentIndexChanged.connect(self._on_provider_changed)
        llm_layout.addWidget(self.backend_combo)

        # Provider info label
        self.provider_info = QLabel("")
        self.provider_info.setWordWrap(True)
        self.provider_info.setStyleSheet(f"color: {C.TEXT_TERTIARY}; font-size: 10px; background: transparent;")
        llm_layout.addWidget(self.provider_info)

        # Model
        llm_layout.addWidget(QLabel("Model Name"))
        self.model_input = QLineEdit()
        self.model_input.setObjectName("model_name_input")
        self.model_input.setPlaceholderText("e.g. nemotron-3-nano or glm-4.7-flash")
        llm_layout.addWidget(self.model_input)

        # Temperature
        temp_row = QHBoxLayout()
        temp_row.addWidget(QLabel("Temperature"))
        self.temp_label = QLabel("0.3")
        self.temp_label.setStyleSheet(f"color: {C.ACCENT}; background: transparent;")
        temp_row.addStretch(1)
        temp_row.addWidget(self.temp_label)
        llm_layout.addLayout(temp_row)

        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 20)
        self.temp_slider.setValue(3)
        self.temp_slider.valueChanged.connect(
            lambda v: self.temp_label.setText(f"{v / 10:.1f}")
        )
        llm_layout.addWidget(self.temp_slider)

        # Context length
        llm_layout.addWidget(QLabel("Context Length"))
        self.ctx_combo = QComboBox()
        self.ctx_combo.addItems(["8192", "16384", "32768", "65536", "131072"])
        self.ctx_combo.setCurrentText("32768")
        llm_layout.addWidget(self.ctx_combo)

        content_layout.addWidget(llm_group)

        # ----- API Keys -----
        api_group = QGroupBox("API Keys")
        api_layout = QVBoxLayout(api_group)
        api_layout.setSpacing(S.SM)

        api_layout.addWidget(QLabel("Isaacus (Kanon 2 Enricher)"))
        self.isaacus_key = QLineEdit()
        self.isaacus_key.setPlaceholderText("iuak_v1_...")
        self.isaacus_key.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        api_layout.addWidget(self.isaacus_key)

        content_layout.addWidget(api_group)

        # ----- Retrieval -----
        retrieval_group = QGroupBox("Retrieval")
        retrieval_layout = QVBoxLayout(retrieval_group)
        retrieval_layout.setSpacing(S.SM)

        retrieval_layout.addWidget(QLabel("Top-K Results"))
        self.topk_combo = QComboBox()
        self.topk_combo.addItems(["3", "5", "8", "10", "15"])
        self.topk_combo.setCurrentText("5")
        retrieval_layout.addWidget(self.topk_combo)

        self.rerank_check = QCheckBox("Enable reranking")
        self.rerank_check.setChecked(True)
        retrieval_layout.addWidget(self.rerank_check)

        self.enrich_check = QCheckBox("Auto-enrich with Kanon 2")
        self.enrich_check.setChecked(True)
        retrieval_layout.addWidget(self.enrich_check)

        content_layout.addWidget(retrieval_group)

        # ----- Matter -----
        matter_group = QGroupBox("Matter")
        matter_layout = QVBoxLayout(matter_group)
        matter_layout.setSpacing(S.SM)

        matter_layout.addWidget(QLabel("Active Matter"))
        self.matter_combo = QComboBox()
        self.matter_combo.addItem("Default")
        matter_layout.addWidget(self.matter_combo)

        new_matter_btn = QPushButton("New Matter")
        matter_layout.addWidget(new_matter_btn)

        content_layout.addWidget(matter_group)

        content_layout.addStretch(1)

        # Save button
        save_btn = QPushButton("Save Settings")
        save_btn.setObjectName("save_settings_btn")
        save_btn.setProperty("class", "primary")
        save_btn.clicked.connect(self._save_settings)
        content_layout.addWidget(save_btn)

        self.save_status = QLabel("")
        self.save_status.setStyleSheet(f"color: {C.SUCCESS}; font-size: 11px; background: transparent;")
        content_layout.addWidget(self.save_status)

        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        # Load current settings
        self._load_settings()

    def _on_provider_changed(self) -> None:
        """Update info label when provider changes."""
        provider = self._current_provider()
        if provider == "llama_cpp":
            self.provider_info.setText(
                "llama-swap on :8000 — Nemotron 3 Nano BF16\n"
                "Speculative decoding enabled (~25-30 t/s)"
            )
            if not self.model_input.text() or self.model_input.text() == "glm-4.7-flash":
                self.model_input.setText("nemotron-3-nano")
        else:
            self.provider_info.setText(
                "Ollama on :11434 — GLM 4.7 Flash\n"
                "Flash Attention + Vulkan backend (~40 t/s)"
            )
            if not self.model_input.text() or self.model_input.text() == "nemotron-3-nano":
                self.model_input.setText("glm-4.7-flash")

    def _current_provider(self) -> str:
        """Return 'llama_cpp' or 'ollama' from the combo."""
        text = self.backend_combo.currentText()
        return "llama_cpp" if "llama" in text.lower() else "ollama"

    def _apply_preset(self) -> None:
        """Apply selected preset and immediately save."""
        preset = self.preset_combo.currentData()
        if not preset:
            self.preset_status.setText("Select a preset first.")
            return

        provider = preset.get("provider", "llama_cpp")
        # Set backend combo
        if provider == "llama_cpp":
            self.backend_combo.setCurrentIndex(0)
        else:
            self.backend_combo.setCurrentIndex(1)

        self.model_input.setText(preset.get("model_name", ""))
        self._save_settings()
        self.preset_status.setText(f"✓ Applied: {preset['label']}")

    def _load_settings(self) -> None:
        """Load settings from config files."""
        try:
            data = _load_runtime()
            model = data.get("model_name", "nemotron-3-nano")
            self.model_input.setText(model)
            temp = data.get("temperature", 0.3)
            self.temp_slider.setValue(int(temp * 10))
            ctx = data.get("n_ctx", 32768)
            idx = self.ctx_combo.findText(str(ctx))
            if idx >= 0:
                self.ctx_combo.setCurrentIndex(idx)

            # Set provider combo
            provider = data.get("provider", "llama_cpp")
            if provider == "ollama":
                self.backend_combo.setCurrentIndex(1)
            else:
                self.backend_combo.setCurrentIndex(0)
            self._on_provider_changed()
        except Exception:
            pass

        # Load Isaacus key from .env
        try:
            env_path = Path(".env")
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("ISAACUS_API_KEY="):
                        self.isaacus_key.setText(line.split("=", 1)[1].strip())
                        break
        except Exception:
            pass

    def _save_settings(self) -> None:
        """Persist settings to config files."""
        try:
            provider = self._current_provider()
            defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["llama_cpp"])
            model = self.model_input.text().strip() or defaults["default_model"]

            _save_runtime({
                "provider": provider,
                "base_url": defaults["base_url"],
                "api_key": defaults["api_key"],
                "model_name": model,
                "temperature": self.temp_slider.value() / 10.0,
                "n_ctx": int(self.ctx_combo.currentText()),
            })
            self.save_status.setText(f"✓ Saved — {provider} / {model}")
        except Exception as e:
            self.save_status.setText(f"✗ Save failed: {e}")

        # Save Isaacus key to .env
        try:
            key = self.isaacus_key.text().strip()
            if key:
                env_path = Path(".env")
                lines = env_path.read_text().splitlines() if env_path.exists() else []
                lines = [l for l in lines if not l.startswith("ISAACUS_API_KEY=")]
                lines.append(f"ISAACUS_API_KEY={key}")
                env_path.write_text("\n".join(lines) + "\n")
        except Exception:
            pass

    def resizeEvent(self, event) -> None:
        """Position at right edge of parent."""
        super().resizeEvent(event)
        if self.parent():
            parent = self.parent()
            self.setGeometry(
                parent.width() - self.PANEL_WIDTH,
                0,
                self.PANEL_WIDTH,
                parent.height(),
            )

    def show(self) -> None:
        """Show panel at right edge."""
        if self.parent():
            parent = self.parent()
            self.setGeometry(
                parent.width() - self.PANEL_WIDTH,
                0,
                self.PANEL_WIDTH,
                parent.height(),
            )
        super().show()
