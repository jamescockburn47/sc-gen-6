"""Top status bar — model switcher, GPU, connection indicators."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Dict, List

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QMessageBox, QWidget

from src.ui7.theme import C, S, T

PRESETS_FILE = Path("config/model_presets.json")
RUNTIME_FILE = Path("config/llm_runtime.json")

PROVIDER_DEFAULTS = {
    "ollama":    {"base_url": "http://localhost:11434/v1", "api_key": "ollama"},
    "llama_cpp": {"base_url": "http://127.0.0.1:8000/v1",  "api_key": "local-llama"},
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
    existing = _load_runtime()
    existing.update(data)
    with RUNTIME_FILE.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Return True if a TCP connection can be made to host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _llama_swap_port() -> int:
    """Read the listen port from llama-swap/config.yaml."""
    try:
        import yaml  # type: ignore
        cfg_path = Path("llama-swap/config.yaml")
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text())
            listen = cfg.get("listen", ":8000")
            return int(listen.split(":")[-1])
    except Exception:
        pass
    return 8000


class StatusDot(QWidget):
    """Tiny coloured dot indicator."""

    def __init__(self, color: str = C.TEXT_TERTIARY, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(8, 8)
        self._set_color(color)

    def _set_color(self, color: str) -> None:
        self.setStyleSheet(f"""
            QWidget {{
                background: {color};
                border-radius: 4px;
                border: none;
            }}
        """)
        self._color = color

    def set_color(self, color: str) -> None:
        self._set_color(color)


class StatusBar(QWidget):
    """Slim top bar: app name + model switcher + status indicators."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(38)
        self.setStyleSheet(f"""
            QWidget#topBar {{
                background-color: {C.BG_SURFACE};
                border-bottom: 1px solid {C.BORDER};
            }}
        """)
        self.setObjectName("topBar")
        self._presets = _load_presets()
        self._suppress_change = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(0)

        # App title — left
        title = QLabel("SCGen7")
        title.setStyleSheet(f"""
            color: {C.TEXT_PRIMARY};
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 1px;
            background: transparent;
            border: none;
        """)
        layout.addWidget(title)
        layout.addSpacing(16)

        # ── Model switcher dropdown ──────────────────────────────────────
        self.model_dot = StatusDot(C.ACCENT)
        layout.addWidget(self.model_dot)
        layout.addSpacing(6)

        self.model_combo = QComboBox()
        self.model_combo.setObjectName("modelSwitcher")
        self.model_combo.setStyleSheet(f"""
            QComboBox#modelSwitcher {{
                background: transparent;
                color: {C.TEXT_SECONDARY};
                border: none;
                font-size: 11px;
                font-weight: 500;
                padding: 0px 4px;
                min-width: 140px;
            }}
            QComboBox#modelSwitcher::drop-down {{
                border: none;
                width: 14px;
            }}
            QComboBox#modelSwitcher::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {C.TEXT_TERTIARY};
                width: 0;
                height: 0;
                margin-right: 4px;
            }}
            QComboBox#modelSwitcher QAbstractItemView {{
                background: {C.BG_RAISED};
                color: {C.TEXT_PRIMARY};
                border: 1px solid {C.BORDER};
                selection-background-color: {C.ACCENT};
                selection-color: white;
                font-size: 11px;
                padding: 4px;
            }}
        """)
        self._populate_model_combo()
        self.model_combo.currentIndexChanged.connect(self._on_model_selected)
        layout.addWidget(self.model_combo)

        layout.addSpacing(16)
        layout.addWidget(self._make_sep())
        layout.addSpacing(16)

        # GPU indicator
        self.gpu_dot = StatusDot(C.SUCCESS)
        self.gpu_label = self._make_indicator_label("96 GB Vulkan")
        layout.addWidget(self.gpu_dot)
        layout.addSpacing(6)
        layout.addWidget(self.gpu_label)

        layout.addSpacing(16)
        layout.addWidget(self._make_sep())
        layout.addSpacing(16)

        # Connection indicator
        self.conn_dot = StatusDot(C.SUCCESS)
        self.conn_label = self._make_indicator_label("Ollama")
        layout.addWidget(self.conn_dot)
        layout.addSpacing(6)
        layout.addWidget(self.conn_label)

        layout.addStretch(1)

        # Kanon 2 enricher badge (right side)
        self.enricher_dot = StatusDot(C.TEXT_TERTIARY)
        self.enricher_label = self._make_indicator_label("Kanon 2")
        layout.addWidget(self.enricher_dot)
        layout.addSpacing(4)
        layout.addWidget(self.enricher_label)

        layout.addSpacing(16)

        # Settings shortcut hint
        kbd = QLabel("Ctrl+,")
        kbd.setStyleSheet(f"""
            color: {C.TEXT_TERTIARY};
            font-size: 10px;
            font-family: "JetBrains Mono", "Consolas", monospace;
            background: {C.BG_RAISED};
            border: 1px solid {C.BORDER};
            border-radius: 3px;
            padding: 1px 6px;
        """)
        layout.addWidget(kbd)

        # Periodic refresh (every 15s)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh_status)
        self._timer.start(15_000)

    # ── Preset combo ──────────────────────────────────────────────────────

    def _populate_model_combo(self) -> None:
        """Fill the model combo from presets + current runtime."""
        self._suppress_change = True
        self.model_combo.clear()

        runtime = _load_runtime()
        current_model = runtime.get("model_name", "")

        best_idx = 0
        for i, preset in enumerate(self._presets):
            label = preset["label"]
            self.model_combo.addItem(label, preset)
            if preset.get("model_name") == current_model:
                best_idx = i

        self.model_combo.setCurrentIndex(best_idx)
        self._suppress_change = False

    def _on_model_selected(self, index: int) -> None:
        """Switch model when user picks from dropdown — checks backend first."""
        if self._suppress_change or index < 0:
            return
        preset = self.model_combo.itemData(index)
        if not preset:
            return

        provider = preset.get("provider", "llama_cpp")
        defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["llama_cpp"])
        model_name = preset.get("model_name", "")

        # ── Check target backend is reachable ──────────────────────────
        if provider == "ollama":
            reachable = _is_port_open("localhost", 11434)
            if not reachable:
                QMessageBox.warning(
                    self, "Ollama Not Running",
                    "Ollama is not running on port 11434.\n\n"
                    "Start it with:\n  sudo systemctl start ollama\n\n"
                    "Model switch cancelled."
                )
                # Revert combo to previous selection
                self._populate_model_combo()
                return
        else:
            # llama_cpp → llama-swap
            port = _llama_swap_port()
            reachable = _is_port_open("127.0.0.1", port)
            if not reachable:
                reply = QMessageBox.question(
                    self, "llama-swap Not Running",
                    f"llama-swap is not running on port {port}.\n\n"
                    "Start it now? (The model will load in the background — "
                    "this may take 1-2 minutes for the 30B model.)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    try:
                        import launch
                        ok = launch.start_llama_swap()
                        if not ok:
                            QMessageBox.critical(
                                self, "Start Failed",
                                f"Failed to start llama-swap.\n"
                                f"Check log: logs/llama-swap.log"
                            )
                            self._populate_model_combo()
                            return
                    except Exception as e:
                        QMessageBox.critical(self, "Start Failed", str(e))
                        self._populate_model_combo()
                        return
                else:
                    self._populate_model_combo()
                    return

        # ── Save and apply ─────────────────────────────────────────────
        _save_runtime({
            "provider": provider,
            "base_url": preset.get("base_url", defaults["base_url"]),
            "api_key": preset.get("api_key", defaults["api_key"]),
            "model_name": model_name,
        })

        self.model_dot.set_color(C.TEXT_TERTIARY)  # dim until confirmed
        self.conn_label.setText("llama-swap" if provider == "llama_cpp" else "Ollama")
        self.conn_dot.set_color(C.SUCCESS)
        QTimer.singleShot(1000, self.refresh_status)

    # ── Polling ───────────────────────────────────────────────────────────

    def refresh_status(self) -> None:
        """Poll the active LLM backend for current model state."""
        try:
            runtime = _load_runtime()
            provider = runtime.get("provider", "llama_cpp")
            model_name = runtime.get("model_name", "")
        except Exception:
            provider = "llama_cpp"
            model_name = ""

        try:
            import requests
            if provider == "ollama":
                resp = requests.get("http://localhost:11434/api/ps", timeout=2)
                data = resp.json()
                models = data.get("models", [])
                if models:
                    name = models[0].get("name", model_name).split(":")[0]
                    self._set_model_display(name, C.ACCENT)
                else:
                    self._set_model_display(model_name or "idle", C.TEXT_TERTIARY)
                self.conn_label.setText("Ollama")
                self.conn_dot.set_color(C.SUCCESS)
            else:
                resp = requests.get("http://127.0.0.1:8000/v1/models", timeout=2)
                data = resp.json()
                loaded = data.get("data", [])
                if loaded:
                    name = loaded[0].get("id", model_name)
                    self._set_model_display(name, C.ACCENT)
                else:
                    self._set_model_display(model_name or "loading…", C.TEXT_TERTIARY)
                self.conn_label.setText("llama-swap")
                self.conn_dot.set_color(C.SUCCESS)
        except Exception:
            self.conn_label.setText("Offline")
            self.conn_dot.set_color(C.ERROR)
            self._set_model_display(model_name or "—", C.TEXT_TERTIARY)

    def _set_model_display(self, name: str, dot_color: str) -> None:
        """Update model combo label without triggering the change signal."""
        self.model_dot.set_color(dot_color)
        # Update the combo text if it doesn't match a preset label
        # (just update the dot — the combo shows the preset label which is more descriptive)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _make_indicator_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"""
            color: {C.TEXT_SECONDARY};
            font-size: 11px;
            font-weight: 500;
            background: transparent;
            border: none;
        """)
        return lbl

    @staticmethod
    def _make_sep() -> QLabel:
        sep = QLabel("\u00B7")
        sep.setStyleSheet(
            f"color: {C.BORDER_STRONG}; font-size: 11px; background: transparent; border: none;"
        )
        return sep

    def set_enricher_status(self, available: bool) -> None:
        self.enricher_dot.set_color(C.SUCCESS if available else C.TEXT_TERTIARY)
