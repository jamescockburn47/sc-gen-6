"""PySide6 widget for managing local LLM providers."""

from __future__ import annotations

import json
import shlex
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.config.llm_config import load_llm_config
from src.config.runtime_store import load_runtime_state, save_runtime_state
from src.llm.client import get_llm_client
from src.llm.constants import LLAMA_SERVER_LOG_PATH
from src.llm.server_manager import manager as llama_manager

PRESETS_FILE = Path("config/model_presets.json")

PROVIDER_DEFAULTS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
    "llama_cpp": {
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "local-llama",
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
    },
}


def _load_presets() -> List[Dict[str, Any]]:
    """Load model presets from config/model_presets.json."""
    try:
        if PRESETS_FILE.exists():
            with PRESETS_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


class LLMControlPanel(QWidget):
    """Control panel for switching providers and managing llama-server.
    
    Supports Ollama, llama_cpp (via llama-swap), and LM Studio.
    Includes a model preset quick-switcher for one-click model changes.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = load_runtime_state()
        self._presets: List[Dict[str, Any]] = _load_presets()

        # Provider + connection fields
        self.provider_combo = QComboBox()
        self.base_url_edit = QLineEdit()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.model_name_edit = QLineEdit()

        # Preset quick-switcher
        self.preset_combo = QComboBox()

        # llama.cpp-specific fields (hidden for Ollama)
        self.llama_exec_edit = QLineEdit()
        self.llama_model_path_edit = QLineEdit()
        self.context_spin = QSpinBox()
        self.context_spin.setRange(1024, 262144)
        self.gpu_layers_spin = QSpinBox()
        self.gpu_layers_spin.setRange(1, 4096)
        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(1, 16)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(32, 4096)
        self.batch_spin.setSingleStep(32)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(60, 7200)
        self.flash_attn_checkbox = QCheckBox("Enable Flash Attention (--flash-attn)")
        self.extra_args_edit = QLineEdit()
        self.extra_args_edit.setPlaceholderText("--no-mmap --mmq")

        self.status_label = QLabel("Status: unknown")

        self._build_ui()
        self._populate_fields()
        self._update_provider_visibility()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Model Preset Quick-Switcher ──────────────────────────────────
        preset_group = QGroupBox("Quick Model Switch")
        preset_layout = QHBoxLayout(preset_group)

        self.preset_combo.setMinimumWidth(320)
        self.preset_combo.addItem("— select preset —", None)
        for preset in self._presets:
            self.preset_combo.addItem(preset["label"], preset)

        apply_preset_btn = QPushButton("Apply Preset")
        apply_preset_btn.setObjectName("apply_preset_btn")
        apply_preset_btn.clicked.connect(self._apply_preset)
        apply_preset_btn.setStyleSheet(
            "QPushButton { background: #2563eb; color: white; font-weight: bold; "
            "padding: 4px 12px; border-radius: 4px; }"
            "QPushButton:hover { background: #1d4ed8; }"
        )

        preset_layout.addWidget(self.preset_combo, stretch=1)
        preset_layout.addWidget(apply_preset_btn)
        layout.addWidget(preset_group)

        # ── Provider + Connection ────────────────────────────────────────
        conn_group = QGroupBox("Provider & Connection")
        conn_layout = QFormLayout(conn_group)

        provider_row = QHBoxLayout()
        self.provider_combo.addItems(["ollama", "llama_cpp", "lmstudio"])
        self.provider_combo.currentTextChanged.connect(self._update_provider_visibility)
        provider_row.addWidget(self.provider_combo)
        conn_layout.addRow("Provider:", provider_row)
        conn_layout.addRow("Base URL:", self.base_url_edit)
        conn_layout.addRow("API Key:", self.api_key_edit)
        conn_layout.addRow("Model Name:", self.model_name_edit)
        layout.addWidget(conn_group)

        # ── llama.cpp Server (hidden for Ollama) ─────────────────────────
        self.llama_group = QGroupBox("llama.cpp / llama-swap Server Settings")
        llama_layout = QFormLayout(self.llama_group)

        llama_layout.addRow(
            self._with_browse("Executable:", self.llama_exec_edit, self._browse_executable)
        )
        llama_layout.addRow(
            self._with_browse("Model Path:", self.llama_model_path_edit, self._browse_model)
        )

        self.context_spin.setSingleStep(1024)
        llama_layout.addRow("Context (tokens):", self.context_spin)
        llama_layout.addRow("GPU Layers:", self.gpu_layers_spin)
        llama_layout.addRow("Parallel Streams:", self.parallel_spin)
        llama_layout.addRow("Prompt Batch Size:", self.batch_spin)
        llama_layout.addRow("Timeout (s):", self.timeout_spin)
        llama_layout.addRow(self.flash_attn_checkbox)
        llama_layout.addRow("Extra CLI Args:", self.extra_args_edit)

        llama_note = QLabel(
            "Note: These settings apply when using the built-in llama.cpp server. "
            "For llama-swap, configure via llama-swap/config.yaml instead."
        )
        llama_note.setWordWrap(True)
        llama_note.setStyleSheet("color: #888; font-style: italic;")
        llama_layout.addRow(llama_note)
        layout.addWidget(self.llama_group)

        # ── Ollama info (shown only for Ollama) ──────────────────────────
        self.ollama_group = QGroupBox("Ollama Info")
        ollama_layout = QVBoxLayout(self.ollama_group)
        ollama_info = QLabel(
            "Ollama manages model loading automatically.\n"
            "Speed settings (Flash Attention, KV cache) are configured via:\n"
            "  /etc/systemd/system/ollama.service.d/speed.conf\n\n"
            "To pull a new model:  ollama pull <model-name>\n"
            "To list models:       ollama list"
        )
        ollama_info.setWordWrap(True)
        ollama_info.setStyleSheet("color: #aaa; font-family: monospace; font-size: 11px;")
        ollama_layout.addWidget(ollama_info)
        layout.addWidget(self.ollama_group)

        # ── Buttons ──────────────────────────────────────────────────────
        button_row = QHBoxLayout()
        save_btn = QPushButton("Save Settings")
        save_btn.setObjectName("save_btn")
        save_btn.clicked.connect(self._save_settings)
        button_row.addWidget(save_btn)

        start_btn = QPushButton("Start llama.cpp")
        start_btn.setObjectName("start_llama_btn")
        start_btn.clicked.connect(self._start_llama)
        button_row.addWidget(start_btn)

        stop_btn = QPushButton("Stop llama.cpp")
        stop_btn.setObjectName("stop_llama_btn")
        stop_btn.clicked.connect(self._stop_llama)
        button_row.addWidget(stop_btn)
        layout.addLayout(button_row)

        control_row = QHBoxLayout()
        test_btn = QPushButton("Test Connection")
        test_btn.setObjectName("test_connection_btn")
        test_btn.clicked.connect(self._test_connection)
        control_row.addWidget(test_btn)

        refresh_btn = QPushButton("List Models")
        refresh_btn.setObjectName("list_models_btn")
        refresh_btn.clicked.connect(self._list_models)
        control_row.addWidget(refresh_btn)
        control_row.addStretch()
        layout.addLayout(control_row)

        self.status_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.status_label)

        note = QLabel("Changes take effect immediately; restart queries to use new provider.")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()

    def _with_browse(self, label: str, line_edit: QLineEdit, handler) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(label))
        layout.addWidget(line_edit, stretch=1)
        browse = QPushButton("Browse")
        browse.clicked.connect(handler)
        layout.addWidget(browse)
        return container

    def _populate_fields(self) -> None:
        provider = self.state.get("provider", "ollama")
        # Ensure provider is in the combo
        idx = self.provider_combo.findText(provider)
        if idx >= 0:
            self.provider_combo.setCurrentIndex(idx)
        self.base_url_edit.setText(self.state.get("base_url", ""))
        self.api_key_edit.setText(self.state.get("api_key", ""))
        self.model_name_edit.setText(self.state.get("model_name", "glm-4.7-flash"))

        llama = self.state.get("llama_server", {})
        self.llama_exec_edit.setText(llama.get("executable", "/home/james/llama.cpp/build/bin/llama-server"))
        self.llama_model_path_edit.setText(llama.get("model_path", ""))
        self.context_spin.setValue(int(llama.get("context", 32768)))
        self.gpu_layers_spin.setValue(int(llama.get("gpu_layers", 99)))
        self.parallel_spin.setValue(int(llama.get("parallel", 4)))
        self.batch_spin.setValue(int(llama.get("batch", 2048)))
        self.timeout_spin.setValue(int(llama.get("timeout", 300)))
        self.flash_attn_checkbox.setChecked(bool(llama.get("flash_attn", True)))
        self.extra_args_edit.setText(llama.get("extra_args", ""))

    @Slot()
    def _update_provider_visibility(self) -> None:
        """Show/hide provider-specific UI sections."""
        provider = self.provider_combo.currentText()
        is_llama = provider == "llama_cpp"
        is_ollama = provider == "ollama"

        self.llama_group.setVisible(is_llama)
        self.ollama_group.setVisible(is_ollama)

        # Auto-fill URL/key defaults when switching providers
        defaults = PROVIDER_DEFAULTS.get(provider, {})
        current_url = self.base_url_edit.text().strip()

        # Only auto-fill if the current URL belongs to a different provider
        url_is_foreign = any(
            current_url == v["base_url"]
            for k, v in PROVIDER_DEFAULTS.items()
            if k != provider
        )
        if not current_url or url_is_foreign:
            self.base_url_edit.setText(defaults.get("base_url", ""))
            self.api_key_edit.setText(defaults.get("api_key", ""))

    @Slot()
    def _apply_preset(self) -> None:
        """Apply the selected model preset, updating all fields at once."""
        preset: Optional[Dict[str, Any]] = self.preset_combo.currentData()
        if not preset:
            return

        provider = preset.get("provider", "ollama")
        idx = self.provider_combo.findText(provider)
        if idx >= 0:
            self.provider_combo.setCurrentIndex(idx)

        # Set connection fields
        self.base_url_edit.setText(preset.get("base_url", PROVIDER_DEFAULTS.get(provider, {}).get("base_url", "")))
        self.api_key_edit.setText(preset.get("api_key", PROVIDER_DEFAULTS.get(provider, {}).get("api_key", "")))
        self.model_name_edit.setText(preset.get("model_name", ""))

        # Set llama.cpp fields if applicable
        if provider == "llama_cpp":
            if preset.get("path"):
                self.llama_model_path_edit.setText(preset["path"])
            if preset.get("context"):
                self.context_spin.setValue(int(preset["context"]))
            if preset.get("gpu_layers"):
                self.gpu_layers_spin.setValue(int(preset["gpu_layers"]))
            self.flash_attn_checkbox.setChecked(preset.get("flash_attn", True))

        self._update_provider_visibility()
        self.status_label.setText(
            f"Preset loaded: {preset['label']} — click 'Save Settings' to apply."
        )

    @Slot()
    def _browse_executable(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select llama-server executable")
        if file_path:
            self.llama_exec_edit.setText(file_path)

    @Slot()
    def _browse_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF model file", filter="GGUF Files (*.gguf);;All Files (*.*)"
        )
        if file_path:
            self.llama_model_path_edit.setText(file_path)

    @Slot()
    def _save_settings(self) -> None:
        state = self._gather_state()
        try:
            save_runtime_state(state)
            self.state = load_runtime_state()
            self.status_label.setText(
                f"✓ Saved — Provider: {state['provider']} | Model: {state['model_name']}"
            )
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Failed to save settings:\n{exc}")

    def _gather_state(self) -> Dict[str, Any]:
        llama_state = {
            "executable": self.llama_exec_edit.text().strip(),
            "model_path": self.llama_model_path_edit.text().strip(),
            "context": self.context_spin.value(),
            "gpu_layers": self.gpu_layers_spin.value(),
            "parallel": self.parallel_spin.value(),
            "batch": self.batch_spin.value(),
            "timeout": self.timeout_spin.value(),
            "flash_attn": self.flash_attn_checkbox.isChecked(),
            "extra_args": self.extra_args_edit.text().strip(),
        }
        return {
            "provider": self.provider_combo.currentText(),
            "base_url": self.base_url_edit.text().strip(),
            "api_key": self.api_key_edit.text().strip(),
            "model_name": self.model_name_edit.text().strip(),
            "llama_server": llama_state,
        }

    @Slot()
    def _start_llama(self) -> None:
        if self.provider_combo.currentText() != "llama_cpp":
            QMessageBox.information(
                self, "Provider",
                "Switch provider to 'llama_cpp' to use the built-in server.\n"
                "For Ollama, ensure 'ollama serve' is running."
            )
            return

        state = self._gather_state()
        llama = state["llama_server"]
        try:
            host, port = _parse_host_port(state["base_url"])
            extra_args: list[str] = []
            if llama.get("flash_attn"):
                extra_args.append("--flash-attn")
            if llama.get("extra_args"):
                extra_args.extend(shlex.split(llama["extra_args"]))
            llama_manager.start(
                executable=llama["executable"],
                model_path=llama["model_path"],
                host=host,
                port=port,
                api_key=state["api_key"],
                context=int(llama["context"]),
                gpu_layers=int(llama["gpu_layers"]),
                parallel=int(llama["parallel"]),
                batch=int(llama["batch"]),
                timeout=int(llama["timeout"]),
                detached=True,
                log_path=LLAMA_SERVER_LOG_PATH,
                extra_args=extra_args,
            )
            self.status_label.setText("Status: llama.cpp starting...")
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Failed to start llama.cpp:\n{exc}")

    @Slot()
    def _stop_llama(self) -> None:
        llama_manager.stop()
        self.status_label.setText("Status: llama.cpp stopped")

    @Slot()
    def _test_connection(self) -> None:
        self.status_label.setText("Status: testing connection...")
        thread = threading.Thread(target=self._run_test_connection, daemon=True)
        thread.start()

    def _run_test_connection(self) -> None:
        try:
            cfg = load_llm_config()
            client = get_llm_client(cfg)
            content = client.generate_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a readiness probe."},
                    {"role": "user", "content": "Say 'ready'."},
                ],
                model=cfg.model_name,
                temperature=0.0,
            )
            self._update_status_label(
                f"✓ Connected — {cfg.provider} / {cfg.model_name} — {content.strip()[:60]}"
            )
        except Exception as exc:
            self._update_status_label(f"✗ Connection failed: {exc}")

    @Slot()
    def _list_models(self) -> None:
        self.status_label.setText("Status: listing models...")
        thread = threading.Thread(target=self._run_list_models, daemon=True)
        thread.start()

    def _run_list_models(self) -> None:
        try:
            cfg = load_llm_config()
            client = get_llm_client(cfg)
            models = client.list_models()
            msg = "\n".join(models) if models else "No models reported."
            self._show_message("Available Models", msg)
            self._update_status_label(f"Status: {len(models)} model(s) listed")
        except Exception as exc:
            self._update_status_label(f"Status: list models failed ({exc})")

    def _show_message(self, title: str, text: str) -> None:
        self._invoke_main_thread(lambda: QMessageBox.information(self, title, text))

    def _update_status_label(self, text: str) -> None:
        self._invoke_main_thread(lambda: self.status_label.setText(text))

    def _invoke_main_thread(self, func) -> None:
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, func)


def _parse_host_port(base_url: str) -> tuple[str, int]:
    from urllib.parse import urlparse
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    return host, port
