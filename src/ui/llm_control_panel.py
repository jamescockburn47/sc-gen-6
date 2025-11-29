"""PySide6 widget for managing local LLM providers."""

from __future__ import annotations

import shlex
import threading
from typing import Any, Dict, Optional

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


class LLMControlPanel(QWidget):
    """Control panel for switching providers and managing llama-server."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = load_runtime_state()

        self.provider_combo = QComboBox()
        self.base_url_edit = QLineEdit()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.model_name_edit = QLineEdit()

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
        self._update_llama_visibility()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        provider_row = QHBoxLayout()
        provider_label = QLabel("Provider:")
        self.provider_combo.addItems(["llama_cpp", "lmstudio"])
        self.provider_combo.currentTextChanged.connect(self._update_llama_visibility)
        provider_row.addWidget(provider_label)
        provider_row.addWidget(self.provider_combo)
        layout.addLayout(provider_row)

        form = QFormLayout()
        form.addRow("Base URL:", self.base_url_edit)
        form.addRow("API Key:", self.api_key_edit)
        form.addRow("Model Name:", self.model_name_edit)
        layout.addLayout(form)

        llama_group = QGroupBox("llama.cpp Server")
        self.llama_group = llama_group
        llama_layout = QFormLayout()

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
        
        llama_note = QLabel("Note: These settings only apply when using the built-in llama.cpp server. For Ollama, configure via 'ollama serve'.")
        llama_note.setWordWrap(True)
        llama_note.setStyleSheet("color: #888; font-style: italic;")
        llama_layout.addRow(llama_note)

        llama_group.setLayout(llama_layout)
        layout.addWidget(llama_group)

        button_row = QHBoxLayout()
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save_settings)
        button_row.addWidget(save_btn)

        start_btn = QPushButton("Start llama.cpp")
        start_btn.clicked.connect(self._start_llama)
        button_row.addWidget(start_btn)

        stop_btn = QPushButton("Stop llama.cpp")
        stop_btn.clicked.connect(self._stop_llama)
        button_row.addWidget(stop_btn)

        layout.addLayout(button_row)

        control_row = QHBoxLayout()
        test_btn = QPushButton("Test Connection")
        test_btn.clicked.connect(self._test_connection)
        control_row.addWidget(test_btn)

        refresh_btn = QPushButton("List Models")
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
        self.provider_combo.setCurrentText(self.state.get("provider", "llama_cpp"))
        self.base_url_edit.setText(self.state.get("base_url", "http://127.0.0.1:8000/v1"))
        self.api_key_edit.setText(self.state.get("api_key", "local-llama"))
        self.model_name_edit.setText(self.state.get("model_name", "gpt-oss-20b"))

        llama = self.state.get("llama_server", {})
        self.llama_exec_edit.setText(llama.get("executable", ""))
        self.llama_model_path_edit.setText(llama.get("model_path", ""))
        self.context_spin.setValue(int(llama.get("context", 65536)))
        self.gpu_layers_spin.setValue(int(llama.get("gpu_layers", 999)))
        self.parallel_spin.setValue(int(llama.get("parallel", 2)))
        self.batch_spin.setValue(int(llama.get("batch", 1024)))
        self.timeout_spin.setValue(int(llama.get("timeout", 1800)))
        self.flash_attn_checkbox.setChecked(bool(llama.get("flash_attn", False)))
        self.extra_args_edit.setText(llama.get("extra_args", ""))

    @Slot()
    def _update_llama_visibility(self) -> None:
        provider = self.provider_combo.currentText()
        self.llama_group.setVisible(provider == "llama_cpp")
        if provider == "lmstudio":
            if not self.base_url_edit.text():
                self.base_url_edit.setText("http://localhost:1234/v1")
            if not self.api_key_edit.text():
                self.api_key_edit.setText("lm-studio")
        else:
            if not self.base_url_edit.text():
                self.base_url_edit.setText("http://127.0.0.1:8000/v1")
            if not self.api_key_edit.text():
                self.api_key_edit.setText("local-llama")

    @Slot()
    def _browse_executable(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select llama-server executable")
        if file_path:
            self.llama_exec_edit.setText(file_path)

    @Slot()
    def _browse_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select GGUF model file", filter="GGUF Files (*.gguf);;All Files (*.*)")
        if file_path:
            self.llama_model_path_edit.setText(file_path)

    @Slot()
    def _save_settings(self) -> None:
        state = self._gather_state()
        try:
            save_runtime_state(state)
            self.state = load_runtime_state()
            QMessageBox.information(self, "Saved", "LLM settings saved.")
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
        base_url = self.base_url_edit.text().strip()
        return {
            "provider": self.provider_combo.currentText(),
            "base_url": base_url,
            "api_key": self.api_key_edit.text().strip(),
            "model_name": self.model_name_edit.text().strip(),
            "llama_server": llama_state,
        }

    @Slot()
    def _start_llama(self) -> None:
        if self.provider_combo.currentText() != "llama_cpp":
            QMessageBox.information(self, "Provider", "Switch provider to llama_cpp to start.")
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
            self._update_status_label(f"Status: ready ({content.strip()})")
        except Exception as exc:
            self._update_status_label(f"Status: connection failed ({exc})")

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
            if models:
                msg = "\n".join(models)
            else:
                msg = "No models reported."
            self._show_message("Models", msg)
            self._update_status_label("Status: models listed")
        except Exception as exc:
            self._update_status_label(f"Status: list models failed ({exc})")

    def _show_message(self, title: str, text: str) -> None:
        def _inner():
            QMessageBox.information(self, title, text)

        self._invoke_main_thread(_inner)

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


