import shlex
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.runtime_store import load_runtime_state  # noqa: E402
from src.llm.server_manager import manager  # noqa: E402
from src.llm.constants import LLAMA_SERVER_LOG_PATH  # noqa: E402

state = load_runtime_state()
llama = state.get('llama_server', {})
base_url = state.get('base_url', 'http://127.0.0.1:8000/v1')
parsed = urlparse(base_url)
host = parsed.hostname or llama.get('host', '127.0.0.1')
port = parsed.port or llama.get('port', 8000)
extra_args = shlex.split(llama.get('extra_args') or '')

manager.start(
    executable=llama['executable'],
    model_path=llama['model_path'],
    host=host,
    port=port,
    api_key=state.get('api_key', 'local-llama'),
    context=int(llama.get('context', 65536)),
    gpu_layers=int(llama.get('gpu_layers', 999)),
    parallel=int(llama.get('parallel', 1)),
    batch=int(llama.get('batch', 1024)),
    timeout=int(llama.get('timeout', 1800)),
    extra_args=extra_args,
    detached=True,
    log_path=LLAMA_SERVER_LOG_PATH,
)
print('llama-server launched with logs at', LLAMA_SERVER_LOG_PATH)
