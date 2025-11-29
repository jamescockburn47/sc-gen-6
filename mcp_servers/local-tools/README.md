# Local Tools MCP Server

Local tooling surface for SC Gen 6 development/debugging. Everything runs fully locally and is restricted to the repository workspace.

## Provided tools

| Tool | Purpose |
| --- | --- |
| `run_program` | Run a whitelisted CLI command (git, python, pytest, npm, etc.). |
| `read_text` | Read a UTF-8 file within the workspace. |
| `list_directory` | Quick directory listings (non/recursive) with workspace-relative paths. |
| `tail_file` | View the last N lines of a log or output file. |
| `run_git` | Execute git commands (`status`, `diff`, etc.). |
| `run_pytest` | Launch `python -m pytest` with optional targets/flags. |
| `run_python_module` | Run `python -m <module>` (e.g., `scripts.foo`). |
| `run_package_manager` | Call npm/pnpm/yarn/uv/pip style commands. |
| `ripgrep_search` | Workspace-limited `rg` search with optional flags. |

The allowlist for executables lives in `ALLOWED_EXECUTABLES` inside `server.py`. Add additional names there if you need more commands (they’re normalized without `.exe`). All paths are enforced to stay under `WORKSPACE_ROOT` to avoid accidental system-wide access.

## Setup

1. Create/refresh the virtual environment (Python 3.11 used here):
   ```powershell
   cd "C:\Users\James\Desktop\SC Gen 6\mcp_servers\local-tools"
   py -3.11 -m venv .venv
   ```
2. Install dependencies inside the venv:
   ```powershell
   .\.venv\Scripts\python -m pip install -r requirements.txt
   ```

## Running manually

```powershell
.\.venv\Scripts\python server.py
```

The server reads/writes over stdio so it will wait for an MCP client to connect.

## Testing & inspection

```powershell
.\.venv\Scripts\mcp dev server.py
```

## Connecting from Cursor

1. Open Cursor → Settings → MCP Servers → Add server.
2. Command: `python`
3. Args: `["C:\\Users\\James\\Desktop\\SC Gen 6\\mcp_servers\\local-tools\\server.py"]`
4. (Optional) Working directory: `C:\Users\James\Desktop\SC Gen 6\mcp_servers\local-tools`
5. Cursor spawns the script inside its own environment and lists the tools.

## Extending safely

- Update `ALLOWED_EXECUTABLES` to cover any new binaries you rely on (format: lowercase name without `.exe`).
- Prefer wrapping each higher-level capability (chunking, embeddings, OCR) behind its own `@server.tool()` function with explicit argument validation instead of exposing raw shell access.
- Add logging/auditing or rate limiting before exposing long-running or privileged commands.
