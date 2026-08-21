#!/bin/bash
# SCGen7 - Linux launcher
# AMD Radeon 8060S (Strix Halo / gfx1151) — Vulkan backend
# ALL models served via llama-swap (llama.cpp) — no Ollama needed
# GLM 4.7 Flash: from Ollama blob (no daemon), Nemotron 3 Nano: from Windows partition

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  SCGen7 — Litigation Support RAG${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# ======== MOUNT WINDOWS PARTITION (for Nemotron model) ========
WINDOWS_MOUNT="/media/james/Windows"
NEMOTRON_PATH="$WINDOWS_MOUNT/ModelStore/nemotron3-nano-30b-q8/Nemotron-3-Nano-30B-A3B-Q8_0.gguf"

if mountpoint -q "$WINDOWS_MOUNT" 2>/dev/null; then
    echo -e "${GREEN}Windows partition already mounted at $WINDOWS_MOUNT${NC}"
elif [ -b /dev/nvme0n1p3 ]; then
    echo -e "${CYAN}Mounting Windows partition (read-only, for Nemotron model)...${NC}"
    sudo mkdir -p "$WINDOWS_MOUNT"
    sudo mount /dev/nvme0n1p3 "$WINDOWS_MOUNT" -o ro,uid=james,gid=james 2>/dev/null && \
        echo -e "${GREEN}Windows partition mounted${NC}" || \
        echo -e "${YELLOW}WARNING: Could not mount Windows partition — Nemotron model unavailable${NC}"
else
    echo -e "${YELLOW}WARNING: /dev/nvme0n1p3 not found — Nemotron model unavailable${NC}"
fi
echo ""

# ======== CLEAR PORTS (kill anything on llama-swap ports) ========
echo -e "${CYAN}Clearing ports 8000 and 8080...${NC}"
fuser -k 8000/tcp 2>/dev/null && echo "  Killed process on :8000" || true
fuser -k 8080/tcp 2>/dev/null && echo "  Killed process on :8080" || true
sleep 0.5

# ======== ACTIVATE VENV ========
if [ -d "$SCRIPT_DIR/.venv" ]; then
    echo -e "${CYAN}Activating virtual environment...${NC}"
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

mkdir -p "$SCRIPT_DIR/logs"

# ======== LAUNCH APP (launch.py starts llama-swap automatically) ========
echo -e "${GREEN}Launching SCGen7...${NC}"
echo ""
python launch.py
