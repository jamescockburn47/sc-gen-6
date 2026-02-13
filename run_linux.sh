#!/bin/bash
# SC Gen 6 - Linux launcher with Ollama (Vulkan backend)
# AMD Radeon 8060S (Strix Halo / gfx1151)
# LLM: Ollama with GLM 4.7 Flash via Vulkan
# Embeddings: sentence-transformers BGE (CPU)
# Reranker: CrossEncoder (CPU)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  SC Gen 6 - GLM 4.7 Flash on Linux${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# ======== CHECK OLLAMA ========
echo -e "${CYAN}Checking Ollama...${NC}"
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}ERROR: Ollama not installed${NC}"
    echo "Install with: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}Starting Ollama service...${NC}"
    sudo systemctl start ollama
    sleep 3
fi

# Check Vulkan config
if ! sudo systemctl cat ollama 2>/dev/null | grep -q "OLLAMA_VULKAN=1"; then
    echo -e "${YELLOW}WARNING: Ollama Vulkan not configured!${NC}"
    echo "  Run: sudo systemctl edit ollama"
    echo "  Add: Environment=\"OLLAMA_VULKAN=1\""
fi

# Check model
MODEL="glm-4.7-flash"
if ! ollama list 2>/dev/null | grep -q "$MODEL"; then
    echo -e "${YELLOW}Model $MODEL not found. Pulling...${NC}"
    ollama pull "$MODEL"
fi

echo -e "${GREEN}Ollama ready with $MODEL${NC}"
echo ""

# ======== ACTIVATE VENV ========
if [ -d "$SCRIPT_DIR/.venv" ]; then
    echo -e "${CYAN}Activating virtual environment...${NC}"
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

mkdir -p "$SCRIPT_DIR/logs"

# ======== LAUNCH APP ========
echo -e "${GREEN}Launching SC Gen 6...${NC}"
echo ""
python launch.py
