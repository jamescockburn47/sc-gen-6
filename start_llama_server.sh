#!/bin/bash
# SC Gen 6 - llama-server launcher for Linux (Vulkan backend)
# AMD Radeon 8060S (Strix Halo / gfx1151)

set -e

# Load environment from .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

# Default values
LLAMA_SERVER="${LLAMA_SERVER_PATH:-/home/james/llama.cpp/build/bin/llama-server}"
MODEL="${LLAMA_MODEL_PATH:-/home/james/.cache/huggingface/gguf/Nemotron-3-Nano-30B-A3B-BF16-00001-of-00002.gguf}"
CONTEXT="${LLAMA_CONTEXT:-32768}"
GPU_LAYERS="${LLAMA_GPU_LAYERS:-99}"
PARALLEL="${LLAMA_PARALLEL:-4}"
BATCH="${LLAMA_BATCH:-2048}"
HOST="${LLAMA_HOST:-127.0.0.1}"
PORT="${LLAMA_PORT:-8000}"
API_KEY="${LLM_API_KEY:-local-llama}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SC Gen 6 - llama-server (Vulkan)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if server exists
if [ ! -f "$LLAMA_SERVER" ]; then
    echo -e "${RED}ERROR: llama-server not found at $LLAMA_SERVER${NC}"
    echo "Build it with: cd ~/llama.cpp && mkdir build && cd build && cmake .. -DGGML_VULKAN=ON && make -j\$(nproc)"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo -e "${RED}ERROR: Model not found at $MODEL${NC}"
    exit 1
fi

# Display configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Server:     $LLAMA_SERVER"
echo "  Model:      $(basename $MODEL)"
echo "  Context:    $CONTEXT tokens"
echo "  GPU Layers: $GPU_LAYERS"
echo "  Parallel:   $PARALLEL slots"
echo "  Batch:      $BATCH"
echo "  Endpoint:   http://$HOST:$PORT/v1"
echo ""

# Check for existing server on port
if lsof -i :$PORT > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Port $PORT is already in use${NC}"
    read -p "Kill existing process? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        fuser -k $PORT/tcp 2>/dev/null || true
        sleep 1
    else
        exit 1
    fi
fi

echo -e "${GREEN}Starting llama-server...${NC}"
echo ""

# Run the server
exec "$LLAMA_SERVER" \
    -m "$MODEL" \
    -c "$CONTEXT" \
    -ngl "$GPU_LAYERS" \
    --host "$HOST" \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --parallel "$PARALLEL" \
    -b "$BATCH" \
    --ubatch-size 512 \
    --flash-attn \
    --cont-batching \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --threads 8 \
    --timeout 300 \
    --metrics
