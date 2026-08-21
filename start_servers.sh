#!/bin/bash
# Start both llama-swap instances for SC Gen 6
# Port 8000: Generation model (nemotron-3-nano)
# Port 8001: Embedding model (nemotron-embed-8b)

cd "$(dirname "$0")/llama-swap"

echo "Starting SC Gen 6 servers..."

# Check if llama-swap binary exists
if [ ! -f "./llama-swap" ]; then
    echo "Error: llama-swap binary not found in $(pwd)"
    exit 1
fi

# Start embedding server (port 8001) - loads first for ingestion
echo "[1/2] Starting embedding server on port 8001..."
./llama-swap --config config-embed.yaml &
EMBED_PID=$!

# Give it a moment to start
sleep 1

# Start generation server (port 8000)
echo "[2/2] Starting generation server on port 8000..."
./llama-swap --config config.yaml &
GEN_PID=$!

echo ""
echo "Both servers started:"
echo "  Embedding server (port 8001): PID $EMBED_PID"
echo "  Generation server (port 8000): PID $GEN_PID"
echo ""
echo "To stop both servers: kill $EMBED_PID $GEN_PID"
echo ""

# Wait for both processes
wait
