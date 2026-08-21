#!/usr/bin/env bash
# SCGen6 pre-query memory flush
# Called via: sudo /home/james/SCGen6/scripts/flush_mem.sh
# Must be run as root (via sudoers NOPASSWD rule).
#
# Actions:
#   1. sync filesystem buffers
#   2. drop page cache (level 1 — safe, keeps dentry/inode)
#   3. if swap used > SWAP_THRESHOLD_MB, flush swap entirely

set -euo pipefail

SWAP_THRESHOLD_MB=${1:-256}   # flush swap if more than this many MB in use

# 1. Sync pending filesystem writes
sync

# 2. Drop page cache (1 = PageCache only; 3 = PageCache + dentries + inodes)
echo 1 > /proc/sys/vm/drop_caches

# 3. Check swap usage and optionally flush
SWAP_USED_KB=$(awk '/SwapFree/{free=$2} /SwapTotal/{total=$2} END{print total-free}' /proc/meminfo)
SWAP_USED_MB=$(( SWAP_USED_KB / 1024 ))

if [ "$SWAP_USED_MB" -gt "$SWAP_THRESHOLD_MB" ]; then
    echo "[flush_mem] Swap used: ${SWAP_USED_MB}MB > threshold ${SWAP_THRESHOLD_MB}MB — flushing swap"
    swapoff -a
    swapon -a
    echo "[flush_mem] Swap flush complete"
else
    echo "[flush_mem] Swap used: ${SWAP_USED_MB}MB — below threshold, skipping swap flush"
fi

echo "[flush_mem] Done"
