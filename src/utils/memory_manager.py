"""Memory management utilities for SCGen6.

Provides pre-query memory optimisation:
  - Drop Linux page cache to reclaim RAM for the GPU model
  - Flush swap if usage exceeds threshold (restores full GPU throughput)

The flush script requires passwordless sudo for the specific script path.
Set up once with: sudo visudo -f /etc/sudoers.d/scgen6-mem
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time

logger = logging.getLogger(__name__)

_FLUSH_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "scripts", "flush_mem.sh"
)

# Minimum MB in swap before we try a full swapoff/swapon flush
SWAP_FLUSH_THRESHOLD_MB = 256

# Don't re-flush more often than this (seconds)
_MIN_INTERVAL_S = 30
_last_flush_time = 0.0
_flush_lock = threading.Lock()


def _read_swap_used_mb() -> int:
    """Return current swap used in MB (fast, no sudo needed)."""
    try:
        total = free = 0
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("SwapTotal:"):
                    total = int(line.split()[1])
                elif line.startswith("SwapFree:"):
                    free = int(line.split()[1])
        return (total - free) // 1024
    except Exception:
        return 0


def flush_for_query(blocking: bool = True, threshold_mb: int = SWAP_FLUSH_THRESHOLD_MB) -> None:
    """Drop page cache and optionally flush swap before an LLM query.

    Safe to call from any thread. Uses a cooldown so rapid successive calls
    (e.g. follow-up questions) don't hammer the kernel.

    Args:
        blocking:     If True, wait for the flush to complete before returning.
                      If False, run in background thread (page cache still dropped
                      synchronously but swap flush is async).
        threshold_mb: Only flush swap if usage exceeds this many MB.
    """
    global _last_flush_time

    with _flush_lock:
        now = time.monotonic()
        if now - _last_flush_time < _MIN_INTERVAL_S:
            logger.debug("[MemFlush] Skipping — last flush was %.0fs ago", now - _last_flush_time)
            return
        _last_flush_time = now

    swap_used = _read_swap_used_mb()
    logger.info("[MemFlush] Swap used: %dMB, threshold: %dMB", swap_used, threshold_mb)

    def _do_flush():
        t0 = time.monotonic()
        try:
            result = subprocess.run(
                ["sudo", _FLUSH_SCRIPT, str(threshold_mb)],
                capture_output=True,
                text=True,
                timeout=120,   # swap flush can take up to 2 min on big swap
            )
            elapsed = (time.monotonic() - t0) * 1000
            if result.returncode == 0:
                logger.info("[MemFlush] Complete in %.0fms: %s",
                            elapsed, result.stdout.strip().split('\n')[-1])
            else:
                logger.warning("[MemFlush] Script failed (rc=%d): %s",
                               result.returncode, result.stderr.strip())
        except FileNotFoundError:
            logger.warning("[MemFlush] sudo or flush script not found — skipping")
        except subprocess.TimeoutExpired:
            logger.warning("[MemFlush] Flush timed out after 120s")
        except Exception as e:
            logger.warning("[MemFlush] Unexpected error: %s", e)

    if blocking:
        _do_flush()
    else:
        t = threading.Thread(target=_do_flush, daemon=True, name="mem-flush")
        t.start()
