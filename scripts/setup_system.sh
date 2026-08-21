#!/usr/bin/env bash
# =============================================================================
# SCGen6 one-time system setup script
# Run as: sudo bash /home/james/SCGen6/scripts/setup_system.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLUSH_SCRIPT="$SCRIPT_DIR/flush_mem.sh"
SWAPFILE="/swapfile16"
USER="${SUDO_USER:-james}"

echo "=== SCGen6 System Setup ==="
echo "User: $USER"
echo ""

# ─── 1. Make flush script executable ──────────────────────────────────────────
echo "[1/5] Making flush script executable..."
chmod +x "$FLUSH_SCRIPT"
echo "      OK: $FLUSH_SCRIPT"

# ─── 2. Sudoers rule ──────────────────────────────────────────────────────────
echo "[2/5] Installing sudoers rule (passwordless flush script for $USER)..."
SUDOERS_FILE="/etc/sudoers.d/scgen6-mem"
cat > "$SUDOERS_FILE" << EOF
# SCGen6: allow $USER to run memory flush script without password
$USER ALL=(ALL) NOPASSWD: $FLUSH_SCRIPT
EOF
chmod 440 "$SUDOERS_FILE"
visudo -c -f "$SUDOERS_FILE" && echo "      OK: $SUDOERS_FILE" || { echo "ERROR: sudoers file invalid, removing"; rm "$SUDOERS_FILE"; exit 1; }

# ─── 3. 16GB swapfile ─────────────────────────────────────────────────────────
echo "[3/5] Creating 16GB swapfile at $SWAPFILE..."
if swapon --show | grep -q "$SWAPFILE"; then
    echo "      Swapfile already active, skipping creation"
elif [ -f "$SWAPFILE" ]; then
    echo "      File exists, activating..."
    swapon "$SWAPFILE"
else
    echo "      Allocating 16GB (this may take a moment)..."
    fallocate -l 16G "$SWAPFILE" || dd if=/dev/zero of="$SWAPFILE" bs=1M count=16384 status=progress
    chmod 600 "$SWAPFILE"
    mkswap "$SWAPFILE"
    swapon "$SWAPFILE"
fi
echo "      OK: $(swapon --show | grep "$SWAPFILE" || echo '(active)')"

# ─── 4. Make swapfile persistent across reboots ───────────────────────────────
echo "[4/5] Adding swapfile to /etc/fstab..."
if grep -q "$SWAPFILE" /etc/fstab; then
    echo "      Already in /etc/fstab, skipping"
else
    echo "$SWAPFILE none swap sw 0 0" >> /etc/fstab
    echo "      OK: added to /etc/fstab"
fi

# ─── 5. Set vm.swappiness=1 (permanent) ──────────────────────────────────────
echo "[5/5] Setting vm.swappiness=1 (minimise swapping)..."
SYSCTL_CONF="/etc/sysctl.d/99-scgen6.conf"
cat > "$SYSCTL_CONF" << EOF
# SCGen6: minimise kernel swapping to keep GPU model weights in RAM
vm.swappiness=1
# Reduce tendency to reclaim from page cache too aggressively
vm.vfs_cache_pressure=50
EOF
sysctl --system | grep -E "swappiness|vfs_cache" || true
echo "      OK: $SYSCTL_CONF"

echo ""
echo "=== Setup complete ==="
free -h
echo ""
echo "Current swap:"
swapon --show
echo ""
echo "Test the flush script with:"
echo "  sudo $FLUSH_SCRIPT"
