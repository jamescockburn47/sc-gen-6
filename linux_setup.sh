#!/bin/bash
# SC Gen 6 - Ubuntu Setup Script
# Run this after installing Ubuntu 25.10
# 
# Prerequisites: Ubuntu 25.10 installed, internet connected
#
# Usage: 
#   chmod +x linux_setup.sh
#   ./linux_setup.sh

set -e  # Exit on error

echo "========================================"
echo "SC Gen 6 - Linux Setup"
echo "========================================"

# Step 1: Update system
echo "[1/8] Updating system..."
sudo apt update && sudo apt upgrade -y

# Step 2: Mount Windows partition
echo "[2/8] Setting up Windows partition mount..."
sudo mkdir -p /mnt/windows
# Find Windows partition (usually the largest NTFS)
WINDOWS_PART=$(lsblk -f | grep ntfs | awk '{print $1}' | head -1)
if [ -n "$WINDOWS_PART" ]; then
    WINDOWS_UUID=$(sudo blkid -s UUID -o value /dev/${WINDOWS_PART})
    echo "UUID=${WINDOWS_UUID} /mnt/windows ntfs-3g defaults,uid=1000,gid=1000,dmask=022,fmask=133 0 0" | sudo tee -a /etc/fstab
    sudo mount -a
    echo "Windows partition mounted at /mnt/windows"
else
    echo "WARNING: Could not auto-detect Windows partition. Mount manually."
fi

# Step 3: Install ROCm
echo "[3/8] Installing ROCm 7.0..."
wget https://repo.radeon.com/amdgpu-install/6.4.1/ubuntu/noble/amdgpu-install_6.4.60401-1_all.deb -O /tmp/amdgpu-install.deb
sudo apt install -y /tmp/amdgpu-install.deb
sudo amdgpu-install -y --usecase=rocm
sudo usermod -a -G render,video $USER

# Step 4: Verify GPU
echo "[4/8] Verifying GPU..."
rocminfo | head -20

# Step 5: Install Python
echo "[5/8] Installing Python and dependencies..."
sudo apt install -y python3.12-venv python3-pip git curl

# Step 6: Install Cursor IDE
echo "[6/8] Installing Cursor IDE..."
curl -fsSL https://cursor.com/download/linux -o /tmp/cursor.deb
sudo apt install -y /tmp/cursor.deb

# Step 7: Install llama-swap
echo "[7/8] Installing llama-swap..."
wget https://github.com/mostlygeek/llama-swap/releases/latest/download/llama-swap_linux_amd64.tar.gz -O /tmp/llama-swap.tar.gz
tar -xzf /tmp/llama-swap.tar.gz -C /tmp
sudo mv /tmp/llama-swap /usr/local/bin/

# Step 8: Setup SC Gen 6
echo "[8/8] Setting up SC Gen 6..."
SC_DIR="/mnt/windows/Users/James/Desktop/SC Gen 6"
if [ -d "$SC_DIR" ]; then
    cd "$SC_DIR"
    python3 -m venv .venv-linux
    source .venv-linux/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "SC Gen 6 Python environment ready!"
else
    echo "WARNING: SC Gen 6 directory not found at expected location"
    echo "Please navigate to your SC Gen 6 folder and run:"
    echo "  python3 -m venv .venv-linux"
    echo "  source .venv-linux/bin/activate"
    echo "  pip install -r requirements.txt"
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. REBOOT your computer (required for ROCm)"
echo "2. Open Cursor IDE"
echo "3. Open SC Gen 6 folder: /mnt/windows/Users/James/Desktop/SC Gen 6"
echo "4. Run: source .venv-linux/bin/activate"
echo "5. Run: python main.py"
echo ""
echo "To use Antigravity: Just use Cursor like normal!"
