# SC Gen 6 - Ubuntu Setup Guide

## 📖 How to View These Instructions in Ubuntu

### Option 1: GitHub (Recommended)
Open Firefox and go to:
```
https://github.com/jamescockburn47/sc-gen-6/blob/main/LINUX_SETUP_GUIDE.md
```

### Option 2: After Mounting Windows
```bash
cat "/mnt/windows/Users/James/Desktop/SC Gen 6/LINUX_SETUP_GUIDE.md"
```

### Option 3: Continue with Antigravity
After installing Cursor in Ubuntu, you can use Antigravity just like on Windows!

---

# STEP-BY-STEP SETUP

## Phase 1: First Boot into Ubuntu

After Ubuntu installation completes and you log in:

### 1.1 Open Terminal
Press `Ctrl + Alt + T`

### 1.2 Update System
```bash
sudo apt update && sudo apt upgrade -y
```

---

## Phase 2: Mount Windows Partition

### 2.1 Find Your Windows Partition
```bash
lsblk -f | grep ntfs
```
Look for the large NTFS partition (probably `nvme0n1p3` or similar).

### 2.2 Create Mount Point
```bash
sudo mkdir -p /mnt/windows
```

### 2.3 Mount It (replace nvme0n1p3 with your partition)
```bash
sudo mount -t ntfs-3g /dev/nvme0n1p3 /mnt/windows
```

### 2.4 Verify
```bash
ls "/mnt/windows/Users/James/Desktop/SC Gen 6"
```
You should see your Windows files!

### 2.5 Make Permanent (optional)
```bash
echo "$(sudo blkid /dev/nvme0n1p3 -s UUID -o export | grep UUID) /mnt/windows ntfs-3g ro,uid=1000,gid=1000 0 0" | sudo tee -a /etc/fstab
```

---

## Phase 3: Install Development Tools

### 3.1 Install Git and Python
```bash
sudo apt install -y git python3.12-venv python3-pip build-essential curl
```

### 3.2 Install Cursor IDE
```bash
# Download Cursor AppImage
wget "https://downloader.cursor.sh/linux/appImage/x64" -O ~/cursor.AppImage
chmod +x ~/cursor.AppImage

# Run Cursor
~/cursor.AppImage &
```
Sign in with your Cursor account. **Now you can use Antigravity in Ubuntu!**

---

## Phase 4: Clone SC Gen 6

### 4.1 Clone from GitHub
```bash
cd ~
git clone https://github.com/jamescockburn47/sc-gen-6.git sc-gen-6
cd sc-gen-6
```

### 4.2 Create Python Virtual Environment
```bash
python3 -m venv .venv-linux
source .venv-linux/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Phase 5: Setup Model Cache

### 5.1 Create Cache Directory
```bash
mkdir -p ~/model-cache
```

### 5.2 Copy Models from Windows (takes time!)
```bash
# Embedding model (required)
rsync -avh --progress "/mnt/windows/Users/James/Desktop/SC Gen 6/models/embeddings/" ~/model-cache/embeddings/

# Main generation model
rsync -avh --progress "/mnt/windows/Users/James/Desktop/SC Gen 6/models/nemotron3-nano-30b-q8/" ~/model-cache/nemotron3-nano-30b-q8/
```

### 5.3 Link Models Directory
```bash
cd ~/sc-gen-6
rm -rf models  # Remove empty/placeholder
ln -s ~/model-cache models
```

---

## Phase 6: Install ROCm (AMD GPU)

### 6.1 Add AMD Repository
```bash
wget https://repo.radeon.com/amdgpu-install/6.4.1/ubuntu/noble/amdgpu-install_6.4.60401-1_all.deb -O /tmp/amdgpu.deb
sudo apt install -y /tmp/amdgpu.deb
```

### 6.2 Install ROCm
```bash
sudo amdgpu-install -y --usecase=rocm
sudo usermod -a -G render,video $USER
```

### 6.3 REBOOT
```bash
sudo reboot
```

### 6.4 Verify GPU (after reboot)
```bash
rocminfo | grep gfx
```
Should show `gfx1151` (Strix Halo).

---

## Phase 7: Build llama.cpp with ROCm

### 7.1 Clone and Build
```bash
cd ~/sc-gen-6
git clone https://github.com/ggerganov/llama.cpp llama-cpp-rocm
cd llama-cpp-rocm
make GGML_HIP=1 -j$(nproc)
```

### 7.2 Verify Build
```bash
./llama-server --help | head -5
```

---

## Phase 8: Install llama-swap

```bash
wget https://github.com/mostlygeek/llama-swap/releases/latest/download/llama-swap_linux_amd64.tar.gz -O /tmp/llama-swap.tar.gz
tar -xzf /tmp/llama-swap.tar.gz -C /tmp
sudo mv /tmp/llama-swap /usr/local/bin/
llama-swap --version
```

---

## Phase 9: Configure and Run

### 9.1 Update llama-swap Config
Create `~/sc-gen-6/llama-cpp/config.yaml`:
```yaml
healthCheckTimeout: 300
logLevel: info

models:
  nemotron-embed-8b:
    cmd: /home/YOUR_USERNAME/sc-gen-6/llama-cpp-rocm/llama-server --port ${PORT} -m /home/YOUR_USERNAME/model-cache/embeddings/llama-embed-nemotron-8b-Q4_K_M.gguf -c 8192 -ngl 99 --embedding
    ttl: 3600
    healthCheckTimeout: 120
    aliases:
      - embedding

  nemotron3-nano-30b-q8:
    cmd: /home/YOUR_USERNAME/sc-gen-6/llama-cpp-rocm/llama-server --port ${PORT} -m /home/YOUR_USERNAME/model-cache/nemotron3-nano-30b-q8/Nemotron-3-Nano-30B-A3B-Q8_0.gguf -c 131072 -ngl 999 --parallel 1 -fa
    ttl: 3600
    healthCheckTimeout: 300
    aliases:
      - nemotron
      - default

groups:
  rag:
    members:
      - nemotron-embed-8b
    swap: false
```

### 9.2 Run SC Gen 6
```bash
cd ~/sc-gen-6
source .venv-linux/bin/activate
python main.py
```

---

## Daily Usage

### Start SC Gen 6
```bash
cd ~/sc-gen-6
source .venv-linux/bin/activate
python main.py
```

### Use Antigravity
```bash
~/cursor.AppImage
# Open folder: ~/sc-gen-6
```

### Switch to Windows
Restart → Select "Windows Boot Manager" from GRUB menu

---

## Troubleshooting

### Windows partition won't mount
```bash
sudo ntfsfix /dev/nvme0n1p3
```

### ROCm not detecting GPU
```bash
groups | grep -E "render|video"
# If missing:
sudo usermod -a -G render,video $USER
sudo reboot
```

### Python packages missing
```bash
source .venv-linux/bin/activate
pip install -r requirements.txt
```
