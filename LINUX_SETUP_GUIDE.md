# Ubuntu 25.10 Setup Guide for SC Gen 6

## How to View This Document in Linux

Once you boot into Linux, this file will be at:
```
/mnt/windows/Users/James/Desktop/SC Gen 6/LINUX_SETUP_GUIDE.md
```

Or view on GitHub:
```
https://github.com/jamescockburn47/sc-gen-6/blob/main/LINUX_SETUP_GUIDE.md
```

---

# PART 1: Windows Preparation (Do Before Rebooting)

## Step 1.1: Download Ubuntu 25.10

1. Go to: https://ubuntu.com/download/desktop
2. Download **Ubuntu 25.10** (should be ~5GB ISO file)
3. Save to Downloads folder

## Step 1.2: Download Rufus

1. Go to: https://rufus.ie/
2. Download **Rufus** (portable version is fine)
3. Save to Downloads folder

## Step 1.3: Create Bootable USB

1. Insert USB drive (8GB or larger) - **This will ERASE the USB!**
2. Open Rufus
3. Device: Select your USB drive
4. Boot selection: Click "SELECT" → choose the Ubuntu ISO you downloaded
5. Leave other settings as default
6. Click "START"
7. Wait for completion (5-10 minutes)

## Step 1.4: Shrink Windows Partition

1. Press `Win + X` → select "Disk Management"
2. Right-click on your **C: drive** (the largest partition)
3. Select "Shrink Volume"
4. Enter **102400** in "Enter the amount of space to shrink in MB" (this is 100GB)
5. Click "Shrink"
6. You should now see "Unallocated" space

## Step 1.5: Disable Fast Startup (Important!)

1. Press `Win + R` → type `powercfg.cpl` → Enter
2. Click "Choose what the power buttons do"
3. Click "Change settings that are currently unavailable"
4. **UNCHECK** "Turn on fast startup"
5. Click "Save changes"

## Step 1.6: Reboot into USB

1. Keep the USB inserted
2. Restart your computer
3. Press **F12** (or F2, Del, Esc - depends on your PC) repeatedly to enter boot menu
4. Select the USB drive
5. You should see the Ubuntu installer

---

# PART 2: Installing Ubuntu

## Step 2.1: Initial Ubuntu Screen

1. Select "Try or Install Ubuntu"
2. Wait for Ubuntu to load (1-2 minutes)

## Step 2.2: Installation Type

1. Select your language → Continue
2. Accessibility options → Continue (or adjust if needed)
3. Keyboard layout → Continue
4. Connect to WiFi if prompted
5. **IMPORTANT**: Select "Install Ubuntu alongside Windows Boot Manager"
   - This keeps Windows and adds Linux
6. Click "Install Now"

## Step 2.3: Partitioning

1. Ubuntu should show a slider to divide space
2. Drag it so Ubuntu gets ~100GB (the space you freed)
3. Click "Install Now"
4. Confirm: "Write changes to disk?" → Continue

## Step 2.4: User Setup

1. Select timezone
2. Create your user:
   - Your name: James
   - Computer name: (anything you like)
   - Username: james (or whatever)
   - Password: (choose a password)
3. Click "Continue"

## Step 2.5: Wait for Installation

- Takes 15-30 minutes
- When done, click "Restart Now"
- Remove the USB when prompted

---

# PART 3: First Boot into Ubuntu

## Step 3.1: Boot Menu (GRUB)

After restart, you'll see a menu:
- **Ubuntu** ← Select this
- Windows Boot Manager ← (This is your Windows, still there!)

Select Ubuntu and press Enter.

## Step 3.2: Log In

Enter your password and log in.

## Step 3.3: Open Terminal

1. Press `Ctrl + Alt + T` to open Terminal
2. Or click the app grid (bottom left) and search "Terminal"

---

# PART 4: System Setup

Copy and paste these commands into Terminal one section at a time.

## Step 4.1: Update System

```bash
sudo apt update && sudo apt upgrade -y
```

## Step 4.2: Mount Windows Partition

First, find your Windows partition:
```bash
lsblk -f | grep ntfs
```

You'll see something like `nvme0n1p3` - note this.

Create mount point and add to fstab:
```bash
sudo mkdir -p /mnt/windows
```

Mount it (replace nvme0n1p3 with your actual partition):
```bash
sudo mount -t ntfs-3g /dev/nvme0n1p3 /mnt/windows
```

Verify it worked:
```bash
ls "/mnt/windows/Users/James/Desktop/SC Gen 6"
```

You should see your SC Gen 6 files!

Make it permanent:
```bash
echo "$(sudo blkid /dev/nvme0n1p3 | grep -o 'UUID="[^"]*"') /mnt/windows ntfs-3g defaults,uid=1000,gid=1000 0 0" | sudo tee -a /etc/fstab
```

## Step 4.3: Install ROCm 7.0

```bash
# Download AMD GPU installer
wget https://repo.radeon.com/amdgpu-install/6.4.1/ubuntu/noble/amdgpu-install_6.4.60401-1_all.deb -O /tmp/amdgpu-install.deb

# Install it  
sudo apt install -y /tmp/amdgpu-install.deb

# Install ROCm
sudo amdgpu-install -y --usecase=rocm

# Add yourself to GPU groups
sudo usermod -a -G render,video $USER
```

**REBOOT NOW:**
```bash
sudo reboot
```

## Step 4.4: Verify GPU (After Reboot)

Log back in, open Terminal:
```bash
rocminfo | grep "gfx"
```

You should see `gfx1151` (Strix Halo).

---

# PART 5: Install Development Tools

## Step 5.1: Install Cursor IDE

```bash
# Download Cursor
wget "https://downloader.cursor.sh/linux/appImage/x64" -O ~/cursor.AppImage

# Make it executable
chmod +x ~/cursor.AppImage

# Run it
~/cursor.AppImage
```

Cursor will open. Sign in with your account.

## Step 5.2: Install llama-swap

```bash
# Download llama-swap
wget https://github.com/mostlygeek/llama-swap/releases/latest/download/llama-swap_linux_amd64.tar.gz -O /tmp/llama-swap.tar.gz

# Extract and install
tar -xzf /tmp/llama-swap.tar.gz -C /tmp
sudo mv /tmp/llama-swap /usr/local/bin/

# Verify
llama-swap --version
```

## Step 5.3: Build llama.cpp with ROCm

```bash
# Install build tools
sudo apt install -y build-essential cmake git

# Clone llama.cpp
cd /mnt/windows/Users/James/Desktop/SC\ Gen\ 6
git clone https://github.com/ggerganov/llama.cpp llama-cpp-linux

# Build with ROCm support
cd llama-cpp-linux
make GGML_HIP=1 -j$(nproc)

# Verify
./llama-server --help | head -5
```

---

# PART 6: Setup SC Gen 6

## Step 6.1: Create Python Virtual Environment

```bash
cd "/mnt/windows/Users/James/Desktop/SC Gen 6"

# Install Python venv
sudo apt install -y python3.12-venv python3-pip

# Create Linux-specific venv
python3 -m venv .venv-linux

# Activate it
source .venv-linux/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 6.2: Update llama-swap Config for Linux

Edit the config file:
```bash
nano llama-cpp/config.yaml
```

Replace the Windows paths with Linux paths:
```yaml
# Change FROM:
# cmd: C:/SCGen6/llama-cpp/llama-server.exe ...

# Change TO:
# cmd: /mnt/windows/Users/James/Desktop/SC Gen 6/llama-cpp-linux/llama-server ...
```

Save with `Ctrl+O`, Enter, then exit with `Ctrl+X`.

## Step 6.3: Run SC Gen 6

```bash
cd "/mnt/windows/Users/James/Desktop/SC Gen 6"
source .venv-linux/bin/activate
python main.py
```

---

# PART 7: Daily Usage

## To Boot Into Linux
1. Restart computer
2. Select "Ubuntu" from GRUB menu

## To Boot Into Windows
1. Restart computer  
2. Select "Windows Boot Manager" from GRUB menu

## To Run SC Gen 6 (Linux)
```bash
cd "/mnt/windows/Users/James/Desktop/SC Gen 6"
source .venv-linux/bin/activate
python main.py
```

## To Use Antigravity
1. Open Cursor IDE
2. Open folder: `/mnt/windows/Users/James/Desktop/SC Gen 6`
3. Use Antigravity exactly like on Windows!

---

# Troubleshooting

## "Windows partition not mounting"
```bash
# Windows might have hibernated. Boot Windows, disable fast startup, and shut down properly.
sudo ntfsfix /dev/nvme0n1p3
```

## "ROCm not detecting GPU"
```bash
# Check if you're in the right groups
groups | grep -E "render|video"

# If not, run and reboot:
sudo usermod -a -G render,video $USER
sudo reboot
```

## "llama-server crashes"
```bash
# Check ROCm is working
rocminfo | head -20

# Try Vulkan fallback
make GGML_VULKAN=1 -j$(nproc)
```
