import urllib.request
import json
import zipfile
import io
import os
import sys

TARGET_DIR = "llama-cpp"
API_URL = "https://api.github.com/repos/ggerganov/llama.cpp/releases"

def update_llama():
    print(f"Fetching releases from {API_URL}...")
    try:
        req = urllib.request.Request(API_URL, headers={'User-Agent': 'Python/3.11'})
        with urllib.request.urlopen(req) as response:
            releases = json.loads(response.read().decode())
            
        target_release = None
        asset_url = None
        asset_name = None
        
        for release in releases[:5]: # Check last 5 releases
            tag = release.get("tag_name")
            print(f"Checking release {tag}...")
            for asset in release.get("assets", []):
                name = asset.get("name", "")
                if "vulkan" in name and "win" in name and "x64" in name:
                    print(f"Found candidate: {name}")
                    asset_url = asset.get("browser_download_url")
                    asset_name = name
                    target_release = tag
                    break
            
            if asset_url:
                break
            
            if asset_url:
                break
                
        if not asset_url:
            print("Error: Could not find suitable Windows binary asset in recent releases.")
            return

        print(f"Found {asset_name} in release {target_release}")
        print(f"Downloading from {asset_url}...")
        
        req = urllib.request.Request(asset_url, headers={'User-Agent': 'Python/3.11'})
        with urllib.request.urlopen(req) as response:
            zip_data = response.read()
            
        print(f"Downloaded {len(zip_data)} bytes.")
        
        print("Extracting...")
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            if not os.path.exists(TARGET_DIR):
                os.makedirs(TARGET_DIR)
            z.extractall(TARGET_DIR)
            
        print(f"Successfully updated llama.cpp to {target_release} in {TARGET_DIR}")
        
    except Exception as e:
        print(f"Error updating llama.cpp: {e}")

if __name__ == "__main__":
    update_llama()
