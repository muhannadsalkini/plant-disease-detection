"""
download_dataset.py
-------------------
Downloads the PlantVillage dataset from Kaggle using the Kaggle API.

Prerequisites:
  1. Install the kaggle package:  pip install kaggle
  2. Place your Kaggle API token at ~/.kaggle/kaggle.json
     (Download from https://www.kaggle.com/settings → API → Create New Token)
  3. chmod 600 ~/.kaggle/kaggle.json   (macOS/Linux)

Usage:
  python download_dataset.py
"""

import os
import sys
import zipfile
import shutil

# ── Dataset identifier on Kaggle ──────────────────────────────────────────────
KAGGLE_DATASET = "emmarex/plantdisease"   # 54,306 images, 38 classes
DOWNLOAD_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EXTRACT_DIR    = os.path.join(DOWNLOAD_DIR, "PlantVillage")

# ─────────────────────────────────────────────────────────────────────────────

def check_kaggle_credentials():
    """Verify that ~/.kaggle/kaggle.json exists."""
    cred_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(cred_path):
        print("❌  Kaggle credentials not found at ~/.kaggle/kaggle.json")
        print("\nTo fix this:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Scroll to 'API' section → click 'Create New Token'")
        print("  3. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json")
        print("  4. Run:  chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)
    print("✅  Kaggle credentials found.")


def download_dataset():
    """Download the PlantVillage dataset zip from Kaggle."""
    import kaggle  # imported here so missing install gives a clean error

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"\n📥  Downloading dataset '{KAGGLE_DATASET}' …")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET,
        path=DOWNLOAD_DIR,
        unzip=False,
        quiet=False,
    )
    print("✅  Download complete.")


def extract_dataset():
    """Extract the downloaded zip into data/PlantVillage/."""
    zip_files = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".zip")]
    if not zip_files:
        print("❌  No zip file found in", DOWNLOAD_DIR)
        sys.exit(1)

    zip_path = os.path.join(DOWNLOAD_DIR, zip_files[0])
    print(f"\n📦  Extracting {zip_path} …")
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(EXTRACT_DIR)

    os.remove(zip_path)
    print(f"✅  Extracted to {EXTRACT_DIR}")


def normalise_structure():
    """
    The Kaggle dataset may be nested one extra level deep.
    Ensure that EXTRACT_DIR contains class-named subdirectories directly.
    e.g.  data/PlantVillage/Apple___Apple_scab/  ...
    """
    entries = os.listdir(EXTRACT_DIR)

    # If there is a single subdirectory and it contains class folders, flatten.
    if len(entries) == 1 and os.path.isdir(os.path.join(EXTRACT_DIR, entries[0])):
        inner = os.path.join(EXTRACT_DIR, entries[0])
        inner_entries = os.listdir(inner)
        print(f"ℹ️   Flattening nested directory: {entries[0]}")
        for item in inner_entries:
            shutil.move(os.path.join(inner, item), os.path.join(EXTRACT_DIR, item))
        os.rmdir(inner)

    # Count class directories
    class_dirs = [
        d for d in os.listdir(EXTRACT_DIR)
        if os.path.isdir(os.path.join(EXTRACT_DIR, d))
    ]
    print(f"\n📊  Found {len(class_dirs)} class directories:")
    for d in sorted(class_dirs):
        count = len(os.listdir(os.path.join(EXTRACT_DIR, d)))
        print(f"   {d:50s}  {count:5d} images")


def main():
    print("=" * 60)
    print("  PlantVillage Dataset Downloader")
    print("=" * 60)

    # Skip download if already extracted
    if os.path.exists(EXTRACT_DIR) and len(os.listdir(EXTRACT_DIR)) > 0:
        print(f"ℹ️   Dataset already exists at {EXTRACT_DIR}")
        normalise_structure()
        return

    check_kaggle_credentials()
    download_dataset()
    extract_dataset()
    normalise_structure()
    print("\n✅  Dataset ready for training!")


if __name__ == "__main__":
    main()
