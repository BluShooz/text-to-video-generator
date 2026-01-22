#!/usr/bin/env python3
"""
Model Download Script
Downloads all required model checkpoints for the text-to-video generator.
"""

import os
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for file downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, description: str = "Downloading"):
    """Download a file with progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"  ✓ {output_path.name} already exists")
        return
    
    print(f"  ↓ Downloading {description}...")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
    
    print(f"  ✓ Saved to {output_path}")


def main():
    print("=" * 60)
    print("Text-to-Video Generator - Model Download")
    print("=" * 60)
    
    # Define checkpoint directory
    project_root = Path(__file__).parent.parent
    checkpoints_dir = project_root / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # ==================== Wav2Lip Models ====================
    print("\n📹 Wav2Lip Models:")
    
    wav2lip_models = {
        "wav2lip_gan.pth": {
            "url": "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth",
            "description": "Wav2Lip GAN checkpoint (for better quality)"
        },
        "s3fd.pth": {
            "url": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
            "description": "S3FD face detection model"
        }
    }
    
    for filename, info in wav2lip_models.items():
        output_path = checkpoints_dir / filename
        try:
            download_file(info["url"], output_path, info["description"])
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {e}")
            print(f"    Please download manually from: {info['url']}")
    
    # ==================== HuggingFace Models ====================
    print("\n🤗 HuggingFace Models (will be auto-downloaded on first run):")
    
    hf_models = [
        ("THUDM/CogVideoX-2b", "CogVideoX 2B text-to-video model"),
        ("coqui/XTTS-v2", "XTTS-v2 text-to-speech model"),
    ]
    
    for model_id, description in hf_models:
        print(f"  → {model_id}: {description}")
        print(f"    (Auto-downloaded when first used)")
    
    # ==================== Real-ESRGAN ====================
    print("\n📺 Real-ESRGAN Models (auto-downloaded from PyPI package):")
    print("  → RealESRGAN_x4plus: General 4x upscaler")
    print("    (Auto-downloaded when first used)")
    
    # ==================== Instructions ====================
    print("\n" + "=" * 60)
    print("SETUP INSTRUCTIONS")
    print("=" * 60)
    
    print("""
1. Create and activate a virtual environment:
   $ python -m venv venv
   $ source venv/bin/activate  # On Windows: venv\\Scripts\\activate

2. Install dependencies:
   $ pip install -r requirements.txt

3. If any downloads failed, manually download and place in ./checkpoints/:
   - wav2lip_gan.pth → https://github.com/Rudrabha/Wav2Lip/releases
   - s3fd.pth → https://www.adrianbulat.com/downloads/python-fan/

4. Copy environment config:
   $ cp .env.example .env

5. Start the server:
   $ uvicorn api.main:app --reload

6. Open http://localhost:8000 in your browser
""")
    
    print("✅ Setup complete! Large models will download on first use.")


if __name__ == "__main__":
    main()
