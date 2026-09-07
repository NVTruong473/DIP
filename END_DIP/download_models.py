from __future__ import annotations

import argparse
from pathlib import Path

from src.model_manager import ModelManager


def main():
    p = argparse.ArgumentParser(description="Download/cache all detector weights in Google Drive")
    p.add_argument("--models-dir", default="/content/drive/MyDrive/DIP/models")
    args = p.parse_args()
    root = Path(args.models_dir); root.mkdir(parents=True, exist_ok=True)
    m = ModelManager(str(root))
    artifacts = {
        "traffic_sign": m.sign_model(),
        "scene_yolo11n": m.scene_model(),
        "helmet": m.helmet_model(),
        "license_plate": m.plate_model(),
    }
    print("Models ready (persistent in Drive):")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
