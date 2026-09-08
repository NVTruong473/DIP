from __future__ import annotations

import argparse
from pathlib import Path

from src.model_manager import ModelManager


def main():
    p = argparse.ArgumentParser(description="Download/cache only the traffic-sign model in Google Drive")
    p.add_argument("--models-dir", default="/content/drive/MyDrive/DIP/models")
    args = p.parse_args()

    root = Path(args.models_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = ModelManager(str(root)).sign_model()
    print("Traffic-sign model ready and persistent in Drive:")
    print(path)


if __name__ == "__main__":
    main()
