from __future__ import annotations

import argparse
from pathlib import Path

from src.model_manager import ModelManager


def main():
    p = argparse.ArgumentParser(
        description="Download/cache the traffic-sign model required by Colab."
    )
    p.add_argument(
        "--models-dir",
        default="/content/drive/MyDrive/DIP/models",
    )
    args = p.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    manager = ModelManager(str(models_dir))
    path = manager.sign_model()
    print("Traffic-sign model ready:")
    print(path)


if __name__ == "__main__":
    main()
