from __future__ import annotations

import argparse
from pathlib import Path

from src.model_manager import ModelManager


def main():
    p = argparse.ArgumentParser(description="Download/cache models required by the final Colab pipeline.")
    p.add_argument("--models-dir", default="/content/drive/MyDrive/DIP/models")
    args = p.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    manager = ModelManager(str(models_dir))
    artifacts = {
        "traffic_sign_model": manager.sign_model(),
        "traffic_sign_mapping": manager.sign_mapping(),
        "license_plate_model": manager.plate_model(),
        "vehicle_model": manager.scene_model(),
    }

    print("Models ready:")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
