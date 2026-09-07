from __future__ import annotations

import argparse
from src.model_manager import ModelManager


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default="/content/drive/MyDrive/DIP/models")
    args = p.parse_args()

    m = ModelManager(args.models_dir)
    print("Traffic sign:", m.sign_model())
    print("Sign mapping:", m.sign_mapping())
    print("Vehicle tracker:", m.scene_model())
    print("License plate:", m.plate_model())


if __name__ == "__main__":
    main()
