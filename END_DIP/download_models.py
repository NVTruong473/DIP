from __future__ import annotations

import argparse
from src.model_manager import ModelManager


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default="/content/drive/MyDrive/DIP/models")
    p.add_argument("--large-helmet", action="store_true")
    args = p.parse_args()

    m = ModelManager(args.models_dir)
    print("Traffic sign:", m.sign_model())
    print("Sign mapping:", m.sign_mapping())
    print("Scene:", m.scene_model())
    print("Helmet:", m.helmet_model_m() if args.large_helmet else m.helmet_model())


if __name__ == "__main__":
    main()
