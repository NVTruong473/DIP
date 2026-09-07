from __future__ import annotations

import argparse
import os
from pathlib import Path

from roboflow import Roboflow


DEFAULT_WORKSPACE = "phms-workspace-ialpp"
DEFAULT_PROJECT = "vietnamese-car-license-plate-dwwrm"
DEFAULT_VERSION = 2


def main():
    p = argparse.ArgumentParser(description="Download Vietnamese car license-plate data from Roboflow Universe")
    p.add_argument("--target", default="/content/drive/MyDrive/DIP/datasets/vn_car_plate")
    p.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--version", type=int, default=DEFAULT_VERSION)
    p.add_argument("--format", default="yolov11")
    args = p.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set ROBOFLOW_API_KEY first. In Colab: "
            "os.environ['ROBOFLOW_API_KEY'] = 'YOUR_KEY'"
        )

    target = Path(args.target)
    target.parent.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    version = rf.workspace(args.workspace).project(args.project).version(args.version)
    dataset = version.download(args.format, location=str(target), overwrite=True)

    location = Path(getattr(dataset, "location", target))
    yaml_candidates = list(location.rglob("data.yaml"))
    if not yaml_candidates:
        raise FileNotFoundError(f"Dataset downloaded but data.yaml was not found under {location}")

    data_yaml = yaml_candidates[0]
    print("Dataset:", location)
    print("YOLO data.yaml:", data_yaml)
    print("Next step:")
    print(
        "python training/train_plate.py "
        f"--data '{data_yaml}' "
        "--models-dir '/content/drive/MyDrive/DIP/models'"
    )


if __name__ == "__main__":
    main()
