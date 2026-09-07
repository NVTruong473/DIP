from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Evaluate the fine-tuned Vietnamese car plate detector")
    p.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    p.add_argument("--model", default="/content/drive/MyDrive/DIP/models/plate_best.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", default="test", choices=["val", "test"])
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        split=args.split,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=True,
    )

    print("mAP50-95:", float(metrics.box.map))
    print("mAP50:", float(metrics.box.map50))
    print("mAP75:", float(metrics.box.map75))


if __name__ == "__main__":
    main()
