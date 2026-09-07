from __future__ import annotations

import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/content/drive/MyDrive/DIP/models/helmet_best.pt")
    p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    model = YOLO(args.model)
    metrics = model.val(data=args.data, split="test", imgsz=args.imgsz)
    print("mAP50:", float(metrics.box.map50))
    print("mAP50-95:", float(metrics.box.map))


if __name__ == "__main__":
    main()
