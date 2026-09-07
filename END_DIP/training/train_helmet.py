from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model_manager import ModelManager


def main():
    p = argparse.ArgumentParser(description="Fine-tune traffic helmet YOLO for a Colab/T4 budget.")
    p.add_argument("--data", required=True)
    p.add_argument("--models-dir", default="/content/drive/MyDrive/DIP/models")
    p.add_argument("--runs-dir", default="/content/drive/MyDrive/DIP/training_runs")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--large-base", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: GPU not detected. Fine-tuning is intended for a Colab GPU runtime.")

    manager = ModelManager(args.models_dir)
    base = manager.helmet_model_m() if args.large_base else manager.helmet_model(prefer_finetuned=False)
    print("Base checkpoint:", base)

    model = YOLO(base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0 if torch.cuda.is_available() else "cpu",
        project=args.runs_dir,
        name="helmet_finetune",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        patience=7,
        amp=True,
        cache=False,
        workers=2,
        fraction=args.fraction,
        cos_lr=True,
        close_mosaic=5,
        hsv_h=0.015,
        hsv_s=0.50,
        hsv_v=0.35,
        degrees=2.0,
        translate=0.10,
        scale=0.40,
        fliplr=0.5,
    )

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training ended but best.pt was not found at {best}")

    dst = Path(args.models_dir) / "helmet_best.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dst)
    print("Fine-tuned model saved to:", dst)


if __name__ == "__main__":
    main()
