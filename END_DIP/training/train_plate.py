from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model_manager import ModelManager


def main():
    p = argparse.ArgumentParser(description="Fine-tune the plate detector on Vietnamese car plates")
    p.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    p.add_argument("--models-dir", default="/content/drive/MyDrive/DIP/models")
    p.add_argument("--runs-dir", default="/content/drive/MyDrive/DIP/training_runs")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=6)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("[WARN] CUDA GPU not detected. Training will be much slower.")

    manager = ModelManager(args.models_dir)
    base_model = manager.plate_model(prefer_finetuned=False)
    print("Base plate model:", base_model)

    model = YOLO(base_model)
    result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=0 if torch.cuda.is_available() else "cpu",
        project=args.runs_dir,
        name="vn_car_plate_finetune",
        exist_ok=True,
        pretrained=True,
        cache=False,
        verbose=True,
    )

    save_dir = Path(result.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found at {best}")

    target = Path(args.models_dir) / "plate_best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target)
    print("Saved fine-tuned model:", target)
    print("The main pipeline will automatically prefer plate_best.pt on the next run.")


if __name__ == "__main__":
    main()
