from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    p.add_argument("--models-dir", default="/content/drive/MyDrive/DIP/models")
    p.add_argument("--dataset-dir", default="/content/drive/MyDrive/DIP/datasets/vn_car_plate_8255")
    p.add_argument("--runs-dir", default="/content/drive/MyDrive/DIP/training_runs")
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    target = Path(args.models_dir)/"plate_best.pt"
    if target.exists():
        print("plate_best.pt already exists; skip training:", target)
        return
    if not args.api_key:
        raise RuntimeError("Set ROBOFLOW_API_KEY to fine-tune the optional Vietnamese plate detector.")

    from roboflow import Roboflow
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace("phms-workspace-ialpp").project("vietnamese-car-license-plate-dwwrm")
    dataset = project.version(2).download("yolov8", location=args.dataset_dir, overwrite=False)
    data_yaml = Path(dataset.location)/"data.yaml"

    from src.model_manager import ModelManager
    base = ModelManager(args.models_dir).plate_model()
    model = YOLO(base)
    result = model.train(
        data=str(data_yaml), epochs=args.epochs, imgsz=640, batch=args.batch,
        device=0 if torch.cuda.is_available() else "cpu",
        project=args.runs_dir, name="vn_plate_finetune", exist_ok=True,
        patience=5, workers=2, amp=True, cache=False,
        degrees=3.0, perspective=0.0005, translate=.08, scale=.35,
        fliplr=.5, mosaic=.7, close_mosaic=3,
    )
    best = Path(result.save_dir)/"weights"/"best.pt"
    if not best.exists(): raise FileNotFoundError(best)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best,target)
    print("Persistent Vietnamese plate detector:", target)


if __name__ == "__main__":
    main()
