from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from ultralytics import YOLO


def find_data_yaml(root: Path):
    for name in ("data.yaml", "dataset.yaml", "helmet.yaml"):
        found = list(root.rglob(name))
        if found:
            return found[0]
    # Fallback for a standard YOLO directory layout.
    train = next(iter(root.rglob("train/images")), None)
    val = next(iter(root.rglob("val/images")), None) or next(iter(root.rglob("valid/images")), None)
    test = next(iter(root.rglob("test/images")), None)
    if train and val:
        cfg = {"train":str(train),"val":str(val),"names":["helmet","no_helmet"],"nc":2}
        if test: cfg["test"] = str(test)
        out = root/"data_autogen.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return out
    raise FileNotFoundError("Could not locate YOLO train/val labels in downloaded helmet dataset")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="/content/drive/MyDrive/DIP/datasets/traffic_helmet_42559")
    p.add_argument("--models-dir", default="/content/drive/MyDrive/DIP/models")
    p.add_argument("--runs-dir", default="/content/drive/MyDrive/DIP/training_runs")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--fraction", type=float, default=1.0)
    args = p.parse_args()

    target = Path(args.models_dir)/"helmet_best.pt"
    if target.exists():
        print("helmet_best.pt already exists; skip training:", target)
        return

    ds = Path(args.dataset_dir); ds.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id="thundarstrom/traffic-helmet-violation", repo_type="dataset", local_dir=str(ds))
    data = find_data_yaml(ds)
    print("Dataset YAML:", data)
    print("Dataset: 42,559 traffic images, about 126k helmet/no-helmet boxes (CC BY 4.0).")

    # Start from a road-rider helmet checkpoint rather than generic COCO.
    from src.model_manager import ModelManager
    base = ModelManager(args.models_dir).helmet_base_model()
    model = YOLO(base)
    result = model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        fraction=args.fraction, device=0 if torch.cuda.is_available() else "cpu",
        project=args.runs_dir, name="helmet_large_finetune", exist_ok=True,
        patience=5, workers=2, cache=False, amp=True,
        hsv_h=.015, hsv_s=.50, hsv_v=.40, degrees=3, translate=.08, scale=.35,
        fliplr=.5, mosaic=.7, close_mosaic=3,
    )
    best = Path(result.save_dir)/"weights"/"best.pt"
    if not best.exists(): raise FileNotFoundError(best)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target)
    print("Persistent fine-tuned helmet model:", target)


if __name__ == "__main__":
    main()
