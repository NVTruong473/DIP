from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
import yaml


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def find_yaml(root: Path) -> Path:
    candidates = list(root.rglob("data.yaml")) or list(root.rglob("*.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No dataset YAML found under {root}")
    return candidates[0]


def parse_names(names):
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    raise ValueError("Unsupported names format in data.yaml")


def rewrite_labels(dataset_root: Path, original_names: dict) -> tuple[int, int]:
    with_id = without_id = None
    for idx, name in original_names.items():
        n = norm(name)
        if "withouthelmet" in n or "nohelmet" in n:
            without_id = idx
        elif "withhelmet" in n or n == "helmet":
            with_id = idx

    if with_id is None or without_id is None:
        raise ValueError(f"Cannot identify With Helmet / Without Helmet classes from: {original_names}")

    mapping = {with_id: 0, without_id: 1}
    rewritten = kept = 0
    for label_path in dataset_root.rglob("*.txt"):
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        output = []
        is_yolo_label = True
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 5:
                is_yolo_label = False
                break
            try:
                cls = int(float(parts[0]))
                [float(x) for x in parts[1:5]]
            except Exception:
                is_yolo_label = False
                break
            if cls in mapping:
                parts[0] = str(mapping[cls])
                output.append(" ".join(parts))
                kept += 1

        if is_yolo_label:
            label_path.write_text("\n".join(output) + ("\n" if output else ""), encoding="utf-8")
            rewritten += 1

    return rewritten, kept


def main():
    p = argparse.ArgumentParser(description="Download Roboflow helmet dataset v5 and convert it to 2 classes.")
    p.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    p.add_argument("--workspace", default="gw-khadatkar-and-sv-wasule")
    p.add_argument("--project", default="helmet-and-no-helmet-rider-detection")
    p.add_argument("--version", type=int, default=5)
    p.add_argument("--target", default="/content/drive/MyDrive/DIP/datasets/helmet_rf_v5")
    args = p.parse_args()

    if not args.api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is required only for optional fine-tuning. Set it in the Colab environment; do not commit it.")

    from roboflow import Roboflow

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    dataset = version.download("yolov8", location=str(target), overwrite=False)
    dataset_root = Path(dataset.location)
    yaml_path = find_yaml(dataset_root)
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = parse_names(cfg["names"])

    print("Original classes:", names)
    rewritten, kept = rewrite_labels(dataset_root, names)

    split_aliases = {"train": ["train/images"], "val": ["valid/images", "val/images"], "test": ["test/images"]}
    for split, candidates in split_aliases.items():
        chosen = None
        for rel in candidates:
            cand = dataset_root / rel
            if cand.exists():
                chosen = cand.resolve()
                break
        if chosen is None and cfg.get(split):
            chosen = (yaml_path.parent / str(cfg[split])).resolve()
        if chosen is not None:
            cfg[split] = str(chosen)

    cfg.pop("path", None)
    cfg["nc"] = 2
    cfg["names"] = ["With Helmet", "Without Helmet"]
    out_yaml = dataset_root / "data_helmet_2class.yaml"
    out_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"Rewrote {rewritten} YOLO label files; kept {kept} helmet/no-helmet boxes.")
    print("Training YAML:", out_yaml)


if __name__ == "__main__":
    main()
