from __future__ import annotations

import argparse
import json

from src.config import AppConfig
from src.video_processor import VideoProcessor


def build_parser():
    p = argparse.ArgumentParser(description="Vietnam traffic-sign detection with short English labels")
    p.add_argument("--input", default="/content/drive/MyDrive/DIP/video1.mp4")
    p.add_argument("--output-dir", default="/content/drive/MyDrive/DIP/outputs")
    p.add_argument("--models-dir", default="/content/drive/MyDrive/DIP/models")
    p.add_argument("--output", default=None)
    p.add_argument("--sign-conf", type=float, default=0.25)
    p.add_argument("--sign-imgsz", type=int, default=640)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--show-hud", action="store_true")
    p.add_argument("--hide-confidence", action="store_true")
    return p


def main():
    a = build_parser().parse_args()
    cfg = AppConfig(
        input_video=a.input,
        output_dir=a.output_dir,
        models_dir=a.models_dir,
        sign_conf=a.sign_conf,
        sign_imgsz=a.sign_imgsz,
        frame_stride=max(1, a.frame_stride),
        show_hud=a.show_hud,
        show_confidence=not a.hide_confidence,
    )
    print(json.dumps({
        "task": "traffic_sign_detection_only",
        "input": cfg.input_video,
        "models_dir": cfg.models_dir,
        "output_dir": cfg.output_dir,
        "sign_conf": cfg.sign_conf,
        "sign_imgsz": cfg.sign_imgsz,
        "tiled_inference": cfg.sign_tiled,
    }, indent=2))
    summary = VideoProcessor(cfg).process(cfg.input_video, output_path=a.output)
    print("\nDONE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
