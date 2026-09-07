from __future__ import annotations

import argparse
import json

from src.config import AppConfig
from src.video_processor import VideoProcessor


def build_parser():
    p = argparse.ArgumentParser(description="Traffic signs + helmet compliance + car plate OCR")
    p.add_argument("--input", default="/content/drive/MyDrive/DIP/video1.mp4")
    p.add_argument("--output-dir", default="/content/drive/MyDrive/DIP/outputs")
    p.add_argument("--models-dir", default="/content/drive/MyDrive/DIP/models")
    p.add_argument("--output", default=None)
    p.add_argument("--sign-conf", type=float, default=0.32)
    p.add_argument("--scene-conf", type=float, default=0.34)
    p.add_argument("--helmet-conf", type=float, default=0.38)
    p.add_argument("--plate-conf", type=float, default=0.34)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--no-signs", action="store_true")
    p.add_argument("--no-helmet", action="store_true")
    p.add_argument("--no-plates", action="store_true")
    p.add_argument("--show-hud", action="store_true")
    p.add_argument("--no-adaptive-lighting", action="store_true")
    return p


def main():
    a = build_parser().parse_args()
    cfg = AppConfig(
        input_video=a.input, output_dir=a.output_dir, models_dir=a.models_dir,
        sign_conf=a.sign_conf, scene_conf=a.scene_conf, helmet_conf=a.helmet_conf, plate_conf=a.plate_conf,
        detect_signs=not a.no_signs, detect_helmet=not a.no_helmet, detect_plates=not a.no_plates,
        show_hud=a.show_hud, adaptive_lighting=not a.no_adaptive_lighting,
        frame_stride=max(1,a.frame_stride),
    )
    print(json.dumps({
        "input":cfg.input_video,"models":cfg.models_dir,"output":cfg.output_dir,
        "tasks":{"signs":cfg.detect_signs,"helmet":cfg.detect_helmet,"plates_ocr":cfg.detect_plates},
        "thresholds":{"sign":cfg.sign_conf,"scene":cfg.scene_conf,"helmet":cfg.helmet_conf,"plate":cfg.plate_conf},
        "adaptive_lighting":cfg.adaptive_lighting,
    }, indent=2))
    summary = VideoProcessor(cfg).process(cfg.input_video, output_path=a.output)
    print("\nDONE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
