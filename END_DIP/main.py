from __future__ import annotations

import argparse
import json

from src.config import AppConfig
from src.video_processor import VideoProcessor


def build_parser():
    p = argparse.ArgumentParser(
        description="Vietnamese traffic-sign detection with short English labels"
    )
    p.add_argument(
        "--input",
        default="/content/drive/MyDrive/DIP/video1.mp4",
        help="Input video path",
    )
    p.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/DIP/outputs",
    )
    p.add_argument(
        "--models-dir",
        default="/content/drive/MyDrive/DIP/models",
    )
    p.add_argument("--output", default=None, help="Optional exact output .mp4 path")
    p.add_argument("--sign-conf", type=float, default=0.25)
    p.add_argument("--sign-imgsz", type=int, default=640)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--show-hud", action="store_true")
    p.add_argument(
        "--dip-enhance",
        action="store_true",
        help="Apply mild CLAHE + unsharp preprocessing before sign YOLO",
    )
    return p


def main():
    args = build_parser().parse_args()
    cfg = AppConfig(
        input_video=args.input,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        sign_conf=args.sign_conf,
        sign_imgsz=args.sign_imgsz,
        frame_stride=max(1, args.frame_stride),
        show_hud=args.show_hud,
        use_dip_enhancement=args.dip_enhance,
    )

    print("Configuration:")
    print(
        json.dumps(
            {
                "input": cfg.input_video,
                "output_dir": cfg.output_dir,
                "models_dir": cfg.models_dir,
                "sign_conf": cfg.sign_conf,
                "sign_imgsz": cfg.sign_imgsz,
                "frame_stride": cfg.frame_stride,
            },
            indent=2,
        )
    )

    processor = VideoProcessor(cfg)
    summary = processor.process(cfg.input_video, output_path=args.output)
    print("\nDONE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
