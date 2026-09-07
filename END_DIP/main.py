from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import AppConfig, DEFAULT_RIGHT_ROAD_ROI, parse_roi_json
from src.video_processor import VideoProcessor


def build_parser():
    p = argparse.ArgumentParser(description="Vietnam traffic-sign + right-road helmet monitoring")
    p.add_argument("--input", default="/content/drive/MyDrive/DIP/video1.mp4", help="Input video path")
    p.add_argument("--output-dir", default="/content/drive/MyDrive/DIP/outputs")
    p.add_argument("--models-dir", default="/content/drive/MyDrive/DIP/models")
    p.add_argument("--output", default=None, help="Optional exact output .mp4 path")
    p.add_argument("--roi-json", default=json.dumps(DEFAULT_RIGHT_ROAD_ROI), help="Normalized polygon [[x,y], ...]")
    p.add_argument("--sign-conf", type=float, default=0.25)
    p.add_argument("--helmet-conf", type=float, default=0.35)
    p.add_argument("--scene-conf", type=float, default=0.30)
    p.add_argument("--sign-imgsz", type=int, default=640)
    p.add_argument("--helmet-imgsz", type=int, default=640)
    p.add_argument("--scene-imgsz", type=int, default=640)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--no-signs", action="store_true")
    p.add_argument("--no-helmet", action="store_true")
    p.add_argument("--hide-roi", action="store_true")
    p.add_argument("--dip-enhance", action="store_true", help="Apply mild CLAHE + unsharp preprocessing before sign YOLO")
    return p


def main():
    args = build_parser().parse_args()
    cfg = AppConfig(
        input_video=args.input,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        sign_conf=args.sign_conf,
        helmet_conf=args.helmet_conf,
        scene_conf=args.scene_conf,
        sign_imgsz=args.sign_imgsz,
        helmet_imgsz=args.helmet_imgsz,
        scene_imgsz=args.scene_imgsz,
        detect_signs=not args.no_signs,
        detect_helmet=not args.no_helmet,
        show_roi=not args.hide_roi,
        use_dip_enhancement=args.dip_enhance,
        right_road_roi=parse_roi_json(args.roi_json),
        frame_stride=max(1, args.frame_stride),
    )

    print("Configuration:")
    print(json.dumps({
        "input": cfg.input_video,
        "output_dir": cfg.output_dir,
        "models_dir": cfg.models_dir,
        "roi": cfg.right_road_roi,
        "sign_conf": cfg.sign_conf,
        "helmet_conf": cfg.helmet_conf,
    }, indent=2, ensure_ascii=False))

    processor = VideoProcessor(cfg)
    summary = processor.process(cfg.input_video, output_path=args.output)
    print("\nDONE")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
