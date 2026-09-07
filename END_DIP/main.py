from __future__ import annotations

import argparse
import json

from src.config import AppConfig
from src.video_processor import VideoProcessor


def build_parser():
    p = argparse.ArgumentParser(description="Vietnam traffic-sign + car license-plate detection")
    p.add_argument("--input", default="/content/drive/MyDrive/DIP/video1.mp4", help="Input video path")
    p.add_argument("--output-dir", default="/content/drive/MyDrive/DIP/outputs")
    p.add_argument("--models-dir", default="/content/drive/MyDrive/DIP/models")
    p.add_argument("--output", default=None, help="Optional exact output .mp4 path")
    p.add_argument("--sign-conf", type=float, default=0.25)
    p.add_argument("--plate-conf", type=float, default=0.30)
    p.add_argument("--vehicle-conf", type=float, default=0.30)
    p.add_argument("--sign-imgsz", type=int, default=640)
    p.add_argument("--plate-imgsz", type=int, default=960)
    p.add_argument("--vehicle-imgsz", type=int, default=640)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--no-signs", action="store_true")
    p.add_argument("--no-plates", action="store_true")
    p.add_argument("--show-hud", action="store_true")
    p.add_argument("--no-plate-crops", action="store_true")
    p.add_argument("--dip-enhance", action="store_true", help="Apply mild CLAHE + unsharp preprocessing before sign YOLO")
    return p


def main():
    args = build_parser().parse_args()
    cfg = AppConfig(
        input_video=args.input,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        sign_conf=args.sign_conf,
        plate_conf=args.plate_conf,
        vehicle_conf=args.vehicle_conf,
        sign_imgsz=args.sign_imgsz,
        plate_imgsz=args.plate_imgsz,
        vehicle_imgsz=args.vehicle_imgsz,
        detect_signs=not args.no_signs,
        detect_plates=not args.no_plates,
        show_hud=args.show_hud,
        save_plate_crops=not args.no_plate_crops,
        use_dip_enhancement=args.dip_enhance,
        frame_stride=max(1, args.frame_stride),
    )

    print("Configuration:")
    print(json.dumps({
        "input": cfg.input_video,
        "output_dir": cfg.output_dir,
        "models_dir": cfg.models_dir,
        "sign_conf": cfg.sign_conf,
        "plate_conf": cfg.plate_conf,
        "vehicle_conf": cfg.vehicle_conf,
        "frame_stride": cfg.frame_stride,
    }, indent=2, ensure_ascii=False))

    processor = VideoProcessor(cfg)
    summary = processor.process(cfg.input_video, output_path=args.output)
    print("\nDONE")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
