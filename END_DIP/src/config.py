from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class AppConfig:
    input_video: str = "/content/drive/MyDrive/DIP/video1.mp4"
    output_dir: str = "/content/drive/MyDrive/DIP/outputs"
    models_dir: str = "/content/drive/MyDrive/DIP/models"

    # Detection thresholds.
    sign_conf: float = 0.25
    plate_conf: float = 0.30
    vehicle_conf: float = 0.30

    # Inference sizes. Plate detection uses a larger size because license plates
    # are much smaller than the containing vehicle in 1080p road footage.
    sign_imgsz: int = 640
    plate_imgsz: int = 960
    vehicle_imgsz: int = 640

    detect_signs: bool = True
    detect_plates: bool = True
    use_dip_enhancement: bool = False
    sign_tiled: bool = True

    # Keep the final video visually clean. Vehicle boxes are used only for
    # association and are not rendered.
    show_hud: bool = False
    save_plate_crops: bool = True

    # Small temporal holds reduce one-frame blinking.
    sign_hold_frames: int = 8
    sign_iou_match: float = 0.30
    plate_hold_frames: int = 4
    plate_iou_match: float = 0.25

    # COCO: car=2, bus=5, truck=7. Motorcycles are intentionally excluded.
    vehicle_classes: Tuple[int, ...] = (2, 5, 7)

    # Process every frame by default. Set to 2 for a faster demo.
    frame_stride: int = 1

    def ensure_dirs(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
