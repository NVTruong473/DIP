from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    input_video: str = "/content/drive/MyDrive/DIP/video1.mp4"
    output_dir: str = "/content/drive/MyDrive/DIP/outputs"
    models_dir: str = "/content/drive/MyDrive/DIP/models"

    # Final project scope: traffic-sign detection only.
    sign_conf: float = 0.25
    sign_imgsz: int = 640
    use_dip_enhancement: bool = False
    sign_tiled: bool = True

    # Short temporal hold reduces one-frame flicker in video.
    sign_hold_frames: int = 8
    sign_iou_match: float = 0.30

    # Keep output clean by default.
    show_hud: bool = False

    # Process every frame by default. Use 2 only when a faster preview is needed.
    frame_stride: int = 1

    def ensure_dirs(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
