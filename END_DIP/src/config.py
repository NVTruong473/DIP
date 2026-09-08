from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    input_video: str = "/content/drive/MyDrive/DIP/video1.mp4"
    output_dir: str = "/content/drive/MyDrive/DIP/outputs"
    models_dir: str = "/content/drive/MyDrive/DIP/models"

    # Current-video scope: traffic-sign detection only.
    sign_conf: float = 0.25
    sign_imgsz: int = 640
    sign_tiled: bool = True

    # Smooth only real current-frame detections. No predicted/ghost boxes.
    sign_track_iou: float = 0.30
    sign_confirm_hits: int = 1
    sign_smooth_alpha: float = 0.70

    # English labels + clean output.
    show_confidence: bool = True
    show_hud: bool = False

    # Process every frame by default.
    frame_stride: int = 1

    def ensure_dirs(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
