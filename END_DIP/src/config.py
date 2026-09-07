from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class AppConfig:
    input_video: str = "/content/drive/MyDrive/DIP/video1.mp4"
    output_dir: str = "/content/drive/MyDrive/DIP/outputs"
    models_dir: str = "/content/drive/MyDrive/DIP/models"

    # Three-task pipeline.
    detect_signs: bool = True
    detect_helmet: bool = True
    detect_plates: bool = True

    # Confidence thresholds. Deliberately conservative to suppress road clutter.
    sign_conf: float = 0.32
    scene_conf: float = 0.34
    helmet_conf: float = 0.38
    plate_conf: float = 0.34

    sign_imgsz: int = 768
    scene_imgsz: int = 640
    helmet_imgsz: int = 640
    plate_imgsz: int = 960

    # Full frame = both traffic lanes. Scene classes are COCO:
    # person=0, motorcycle=3, car=2, bus=5, truck=7.
    scene_classes: Tuple[int, ...] = (0, 2, 3, 5, 7)

    # Traffic signs: tiled inference helps small distant objects.
    sign_tiled: bool = True
    sign_track_iou: float = 0.28
    sign_confirm_hits: int = 2

    # Rider/helmet association.
    rider_max_motorcycle_distance: float = 1.25
    helmet_vote_window: int = 10
    helmet_min_votes: int = 4
    helmet_stable_ratio: float = 0.70

    # License plate association + OCR quality gates.
    plate_track_iou: float = 0.25
    plate_confirm_hits: int = 2
    plate_min_width_px: int = 42
    plate_min_height_px: int = 14
    plate_min_sharpness: float = 30.0
    ocr_every_n_frames: int = 3
    ocr_min_conf: float = 0.35
    ocr_min_votes: int = 2

    # Lighting adaptation is used only for inference copies; the saved video
    # remains visually faithful to the original input.
    adaptive_lighting: bool = True
    night_v_threshold: float = 68.0
    glare_v_threshold: float = 205.0

    # Visual policy: never draw internal person/vehicle/motorcycle boxes.
    # Only signs, helmet status near a rider's head, and license plates appear.
    show_hud: bool = False
    show_confidence: bool = True

    # Inference every frame by default. Set 2 for a faster preview.
    frame_stride: int = 1

    def ensure_dirs(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
