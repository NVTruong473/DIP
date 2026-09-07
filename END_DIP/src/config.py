from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import json
import os

Point = Tuple[float, float]

# Tuned from the supplied 1920x1080 road video. Coordinates are normalized
# so the same polygon scales to any resolution. The polygon intentionally
# covers the right-hand carriageway rather than blindly using x > width / 2.
DEFAULT_RIGHT_ROAD_ROI: List[Point] = [
    (0.48, 0.36),
    (1.00, 0.36),
    (1.00, 1.00),
    (0.38, 1.00),
]


def default_drive_root() -> Path:
    drive = Path("/content/drive/MyDrive/DIP")
    return drive if drive.parent.exists() else Path.cwd()


def parse_roi_json(value: str | None) -> List[Point]:
    if not value:
        return list(DEFAULT_RIGHT_ROAD_ROI)
    data = json.loads(value)
    if not isinstance(data, list) or len(data) < 3:
        raise ValueError("ROI must be a JSON list with at least 3 [x, y] points.")
    pts: List[Point] = []
    for p in data:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError("Each ROI point must be [x, y].")
        x, y = float(p[0]), float(p[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("ROI coordinates must be normalized to [0, 1].")
        pts.append((x, y))
    return pts


@dataclass
class AppConfig:
    input_video: str = "/content/drive/MyDrive/DIP/video1.mp4"
    output_dir: str = "/content/drive/MyDrive/DIP/outputs"
    models_dir: str = "/content/drive/MyDrive/DIP/models"

    # Primary detection thresholds.
    sign_conf: float = 0.25
    helmet_conf: float = 0.35
    scene_conf: float = 0.30

    # Exported sign model uses 640; tiled inference preserves more pixels for small signs.
    sign_imgsz: int = 640
    helmet_imgsz: int = 640
    scene_imgsz: int = 640

    detect_signs: bool = True
    detect_helmet: bool = True
    show_roi: bool = True
    use_dip_enhancement: bool = False
    sign_tiled: bool = True

    # Right-road polygon, normalized to frame width/height.
    right_road_roi: List[Point] = field(default_factory=lambda: list(DEFAULT_RIGHT_ROAD_ROI))

    # Temporal logic: never infer "no helmet" simply because a helmet was missed.
    vote_window: int = 12
    min_votes: int = 3
    stable_ratio: float = 0.65
    state_ttl_frames: int = 15
    violation_snapshot_cooldown_sec: float = 3.0

    # Traffic sign temporal hold reduces frame-to-frame blinking.
    sign_hold_frames: int = 8
    sign_iou_match: float = 0.30

    # Scene model uses COCO: person=0, bicycle=1, motorcycle=3.
    scene_classes: Tuple[int, ...] = (0, 1, 3)

    # Process every frame by default. Set to 2 for a faster demo if needed.
    frame_stride: int = 1

    def ensure_dirs(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
