from __future__ import annotations

from typing import List

import torch
from ultralytics import YOLO

from src.common import Detection
from src.model_manager import ModelManager


class LicensePlateDetector:
    """YOLO license-plate detector.

    The model detects plates globally. A separate association step keeps only
    plates belonging to COCO car/bus/truck detections, so motorcycle plates are
    not shown in the final output.
    """

    def __init__(self, manager: ModelManager, conf: float = 0.30, imgsz: int = 960):
        self.conf = conf
        self.imgsz = imgsz
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model_path = manager.plate_model()
        self.model = YOLO(self.model_path, task="detect")

    def detect(self, frame) -> List[Detection]:
        result = self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        out: List[Detection] = []
        if result.boxes is None:
            return out

        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cls = int(box.cls[0])
            raw = str(names.get(cls, cls) if isinstance(names, dict) else names[cls])
            out.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    label="Biển số ô tô",
                    raw_label=raw,
                    confidence=float(box.conf[0]),
                    class_id=cls,
                )
            )
        return out
