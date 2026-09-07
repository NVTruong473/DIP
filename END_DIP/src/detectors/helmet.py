from __future__ import annotations

from typing import List, Optional, Tuple
import re
import torch
from ultralytics import YOLO

from src.common import Detection
from src.model_manager import ModelManager


def normalize_helmet_label(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "", label.lower())
    if "without" in s or "nohelmet" in s or s in {"face", "barehead"}:
        return "NO_HELMET"
    if "withhelmet" in s or s == "helmet" or "helmet" in s:
        return "HELMET"
    return "UNKNOWN"


class HelmetDetector:
    def __init__(self, manager: ModelManager, conf: float = 0.35, imgsz: int = 640):
        self.conf = conf
        self.imgsz = imgsz
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model_path = manager.helmet_model(prefer_finetuned=True)
        self.model = YOLO(self.model_path)

    def detect(self, frame, crop_box: Optional[Tuple[int, int, int, int]] = None) -> List[Detection]:
        ox = oy = 0
        source = frame
        if crop_box is not None:
            x1, y1, x2, y2 = crop_box
            ox, oy = x1, y1
            source = frame[y1:y2, x1:x2]
            if source.size == 0:
                return []

        result = self.model.predict(
            source=source,
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
            bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0].tolist()]
            cls = int(box.cls[0])
            raw = str(names.get(cls, cls) if isinstance(names, dict) else names[cls])
            status = normalize_helmet_label(raw)
            out.append(
                Detection(
                    box=(bx1 + ox, by1 + oy, bx2 + ox, by2 + oy),
                    label=status,
                    raw_label=raw,
                    confidence=float(box.conf[0]),
                    class_id=cls,
                )
            )
        return out
