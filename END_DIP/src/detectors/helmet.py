from __future__ import annotations

from typing import List
import re
import torch
from ultralytics import YOLO

from src.common import Detection
from src.model_manager import ModelManager


def _norm(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "", label.lower())
    if "without" in s or "nohelmet" in s or s in {"barehead", "face"}:
        return "NO HELMET"
    if "withhelmet" in s or s == "helmet" or "helmet" in s:
        return "HELMET"
    return "UNKNOWN"


class HelmetDetector:
    def __init__(self, manager: ModelManager, conf=0.38, imgsz=640):
        self.model = YOLO(manager.helmet_model())
        self.conf = conf
        self.imgsz = imgsz
        self.device = 0 if torch.cuda.is_available() else "cpu"

    def detect(self, frame) -> List[Detection]:
        result = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        out: List[Detection] = []
        if result.boxes is None:
            return out
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            raw = str(names.get(cls, cls) if isinstance(names, dict) else names[cls])
            status = _norm(raw)
            if status == "UNKNOWN":
                continue
            out.append(Detection((x1, y1, x2, y2), status, float(box.conf[0]), cls, None, raw, {}))
        return out
