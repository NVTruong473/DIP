from __future__ import annotations

from typing import List
import torch
from ultralytics import YOLO

from src.common import Detection
from src.model_manager import ModelManager


class SceneDetector:
    """Hidden parent-object detector used only for semantic validation.

    ByteTrack IDs are important for temporal voting, but these boxes are never
    drawn in the final video to avoid visual clutter.
    """

    def __init__(self, manager: ModelManager, conf=0.34, imgsz=640, classes=(0, 2, 3, 5, 7)):
        self.model = YOLO(manager.scene_model())
        self.conf = conf
        self.imgsz = imgsz
        self.classes = list(classes)
        self.device = 0 if torch.cuda.is_available() else "cpu"

    def detect(self, frame) -> List[Detection]:
        result = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )[0]
        out: List[Detection] = []
        if result.boxes is None:
            return out
        names = result.names
        ids = result.boxes.id
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            label = str(names.get(cls, cls) if isinstance(names, dict) else names[cls])
            tid = int(ids[i].item()) if ids is not None else None
            out.append(Detection((x1, y1, x2, y2), label, float(box.conf[0]), cls, tid, label, {}))
        return out
