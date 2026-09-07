from __future__ import annotations

from typing import List, Optional, Tuple
import torch
from ultralytics import YOLO

from src.common import Detection
from src.model_manager import ModelManager


class SceneDetector:
    """COCO car/bus/truck detector with ByteTrack IDs.

    Vehicle boxes are used internally to decide whether a detected plate belongs
    to a four-wheel vehicle. They do not need to be rendered in the final video.
    """

    def __init__(self, manager: ModelManager, conf: float = 0.30, imgsz: int = 640, classes=(2, 5, 7)):
        self.conf = conf
        self.imgsz = imgsz
        self.classes = list(classes)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(manager.scene_model())

    def detect(self, frame, crop_box: Optional[Tuple[int, int, int, int]] = None) -> List[Detection]:
        ox = oy = 0
        source = frame
        if crop_box is not None:
            x1, y1, x2, y2 = crop_box
            ox, oy = x1, y1
            source = frame[y1:y2, x1:x2]
            if source.size == 0:
                return []

        result = self.model.track(
            source=source,
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
            bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0].tolist()]
            cls = int(box.cls[0])
            label = str(names.get(cls, cls) if isinstance(names, dict) else names[cls])
            track_id = int(ids[i].item()) if ids is not None else None
            out.append(
                Detection(
                    box=(bx1 + ox, by1 + oy, bx2 + ox, by2 + oy),
                    label=label,
                    raw_label=label,
                    confidence=float(box.conf[0]),
                    class_id=cls,
                    track_id=track_id,
                )
            )
        return out
