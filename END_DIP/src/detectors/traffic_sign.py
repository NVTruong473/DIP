from __future__ import annotations

from typing import Dict, List
import re
import torch
from ultralytics import YOLO

from src.common import Detection
from src.model_manager import ModelManager
from src.utils.dip import analyze_sign_color
from src.utils.geometry import box_iou


def compact_label(name: str) -> str:
    s = re.sub(r"\s+", " ", str(name)).strip()
    replacements = {
        "No Stopping & No Parking": "No Stop/Parking",
        "No U-Turn and No Left Turn": "No U-Turn/Left",
        "No U-Turn and No Right Turn": "No U-Turn/Right",
        "No U-Turn and Left Turn for Cars": "No U-Turn/Left (Cars)",
        "Intersection with a Minor Road": "Minor Road Junction",
        "Intersection with Equal Roads": "Equal Roads Junction",
        "Intersection with a Priority Road": "Priority Junction",
        "Level Crossing with Barriers": "Rail Crossing",
        "No Two or Three-wheeled Vehicles": "No 2/3-Wheelers",
        "Road with Surveillance Camera": "Camera Ahead",
        "Double curve first to right": "Double Curve Right",
        "sparsely populated area": "Sparse Area",
    }
    return replacements.get(s, s[:34])


class TrafficSignDetector:
    def __init__(self, manager: ModelManager, conf=0.32, imgsz=768, tiled=True):
        self.conf = conf
        self.imgsz = imgsz
        self.tiled = tiled
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(manager.sign_model())

    @staticmethod
    def _plausible_box(x1, y1, x2, y2, frame_w, frame_h):
        bw, bh = max(1, x2-x1), max(1, y2-y1)
        aspect = bw/float(bh)
        area_ratio = (bw*bh)/float(frame_w*frame_h)
        # Reject tiny noise and implausibly huge/elongated detections. Limits are
        # permissive enough for rectangular supplementary/information signs.
        if bw < 7 or bh < 7:
            return False
        if not 0.20 <= aspect <= 5.0:
            return False
        if area_ratio > 0.16:
            return False
        return True

    def _predict_tile(self, tile, ox: int, oy: int, original_frame) -> List[Detection]:
        result = self.model.predict(tile, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        out: List[Detection] = []
        if result.boxes is None:
            return out
        names = result.names
        h, w = original_frame.shape[:2]
        for box in result.boxes:
            tx1, ty1, tx2, ty2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, tx1+ox), max(0, ty1+oy)
            x2, y2 = min(w-1, tx2+ox), min(h-1, ty2+oy)
            if not self._plausible_box(x1,y1,x2,y2,w,h):
                continue
            cls = int(box.cls[0])
            raw = str(names.get(cls, cls) if isinstance(names, dict) else names[cls])
            roi = original_frame[y1:y2, x1:x2]
            dip = analyze_sign_color(roi)
            out.append(Detection((x1, y1, x2, y2), compact_label(raw), float(box.conf[0]), cls, None, raw, dip))
        return out

    @staticmethod
    def _nms(dets: List[Detection], threshold=0.45):
        groups: Dict[str, List[Detection]] = {}
        for d in dets:
            groups.setdefault(d.raw_label or d.label, []).append(d)
        out = []
        for group in groups.values():
            remaining = sorted(group, key=lambda d: d.confidence, reverse=True)
            while remaining:
                best = remaining.pop(0)
                out.append(best)
                remaining = [d for d in remaining if box_iou(best.box, d.box) < threshold]
        return out

    def detect(self, frame):
        _, w = frame.shape[:2]
        if not self.tiled or w < 1000:
            return self._predict_tile(frame, 0, 0, frame)
        tile_w = int(round(w * 0.62))
        starts = [0, max(0, w-tile_w)]
        dets = []
        for x0 in starts:
            dets.extend(self._predict_tile(frame[:, x0:x0+tile_w], x0, 0, frame))
        return self._nms(dets)
