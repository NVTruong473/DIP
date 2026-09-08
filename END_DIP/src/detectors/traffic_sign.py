from __future__ import annotations

from typing import Dict, List
import torch
from ultralytics import YOLO

from src.common import Detection
from src.model_manager import ModelManager
from src.utils.dip import analyze_sign_color
from src.utils.geometry import box_iou


ENGLISH_LABELS: Dict[str, str] = {
    "DP-135": "End Restrictions",
    "P-102": "No Entry",
    "P-103A": "No Cars",
    "P-103B": "No Left Turn",
    "P-103C": "No Right Turn",
    "P-104": "No Motorcycles",
    "P-106A": "No Trucks",
    "P-106B": "Truck Weight Limit",
    "P-107A": "No Buses",
    "P-112": "No Pedestrians",
    "P-115": "Weight Limit",
    "P-117": "Height Limit",
    "P-123A": "No Left Turn",
    "P-123B": "No Right Turn",
    "P-124A": "No U-Turn",
    "P-124B": "No U-Turn",
    "P-124C": "No Left/U-Turn",
    "P-127": "Speed Limit",
    "P-128": "No Horn",
    "P-130": "No Stop/Parking",
    "P-131A": "No Parking",
    "P-137": "No Left/Right Turn",
    "P-245A": "Slow Down",
    "R-301C": "Left Only",
    "R-301D": "Right Only",
    "R-301E": "Left Only",
    "R-302A": "Keep Right",
    "R-302B": "Keep Left",
    "R-303": "Roundabout",
    "R-407A": "One Way",
    "R-409": "U-Turn Point",
    "R-425": "Hospital",
    "R-434": "Bus Stop",
    "S-509A": "Safe Height",
    "W-201A": "Bend Left",
    "W-201B": "Bend Right",
    "W-202A": "Winding Road",
    "W-202B": "Winding Road",
    "W-203B": "Road Narrows",
    "W-203C": "Road Narrows",
    "W-205A": "Crossroads",
    "W-205B": "Junction Ahead",
    "W-205D": "Junction Ahead",
    "W-207A": "Side Junction",
    "W-207B": "Side Junction",
    "W-207C": "Side Junction",
    "W-208": "Priority Junction",
    "W-209": "Traffic Lights",
    "W-210": "Rail Crossing",
    "W-219": "Steep Descent",
    "W-224": "Pedestrian Crossing",
    "W-225": "Children",
    "W-227": "Road Works",
    "W-233": "Other Danger",
    "W-235": "Divided Road",
    "W-245A": "Slow Down",
}


def canonical_code(value: str) -> str:
    code = str(value).strip().upper().replace(".", "-").replace("_", "-")
    while "--" in code:
        code = code.replace("--", "-")
    return code


def english_name(raw: str) -> str:
    code = canonical_code(raw)
    if code in ENGLISH_LABELS:
        return ENGLISH_LABELS[code]
    if code.startswith("P-"):
        return "Prohibition Sign"
    if code.startswith("R-"):
        return "Mandatory Sign"
    if code.startswith("W-"):
        return "Warning Sign"
    if code.startswith("S-"):
        return "Supplementary Sign"
    return "Traffic Sign"


class TrafficSignDetector:
    def __init__(self, manager: ModelManager, conf: float = 0.25, imgsz: int = 640, tiled: bool = True):
        self.conf = conf
        self.imgsz = imgsz
        self.tiled = tiled
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(manager.sign_model(), task="detect")

    @staticmethod
    def _plausible_box(x1, y1, x2, y2, frame_w, frame_h):
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        aspect = bw / float(bh)
        area_ratio = (bw * bh) / float(frame_w * frame_h)
        if bw < 7 or bh < 7:
            return False
        if not 0.20 <= aspect <= 5.0:
            return False
        if area_ratio > 0.16:
            return False
        return True

    def _predict_tile(self, tile, ox: int, oy: int, original_frame) -> List[Detection]:
        result = self.model.predict(
            source=tile,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        detections: List[Detection] = []
        if result.boxes is None:
            return detections

        names = result.names
        h, w = original_frame.shape[:2]
        for box in result.boxes:
            tx1, ty1, tx2, ty2 = [int(v) for v in box.xyxy[0].tolist()]
            x1, y1 = max(0, tx1 + ox), max(0, ty1 + oy)
            x2, y2 = min(w - 1, tx2 + ox), min(h - 1, ty2 + oy)
            if not self._plausible_box(x1, y1, x2, y2, w, h):
                continue

            cls = int(box.cls[0])
            conf = float(box.conf[0])
            raw = str(names.get(cls, cls) if isinstance(names, dict) else names[cls])
            code = canonical_code(raw)
            roi = original_frame[y1:y2, x1:x2]
            extra = analyze_sign_color(roi)
            extra["sign_code"] = code

            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    label=english_name(code),
                    confidence=conf,
                    class_id=cls,
                    raw_label=code,
                    extra=extra,
                )
            )
        return detections

    @staticmethod
    def _class_aware_nms(dets: List[Detection], threshold: float = 0.45) -> List[Detection]:
        groups: Dict[str, List[Detection]] = {}
        for det in dets:
            groups.setdefault(det.raw_label or det.label, []).append(det)

        out: List[Detection] = []
        for group in groups.values():
            remaining = sorted(group, key=lambda d: d.confidence, reverse=True)
            while remaining:
                best = remaining.pop(0)
                out.append(best)
                remaining = [d for d in remaining if box_iou(best.box, d.box) < threshold]
        return out

    def detect(self, frame) -> List[Detection]:
        _, w = frame.shape[:2]
        if not self.tiled or w < 1000:
            return self._predict_tile(frame, 0, 0, frame)

        # Two overlapping vertical tiles preserve more detail for small signs.
        tile_w = int(round(w * 0.62))
        starts = [0, max(0, w - tile_w)]
        dets: List[Detection] = []
        for x0 in starts:
            dets.extend(self._predict_tile(frame[:, x0:x0 + tile_w], x0, 0, frame))
        return self._class_aware_nms(dets)
