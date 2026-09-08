from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from src.common import Detection
from src.utils.geometry import box_iou


class CurrentOnlySmoother:
    """EMA-smooth matched current detections without predicting missing boxes.

    This keeps bounding boxes visually steadier while avoiding ghost boxes: if
    YOLO does not detect a sign in the current frame, nothing is extrapolated.
    """

    def __init__(self, iou=0.30, alpha=0.70, confirm_hits=1, class_aware=True):
        self.iou = float(iou)
        self.alpha = float(alpha)
        self.confirm_hits = max(1, int(confirm_hits))
        self.class_aware = bool(class_aware)
        self.prev: List[Detection] = []
        self.hits: Dict[str, int] = defaultdict(int)

    def _key(self, det: Detection) -> str:
        return det.raw_label or det.label

    def update(self, detections: List[Detection]) -> List[Detection]:
        out: List[Detection] = []
        current_keys = set()

        for det in detections:
            best = None
            best_iou = 0.0
            for old in self.prev:
                if self.class_aware and (old.raw_label or old.label) != (det.raw_label or det.label):
                    continue
                iou = box_iou(old.box, det.box)
                if iou >= self.iou and iou > best_iou:
                    best, best_iou = old, iou

            if best is not None:
                a = self.alpha
                det.box = tuple(
                    int(round(a * new + (1.0 - a) * old))
                    for new, old in zip(det.box, best.box)
                )

            key = self._key(det)
            current_keys.add(key)
            self.hits[key] += 1
            if self.hits[key] >= self.confirm_hits or det.confidence >= 0.65:
                out.append(det)

        # Reset consecutive-hit counters for classes absent from this frame.
        for key in list(self.hits):
            if key not in current_keys:
                self.hits[key] = 0

        self.prev = detections
        return out
