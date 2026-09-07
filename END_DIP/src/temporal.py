from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.common import Detection
from src.utils.geometry import box_iou


@dataclass
class _HeldDetection:
    det: Detection
    ttl: int


class DetectionTemporalHold:
    """Small temporal buffer that suppresses one-frame detection blinking."""

    def __init__(self, hold_frames: int = 6, iou_match: float = 0.30, class_aware: bool = True):
        self.hold_frames = max(1, int(hold_frames))
        self.iou_match = float(iou_match)
        self.class_aware = bool(class_aware)
        self.items: List[_HeldDetection] = []

    def _same_class(self, a: Detection, b: Detection) -> bool:
        if not self.class_aware:
            return True
        return (a.raw_label or a.label) == (b.raw_label or b.label)

    def update(self, detections: List[Detection]) -> List[Detection]:
        next_items: List[_HeldDetection] = []
        used_old = set()

        for det in detections:
            best_i = None
            best_iou = 0.0
            for i, item in enumerate(self.items):
                if i in used_old:
                    continue
                if not self._same_class(item.det, det):
                    continue
                iou = box_iou(item.det.box, det.box)
                if iou >= self.iou_match and iou > best_iou:
                    best_i, best_iou = i, iou

            if best_i is not None:
                used_old.add(best_i)
            next_items.append(_HeldDetection(det=det, ttl=self.hold_frames))

        for i, item in enumerate(self.items):
            if i in used_old:
                continue
            ttl = item.ttl - 1
            if ttl > 0:
                next_items.append(_HeldDetection(det=item.det, ttl=ttl))

        self.items = next_items
        return [x.det for x in self.items]


# Backward-compatible alias for older imports/documentation.
SignTemporalHold = DetectionTemporalHold
