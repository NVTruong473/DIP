from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Dict, List

from src.common import Detection
from src.utils.geometry import box_iou


class CurrentOnlySmoother:
    """Smooth matched boxes without extrapolating missing detections.

    This is intentionally *not* a predictive tracker: if the detector misses an
    object, no stale/guessed box is drawn. It only EMA-smooths a box when a real
    current-frame detection can be IoU-matched to recent evidence.
    """

    def __init__(self, iou=0.28, alpha=0.68, confirm_hits=2, class_aware=True):
        self.iou = iou
        self.alpha = alpha
        self.confirm_hits = confirm_hits
        self.class_aware = class_aware
        self.prev: List[Detection] = []
        self.hits: Dict[str, int] = defaultdict(int)

    def _key(self, d: Detection):
        return str(d.track_id) if d.track_id is not None else (d.raw_label or d.label)

    def update(self, detections: List[Detection]) -> List[Detection]:
        out = []
        for det in detections:
            best = None
            best_iou = 0.0
            for old in self.prev:
                if self.class_aware and (old.raw_label or old.label) != (det.raw_label or det.label):
                    continue
                if det.track_id is not None and old.track_id is not None and det.track_id != old.track_id:
                    continue
                i = box_iou(old.box, det.box)
                if i >= self.iou and i > best_iou:
                    best, best_iou = old, i
            if best is not None:
                a = self.alpha
                det.box = tuple(int(round(a*n + (1-a)*o)) for n, o in zip(det.box, best.box))
            key = self._key(det)
            self.hits[key] += 1
            if self.hits[key] >= self.confirm_hits or det.confidence >= 0.65:
                out.append(det)
        self.prev = detections
        return out


class HelmetVoter:
    def __init__(self, window=10, min_votes=4, stable_ratio=0.70):
        self.window = window
        self.min_votes = min_votes
        self.stable_ratio = stable_ratio
        self.history = defaultdict(lambda: deque(maxlen=window))

    def update(self, track_id, status: str):
        if track_id is None:
            return status if status in ("HELMET", "NO HELMET") else "UNKNOWN"
        if status in ("HELMET", "NO HELMET"):
            self.history[track_id].append(status)
        hist = self.history[track_id]
        if len(hist) < self.min_votes:
            return "UNKNOWN"
        winner, count = Counter(hist).most_common(1)[0]
        return winner if count / len(hist) >= self.stable_ratio else "UNKNOWN"
