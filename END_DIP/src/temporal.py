from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, List

from src.common import Detection
from src.utils.geometry import box_iou


class HelmetTemporalVoter:
    def __init__(self, window: int = 12, min_votes: int = 3, stable_ratio: float = 0.65, ttl_frames: int = 15):
        self.window = window
        self.min_votes = min_votes
        self.stable_ratio = stable_ratio
        self.ttl_frames = ttl_frames
        self.history: Dict[int, deque] = {}
        self.last_seen: Dict[int, int] = {}
        self.last_stable: Dict[int, str] = {}

    def update(self, track_id: int | None, status: str, frame_idx: int) -> str:
        if track_id is None:
            return status if status in ("HELMET", "NO_HELMET") else "UNKNOWN"

        self.last_seen[track_id] = frame_idx
        hist = self.history.setdefault(track_id, deque(maxlen=self.window))
        if status in ("HELMET", "NO_HELMET"):
            hist.append(status)

        if len(hist) >= self.min_votes:
            counts = Counter(hist)
            winner, n = counts.most_common(1)[0]
            if n / len(hist) >= self.stable_ratio:
                self.last_stable[track_id] = winner

        stable = self.last_stable.get(track_id, "UNKNOWN")
        self._cleanup(frame_idx)
        return stable

    def get(self, track_id: int | None) -> str:
        if track_id is None:
            return "UNKNOWN"
        return self.last_stable.get(track_id, "UNKNOWN")

    def _cleanup(self, frame_idx: int) -> None:
        stale = [k for k, v in self.last_seen.items() if frame_idx - v > self.ttl_frames]
        for k in stale:
            self.last_seen.pop(k, None)
            self.history.pop(k, None)
            self.last_stable.pop(k, None)


@dataclass
class _HeldSign:
    det: Detection
    ttl: int


class SignTemporalHold:
    """Small temporal buffer that suppresses one-frame traffic-sign blinking."""
    def __init__(self, hold_frames: int = 8, iou_match: float = 0.30):
        self.hold_frames = hold_frames
        self.iou_match = iou_match
        self.items: List[_HeldSign] = []

    def update(self, detections: List[Detection]) -> List[Detection]:
        next_items: List[_HeldSign] = []
        used_old = set()

        for det in detections:
            best_i = None
            best_iou = 0.0
            for i, item in enumerate(self.items):
                if i in used_old:
                    continue
                if (item.det.raw_label or item.det.label) != (det.raw_label or det.label):
                    continue
                iou = box_iou(item.det.box, det.box)
                if iou >= self.iou_match and iou > best_iou:
                    best_i, best_iou = i, iou
            if best_i is not None:
                used_old.add(best_i)
            next_items.append(_HeldSign(det=det, ttl=self.hold_frames))

        for i, item in enumerate(self.items):
            if i in used_old:
                continue
            ttl = item.ttl - 1
            if ttl > 0:
                next_items.append(_HeldSign(item.det, ttl))

        self.items = next_items
        return [x.det for x in self.items]
