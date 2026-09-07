from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

Box = Tuple[int, int, int, int]


@dataclass
class Detection:
    box: Box
    label: str
    confidence: float
    class_id: int = -1
    track_id: Optional[int] = None
    raw_label: Optional[str] = None
    extra: Optional[dict] = None

    @property
    def x1(self) -> int:
        return self.box[0]

    @property
    def y1(self) -> int:
        return self.box[1]

    @property
    def x2(self) -> int:
        return self.box[2]

    @property
    def y2(self) -> int:
        return self.box[3]

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, float(self.y2))
