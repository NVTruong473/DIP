from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple
import cv2
import numpy as np

from src.common import Box, Detection


def normalized_polygon_to_pixels(points: Sequence[Tuple[float, float]], width: int, height: int) -> np.ndarray:
    pts = np.array([(int(round(x * width)), int(round(y * height))) for x, y in points], dtype=np.int32)
    return pts.reshape((-1, 1, 2))


def point_in_polygon(point: Tuple[float, float], polygon_px: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon_px, (float(point[0]), float(point[1])), False) >= 0


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ba = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1e-9, aa + ba - inter)


def expand_box(box: Box, frame_shape, x_scale: float = 1.3, y_scale: float = 1.3, upward_bias: float = 0.0) -> Box:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * x_scale
    bh = (y2 - y1) * y_scale
    cy -= upward_bias * (y2 - y1)
    nx1 = max(0, int(cx - bw / 2))
    nx2 = min(w - 1, int(cx + bw / 2))
    ny1 = max(0, int(cy - bh / 2))
    ny2 = min(h - 1, int(cy + bh / 2))
    return nx1, ny1, nx2, ny2


def point_in_box(point: Tuple[float, float], box: Box) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def head_region(person_box: Box, frame_shape) -> Box:
    x1, y1, x2, y2 = person_box
    height = max(1, y2 - y1)
    width = max(1, x2 - x1)
    hx1 = x1 - int(0.12 * width)
    hx2 = x2 + int(0.12 * width)
    hy1 = y1 - int(0.08 * height)
    hy2 = y1 + int(0.60 * height)
    h, w = frame_shape[:2]
    return max(0, hx1), max(0, hy1), min(w - 1, hx2), min(h - 1, hy2)


def filter_detections_in_roi(detections: Iterable[Detection], polygon_px: np.ndarray, anchor: str = "bottom") -> List[Detection]:
    out = []
    for d in detections:
        p = d.bottom_center if anchor == "bottom" else d.center
        if point_in_polygon(p, polygon_px):
            out.append(d)
    return out
