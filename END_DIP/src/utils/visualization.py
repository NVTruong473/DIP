from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np


def draw_label(frame, box, text: str, color: Tuple[int, int, int], thickness: int = 2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    ty = max(th + 6, y1)
    cv2.rectangle(frame, (x1, ty - th - 6), (min(frame.shape[1] - 1, x1 + tw + 6), ty + 2), color, -1)
    cv2.putText(frame, text, (x1 + 3, ty - 3), font, scale, (255, 255, 255), 2, cv2.LINE_AA)


def draw_roi(frame, polygon_px: np.ndarray):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon_px], (40, 180, 255))
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, dst=frame)
    cv2.polylines(frame, [polygon_px], True, (40, 180, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "HELMET ROI - RIGHT ROAD", tuple(polygon_px.reshape(-1, 2)[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 180, 255), 2, cv2.LINE_AA)


def status_color(status: str):
    s = status.lower()
    if "no_helmet" in s or "without" in s or "violation" in s:
        return (0, 0, 255)
    if "helmet" in s:
        return (0, 180, 0)
    return (0, 200, 255)
