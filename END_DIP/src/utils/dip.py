from __future__ import annotations

import cv2
import numpy as np


def enhance_frame_clahe(frame: np.ndarray) -> np.ndarray:
    """Mild DIP enhancement. Kept optional because learned detectors can be hurt by overly aggressive preprocessing."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.12, blur, -0.12, 0)


def analyze_sign_color(roi: np.ndarray) -> dict:
    """Secondary DIP cue for a YOLO-detected sign; diagnostic, not a hard gate."""
    if roi is None or roi.size == 0:
        return {"dominant_color": "unknown", "color_ratio": 0.0}
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    masks = {
        "red": cv2.inRange(hsv, np.array([0, 70, 45]), np.array([10, 255, 255])) | cv2.inRange(hsv, np.array([165, 70, 45]), np.array([180, 255, 255])),
        "blue": cv2.inRange(hsv, np.array([90, 65, 45]), np.array([135, 255, 255])),
        "yellow": cv2.inRange(hsv, np.array([15, 70, 70]), np.array([38, 255, 255])),
    }
    counts = {k: int(cv2.countNonZero(v)) for k, v in masks.items()}
    dominant = max(counts, key=counts.get)
    ratio = counts[dominant] / float(max(1, roi.shape[0] * roi.shape[1]))
    return {"dominant_color": dominant, "color_ratio": round(ratio, 4)}
