from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple
import os
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=16)
def _unicode_font(size: int):
    """Return a TrueType font with Vietnamese glyph support.

    Google Colab normally ships DejaVu Sans. We also probe common Noto and
    Liberation locations and finally ask fontconfig. TRAFFIC_FONT_PATH can be
    set when running on another platform.
    """
    candidates = [
        os.environ.get("TRAFFIC_FONT_PATH", ""),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]

    for path in candidates:
        if path and Path(path).is_file():
            return ImageFont.truetype(path, size=size)

    try:
        matched = subprocess.check_output(
            ["fc-match", "-f", "%{file}", "DejaVu Sans"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if matched and Path(matched).is_file():
            return ImageFont.truetype(matched, size=size)
    except Exception:
        pass

    # Last-resort fallback. Colab should never reach this branch.
    return ImageFont.load_default()


def _font_size(frame) -> int:
    # ~20 px at 1080p, but remains readable on smaller/larger videos.
    return max(16, min(30, int(round(frame.shape[0] * 0.019))))


def _draw_pil_text_on_roi(frame, rect, text: str, font) -> None:
    """Draw Unicode text only inside a small ROI to avoid full-frame PIL copies."""
    x1, y1, x2, y2 = rect
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(frame.shape[1], int(x2))
    y2 = min(frame.shape[0], int(y2))
    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    bbox = font.getbbox(text)
    draw.text((4 - bbox[0], 3 - bbox[1]), text, font=font, fill=(255, 255, 255))
    frame[y1:y2, x1:x2] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def draw_label(frame, box, text: str, color: Tuple[int, int, int], thickness: int = 2):
    """Draw a bounding box and UTF-8 label, including Vietnamese diacritics."""
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    text = str(text)
    font = _unicode_font(_font_size(frame))
    bbox = font.getbbox(text)
    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])
    pad_x, pad_y = 8, 6
    label_w = min(frame.shape[1] - x1, tw + pad_x)
    label_h = th + pad_y

    label_x1 = max(0, x1)
    label_x2 = min(frame.shape[1], label_x1 + label_w)
    if y1 - label_h >= 0:
        label_y1 = y1 - label_h
        label_y2 = y1
    else:
        label_y1 = max(0, y1)
        label_y2 = min(frame.shape[0], label_y1 + label_h)

    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    _draw_pil_text_on_roi(frame, (label_x1, label_y1, label_x2, label_y2), text, font)


def draw_roi(frame, polygon_px: np.ndarray):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon_px], (40, 180, 255))
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, dst=frame)
    cv2.polylines(frame, [polygon_px], True, (40, 180, 255), 2, cv2.LINE_AA)

    # This text is currently ASCII, but use the same Unicode renderer so it is
    # safe to rename the ROI in Vietnamese later.
    x, y = [int(v) for v in polygon_px.reshape(-1, 2)[0]]
    font = _unicode_font(_font_size(frame))
    text = "HELMET ROI - RIGHT ROAD"
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    rx1 = max(0, x)
    ry1 = max(0, y - th - 10)
    rx2 = min(frame.shape[1], rx1 + tw + 8)
    ry2 = min(frame.shape[0], ry1 + th + 6)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (40, 180, 255), -1)
    _draw_pil_text_on_roi(frame, (rx1, ry1, rx2, ry2), text, font)


def status_color(status: str):
    s = status.lower()
    if "no_helmet" in s or "without" in s or "violation" in s:
        return (0, 0, 255)
    if "helmet" in s:
        return (0, 180, 0)
    return (0, 200, 255)
