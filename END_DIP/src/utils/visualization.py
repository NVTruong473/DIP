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
    """Return a TrueType font with Vietnamese glyph support."""
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

    return ImageFont.load_default()


def _font_size(frame) -> int:
    # About 20 px at 1080p, readable without covering small traffic objects.
    return max(16, min(30, int(round(frame.shape[0] * 0.019))))


def _draw_pil_text_on_roi(frame, rect, text: str, font) -> None:
    """Draw UTF-8 text only inside a small ROI to avoid full-frame PIL copies."""
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
    """Draw a compact bounding box and Vietnamese Unicode label."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(frame.shape[1] - 1, x1))
    y1 = max(0, min(frame.shape[0] - 1, y1))
    x2 = max(0, min(frame.shape[1] - 1, x2))
    y2 = max(0, min(frame.shape[0] - 1, y2))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    text = str(text)
    font = _unicode_font(_font_size(frame))
    bbox = font.getbbox(text)
    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])
    pad_x, pad_y = 8, 6

    label_w = min(frame.shape[1] - x1, tw + pad_x)
    label_h = th + pad_y
    label_x1 = x1
    label_x2 = min(frame.shape[1], label_x1 + label_w)

    if y1 - label_h >= 0:
        label_y1 = y1 - label_h
        label_y2 = y1
    else:
        label_y1 = y1
        label_y2 = min(frame.shape[0], label_y1 + label_h)

    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    _draw_pil_text_on_roi(frame, (label_x1, label_y1, label_x2, label_y2), text, font)
