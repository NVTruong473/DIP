from __future__ import annotations

from typing import Tuple

import cv2


def draw_label(
    frame,
    box,
    text: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
):
    """Draw a compact bounding box and short English label with OpenCV."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    text = str(text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.48, min(0.72, 0.58 * h / 1080.0))
    font_thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, font_thickness)
    pad_x = 6
    pad_y = 5
    label_w = tw + pad_x * 2
    label_h = th + baseline + pad_y * 2

    label_x1 = max(0, min(w - 1, x1))
    label_x2 = min(w - 1, label_x1 + label_w)

    if y1 - label_h >= 0:
        label_y1 = y1 - label_h
        label_y2 = y1
        text_y = label_y2 - baseline - pad_y
    else:
        label_y1 = y1
        label_y2 = min(h - 1, y1 + label_h)
        text_y = min(label_y2 - baseline - pad_y, label_y1 + th + pad_y)

    cv2.rectangle(
        frame,
        (label_x1, label_y1),
        (label_x2, label_y2),
        color,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (label_x1 + pad_x, max(th, text_y)),
        font,
        scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA,
    )
