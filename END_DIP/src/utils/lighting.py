from __future__ import annotations

import cv2
import numpy as np


def _gamma(img, gamma: float):
    inv = 1.0 / max(1e-6, gamma)
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def _clahe_luma(frame, clip=2.0, grid=(8, 8)):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def adapt_for_inference(frame, night_threshold=68.0, glare_threshold=205.0):
    """Return an inference-only image plus a short environment label.

    Coordinates are preserved because all operations are photometric. The
    original frame must still be used for rendering/output.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    mean_v = float(v.mean())
    p95_v = float(np.percentile(v, 95))

    if mean_v < night_threshold:
        # Night: lift dark regions, then CLAHE. Avoid denoising too heavily
        # because tiny signs/plates already contain few pixels.
        out = _gamma(frame, 1.45)
        out = _clahe_luma(out, clip=2.4)
        return out, "NIGHT"

    if mean_v > glare_threshold or p95_v > 248:
        # Strong sun / glare: compress highlights and restore local contrast.
        out = _gamma(frame, 0.82)
        out = _clahe_luma(out, clip=1.8)
        return out, "GLARE"

    # Daylight: only a very mild local-contrast correction.
    return _clahe_luma(frame, clip=1.25), "DAY"


def crop_quality(crop):
    if crop is None or crop.size == 0:
        return {"sharpness": 0.0, "brightness": 0.0, "contrast": 0.0}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return {
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
    }
