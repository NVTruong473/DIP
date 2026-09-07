from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Tuple
import re

import cv2

ALLOWED = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class PlateOCR:
    def __init__(self, gpu=True, min_conf=0.35, min_votes=2, history=12, model_dir=None):
        import easyocr
        storage = None
        if model_dir:
            storage = str(Path(model_dir))
            Path(storage).mkdir(parents=True, exist_ok=True)
        self.reader = easyocr.Reader(
            ["en"],
            gpu=gpu,
            verbose=False,
            model_storage_directory=storage,
            user_network_directory=storage,
        )
        self.min_conf = min_conf
        self.min_votes = min_votes
        self.history = defaultdict(lambda: deque(maxlen=history))

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    @staticmethod
    def _valid(text: str) -> bool:
        return bool(re.fullmatch(r"[0-9]{2}[A-Z]{1,2}[0-9]{4,5}", text))

    @staticmethod
    def _variants(crop):
        up = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        blur = cv2.GaussianBlur(clahe, (3, 3), 0)
        thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
        return [up, clahe, thr]

    def recognize(self, crop) -> Tuple[str, float]:
        best = ("", 0.0)
        for img in self._variants(crop):
            for _, text, conf in self.reader.readtext(img, detail=1, paragraph=False, allowlist=ALLOWED, decoder="greedy"):
                cleaned = self._clean(text)
                conf = float(conf)
                if conf >= self.min_conf and self._valid(cleaned) and conf > best[1]:
                    best = (cleaned, conf)
        return best

    def update(self, track_id, text: str, conf: float):
        if track_id is not None and text and conf >= self.min_conf:
            self.history[track_id].append((text, conf))
        return self.stable(track_id)

    def stable(self, track_id):
        if track_id is None or not self.history.get(track_id):
            return ""
        scores: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        for text, conf in self.history[track_id]:
            scores[text] += conf
            counts[text] += 1
        winner = max(scores, key=scores.get)
        return winner if counts[winner] >= self.min_votes else ""
