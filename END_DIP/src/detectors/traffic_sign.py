from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from ultralytics import YOLO

from src.common import Detection
from src.model_manager import ModelManager
from src.utils.dip import analyze_sign_color, enhance_frame_clahe
from src.utils.geometry import box_iou


def ascii_text(value: str) -> str:
    # cv2.putText uses Hershey fonts and cannot render Vietnamese Unicode.
    value = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in value if not unicodedata.combining(ch)).encode("ascii", "ignore").decode("ascii")


class TrafficSignDetector:
    def __init__(
        self,
        manager: ModelManager,
        conf: float = 0.25,
        imgsz: int = 640,
        use_dip_enhancement: bool = False,
        tiled: bool = True,
    ):
        self.conf = conf
        self.imgsz = imgsz
        self.use_dip_enhancement = use_dip_enhancement
        self.tiled = tiled
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model_path = manager.sign_model()
        self.mapping_path = manager.sign_mapping()
        self.model = YOLO(self.model_path, task="detect")
        self.mapping = self._load_mapping(self.mapping_path)

    @staticmethod
    def _load_mapping(path: str) -> Dict[str, str]:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}

        out: Dict[str, str] = {}
        for key, value in data.items():
            if isinstance(value, str):
                out[str(key)] = value
            elif isinstance(value, dict):
                text = value.get("vi") or value.get("vn") or value.get("name_vi") or value.get("description") or value.get("name")
                out[str(key)] = str(text) if text else str(key)
            else:
                out[str(key)] = str(value)
        return out

    def _predict_tile(self, tile, ox: int, oy: int, original_frame) -> List[Detection]:
        source = enhance_frame_clahe(tile) if self.use_dip_enhancement else tile
        result = self.model.predict(
            source=source,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        if result.boxes is None:
            return detections

        names = result.names
        h, w = original_frame.shape[:2]
        for box in result.boxes:
            tx1, ty1, tx2, ty2 = [int(v) for v in box.xyxy[0].tolist()]
            x1, y1 = max(0, tx1 + ox), max(0, ty1 + oy)
            x2, y2 = min(w - 1, tx2 + ox), min(h - 1, ty2 + oy)
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            raw = str(names.get(cls, cls) if isinstance(names, dict) else names[cls])
            desc_vi = self.mapping.get(raw, raw)
            friendly = f"{raw}: {ascii_text(desc_vi)}" if desc_vi != raw else raw
            roi = original_frame[y1:y2, x1:x2]
            dip = analyze_sign_color(roi)
            dip["description_vi"] = desc_vi
            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    label=friendly,
                    raw_label=raw,
                    confidence=conf,
                    class_id=cls,
                    extra=dip,
                )
            )
        return detections

    @staticmethod
    def _class_aware_nms(dets: List[Detection], threshold: float = 0.45) -> List[Detection]:
        out: List[Detection] = []
        groups: Dict[str, List[Detection]] = {}
        for d in dets:
            groups.setdefault(d.raw_label or d.label, []).append(d)
        for _, group in groups.items():
            remaining = sorted(group, key=lambda d: d.confidence, reverse=True)
            while remaining:
                best = remaining.pop(0)
                out.append(best)
                remaining = [d for d in remaining if box_iou(best.box, d.box) < threshold]
        return out

    def detect(self, frame) -> List[Detection]:
        h, w = frame.shape[:2]
        if not self.tiled or w < 1000:
            return self._predict_tile(frame, 0, 0, frame)

        # Two overlapping vertical tiles make small distant signs ~1.6x larger
        # than a full 1920x1080 -> 640 letterbox inference, while adding only
        # one extra pass through the lightweight sign model.
        tile_w = int(round(w * 0.62))
        starts = [0, max(0, w - tile_w)]
        detections: List[Detection] = []
        for x0 in starts:
            tile = frame[:, x0:x0 + tile_w]
            detections.extend(self._predict_tile(tile, x0, 0, frame))
        return self._class_aware_nms(detections)
