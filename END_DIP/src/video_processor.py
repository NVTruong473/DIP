from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
from tqdm.auto import tqdm

from src.config import AppConfig
from src.detectors.traffic_sign import TrafficSignDetector
from src.model_manager import ModelManager
from src.temporal import CurrentOnlySmoother
from src.utils.visualization import draw_label

CSV_FIELDS = [
    "frame", "time_sec", "type", "track_id", "class", "confidence",
    "x1", "y1", "x2", "y2", "extra",
]


class VideoProcessor:
    """Traffic-sign-only pipeline for the current road video."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        cfg.ensure_dirs()
        manager = ModelManager(cfg.models_dir)
        self.sign = TrafficSignDetector(
            manager,
            conf=cfg.sign_conf,
            imgsz=cfg.sign_imgsz,
            tiled=cfg.sign_tiled,
        )
        self.smoother = CurrentOnlySmoother(
            iou=cfg.sign_track_iou,
            alpha=cfg.sign_smooth_alpha,
            confirm_hits=cfg.sign_confirm_hits,
            class_aware=True,
        )

    @staticmethod
    def _mux_h264(temp_video: str, source_video: str, final_video: str) -> None:
        base = [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-tag:v", "avc1", "-movflags", "+faststart",
        ]
        with_audio = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", temp_video, "-i", source_video,
            "-map", "0:v:0", "-map", "1:a?",
            *base, "-c:a", "aac", "-b:a", "128k", "-shortest", final_video,
        ]
        video_only = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", temp_video, "-map", "0:v:0", *base, "-an", final_video,
        ]
        try:
            subprocess.run(with_audio, check=True)
            os.remove(temp_video)
        except Exception:
            try:
                subprocess.run(video_only, check=True)
                os.remove(temp_video)
            except Exception as exc:
                print(f"[WARN] H.264 encode failed: {exc}")
                shutil.move(temp_video, final_video)

    @staticmethod
    def _log(log, frame_idx: int, fps: float, det) -> None:
        x1, y1, x2, y2 = det.box
        log.writerow({
            "frame": frame_idx,
            "time_sec": round(frame_idx / fps, 3),
            "type": "traffic_sign",
            "track_id": "",
            "class": det.label,
            "confidence": round(det.confidence, 4),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "extra": json.dumps({**(det.extra or {}), "raw_code": det.raw_label}, ensure_ascii=True),
        })

    def process(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict:
        input_path = str(input_path)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        stem = Path(input_path).stem
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        final = Path(output_path) if output_path else out_dir / f"{stem}_result.mp4"
        temp = final.with_name(final.stem + "_temp.mp4")
        csv_path = final.with_suffix(".csv")

        writer = cv2.VideoWriter(
            str(temp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Cannot create output video writer")

        sign_rows = 0
        unique_signs = set()
        frame_idx = 0

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            log = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            log.writeheader()
            bar = tqdm(total=total or None, desc="Traffic sign detection", unit="frame")

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                do_detect = frame_idx % max(1, self.cfg.frame_stride) == 0
                signs = self.smoother.update(self.sign.detect(frame)) if do_detect else []

                for det in signs:
                    label = det.label
                    if self.cfg.show_confidence:
                        label = f"{label} {det.confidence:.2f}"
                    draw_label(frame, det.box, label, (0, 165, 255), thickness=2)
                    self._log(log, frame_idx, fps, det)
                    sign_rows += 1
                    unique_signs.add(det.raw_label or det.label)

                if self.cfg.show_hud:
                    cv2.putText(
                        frame,
                        f"signs={len(signs)} | t={frame_idx / fps:.1f}s",
                        (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (255, 255, 255), 2, cv2.LINE_AA,
                    )

                writer.write(frame)
                frame_idx += 1
                bar.update(1)
                if progress_callback and (frame_idx % 15 == 0 or frame_idx == total):
                    progress_callback(frame_idx, total)

            bar.close()

        cap.release()
        writer.release()
        self._mux_h264(str(temp), input_path, str(final))

        return {
            "input": input_path,
            "output_video": str(final),
            "csv": str(csv_path),
            "frames": frame_idx,
            "fps": fps,
            "duration_sec": frame_idx / fps if fps else 0,
            "traffic_sign_rows": sign_rows,
            "unique_sign_codes": sorted(unique_signs),
        }
