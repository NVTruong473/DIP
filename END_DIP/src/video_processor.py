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
from src.temporal import DetectionTemporalHold
from src.utils.visualization import draw_label


CSV_FIELDS = [
    "frame",
    "time_sec",
    "type",
    "track_id",
    "class",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "extra",
]


class VideoProcessor:
    """Traffic-sign-only video pipeline for Google Colab."""

    def __init__(self, config: AppConfig):
        self.cfg = config
        self.cfg.ensure_dirs()
        self.manager = ModelManager(config.models_dir)
        self.sign_detector = TrafficSignDetector(
            self.manager,
            conf=config.sign_conf,
            imgsz=config.sign_imgsz,
            use_dip_enhancement=config.use_dip_enhancement,
            tiled=config.sign_tiled,
        )
        self.sign_hold = DetectionTemporalHold(
            config.sign_hold_frames,
            config.sign_iou_match,
            class_aware=True,
        )

    @staticmethod
    def _mux_h264(temp_video: str, source_video: str, final_video: str) -> None:
        """Create a Drive/Chrome/Colab-friendly H.264 MP4."""
        common_video = [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-level",
            "4.1",
            "-tag:v",
            "avc1",
            "-movflags",
            "+faststart",
        ]

        with_audio = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            temp_video,
            "-i",
            source_video,
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            *common_video,
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            final_video,
        ]

        video_only = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            temp_video,
            "-map",
            "0:v:0",
            *common_video,
            "-an",
            final_video,
        ]

        try:
            subprocess.run(with_audio, check=True)
            os.remove(temp_video)
            return
        except Exception as first_error:
            print(f"[WARN] Audio mux failed, retrying video-only: {first_error}")

        try:
            subprocess.run(video_only, check=True)
            os.remove(temp_video)
            return
        except Exception as second_error:
            print(f"[WARN] H.264 encoding failed: {second_error}")
            print("[WARN] Falling back to OpenCV mp4v output.")
            shutil.move(temp_video, final_video)

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
            raise RuntimeError(f"OpenCV cannot open video: {input_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        stem = Path(input_path).stem
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        final_video = Path(output_path) if output_path else out_dir / f"{stem}_result.mp4"
        final_video.parent.mkdir(parents=True, exist_ok=True)
        temp_video = final_video.with_name(final_video.stem + "_temp.mp4")
        csv_path = final_video.with_suffix(".csv")

        writer = cv2.VideoWriter(
            str(temp_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Cannot create output video writer.")

        last_signs = []
        sign_rows = 0
        unique_codes = set()

        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            log = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            log.writeheader()
            bar = tqdm(
                total=total if total > 0 else None,
                desc="Detecting traffic signs",
                unit="frame",
            )
            frame_idx = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                do_detect = frame_idx % max(1, self.cfg.frame_stride) == 0
                current_signs = self.sign_detector.detect(frame) if do_detect else []
                last_signs = self.sign_hold.update(current_signs)

                if do_detect:
                    for sign in current_signs:
                        self._log_detection(log, frame_idx, fps, sign)
                        sign_rows += 1
                        unique_codes.add(sign.raw_label or sign.label)

                for sign in last_signs:
                    draw_label(
                        frame,
                        sign.box,
                        f"{sign.label} {sign.confidence:.2f}",
                        (0, 165, 255),
                        thickness=2,
                    )

                if self.cfg.show_hud:
                    cv2.putText(
                        frame,
                        f"t={frame_idx / fps:6.1f}s | signs={len(last_signs)}",
                        (18, 34),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                writer.write(frame)
                frame_idx += 1
                bar.update(1)

                if progress_callback and (frame_idx % 15 == 0 or frame_idx == total):
                    progress_callback(frame_idx, total)

            bar.close()

        cap.release()
        writer.release()
        self._mux_h264(str(temp_video), input_path, str(final_video))

        return {
            "input": input_path,
            "output_video": str(final_video),
            "csv": str(csv_path),
            "frames": frame_idx,
            "fps": fps,
            "duration_sec": frame_idx / fps if fps else 0,
            "traffic_sign_rows": sign_rows,
            "unique_sign_codes": sorted(unique_codes),
        }

    @staticmethod
    def _log_detection(log, frame_idx: int, fps: float, det) -> None:
        x1, y1, x2, y2 = det.box
        log.writerow(
            {
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
                "extra": json.dumps(
                    {
                        **(det.extra or {}),
                        "raw_code": det.raw_label,
                    },
                    ensure_ascii=True,
                ),
            }
        )
