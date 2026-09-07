from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import cv2
from tqdm.auto import tqdm

from src.config import AppConfig
from src.detectors.helmet import HelmetDetector
from src.detectors.scene import SceneDetector
from src.detectors.traffic_sign import TrafficSignDetector
from src.model_manager import ModelManager
from src.rider_logic import associate_riders
from src.temporal import HelmetTemporalVoter, SignTemporalHold
from src.utils.geometry import filter_detections_in_roi, normalized_polygon_to_pixels
from src.utils.visualization import draw_label, draw_roi, status_color


CSV_FIELDS = ["frame", "time_sec", "type", "track_id", "class", "confidence", "x1", "y1", "x2", "y2", "extra"]


class VideoProcessor:
    def __init__(self, config: AppConfig):
        self.cfg = config
        self.cfg.ensure_dirs()
        self.manager = ModelManager(config.models_dir)

        self.sign_detector = (
            TrafficSignDetector(
                self.manager,
                conf=config.sign_conf,
                imgsz=config.sign_imgsz,
                use_dip_enhancement=config.use_dip_enhancement,
                tiled=config.sign_tiled,
            ) if config.detect_signs else None
        )
        self.scene_detector = (
            SceneDetector(self.manager, conf=config.scene_conf, imgsz=config.scene_imgsz, classes=config.scene_classes)
            if config.detect_helmet else None
        )
        self.helmet_detector = (
            HelmetDetector(self.manager, conf=config.helmet_conf, imgsz=config.helmet_imgsz)
            if config.detect_helmet else None
        )

        self.voter = HelmetTemporalVoter(
            window=config.vote_window,
            min_votes=config.min_votes,
            stable_ratio=config.stable_ratio,
            ttl_frames=config.state_ttl_frames,
        )
        self.sign_hold = SignTemporalHold(config.sign_hold_frames, config.sign_iou_match)

    @staticmethod
    def _crop_from_polygon(poly, width: int, height: int) -> Tuple[int, int, int, int]:
        x, y, w, h = cv2.boundingRect(poly)
        pad_x = int(0.03 * width)
        pad_y = int(0.03 * height)
        return max(0, x - pad_x), max(0, y - pad_y), min(width, x + w + pad_x), min(height, y + h + pad_y)

    @staticmethod
    def _mux_h264(temp_video: str, source_video: str, final_video: str) -> None:
        """Encode a browser/Colab-friendly MP4.

        OpenCV's temporary `mp4v` output is not reliably playable in Chrome.
        Force H.264 + yuv420p + avc1 and move the moov atom to the front so the
        final Drive file can be streamed by HTML5/Colab without downloading it.
        """
        common_video = [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-level", "4.1",
            "-tag:v", "avc1",
            "-movflags", "+faststart",
        ]

        with_audio = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", temp_video,
            "-i", source_video,
            "-map", "0:v:0",
            "-map", "1:a?",
            *common_video,
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            final_video,
        ]

        video_only = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", temp_video,
            "-map", "0:v:0",
            *common_video,
            "-an",
            final_video,
        ]

        try:
            subprocess.run(with_audio, check=True)
            os.remove(temp_video)
            return
        except Exception as first_error:
            print(f"[WARN] H.264 mux with source audio failed: {first_error}")

        try:
            subprocess.run(video_only, check=True)
            os.remove(temp_video)
            return
        except Exception as second_error:
            print(f"[WARN] H.264 browser encoding failed: {second_error}")
            print("[WARN] Falling back to OpenCV mp4v output; browser playback may be unavailable.")
            shutil.move(temp_video, final_video)

    def process(self, input_path: str, output_path: Optional[str] = None, progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict:
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
        snapshots_dir = out_dir / f"{stem}_violations"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        final_video = Path(output_path) if output_path else out_dir / f"{stem}_result.mp4"
        final_video.parent.mkdir(parents=True, exist_ok=True)
        temp_video = final_video.with_name(final_video.stem + "_temp.mp4")
        csv_path = final_video.with_suffix(".csv")

        writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Cannot create output video writer.")

        polygon_px = normalized_polygon_to_pixels(self.cfg.right_road_roi, width, height)
        crop_box = self._crop_from_polygon(polygon_px, width, height)

        last_signs = []
        last_scene = []
        last_helmets = []
        last_snapshot_time: Dict[int, float] = {}
        snapshot_count = 0
        nohelmet_tracks = set()
        sign_events = 0
        rider_observations = 0

        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            log = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            log.writeheader()
            bar = tqdm(total=total if total > 0 else None, desc="Processing video", unit="frame")
            frame_idx = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                do_detect = frame_idx % max(1, self.cfg.frame_stride) == 0

                if self.sign_detector is not None and do_detect:
                    last_signs = self.sign_hold.update(self.sign_detector.detect(frame))
                elif self.sign_detector is not None:
                    last_signs = self.sign_hold.update([])

                if self.scene_detector is not None and self.helmet_detector is not None and do_detect:
                    scene = self.scene_detector.detect(frame, crop_box=crop_box)
                    helmets = self.helmet_detector.detect(frame, crop_box=crop_box)
                    last_scene = filter_detections_in_roi(scene, polygon_px, anchor="bottom")
                    last_helmets = filter_detections_in_roi(helmets, polygon_px, anchor="center")

                if self.cfg.show_roi and self.cfg.detect_helmet:
                    draw_roi(frame, polygon_px)

                for sign in last_signs:
                    extra = sign.extra or {}
                    color_hint = extra.get("dominant_color", "")
                    suffix = f" [{color_hint}]" if color_hint and color_hint != "unknown" else ""
                    draw_label(frame, sign.box, f"{sign.label} {sign.confidence:.2f}{suffix}", (255, 120, 0))
                    if do_detect:
                        self._log_detection(log, frame_idx, fps, "traffic_sign", sign)
                        sign_events += 1

                riders = associate_riders(last_scene, last_helmets, frame.shape) if self.cfg.detect_helmet else []
                current_violation_tracks = []
                for rider in riders:
                    stable = self.voter.update(rider.track_id, rider.instantaneous_status, frame_idx) if do_detect else self.voter.get(rider.track_id)
                    rider.stable_status = stable
                    shown = stable if stable != "UNKNOWN" else rider.instantaneous_status
                    track_txt = f" ID:{rider.track_id}" if rider.track_id is not None else ""
                    draw_label(frame, rider.person.box, f"RIDER{track_txt} | {shown}", status_color(shown), thickness=3 if shown == "NO_HELMET" else 2)
                    if rider.helmet is not None:
                        draw_label(frame, rider.helmet.box, f"{rider.helmet.label} {rider.helmet.confidence:.2f}", status_color(rider.helmet.label))

                    if do_detect:
                        self._log_rider(log, frame_idx, fps, rider)
                        rider_observations += 1

                    if stable == "NO_HELMET" and rider.track_id is not None:
                        nohelmet_tracks.add(rider.track_id)
                        current_violation_tracks.append(rider.track_id)

                cv2.putText(frame, f"t={frame_idx / fps:6.1f}s | signs={len(last_signs)} | riders={len(riders)}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

                now_sec = frame_idx / fps
                for track_id in current_violation_tracks:
                    last = last_snapshot_time.get(track_id, -1e9)
                    if now_sec - last >= self.cfg.violation_snapshot_cooldown_sec:
                        snap = snapshots_dir / f"no_helmet_track{track_id}_t{now_sec:07.2f}.jpg"
                        cv2.imwrite(str(snap), frame)
                        last_snapshot_time[track_id] = now_sec
                        snapshot_count += 1

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
            "snapshots_dir": str(snapshots_dir),
            "frames": frame_idx,
            "fps": fps,
            "duration_sec": frame_idx / fps if fps else 0,
            "sign_detection_rows": sign_events,
            "rider_observation_rows": rider_observations,
            "unique_no_helmet_tracks": len(nohelmet_tracks),
            "violation_snapshots": snapshot_count,
            "roi": self.cfg.right_road_roi,
        }

    @staticmethod
    def _log_detection(log, frame_idx: int, fps: float, kind: str, det) -> None:
        x1, y1, x2, y2 = det.box
        log.writerow({
            "frame": frame_idx,
            "time_sec": round(frame_idx / fps, 3),
            "type": kind,
            "track_id": det.track_id if det.track_id is not None else "",
            "class": det.label,
            "confidence": round(det.confidence, 4),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "extra": json.dumps(det.extra or {}, ensure_ascii=False),
        })

    @staticmethod
    def _log_rider(log, frame_idx: int, fps: float, rider) -> None:
        x1, y1, x2, y2 = rider.person.box
        hconf = rider.helmet.confidence if rider.helmet is not None else 0.0
        log.writerow({
            "frame": frame_idx,
            "time_sec": round(frame_idx / fps, 3),
            "type": "rider_helmet_status",
            "track_id": rider.track_id if rider.track_id is not None else "",
            "class": rider.stable_status,
            "confidence": round(hconf, 4),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "extra": json.dumps({
                "instantaneous": rider.instantaneous_status,
                "helmet_raw": rider.helmet.raw_label if rider.helmet else None,
                "vehicle_track_id": rider.vehicle.track_id,
            }, ensure_ascii=False),
        })
