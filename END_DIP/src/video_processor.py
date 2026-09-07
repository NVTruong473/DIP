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
from src.detectors.license_plate import LicensePlateDetector
from src.detectors.scene import SceneDetector
from src.detectors.traffic_sign import TrafficSignDetector
from src.model_manager import ModelManager
from src.plate_logic import associate_car_plates
from src.temporal import DetectionTemporalHold
from src.utils.visualization import draw_label


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
            )
            if config.detect_signs
            else None
        )

        # Vehicle boxes are internal only. They filter out motorcycle plates and
        # provide stable ByteTrack IDs, but are never drawn on the output video.
        self.vehicle_detector = (
            SceneDetector(
                self.manager,
                conf=config.vehicle_conf,
                imgsz=config.vehicle_imgsz,
                classes=config.vehicle_classes,
            )
            if config.detect_plates
            else None
        )
        self.plate_detector = (
            LicensePlateDetector(self.manager, conf=config.plate_conf, imgsz=config.plate_imgsz)
            if config.detect_plates
            else None
        )

        self.sign_hold = DetectionTemporalHold(config.sign_hold_frames, config.sign_iou_match, class_aware=True)
        self.plate_hold = DetectionTemporalHold(config.plate_hold_frames, config.plate_iou_match, class_aware=False)

    @staticmethod
    def _mux_h264(temp_video: str, source_video: str, final_video: str) -> None:
        """Encode a Chrome/Colab-friendly H.264 MP4 with optional source audio."""
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
            print("[WARN] Falling back to OpenCV mp4v output; inline browser playback may be unavailable.")
            shutil.move(temp_video, final_video)

    @staticmethod
    def _safe_crop(frame, box, pad: float = 0.08):
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame.shape[:2]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        px = int(round(bw * pad))
        py = int(round(bh * pad))
        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w, x2 + px)
        y2 = min(h, y2 + py)
        return frame[y1:y2, x1:x2]

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
        plate_crops_dir = out_dir / f"{stem}_plates"
        if self.cfg.save_plate_crops and self.cfg.detect_plates:
            plate_crops_dir.mkdir(parents=True, exist_ok=True)

        final_video = Path(output_path) if output_path else out_dir / f"{stem}_result.mp4"
        final_video.parent.mkdir(parents=True, exist_ok=True)
        temp_video = final_video.with_name(final_video.stem + "_temp.mp4")
        csv_path = final_video.with_suffix(".csv")

        writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Cannot create output video writer.")

        last_signs = []
        last_plates = []
        sign_rows = 0
        plate_rows = 0
        saved_plate_crops = 0
        vehicle_tracks_with_plate = set()
        best_crop_conf: Dict[str, float] = {}

        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            log = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            log.writeheader()
            bar = tqdm(total=total if total > 0 else None, desc="Processing video", unit="frame")
            frame_idx = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                raw_frame = frame.copy() if self.cfg.save_plate_crops and self.cfg.detect_plates else frame
                do_detect = frame_idx % max(1, self.cfg.frame_stride) == 0

                if self.sign_detector is not None:
                    current_signs = self.sign_detector.detect(frame) if do_detect else []
                    last_signs = self.sign_hold.update(current_signs)
                    if do_detect:
                        for sign in current_signs:
                            self._log_detection(log, frame_idx, fps, "traffic_sign", sign)
                            sign_rows += 1

                current_pairs = []
                if self.vehicle_detector is not None and self.plate_detector is not None:
                    if do_detect:
                        vehicles = self.vehicle_detector.detect(frame)
                        candidate_plates = self.plate_detector.detect(frame)
                        current_pairs = associate_car_plates(vehicles, candidate_plates)
                        current_plates = [pair.plate for pair in current_pairs]
                    else:
                        current_plates = []
                    last_plates = self.plate_hold.update(current_plates)

                # Keep the output intentionally sparse: only traffic signs and
                # actual car/bus/truck license plates are rendered.
                for sign in last_signs:
                    draw_label(frame, sign.box, f"{sign.label} {sign.confidence:.2f}", (255, 120, 0))

                for plate in last_plates:
                    draw_label(frame, plate.box, f"Biển số ô tô {plate.confidence:.2f}", (0, 190, 0))

                if do_detect:
                    for pair in current_pairs:
                        plate = pair.plate
                        self._log_detection(log, frame_idx, fps, "car_license_plate", plate)
                        plate_rows += 1

                        if pair.vehicle.track_id is not None:
                            vehicle_tracks_with_plate.add(pair.vehicle.track_id)
                            crop_key = f"track_{pair.vehicle.track_id}"
                        else:
                            crop_key = f"frame_{frame_idx}_{plate.box[0]}_{plate.box[1]}"

                        if self.cfg.save_plate_crops:
                            old_conf = best_crop_conf.get(crop_key, -1.0)
                            if plate.confidence > old_conf + 0.02:
                                crop = self._safe_crop(raw_frame, plate.box)
                                if crop.size and crop.shape[1] >= 18 and crop.shape[0] >= 8:
                                    crop_path = plate_crops_dir / f"{crop_key}_conf{plate.confidence:.2f}.jpg"
                                    # Remove older best crop(s) for this vehicle track.
                                    if pair.vehicle.track_id is not None:
                                        for old in plate_crops_dir.glob(f"{crop_key}_conf*.jpg"):
                                            try:
                                                old.unlink()
                                            except OSError:
                                                pass
                                    cv2.imwrite(str(crop_path), crop)
                                    best_crop_conf[crop_key] = plate.confidence
                                    saved_plate_crops += 1

                if self.cfg.show_hud:
                    cv2.putText(
                        frame,
                        f"t={frame_idx / fps:6.1f}s | signs={len(last_signs)} | car plates={len(last_plates)}",
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
            "plate_crops_dir": str(plate_crops_dir) if self.cfg.save_plate_crops and self.cfg.detect_plates else None,
            "frames": frame_idx,
            "fps": fps,
            "duration_sec": frame_idx / fps if fps else 0,
            "traffic_sign_rows": sign_rows,
            "car_license_plate_rows": plate_rows,
            "unique_vehicle_tracks_with_plate": len(vehicle_tracks_with_plate),
            "saved_plate_crop_updates": saved_plate_crops,
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
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "extra": json.dumps(det.extra or {}, ensure_ascii=False),
        })
