from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import torch
from tqdm.auto import tqdm

from src.associations import associate_car_plates, associate_riders
from src.config import AppConfig
from src.detectors.helmet import HelmetDetector
from src.detectors.license_plate import LicensePlateDetector
from src.detectors.scene import SceneDetector
from src.detectors.traffic_sign import TrafficSignDetector
from src.model_manager import ModelManager
from src.plate_ocr import PlateOCR
from src.temporal import CurrentOnlySmoother, HelmetVoter
from src.utils.lighting import adapt_for_inference, crop_quality
from src.utils.visualization import draw_label

CSV_FIELDS = ["frame","time_sec","type","track_id","class","confidence","x1","y1","x2","y2","extra"]


class VideoProcessor:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        cfg.ensure_dirs()
        m = ModelManager(cfg.models_dir)
        self.sign = TrafficSignDetector(m, cfg.sign_conf, cfg.sign_imgsz, cfg.sign_tiled) if cfg.detect_signs else None
        self.scene = SceneDetector(m, cfg.scene_conf, cfg.scene_imgsz, cfg.scene_classes) if (cfg.detect_helmet or cfg.detect_plates) else None
        self.helmet = HelmetDetector(m, cfg.helmet_conf, cfg.helmet_imgsz) if cfg.detect_helmet else None
        self.plate = LicensePlateDetector(m, cfg.plate_conf, cfg.plate_imgsz) if cfg.detect_plates else None
        self.ocr = PlateOCR(
            gpu=torch.cuda.is_available(),
            min_conf=cfg.ocr_min_conf,
            min_votes=cfg.ocr_min_votes,
            model_dir=str(Path(cfg.models_dir)/"easyocr"),
        ) if cfg.detect_plates else None

        # Current-detection-only smoothing: no stale/predicted boxes are rendered.
        self.sign_smooth = CurrentOnlySmoother(cfg.sign_track_iou, alpha=0.68, confirm_hits=cfg.sign_confirm_hits, class_aware=True)
        self.helmet_smooth = CurrentOnlySmoother(0.22, alpha=0.72, confirm_hits=1, class_aware=True)
        self.plate_smooth = CurrentOnlySmoother(cfg.plate_track_iou, alpha=0.72, confirm_hits=cfg.plate_confirm_hits, class_aware=False)
        self.helmet_vote = HelmetVoter(cfg.helmet_vote_window, cfg.helmet_min_votes, cfg.helmet_stable_ratio)

    @staticmethod
    def _mux_h264(temp_video, source_video, final_video):
        base = ["-c:v","libx264","-preset","veryfast","-crf","22","-pix_fmt","yuv420p","-tag:v","avc1","-movflags","+faststart"]
        with_audio = ["ffmpeg","-y","-loglevel","error","-i",temp_video,"-i",source_video,"-map","0:v:0","-map","1:a?",*base,"-c:a","aac","-b:a","128k","-shortest",final_video]
        video_only = ["ffmpeg","-y","-loglevel","error","-i",temp_video,"-map","0:v:0",*base,"-an",final_video]
        try:
            subprocess.run(with_audio, check=True)
            os.remove(temp_video)
        except Exception:
            try:
                subprocess.run(video_only, check=True)
                os.remove(temp_video)
            except Exception:
                shutil.move(temp_video, final_video)

    @staticmethod
    def _crop(frame, box, pad=0.08):
        x1,y1,x2,y2 = map(int, box)
        h,w = frame.shape[:2]
        bw,bh = max(1,x2-x1), max(1,y2-y1)
        px,py = int(bw*pad), int(bh*pad)
        return frame[max(0,y1-py):min(h,y2+py), max(0,x1-px):min(w,x2+px)]

    @staticmethod
    def _log(log, frame_idx, fps, kind, det, extra=None):
        x1,y1,x2,y2 = det.box
        log.writerow({
            "frame":frame_idx,"time_sec":round(frame_idx/fps,3),"type":kind,
            "track_id":det.track_id if det.track_id is not None else "",
            "class":det.label,"confidence":round(det.confidence,4),
            "x1":x1,"y1":y1,"x2":x2,"y2":y2,
            "extra":json.dumps({**(det.extra or {}), **(extra or {})}, ensure_ascii=True),
        })

    def process(self, input_path: str, output_path: Optional[str]=None, progress_callback: Optional[Callable[[int,int],None]]=None) -> Dict:
        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width,height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stem = Path(input_path).stem
        out_dir = Path(self.cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        final = Path(output_path) if output_path else out_dir/f"{stem}_result.mp4"
        temp = final.with_name(final.stem+"_temp.mp4")
        csv_path = final.with_suffix(".csv")
        crops_dir = out_dir/f"{stem}_plates"; crops_dir.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Cannot create output writer")

        counts = {"traffic_sign":0,"helmet":0,"no_helmet":0,"car_plate":0,"ocr":0}
        best_crop: Dict[str,float] = {}
        env_counts: Dict[str,int] = {"DAY":0,"NIGHT":0,"GLARE":0}

        with open(csv_path,"w",newline="",encoding="utf-8") as f:
            log = csv.DictWriter(f, fieldnames=CSV_FIELDS); log.writeheader()
            bar = tqdm(total=total or None, desc="3-task traffic analysis", unit="frame")
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                raw = frame.copy()
                infer, env = adapt_for_inference(frame, self.cfg.night_v_threshold, self.cfg.glare_v_threshold) if self.cfg.adaptive_lighting else (frame,"DAY")
                env_counts[env] = env_counts.get(env,0)+1
                do_detect = frame_idx % max(1,self.cfg.frame_stride)==0

                signs=[]; scene=[]; helmets=[]; plate_pairs=[]; riders=[]
                if do_detect:
                    if self.sign:
                        signs = self.sign_smooth.update(self.sign.detect(infer))
                    if self.scene:
                        scene = self.scene.detect(infer)
                    if self.helmet:
                        helmets = self.helmet_smooth.update(self.helmet.detect(infer))
                        riders = associate_riders(scene, helmets)
                    if self.plate:
                        candidates = self.plate.detect(infer)
                        plate_pairs = associate_car_plates(scene, candidates)
                        current_plates = self.plate_smooth.update([p.plate for p in plate_pairs])
                        allowed_ids = {id(p) for p in current_plates}
                        plate_pairs = [p for p in plate_pairs if id(p.plate) in allowed_ids]

                # 1) Traffic signs.
                for d in signs:
                    txt = f"{d.label} {d.confidence:.2f}" if self.cfg.show_confidence else d.label
                    draw_label(frame,d.box,txt,(0,165,255),2)
                    self._log(log,frame_idx,fps,"traffic_sign",d,{"environment":env})
                    counts["traffic_sign"]+=1

                # 2) Helmet compliance. No parent boxes; no evidence => no guess.
                if do_detect and self.cfg.detect_helmet:
                    for r in riders:
                        instant = r.helmet.label if r.helmet is not None else "UNKNOWN"
                        stable = self.helmet_vote.update(r.track_id, instant)
                        if r.helmet is None:
                            continue
                        shown = stable if stable != "UNKNOWN" else instant
                        color = (0,180,0) if shown=="HELMET" else (0,0,255)
                        txt = shown + (f" {r.helmet.confidence:.2f}" if self.cfg.show_confidence else "")
                        draw_label(frame,r.helmet.box,txt,color,2)
                        self._log(log,frame_idx,fps,"rider_helmet",r.helmet,{"stable":stable,"rider_track_id":r.track_id,"motorcycle_track_id":r.motorcycle.track_id,"environment":env})
                        counts["helmet" if shown=="HELMET" else "no_helmet"]+=1

                # 3) Car license plates + validated OCR.
                if do_detect and self.cfg.detect_plates:
                    for pair in plate_pairs:
                        d = pair.plate
                        crop = self._crop(raw,d.box)
                        q = crop_quality(crop)
                        text = self.ocr.stable(pair.track_id)
                        h,w = crop.shape[:2] if crop.size else (0,0)
                        quality_ok = (
                            w>=self.cfg.plate_min_width_px and
                            h>=self.cfg.plate_min_height_px and
                            q["sharpness"]>=self.cfg.plate_min_sharpness and
                            20<=q["brightness"]<=240
                        )
                        if quality_ok and frame_idx % max(1,self.cfg.ocr_every_n_frames)==0:
                            candidate, oconf = self.ocr.recognize(crop)
                            text = self.ocr.update(pair.track_id,candidate,oconf)
                            if candidate:
                                counts["ocr"]+=1
                        label = f"PLATE {text}" if text else "PLATE"
                        if self.cfg.show_confidence and not text:
                            label += f" {d.confidence:.2f}"
                        draw_label(frame,d.box,label,(255,180,0),2)
                        self._log(log,frame_idx,fps,"car_license_plate",d,{"ocr":text,"quality":q,"environment":env})
                        counts["car_plate"]+=1

                        key = str(pair.track_id) if pair.track_id is not None else f"f{frame_idx}_{d.box[0]}"
                        score = d.confidence + min(q["sharpness"],200)/1000.0
                        if quality_ok and score > best_crop.get(key,-1):
                            for old in crops_dir.glob(f"track_{key}_*.jpg"):
                                try: old.unlink()
                                except OSError: pass
                            cv2.imwrite(str(crops_dir/f"track_{key}_{text or 'UNKNOWN'}_{d.confidence:.2f}.jpg"),crop)
                            best_crop[key]=score

                if self.cfg.show_hud:
                    cv2.putText(frame,f"{env} | signs {len(signs)} | riders {len(riders)} | plates {len(plate_pairs)}",(18,32),cv2.FONT_HERSHEY_SIMPLEX,0.62,(255,255,255),2,cv2.LINE_AA)

                writer.write(frame)
                frame_idx+=1
                bar.update(1)
                if progress_callback and (frame_idx%15==0 or frame_idx==total):
                    progress_callback(frame_idx,total)
            bar.close()

        cap.release(); writer.release(); self._mux_h264(str(temp),str(input_path),str(final))
        return {
            "input":str(input_path),"output_video":str(final),"csv":str(csv_path),
            "plate_crops_dir":str(crops_dir),"frames":frame_idx,"fps":fps,
            "duration_sec":frame_idx/fps if fps else 0,"counts":counts,
            "lighting_frames":env_counts,
        }
