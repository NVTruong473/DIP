from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from src.config import AppConfig, DEFAULT_RIGHT_ROAD_ROI, parse_roi_json
from src.video_processor import VideoProcessor


def drive_root() -> Path:
    p = Path("/content/drive/MyDrive/DIP")
    return p if p.exists() else Path.cwd()


def process_video(video_path, detect_signs, detect_helmet, show_roi, sign_conf, helmet_conf, scene_conf, roi_json, speed_mode, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Please upload/select a video first.")

    root = drive_root()
    out_dir = root / "outputs"
    models_dir = root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        roi = parse_roi_json(roi_json)
    except Exception as exc:
        raise gr.Error(f"Invalid ROI JSON: {exc}")

    stride = 2 if speed_mode else 1
    cfg = AppConfig(
        input_video=str(video_path),
        output_dir=str(out_dir),
        models_dir=str(models_dir),
        sign_conf=float(sign_conf),
        helmet_conf=float(helmet_conf),
        scene_conf=float(scene_conf),
        detect_signs=bool(detect_signs),
        detect_helmet=bool(detect_helmet),
        show_roi=bool(show_roi),
        right_road_roi=roi,
        frame_stride=stride,
    )

    progress(0.01, desc="Loading YOLO models...")
    processor = VideoProcessor(cfg)

    def cb(i, total):
        if total > 0:
            progress(min(0.99, i / total), desc=f"Processing frame {i}/{total}")

    summary = processor.process(str(video_path), progress_callback=cb)
    progress(1.0, desc="Done")
    report = json.dumps(summary, indent=2, ensure_ascii=False)
    return summary["output_video"], summary["csv"], report


with gr.Blocks(title="Traffic Safety Monitoring - DIP + YOLO") as demo:
    gr.Markdown("""
# Traffic Safety Monitoring — DIP + YOLO
**Traffic signs:** Vietnamese 56-class YOLO detector + DIP color cue  
**Helmet:** only riders associated with motorcycles/bicycles inside the right-road ROI  
**Important:** a missed helmet detection is `UNKNOWN`, never automatically a violation.
    """)

    with gr.Row():
        video_in = gr.Video(label="Input video", sources=["upload"])
        video_out = gr.Video(label="Processed output")

    with gr.Row():
        detect_signs = gr.Checkbox(True, label="Detect Vietnamese traffic signs")
        detect_helmet = gr.Checkbox(True, label="Detect helmet compliance")
        show_roi = gr.Checkbox(True, label="Show right-road ROI")
        speed_mode = gr.Checkbox(False, label="Fast demo (infer every 2nd frame)")

    with gr.Row():
        sign_conf = gr.Slider(0.10, 0.80, value=0.25, step=0.01, label="Traffic-sign confidence")
        helmet_conf = gr.Slider(0.10, 0.80, value=0.35, step=0.01, label="Helmet confidence")
        scene_conf = gr.Slider(0.10, 0.80, value=0.30, step=0.01, label="Rider/vehicle confidence")

    roi_json = gr.Textbox(value=json.dumps(DEFAULT_RIGHT_ROAD_ROI), label="Right-road polygon (normalized JSON)", info="Points are [x,y] in [0,1]. Change this when the camera/road geometry changes.")
    run_btn = gr.Button("Process video", variant="primary")
    csv_out = gr.File(label="Detection CSV")
    summary_out = gr.Code(label="Run summary", language="json")

    run_btn.click(
        fn=process_video,
        inputs=[video_in, detect_signs, detect_helmet, show_roi, sign_conf, helmet_conf, scene_conf, roi_json, speed_mode],
        outputs=[video_out, csv_out, summary_out],
    )


if __name__ == "__main__":
    demo.launch(share=True, debug=False)
