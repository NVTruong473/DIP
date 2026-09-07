from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from src.config import AppConfig
from src.video_processor import VideoProcessor


def drive_root() -> Path:
    p = Path("/content/drive/MyDrive/DIP")
    return p if p.exists() else Path.cwd()


def process_video(
    video_path,
    detect_signs,
    detect_plates,
    sign_conf,
    plate_conf,
    vehicle_conf,
    speed_mode,
    show_hud,
    progress=gr.Progress(),
):
    if not video_path:
        raise gr.Error("Please upload/select a video first.")

    root = drive_root()
    out_dir = root / "outputs"
    models_dir = root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    cfg = AppConfig(
        input_video=str(video_path),
        output_dir=str(out_dir),
        models_dir=str(models_dir),
        sign_conf=float(sign_conf),
        plate_conf=float(plate_conf),
        vehicle_conf=float(vehicle_conf),
        detect_signs=bool(detect_signs),
        detect_plates=bool(detect_plates),
        show_hud=bool(show_hud),
        frame_stride=2 if speed_mode else 1,
    )

    progress(0.01, desc="Loading models...")
    processor = VideoProcessor(cfg)

    def cb(i, total):
        if total > 0:
            progress(min(0.99, i / total), desc=f"Processing frame {i}/{total}")

    summary = processor.process(str(video_path), progress_callback=cb)
    progress(1.0, desc="Done")
    report = json.dumps(summary, indent=2, ensure_ascii=False)
    return summary["output_video"], summary["csv"], report


with gr.Blocks(title="Traffic Sign + Car License Plate Detection") as demo:
    gr.Markdown("""
# Traffic Sign + Car License Plate Detection

- **Biển báo:** YOLO nhận diện biển báo giao thông Việt Nam + DIP color cue.
- **Biển số ô tô:** YOLO phát hiện biển số; YOLO11 + ByteTrack chỉ dùng ẩn để xác nhận biển số thuộc `car / bus / truck`.
- **Video sạch:** không vẽ box xe, không ROI, không helmet, không Student ID.
    """)

    with gr.Row():
        video_in = gr.Video(label="Input video", sources=["upload"])
        video_out = gr.Video(label="Processed output")

    with gr.Row():
        detect_signs = gr.Checkbox(True, label="Detect Vietnamese traffic signs")
        detect_plates = gr.Checkbox(True, label="Detect car license plates")
        speed_mode = gr.Checkbox(False, label="Fast demo (infer every 2nd frame)")
        show_hud = gr.Checkbox(False, label="Show small HUD")

    with gr.Row():
        sign_conf = gr.Slider(0.10, 0.80, value=0.25, step=0.01, label="Traffic-sign confidence")
        plate_conf = gr.Slider(0.10, 0.80, value=0.30, step=0.01, label="License-plate confidence")
        vehicle_conf = gr.Slider(0.10, 0.80, value=0.30, step=0.01, label="Internal vehicle confidence")

    run_btn = gr.Button("Process video", variant="primary")
    csv_out = gr.File(label="Detection CSV")
    summary_out = gr.Code(label="Run summary", language="json")

    run_btn.click(
        fn=process_video,
        inputs=[
            video_in,
            detect_signs,
            detect_plates,
            sign_conf,
            plate_conf,
            vehicle_conf,
            speed_mode,
            show_hud,
        ],
        outputs=[video_out, csv_out, summary_out],
    )


if __name__ == "__main__":
    demo.launch(share=True, debug=False)
