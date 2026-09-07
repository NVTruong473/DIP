from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from src.config import AppConfig
from src.video_processor import VideoProcessor


def drive_root() -> Path:
    path = Path("/content/drive/MyDrive/DIP")
    return path if path.exists() else Path.cwd()


def process_video(
    video_path,
    sign_conf,
    speed_mode,
    show_hud,
    progress=gr.Progress(),
):
    if not video_path:
        raise gr.Error("Please upload a video first.")

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
        frame_stride=2 if speed_mode else 1,
        show_hud=bool(show_hud),
    )

    progress(0.01, desc="Loading traffic-sign model...")
    processor = VideoProcessor(cfg)

    def cb(i, total):
        if total > 0:
            progress(min(0.99, i / total), desc=f"Processing frame {i}/{total}")

    summary = processor.process(str(video_path), progress_callback=cb)
    progress(1.0, desc="Done")
    return (
        summary["output_video"],
        summary["csv"],
        json.dumps(summary, indent=2),
    )


with gr.Blocks(title="Traffic Sign Detection - DIP + YOLO") as demo:
    gr.Markdown(
        """
# Traffic Sign Detection — DIP + YOLO

Vietnamese traffic signs are detected with YOLO and displayed using short **English labels**.
The classic DIP color cue remains in the CSV metadata. No helmet, vehicle, license-plate, ROI, or Student ID overlays are used.
"""
    )

    with gr.Row():
        video_in = gr.Video(label="Input video", sources=["upload"])
        video_out = gr.Video(label="Processed output")

    with gr.Row():
        sign_conf = gr.Slider(
            0.10,
            0.80,
            value=0.25,
            step=0.01,
            label="Traffic-sign confidence",
        )
        speed_mode = gr.Checkbox(
            False,
            label="Fast preview (infer every 2nd frame)",
        )
        show_hud = gr.Checkbox(False, label="Show small HUD")

    run_btn = gr.Button("Process video", variant="primary")
    csv_out = gr.File(label="Detection CSV")
    summary_out = gr.Code(label="Run summary", language="json")

    run_btn.click(
        fn=process_video,
        inputs=[video_in, sign_conf, speed_mode, show_hud],
        outputs=[video_out, csv_out, summary_out],
    )


if __name__ == "__main__":
    demo.launch(share=True, debug=False)
