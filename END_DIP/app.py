from __future__ import annotations

import json
from pathlib import Path
import gradio as gr

from src.config import AppConfig
from src.video_processor import VideoProcessor


def drive_root():
    p = Path('/content/drive/MyDrive/DIP')
    return p if p.exists() else Path.cwd()


def process_video(video_path, signs, helmet, plates, sign_conf, helmet_conf, plate_conf, speed_mode, show_hud, progress=gr.Progress()):
    if not video_path:
        raise gr.Error('Please upload a video first.')
    root = drive_root(); out = root/'outputs'; models = root/'models'
    out.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig(
        input_video=str(video_path), output_dir=str(out), models_dir=str(models),
        detect_signs=bool(signs), detect_helmet=bool(helmet), detect_plates=bool(plates),
        sign_conf=float(sign_conf), helmet_conf=float(helmet_conf), plate_conf=float(plate_conf),
        frame_stride=2 if speed_mode else 1, show_hud=bool(show_hud),
    )
    progress(0.01, desc='Loading models...')
    proc = VideoProcessor(cfg)
    def cb(i,total):
        if total: progress(min(0.99,i/total), desc=f'Frame {i}/{total}')
    summary = proc.process(str(video_path), progress_callback=cb)
    progress(1.0, desc='Done')
    return summary['output_video'], summary['csv'], json.dumps(summary, indent=2)


with gr.Blocks(title='Traffic Intelligence - 3 Tasks') as demo:
    gr.Markdown('''
# Traffic Intelligence — 3 Tasks
**Traffic signs** + **motorcycle helmet compliance** + **car license-plate detection/OCR**.  
Person/motorcycle/car boxes are used internally and hidden to keep the video readable.
''')
    with gr.Row():
        video_in = gr.Video(label='Input video', sources=['upload'])
        video_out = gr.Video(label='Processed output')
    with gr.Row():
        signs = gr.Checkbox(True, label='Traffic signs')
        helmet = gr.Checkbox(True, label='Helmet compliance')
        plates = gr.Checkbox(True, label='Car plates + OCR')
        speed = gr.Checkbox(False, label='Fast preview (stride 2)')
        hud = gr.Checkbox(False, label='Show HUD')
    with gr.Row():
        sign_conf = gr.Slider(.15,.75,value=.32,step=.01,label='Sign confidence')
        helmet_conf = gr.Slider(.15,.75,value=.38,step=.01,label='Helmet confidence')
        plate_conf = gr.Slider(.15,.75,value=.34,step=.01,label='Plate confidence')
    run = gr.Button('Process video', variant='primary')
    csv_out = gr.File(label='CSV')
    report = gr.Code(label='Summary', language='json')
    run.click(process_video,[video_in,signs,helmet,plates,sign_conf,helmet_conf,plate_conf,speed,hud],[video_out,csv_out,report])

if __name__ == '__main__':
    demo.launch(share=True, debug=False)
