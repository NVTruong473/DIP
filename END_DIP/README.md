# Traffic Safety Monitoring System — DIP + YOLO

Redesign of the original `END_DIP/521H0324_521H0461.py` pipeline for real road video and Google Colab.

The original baseline uses HSV thresholding -> contours -> template matching. It remains in the repository for comparison/reporting, but the new pipeline uses learned detectors as the primary perception stack and keeps classic DIP as an interpretable secondary cue.

## Architecture

```text
VIDEO
 ├─> Vietnamese traffic-sign YOLO (56 classes)
 │     ├─> overlapping-tile inference for small distant signs
 │     └─> classic DIP color analysis (red / blue / yellow)
 │
 └─> RIGHT-ROAD POLYGON ROI
       ├─> YOLO11n scene detector + ByteTrack
       │     └─> person <-> motorcycle/bicycle association
       └─> YOLO11 helmet detector
             └─> head-region association
                   └─> temporal voting
                        ├─ HELMET
                        ├─ NO_HELMET -> violation snapshot
                        └─ UNKNOWN -> never guessed as a violation
```

### Critical design rule

**Missing a helmet detection does not mean `NO_HELMET`.** A missed detection is `UNKNOWN`. A violation is emitted only after explicit `NO_HELMET` observations become stable over multiple frames.

## Supplied video inspection

The supplied video is approximately:

- 1920 x 1080
- 29.97 FPS
- 103 seconds

Because the divider/road geometry moves in image coordinates, helmet inference uses a configurable normalized polygon rather than `x > width/2`.

Default ROI:

```json
[[0.48, 0.36], [1.0, 0.36], [1.0, 1.0], [0.38, 1.0]]
```

Change this in Gradio for a different camera view.

## Models

### Vietnamese traffic signs

Default: `liamxdev/vtsr` (YOLOv8n, 56 Vietnamese traffic-sign classes). The project uses the public TorchScript model and `label-mapping.json`. Two overlapping image tiles preserve more pixels for small signs in a 1920x1080 frame.

Source: https://huggingface.co/liamxdev/vtsr

### Helmet / no-helmet

Default: `nnsohamnn/helmet-detection-yolo11/yolov11s(80 epochs).pt`. The same model repository also has a larger YOLO11m checkpoint. The small model is used by default so a Colab T4 can run the complete multi-stage pipeline.

Source: https://huggingface.co/nnsohamnn/helmet-detection-yolo11

### Person / motorcycle tracking

YOLO11n COCO detects `person`, `bicycle`, and `motorcycle`. ByteTrack provides track IDs for temporal voting.

## Google Drive layout

Keep large artifacts out of GitHub:

```text
MyDrive/
└── DIP/
    ├── video1.mp4
    ├── models/
    │   └── helmet_best.pt       # optional fine-tuned checkpoint
    ├── datasets/
    ├── training_runs/
    └── outputs/
        ├── video1_result.mp4
        ├── video1_result.csv
        └── video1_violations/
```

Anything in Drive survives a Colab runtime reset.

## Google Colab — normal inference

### 1. Enable a GPU

Use a T4 GPU or better.

### 2. Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Put your input at:

```text
/content/drive/MyDrive/DIP/video1.mp4
```

### 3. Clone this branch

```bash
!git clone -b feature/yolo-traffic-safety https://github.com/NVTruong473/DIP.git /content/DIP
%cd /content/DIP/END_DIP
```

### 4. Install

```bash
!pip install -q -r requirements.txt
```

### 5. Download model files once

```bash
!python download_models.py --models-dir '/content/drive/MyDrive/DIP/models'
```

### 6. Process video1

```bash
!python main.py \
  --input '/content/drive/MyDrive/DIP/video1.mp4' \
  --output-dir '/content/drive/MyDrive/DIP/outputs' \
  --models-dir '/content/drive/MyDrive/DIP/models'
```

Persistent results:

```text
/content/drive/MyDrive/DIP/outputs/video1_result.mp4
/content/drive/MyDrive/DIP/outputs/video1_result.csv
/content/drive/MyDrive/DIP/outputs/video1_violations/
```

## Gradio UI

Run:

```bash
!python app.py
```

The UI lets you upload another video, enable/disable each detector, tune confidence thresholds, edit the right-road polygon, toggle ROI display, select fast demo mode, preview the processed result, and download the CSV.

## Output CSV

Columns:

```text
frame,time_sec,type,track_id,class,confidence,x1,y1,x2,y2,extra
```

This makes the project measurable rather than only a bounding-box demo.

## Optional helmet fine-tuning (Colab/T4-oriented)

Normal inference works immediately with the pretrained helmet checkpoint. Fine-tuning is optional.

The provided script targets the public Roboflow project `Helmet and no helmet rider detection`, version 5. It removes the licence-plate class and remaps the data to exactly two classes: `With Helmet` and `Without Helmet`.

### Prepare dataset

Set your Roboflow key only in the Colab environment; never commit it:

```python
import os
os.environ['ROBOFLOW_API_KEY'] = 'YOUR_KEY'
```

Then:

```bash
!python training/prepare_roboflow_dataset.py \
  --target '/content/drive/MyDrive/DIP/datasets/helmet_rf_v5'
```

### Fine-tune

```bash
!python training/train_helmet.py \
  --data '/content/drive/MyDrive/DIP/datasets/helmet_rf_v5/data_helmet_2class.yaml' \
  --models-dir '/content/drive/MyDrive/DIP/models' \
  --runs-dir '/content/drive/MyDrive/DIP/training_runs' \
  --epochs 20 \
  --batch 16
```

If the T4 runtime is constrained, use `--epochs 15 --fraction 0.75`.

The final checkpoint is copied to:

```text
/content/drive/MyDrive/DIP/models/helmet_best.pt
```

Subsequent inference automatically prefers `helmet_best.pt`.

### Evaluate

```bash
!python training/evaluate_helmet.py \
  --model '/content/drive/MyDrive/DIP/models/helmet_best.pt' \
  --data '/content/drive/MyDrive/DIP/datasets/helmet_rf_v5/data_helmet_2class.yaml'
```

## Source tree

```text
END_DIP/
├── main.py
├── app.py
├── download_models.py
├── requirements.txt
├── colab_demo.ipynb
├── src/
│   ├── config.py
│   ├── common.py
│   ├── model_manager.py
│   ├── rider_logic.py
│   ├── temporal.py
│   ├── video_processor.py
│   ├── detectors/
│   │   ├── traffic_sign.py
│   │   ├── scene.py
│   │   └── helmet.py
│   └── utils/
│       ├── dip.py
│       ├── geometry.py
│       └── visualization.py
└── training/
    ├── prepare_roboflow_dataset.py
    ├── train_helmet.py
    └── evaluate_helmet.py
```

The original template-matching code and `sign_templates/` are intentionally retained as the classic-DIP baseline for the final report/demo comparison.
