# Traffic Sign Recognition + Car License Plate Detection — DIP + YOLO

Redesign of the original `END_DIP/521H0324_521H0461.py` project for real road video and Google Colab.

The original baseline uses HSV thresholding -> contours -> template matching. It is kept in the repository for the Digital Image Processing comparison/report, while the new pipeline uses learned detectors as the primary perception stack and keeps classic DIP as an interpretable secondary cue for traffic signs.

## New project scope

The previous rider/helmet task has been removed. The output now contains only:

1. **Vietnamese traffic signs**
2. **License plates belonging to cars / buses / trucks**

Vehicle detections are used internally for filtering/tracking but are **not drawn**, so the final video is much less cluttered. There is no helmet ROI and no Student ID overlay.

## Architecture

```text
VIDEO
 │
 ├── Vietnamese traffic-sign YOLO (56 classes)
 │     ├── overlapping tiled inference for small distant signs
 │     ├── Vietnamese Unicode labels
 │     └── classic DIP red / blue / yellow color analysis
 │
 └── Car license-plate branch
       ├── YOLO11n + ByteTrack: car / bus / truck (hidden boxes)
       ├── YOLOv8n license-plate detector
       └── plate <-> vehicle geometric association
              ├── keep plate if it belongs to car/bus/truck
              ├── discard motorcycle/non-vehicle false matches
              └── save best plate crop per tracked vehicle

FINAL VIDEO
 ├── traffic-sign boxes
 └── car-license-plate boxes
```

## Why this is cleaner than helmet detection

The helmet version required person + motorcycle + helmet boxes plus a large road ROI. The current design renders only two object types that are visually small and semantically related to road infrastructure/vehicles.

The car/bus/truck detector still runs, but its boxes stay hidden and are used only to verify plate ownership.

## Supplied video

The supplied video is approximately:

- 1920 x 1080
- 29.97 FPS
- 103 seconds

There are cars at multiple distances. Plates on nearby vehicles contain enough pixels for detection; very distant plates are naturally harder. For this reason plate inference defaults to a larger `960` input size.

## Models

### Vietnamese traffic signs

Default: `liamxdev/vtsr` — YOLOv8n, 56 Vietnamese traffic-sign classes.

Source: https://huggingface.co/liamxdev/vtsr

### License plates

Default: `Koushim/yolov8-license-plate-detection/best.pt` — lightweight YOLOv8n one-class license-plate detector.

Source: https://huggingface.co/Koushim/yolov8-license-plate-detection

A custom `plate_best.pt` in `MyDrive/DIP/models/` automatically overrides the default model.

### Vehicle filtering / tracking

YOLO11n COCO detects only:

- `car` — class 2
- `bus` — class 5
- `truck` — class 7

ByteTrack gives stable vehicle IDs. These vehicle boxes are not rendered.

## Google Drive layout

```text
MyDrive/
└── DIP/
    ├── video1.mp4
    ├── models/
    │   └── plate_best.pt            # optional Vietnamese fine-tuned checkpoint
    ├── datasets/
    │   └── vn_car_plate/
    ├── training_runs/
    └── outputs/
        ├── video1_result.mp4
        ├── video1_result.csv
        └── video1_plates/
            └── track_*_conf*.jpg
```

All important artifacts remain on Drive after a Colab runtime reset.

## Google Colab — normal inference

The easiest option is `colab_demo.ipynb`.

### 1. Enable GPU

Use **Runtime -> Change runtime type -> T4 GPU** or better.

### 2. Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Place the input at:

```text
/content/drive/MyDrive/DIP/video1.mp4
```

### 3. Clone the development branch

```bash
!rm -rf /content/DIP
!git clone -b feature/yolo-traffic-safety https://github.com/NVTruong473/DIP.git /content/DIP
%cd /content/DIP/END_DIP
```

### 4. Install

```bash
!pip install -q -r requirements.txt
```

### 5. Download models once

```bash
!python download_models.py --models-dir '/content/drive/MyDrive/DIP/models'
```

### 6. Process video

```bash
!python main.py \
  --input '/content/drive/MyDrive/DIP/video1.mp4' \
  --output-dir '/content/drive/MyDrive/DIP/outputs' \
  --models-dir '/content/drive/MyDrive/DIP/models' \
  --sign-conf 0.25 \
  --plate-conf 0.30 \
  --vehicle-conf 0.30
```

Results:

```text
/content/drive/MyDrive/DIP/outputs/video1_result.mp4
/content/drive/MyDrive/DIP/outputs/video1_result.csv
/content/drive/MyDrive/DIP/outputs/video1_plates/
```

The output MP4 is re-encoded to H.264/yuv420p/avc1 with `faststart` for Chrome and Google Colab playback.

## Visually clean defaults

By default the output does **not** show:

- Student ID
- road ROI
- car/bus/truck boxes
- frame statistics/HUD
- motorcycle/person/helmet boxes

Only traffic signs and verified car license plates are drawn.

If you want the small diagnostic HUD:

```bash
!python main.py ... --show-hud
```

## Gradio UI

```bash
!python app.py
```

The UI lets you:

- upload a replacement video
- enable/disable traffic-sign detection
- enable/disable car-license-plate detection
- tune traffic-sign confidence
- tune plate confidence
- tune internal vehicle confidence
- run a faster every-second-frame demo
- optionally show a small HUD
- preview the processed video
- download the CSV

## CSV output

```text
frame,time_sec,type,track_id,class,confidence,x1,y1,x2,y2,extra
```

Types include:

```text
traffic_sign
car_license_plate
```

For a plate row, `extra` also records the associated vehicle class and vehicle track ID.

## Optional Vietnamese car-plate fine-tuning

Normal inference works immediately with the pretrained plate checkpoint. Fine-tuning is optional.

The provided preparation script defaults to the public Roboflow dataset:

```text
Workspace: phms-workspace-ialpp
Project:   vietnamese-car-license-plate-dwwrm
Version:   2
```

### Prepare dataset

```python
import os
os.environ['ROBOFLOW_API_KEY'] = 'YOUR_KEY'
```

```bash
!python training/prepare_plate_dataset.py \
  --target '/content/drive/MyDrive/DIP/datasets/vn_car_plate'
```

Use the `data.yaml` path printed by the script.

### Fine-tune

```bash
!python training/train_plate.py \
  --data '/content/drive/MyDrive/DIP/datasets/vn_car_plate/data.yaml' \
  --models-dir '/content/drive/MyDrive/DIP/models' \
  --runs-dir '/content/drive/MyDrive/DIP/training_runs' \
  --epochs 20 \
  --batch 16
```

The best checkpoint is copied to:

```text
/content/drive/MyDrive/DIP/models/plate_best.pt
```

The next inference run automatically uses it.

### Evaluate

```bash
!python training/evaluate_plate.py \
  --model '/content/drive/MyDrive/DIP/models/plate_best.pt' \
  --data '/content/drive/MyDrive/DIP/datasets/vn_car_plate/data.yaml'
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
│   ├── plate_logic.py
│   ├── temporal.py
│   ├── video_processor.py
│   ├── detectors/
│   │   ├── traffic_sign.py
│   │   ├── scene.py
│   │   └── license_plate.py
│   └── utils/
│       ├── dip.py
│       ├── geometry.py
│       └── visualization.py
└── training/
    ├── prepare_plate_dataset.py
    ├── train_plate.py
    └── evaluate_plate.py
```

The old template-matching script and `sign_templates/` remain only as the classic-DIP baseline for comparison in the final report.
