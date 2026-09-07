# Traffic Sign Recognition + Car License Plate Detection — DIP + YOLO

> **Google Colab final run:** open `END_DIP/colab_demo.ipynb` on branch `feature/yolo-traffic-safety` and run every cell from top to bottom. The notebook now clears stale outputs, runs the final traffic-sign + car-plate pipeline, validates the Drive result, and shows a lightweight inline preview.

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
    ├── datasets/
    ├── training_runs/
    └── outputs/
        ├── video1_result.mp4
        ├── video1_result.csv
        └── video1_plates/
```

## Recommended final workflow

Use `END_DIP/colab_demo.ipynb` rather than manually copying commands. The notebook performs these steps:

1. Mount Google Drive.
2. Fresh-clone `feature/yolo-traffic-safety`.
3. Install dependencies.
4. Verify T4 GPU and `MyDrive/DIP/video1.mp4`.
5. Download/cache the three required models.
6. Delete stale previous outputs and run `main.py`.
7. Validate that the result and CSV were written to Drive.
8. Create an H.264 preview under `/content` and show it directly below the Colab cell.

Persistent result:

```text
/content/drive/MyDrive/DIP/outputs/video1_result.mp4
```

The final MP4 is re-encoded as H.264 / yuv420p / avc1 with fast-start metadata so it is suitable for Google Drive and browser playback.

## CLI equivalent

```bash
python main.py \
  --input '/content/drive/MyDrive/DIP/video1.mp4' \
  --output-dir '/content/drive/MyDrive/DIP/outputs' \
  --models-dir '/content/drive/MyDrive/DIP/models' \
  --sign-conf 0.25 \
  --plate-conf 0.30 \
  --vehicle-conf 0.30
```

## Gradio UI

`app.py` remains available for testing replacement videos. It is optional for the default final run.

```bash
python app.py
```

## Output CSV

Columns:

```text
frame,time_sec,type,track_id,class,confidence,x1,y1,x2,y2,extra
```

## Optional Vietnamese plate fine-tuning

Normal inference works with the default pretrained detector. Optional scripts remain under `training/` to download a Vietnamese car-plate dataset, fine-tune the detector, and evaluate it. A successful fine-tune is copied to:

```text
/content/drive/MyDrive/DIP/models/plate_best.pt
```

The main pipeline automatically prefers that file on later runs.

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

The original template-matching code and `sign_templates/` remain only as the classic-DIP baseline for comparison in the report/demo.
