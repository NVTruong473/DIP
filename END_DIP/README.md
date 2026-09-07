# Traffic Intelligence System — DIP + YOLO + OCR

Google Colab project integrating three tasks on the same road video:

1. **Vietnamese traffic-sign detection** — YOLO11s, 82 English-labeled classes.
2. **Motorcycle helmet compliance** — tracked person/motorcycle association + explicit `HELMET / NO HELMET` detector.
3. **Car license-plate detection + OCR** — car/bus/truck validation + YOLO plate detector + conservative EasyOCR recognition.

The complete frame is processed, so both traffic lanes are covered. Parent person/motorcycle/car boxes are intentionally hidden to keep the video readable.

## Design rules that reduce false positives

- A person is considered a rider only when spatially associated with a tracked motorcycle.
- Multiple riders can be associated with one motorcycle, but one person cannot belong to two motorcycles.
- Missing helmet evidence is `UNKNOWN`; it is never converted to `NO HELMET`.
- Helmet status requires repeated temporal evidence before it becomes stable.
- License plates are kept only when they are geometrically plausible inside the lower region of a tracked `car / bus / truck`.
- Plate aspect ratio, relative size, location, sharpness, brightness and OCR confidence are checked.
- OCR text must match Vietnamese car-plate grammar and appear repeatedly on the same vehicle track before being shown.
- Bounding boxes are EMA-smoothed only when a real current-frame detection exists. Missing detections are not extrapolated or guessed.
- Traffic-sign tile overlap + class-aware NMS improves distant-sign detection while removing duplicate tile boxes.

## Lighting robustness

The original video is never altered for output. A photometric inference-only copy is created per frame:

- `DAY`: mild CLAHE.
- `NIGHT`: gamma lift + stronger CLAHE.
- `GLARE`: highlight compression + CLAHE.

Because these operations do not resize or warp the image, predicted coordinates still map directly to the original frame.

## Models / data

### Traffic signs

Default model: `star092304/traffic-sign-detection-vietnam-yolo`

- YOLO11s
- 82 Vietnamese traffic-sign classes
- English class names
- trained on 10,157 images
- reported mAP50 about 0.9806

### Helmet compliance

Default inference model: `nnsohamnn/helmet-detection-yolo11`.

Optional one-time fine-tuning script:

```text
training/train_helmet_large.py
```

Selected dataset: `thundarstrom/traffic-helmet-violation`:

- 42,559 traffic images
- about 126,000 bounding boxes
- `helmet` / `no_helmet`
- train/val/test = 80/10/10
- traffic/two-wheeler domain rather than construction PPE

Fine-tuned output is copied to:

```text
MyDrive/DIP/models/helmet_best.pt
```

and is automatically preferred on later runs.

### Vietnamese car license plates

Default detector: `Koushim/yolov8-license-plate-detection`.

Optional VN-specific fine-tuning script:

```text
training/train_plate_vn.py
```

Dataset: `phms-workspace-ialpp/vietnamese-car-license-plate-dwwrm`, 8,255 images. A successful fine-tune becomes:

```text
MyDrive/DIP/models/plate_best.pt
```

OCR uses EasyOCR with an alphanumeric allowlist, multi-preprocessing variants, quality gates, Vietnamese car-plate regex validation and vehicle-track voting.

## Persistent Google Drive layout

```text
MyDrive/DIP/
├── video1.mp4
├── models/
│   ├── best.pt / downloaded public checkpoints
│   ├── helmet_best.pt       # optional custom fine-tune
│   ├── plate_best.pt        # optional VN plate fine-tune
│   └── easyocr/
├── datasets/
├── training_runs/
└── outputs/
    ├── video1_result.mp4
    ├── video1_result.csv
    └── video1_plates/
```

Detector weights, optional fine-tuned models and OCR weights live on Drive, so Colab runtime loss does not delete them.

## Google Colab

Open `END_DIP/colab_demo.ipynb` from branch `feature/yolo-traffic-safety`, enable a T4 GPU and use **Runtime → Run all**.

The notebook:

```text
mount Drive
→ clone repo
→ install
→ cache all model weights in Drive
→ optional one-time helmet fine-tune
→ run all 3 tasks
→ save H.264 result + CSV + best plate crops
→ show preview directly in Colab
```

The notebook ends with a **RECOVERY CELL**. After a Colab disconnect, run only that last cell. It remounts Drive, restores the repo/runtime, reuses the models already saved in `MyDrive/DIP/models`, processes `video1.mp4` again and shows the preview. No retraining is required.

## Default CLI

```bash
python main.py \
  --input /content/drive/MyDrive/DIP/video1.mp4 \
  --output-dir /content/drive/MyDrive/DIP/outputs \
  --models-dir /content/drive/MyDrive/DIP/models \
  --sign-conf 0.32 \
  --scene-conf 0.34 \
  --helmet-conf 0.38 \
  --plate-conf 0.34
```

## Visual policy

The saved video shows only:

- orange: traffic sign box + short English name;
- green/red: explicit helmet/no-helmet evidence near the rider's head;
- cyan: car license plate box + stable OCR string when validated.

No full person, motorcycle, car/bus/truck, ROI, or Student ID boxes are rendered.
