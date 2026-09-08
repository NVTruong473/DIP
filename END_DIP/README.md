# Traffic Sign Detection — DIP + YOLO

This branch is simplified again for the current road video: **traffic-sign detection only**.

## Active pipeline

```text
video1.mp4
  ↓
2 overlapping vertical tiles
  ↓
Vietnamese traffic-sign YOLO
  ↓
class-aware NMS
  ↓
current-frame-only EMA smoothing
  ↓
short English label + confidence
  ↓
H.264 result + CSV
```

The output no longer includes helmet, rider, vehicle, license-plate, OCR, ROI, or Student ID overlays.

## Why this version is cleaner

- Only traffic-sign boxes are drawn.
- Labels are short English names, avoiding Vietnamese font issues and long text blocks.
- Overlapping tiled inference preserves more pixels for small/distant signs in 1080p footage.
- Duplicate detections from overlapping tiles are removed with class-aware NMS.
- Bounding boxes are EMA-smoothed only when a real detection exists in the current frame. Missing detections are **not** extrapolated, so no guessed/ghost boxes are drawn.
- Classic DIP color analysis is still stored in CSV metadata for the Digital Image Processing report.

## Model persistence

The detector is cached in Google Drive under:

```text
MyDrive/DIP/models/traffic_sign/
```

Default model: `liamxdev/vtsr` (`vtsr.torchscript`).

After the first download, a Colab restart does not require training and does not delete the model. The recovery cell in the notebook reuses the cached file.

## Google Drive layout

```text
MyDrive/DIP/
├── video1.mp4
├── models/
│   └── traffic_sign/
└── outputs/
    ├── video1_result.mp4
    └── video1_result.csv
```

## Google Colab

Open `END_DIP/colab_demo.ipynb` from branch `feature/yolo-traffic-safety`, enable **T4 GPU**, then choose **Runtime → Run all**.

The notebook automatically:

1. mounts Google Drive;
2. fresh-clones this branch;
3. installs only the required dependencies;
4. verifies T4 GPU and `MyDrive/DIP/video1.mp4`;
5. downloads/reuses the traffic-sign model in Drive;
6. removes stale output;
7. processes the video;
8. saves the full H.264 result to Drive;
9. shows a lightweight preview directly in Colab.

After a runtime disconnect, the last **RECOVERY / RERUN** cell can be run by itself. It restores the runtime, reuses the model already stored in Drive, and runs the same video again.

## CLI

```bash
python main.py \
  --input /content/drive/MyDrive/DIP/video1.mp4 \
  --output-dir /content/drive/MyDrive/DIP/outputs \
  --models-dir /content/drive/MyDrive/DIP/models \
  --sign-conf 0.25 \
  --sign-imgsz 640
```

The original HSV/contour/template-matching file and `sign_templates/` remain in the repository as the classic DIP baseline for comparison.
