# Traffic Sign Detection — DIP + YOLO

Final redesign of `END_DIP` for Google Colab.

## Final scope

The project now performs **traffic-sign detection only**.

- YOLO is the primary detector for Vietnamese traffic signs.
- Labels shown on the video are short **English** names.
- Classic Digital Image Processing (DIP) color analysis is retained as secondary metadata for the report/CSV.
- Overlapping tiled inference improves small/distant sign detection in 1080p road footage.
- A short temporal hold reduces one-frame box flicker.
- No helmet detection.
- No rider/vehicle boxes.
- No license-plate detection.
- No ROI overlay.
- No Student ID overlay.

The original HSV/contour/template-matching script and `sign_templates/` are kept in the repository as the classic-DIP baseline for comparison.

## Pipeline

```text
video
  ↓
optional DIP enhancement
  ↓
2 overlapping image tiles
  ↓
Vietnamese Traffic Sign YOLO
  ↓
class-aware NMS
  ↓
temporal hold
  ↓
short English label + confidence
  ↓
H.264 MP4 + CSV
```

## Model

Traffic-sign detector: `liamxdev/vtsr`.

The final Colab pipeline downloads only:

```text
vtsr.torchscript
```

Large model/output files are stored in Google Drive rather than committed to GitHub.

## Google Drive layout

```text
MyDrive/
└── DIP/
    ├── video1.mp4
    ├── models/
    └── outputs/
        ├── video1_result.mp4
        └── video1_result.csv
```

## Recommended Google Colab workflow

Open `END_DIP/colab_demo.ipynb` from branch `feature/yolo-traffic-safety`, choose a **T4 GPU**, then use **Runtime → Run all**.

The notebook performs the complete workflow automatically:

```text
Mount Drive
→ fresh clone
→ install dependencies
→ verify GPU + video
→ download/cache traffic-sign model
→ delete stale result
→ process video1.mp4
→ save full result to Drive
→ verify codec
→ show lightweight preview directly in Colab
```

Default input:

```text
/content/drive/MyDrive/DIP/video1.mp4
```

Full result:

```text
/content/drive/MyDrive/DIP/outputs/video1_result.mp4
```

Detection CSV:

```text
/content/drive/MyDrive/DIP/outputs/video1_result.csv
```

To test another video already in `MyDrive/DIP/`, change only this notebook line:

```python
VIDEO_NAME = 'video1.mp4'
```

## CLI

```bash
python main.py \
  --input /content/drive/MyDrive/DIP/video1.mp4 \
  --output-dir /content/drive/MyDrive/DIP/outputs \
  --models-dir /content/drive/MyDrive/DIP/models \
  --sign-conf 0.25
```

Useful options:

```text
--sign-conf      YOLO confidence threshold
--sign-imgsz     inference size, default 640
--frame-stride   infer every Nth frame, default 1
--dip-enhance    enable mild CLAHE/unsharp preprocessing
--show-hud       display a small time/sign-count HUD
```

## Gradio

Optional upload UI:

```bash
python app.py
```

The normal `colab_demo.ipynb` flow does not launch Gradio, so **Run all** completes without waiting for an interactive server.

## Source tree

```text
END_DIP/
├── main.py
├── app.py
├── download_models.py
├── requirements.txt
├── colab_demo.ipynb
├── src/
│   ├── common.py
│   ├── config.py
│   ├── model_manager.py
│   ├── temporal.py
│   ├── video_processor.py
│   ├── detectors/
│   │   └── traffic_sign.py
│   └── utils/
│       ├── dip.py
│       ├── geometry.py
│       └── visualization.py
└── sign_templates/        # original DIP baseline assets
```
