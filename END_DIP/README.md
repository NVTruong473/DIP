# END_DIP — Robust Traffic Sign Detection v3

Final scope on branch `feature/yolo-traffic-safety`: **traffic-sign detection only** for `MyDrive/DIP/video1.mp4`.

The active implementation is intentionally contained in one file:

```text
END_DIP/colab_demo.ipynb
```

## Why v3 changed the architecture

Reviewing the latest rendered video showed that the dominant failure was no longer just missed far-away signs. The more serious issue was **semantic conflict**: the same physical sign could receive different labels from different passes/models, producing stacked labels and visibly wrong classes.

v3 therefore removes the previous two-model semantic ensemble.

## v3 pipeline

- **One semantic model:** `star092304/traffic-sign-detection-vietnam-yolo` (YOLO11s, 82 Vietnam traffic-sign classes).
- **Global pass every frame** for nearby/medium signs.
- **Three overlapping upper-road slices every frame** for far signs, following the sliced-inference idea used by SAHI: crop first, then infer at high resolution.
- **Class-agnostic geometry clustering first.** Predictions are grouped as the same physical object before deciding the class.
- **Weighted-box-fusion style localization.** Overlapping global/slice boxes are confidence-weighted into one current-frame box.
- **Two-stage refinement.** If class evidence conflicts, the fused candidate is cropped with padding and re-run through the same YOLO11s model at a larger relative scale.
- **DIP is validation only.** HSV red/blue/yellow/white ratios are soft class plausibility priors; they never create a class.
- **Official Vietnamese sign templates** are downloaded/cached in Drive and used only as a small tie-breaker for ambiguous YOLO classes.
- **Class-agnostic temporal tracks.** Geometry tracks are matched independent of the current predicted label, then class votes are accumulated over real detections.
- **Label hysteresis.** Once a class is stable, it does not switch unless a competing class wins strongly for multiple frames.
- **Generic far-sign mode.** If object presence is stable but class evidence is weak, output `Sign` rather than a confident wrong class.
- **No ghost boxes.** Missing current-frame detections are never extrapolated or drawn.
- **Compact visualization.** Thin boxes, small English labels, no large orange label blocks; confidence remains in CSV.

## Dataset/model note

The YOLO11s model is trained on the Vietnam Traffic Sign Detection dataset: 10,157 images and 82 classes. The model card reports precision 0.9642, recall 0.9615 and mAP50 0.9806.

The dataset label `Turn Right Only` is displayed as `Keep Right` because its Vietnamese description is `Đi về bên phải`, while the dataset has a separate `Turn Right` class described as `Rẽ phải`.

## Drive persistence / cleanup

On Run all, the notebook keeps only the v3 runtime artifacts:

```text
MyDrive/DIP/models/
├── traffic_sign_yolo11s/
└── sign_templates_v3/
```

It removes obsolete traffic-sign primary/secondary folders and previous helmet/plate/OCR/scene artifacts. No training is required. After a Colab disconnect, reopen the notebook and use `Runtime → Run all`; cached YOLO11s weights and templates are reused.

## Outputs

```text
MyDrive/DIP/outputs/video1_result.mp4
MyDrive/DIP/outputs/video1_result.csv
MyDrive/DIP/outputs/video1_audit.jpg
```

`video1_audit.jpg` contains representative `ORIGINAL | RESULT` pairs so later tuning can be based on evidence instead of blind threshold changes.
