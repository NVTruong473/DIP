# END_DIP — Robust Traffic Sign Detection

Final scope on branch `feature/yolo-traffic-safety`: **traffic-sign detection only** for the current `MyDrive/DIP/video1.mp4`.

The active implementation is intentionally contained in one file:

```text
END_DIP/colab_demo.ipynb
```

## Key design

- Primary: `liamxdev/vtsr` for the sign classes that already worked well on the current video.
- Secondary: `star092304/traffic-sign-detection-vietnam-yolo` YOLO11s for complementary Vietnamese traffic-sign coverage.
- Nearby signs: global inference every frame.
- Distant signs: two overlapping upper-road crops are processed separately, giving roughly 1.7–1.9× more effective sign pixels than a full-frame pass.
- DIP rescue: red/blue/yellow + contour geometry proposes at most one extra crop every three frames; YOLO still has to confirm it.
- Template bank: legacy project templates plus Vietnamese sign artwork cached in Drive. Templates only verify a YOLO-proposed class and never classify an object by themselves.
- False-positive controls: geometry gate, color/shape evidence, template support, cross-pass conflict suppression, NMS-style merging, and short temporal confirmation.
- Bounding boxes: EMA is applied only to a real current-frame detection. Missing detections are never extrapolated or rendered as ghost boxes.
- Labels: compact English only.

## Important TorchScript fix

`vtsr.torchscript` is a static export whose detection head expects the native 640×640 anchor layout. Calling the model at arbitrary sizes such as 768 or 896 can fail with an anchor tensor mismatch (`8400` vs another anchor count).

The notebook therefore manually letterboxes **every primary input to exactly 640×640**. Far-sign zoom is created by cropping a smaller scene region first, not by changing the TorchScript inference size.

Before the full 3087-frame loop, the notebook runs a **smoke test** on:

1. primary global 640×640,
2. primary left/right far crops 640×640,
3. secondary global,
4. secondary far crop.

If the primary TorchScript still fails on a future Colab runtime, the notebook automatically disables it and continues using the dynamic YOLO11s `.pt` model instead of crashing.

## Google Drive persistence

Weights and template cache are stored in:

```text
MyDrive/DIP/models/
├── traffic_sign_primary/
├── traffic_sign_secondary/
└── sign_templates/
```

No training is required. After a Colab disconnect, open the notebook again and use `Runtime → Run all`; cached files in Drive are reused.

Obsolete helmet, plate, OCR and scene-model artifacts are removed by the setup cell when present.

## Output

```text
MyDrive/DIP/outputs/video1_result.mp4
MyDrive/DIP/outputs/video1_result.csv
```

The final MP4 is encoded as H.264/yuv420p and the last cell creates a smaller inline preview for Google Colab.
