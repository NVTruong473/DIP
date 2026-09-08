# Traffic Sign Detection — Single Colab Notebook

Final scope for `END_DIP` on branch `feature/yolo-traffic-safety` is **traffic-sign detection only** for the current `video1.mp4`.

The active implementation is intentionally contained in **one file**:

```text
END_DIP/colab_demo.ipynb
```

No `main.py`, `src/`, detector modules, OCR modules, helmet modules, plate modules, or training scripts are used on this branch. This avoids stale imports and version conflicts in Google Colab.

## Why the pipeline was redesigned again

The previous result was already reasonable for nearby, repeated signs, but distant signs were often missed because a small sign could occupy only a few pixels before a full 1920x1080 frame was resized for YOLO inference. Lowering confidence alone would mostly increase false positives.

The final notebook therefore treats **far-sign recall as a scale problem**, not merely a threshold problem.

## Final pipeline

```text
video1.mp4
   |
   +-- Primary VTSR global pass
   |      -> nearby / medium-distance signs
   |
   +-- Two overlapping far-zone slices
   |      -> enlarged upper-road regions
   |      -> mild LAB CLAHE + unsharp inference copy
   |      -> primary VTSR again at larger inference size
   |
   +-- Complementary YOLO11s detector
   |      -> 82-class Vietnamese traffic-sign model
   |      -> slower schedule to add recall without excessive runtime
   |
   +-- DIP rescue proposals
   |      -> HSV red / blue / yellow
   |      -> contour shape / circularity filters
   |      -> at most two small candidate crops
   |      -> YOLO must still confirm the object
   |
   +-- Template verifier
   |      -> legacy project templates
   |      -> official Vietnamese sign artwork from Wikimedia/QCVN
   |      -> used only to strengthen weak YOLO evidence
   |      -> NEVER assigns a class by itself
   |
   +-- Cross-pass NMS / conflict resolution
   |
   +-- strict temporal confirmation
          -> strong evidence can appear immediately
          -> ordinary evidence needs repeated hits
          -> weak evidence needs repeated hits + template/color support
          -> no current detection = no box is rendered
          -> EMA smooths only real current detections
```

## Models

Primary:

```text
liamxdev/vtsr
vtsr.torchscript
```

This model is retained because it works well for several sign classes that occur repeatedly in the supplied road video, including official-code classes such as `P-102`, `P-130`, `P-131A`, and `R-302A`.

Complementary detector:

```text
star092304/traffic-sign-detection-vietnam-yolo
best.pt
```

It is a YOLO11s model trained on 10,157 Vietnamese traffic-scene images with 82 traffic-sign classes. It is used as a second source of evidence rather than blindly replacing the first model.

## Template bank

The notebook caches templates permanently under:

```text
MyDrive/DIP/models/sign_templates/
```

It combines the original project templates with official Vietnamese road-sign artwork resolved from Wikimedia Commons, including examples such as:

- No Entry / P102
- No Stopping or Parking / P130
- No Parking / P131a
- No Left Turn / P123a
- No Right Turn / P123b
- No U-Turn / P124a1
- Keep Right / R302a
- Children / W225
- Road Works / W227
- No Overtaking / P125

The template subsystem is deliberately a **verifier**, not a standalone classifier. A colored advertisement or circular object cannot become a traffic sign solely because it resembles a template.

## False-positive controls

The final notebook uses all of the following safeguards:

- low confidence is not accepted by itself;
- minimum object size and plausible aspect-ratio checks;
- red / blue / yellow color evidence is only supporting evidence;
- DIP contours create proposals but cannot assign a label;
- secondary traffic-light classes (`Green Light`, `Red Light`) are excluded because the project scope is road signs, not signal lamps;
- overlapping slice detections are deduplicated;
- conflicting model predictions are resolved conservatively;
- low-confidence detections require temporal repetition;
- very weak detections additionally require template or color evidence;
- missing detections are not extrapolated;
- bounding boxes are smoothed only when a real box is present in the current frame.

## Drive persistence and cleanup

Input:

```text
/content/drive/MyDrive/DIP/video1.mp4
```

Persistent models:

```text
MyDrive/DIP/models/
├── traffic_sign_primary/
├── traffic_sign_secondary/
└── sign_templates/
```

The notebook removes obsolete artifacts from previous helmet / license-plate / OCR versions when it starts, including their old model directories and old plate/violation output folders.

Output:

```text
MyDrive/DIP/outputs/video1_result.mp4
MyDrive/DIP/outputs/video1_result.csv
```

The result is encoded to H.264/yuv420p so it can be opened from Google Drive and previewed directly in Colab.

## Run

Open `END_DIP/colab_demo.ipynb`, enable a **T4 GPU**, then select:

```text
Runtime -> Run all
```

The model files and template bank remain on Google Drive. If the Colab runtime disconnects, reopening the notebook does **not** require training again; the cached model files are reused and only video inference runs again.

## Files intentionally kept

`sign_templates/` is retained because it is now an input to the notebook's conservative template-verification stage.

`521H0461_521H0324.pdf` is kept only as the original academic report/reference and is not executed by the pipeline.
