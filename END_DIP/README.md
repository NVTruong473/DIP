# Robust Vietnamese Traffic Sign Detection in Road Videos

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVTruong473/DIP/blob/feature/yolo-traffic-safety/END_DIP/colab_demo.ipynb)

A **Computer Vision + Digital Image Processing** project for detecting and recognizing Vietnamese traffic signs from dashcam/road video. The current implementation is optimized for Google Colab and is intentionally kept inside a **single notebook** so that the complete pipeline can be inspected, executed, modified, and demonstrated without managing multiple Python modules.

> Current scope: **traffic-sign detection and recognition only**.
>
> Default input: `MyDrive/DIP/video1.mp4`

---

## 1. Project overview

Traffic-sign detection in real road videos is more difficult than detecting large, clean objects in still images. Signs can be:

- very small when far from the camera,
- partially occluded,
- blurred by camera/vehicle motion,
- affected by strong sunlight, shadows, haze, or low contrast,
- visually similar to advertisements, lamps, logos, road markings, and other colored objects,
- detected multiple times by overlapping crops,
- classified inconsistently from one frame to another.

This project therefore does **not** rely on a single YOLO prediction per frame. Instead, it combines deep-learning detection with classical image-processing and temporal rules designed specifically for video.

The main goals are:

1. detect nearby and medium-distance signs reliably,
2. improve recall for distant/small signs without simply lowering confidence thresholds,
3. reduce duplicate and conflicting bounding boxes,
4. reduce label flickering between consecutive frames,
5. avoid drawing stale or predicted boxes when there is no current visual evidence,
6. prefer an uncertain generic `Sign` label over confidently displaying the wrong class.

---

## 2. Computer Vision or Digital Image Processing?

This project is primarily a **Computer Vision object-detection/recognition system**, while **Digital Image Processing (DIP)** techniques are used to improve and validate the vision pipeline.

### Computer Vision components

- YOLO11 object detection
- multi-scale / sliced inference
- object-level box fusion
- crop re-inference
- temporal tracking and class voting
- traffic-sign recognition

### Digital Image Processing components

- image cropping and rescaling
- contrast enhancement
- HSV color analysis
- color-ratio validation
- edge/template comparison
- geometric filtering
- frame-by-frame video processing
- visualization and video encoding

The final system can therefore be described as:

> **Traffic-sign detection in road videos using YOLO-based Computer Vision enhanced by Digital Image Processing and temporal consistency.**

---

## 3. Current v3 architecture

The latest result review showed that the largest problem was not only missed distant signs. A more serious failure occurred when multiple models/passes assigned **different semantic classes to the same physical sign**, producing stacked boxes and unstable labels.

v3 changes the design so that **one model is the semantic authority**.

```text
                         Input video frame
                                |
                    +-----------+-----------+
                    |                       |
               Global YOLO11s          Far-sign slices
                    |                 (3 overlapping crops)
                    |                       |
                    +-----------+-----------+
                                |
                    Class-agnostic filtering
                                |
                  Geometry-first object clustering
                                |
                 Confidence-weighted box fusion
                                |
                    Is the class ambiguous?
                         /              \
                       no                yes
                       |                  |
                       |          padded crop re-inference
                       |                  |
                       +--------+---------+
                                |
                    DIP plausibility checks
                    (HSV / color / template)
                                |
                    Class-agnostic temporal track
                                |
                       accumulated class votes
                                |
                         label hysteresis
                                |
                     stable current-frame box
                                |
                         Annotated video
```

### Key design rules

- **One semantic model only** to avoid class conflicts between different label ontologies.
- **Global inference every frame** for nearby and medium-distance signs.
- **Three overlapping upper-road slices** for small and distant signs.
- Predictions are grouped by **physical geometry before class selection**.
- Multiple observations of the same sign are combined using **confidence-weighted box fusion**.
- Ambiguous detections are cropped with padding and sent through the **same model again at a larger relative scale**.
- DIP evidence can support or penalize a class, but **DIP never invents a class by itself**.
- Temporal tracking is **class-agnostic**, so one sign remains the same object even if frame-level class predictions fluctuate.
- Stable labels use **hysteresis**; a one-frame challenger cannot immediately replace an established class.
- If a distant sign is visually real but the class is not reliable yet, the system can display **`Sign`** instead of guessing.
- No current detection = **no rendered box**. The pipeline does not extrapolate ghost boxes.

---

## 4. Model and dataset

### Semantic detector

The project currently uses:

**YOLO11s Vietnamese Traffic Sign Detection**  
Model: `star092304/traffic-sign-detection-vietnam-yolo`

Model page:

https://huggingface.co/star092304/traffic-sign-detection-vietnam-yolo

Dataset page:

https://huggingface.co/datasets/star092304/Traffic-sign-detection-VietNam

The published model card describes a Vietnamese traffic-sign dataset with:

- **10,157 images**
- **82 traffic-sign classes**
- reported Precision: **0.9642**
- reported Recall: **0.9615**
- reported mAP@0.5: **0.9806**

These values are the model author's reported evaluation results and should not be interpreted as the measured accuracy of this project on every road video.

### Label normalization

Some raw dataset labels are shortened for cleaner video visualization. For example, the dataset includes both concepts corresponding to Vietnamese descriptions `Đi về bên phải` and `Rẽ phải`; the former is displayed as **`Keep Right`** to reduce ambiguity in the rendered result.

All rendered labels use short English text to avoid Unicode/font problems and to reduce visual clutter.

---

## 5. Why sliced inference is used

A 1920x1080 road frame contains many pixels, but a far-away traffic sign may occupy only a tiny region. Resizing the complete frame to a normal detector input size can remove important visual details.

Instead of increasing the full-frame input indefinitely, the notebook uses a **SAHI-style sliced-inference idea**:

1. run one global detection pass,
2. crop several overlapping regions from the part of the image where distant road signs are likely to appear,
3. run the same detector on those crops,
4. transform their coordinates back to the original frame,
5. fuse duplicate detections.

Reference concept:

- SAHI: *Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection*  
  https://arxiv.org/abs/2202.06934

This improves the number of effective pixels available to the detector for distant objects while keeping the semantic model unchanged.

---

## 6. Bounding-box stabilization

Simply retaining a box from previous frames can make a video look smooth, but it can also create false detections after an object disappears. This project deliberately avoids that behavior.

The stabilization rules are:

- detections are associated using spatial overlap and center-distance criteria,
- bounding boxes from the current frame can be smoothed with recent real detections,
- class evidence is accumulated over several frames,
- established labels use hysteresis before switching to a competing class,
- **a missing current-frame detection is not rendered**.

This gives a more conservative result: slightly less visually continuous than aggressive tracking, but much less likely to show a bounding box where the detector has no evidence.

---

## 7. DIP validation layer

Classical image processing is used as a **soft validation layer**, not as an independent classifier.

Examples:

- prohibition signs often contain significant red regions,
- mandatory direction signs are often dominated by blue,
- no-parking/no-stopping signs usually combine red and blue,
- warning signs often contain characteristic red/yellow structures.

The notebook computes HSV-based appearance evidence and can use traffic-sign templates as a small tie-breaker when several YOLO classes compete for the same object.

This is intentionally conservative:

```text
YOLO proposes candidate classes
          |
          v
DIP checks whether visual appearance is plausible
          |
          v
DIP re-ranks / validates
          |
          X
DIP does NOT independently assign a traffic-sign class
```

This prevents red advertisements, blue logos, lamps, or other colorful road objects from being classified as traffic signs merely because their color resembles one.

---

## 8. Repository structure

The active project is intentionally minimal:

```text
END_DIP/
├── .gitignore
├── 521H0461_521H0324.pdf     # previous academic/report reference
├── README.md                 # this documentation
└── colab_demo.ipynb          # complete executable implementation
```

All active runtime logic is inside `colab_demo.ipynb`.

The feature branch removes old split runtime modules and obsolete helmet/license-plate/OCR implementations so that the notebook cannot accidentally load stale code.

---

## 9. Requirements

Recommended environment:

- Google Colab
- NVIDIA T4 GPU or better
- Google Drive
- Python provided by Colab

Main libraries installed automatically by the notebook:

- `ultralytics`
- `torch`
- `opencv-python-headless`
- `numpy`
- `huggingface_hub`
- `requests`
- `tqdm`

FFmpeg is used for final H.264 video encoding and is already available in standard Google Colab environments.

---

## 10. Google Drive layout

Before running the notebook, place the input video here:

```text
MyDrive/
└── DIP/
    └── video1.mp4
```

The notebook automatically creates and reuses:

```text
MyDrive/DIP/
├── video1.mp4
│
├── models/
│   ├── traffic_sign_yolo11s/
│   └── sign_templates_v3/
│
└── outputs/
    ├── video1_result.mp4
    ├── video1_result.csv
    └── video1_audit.jpg
```

Model files are stored on Google Drive, so if the Colab runtime disconnects, the model does **not** need to be trained or downloaded from scratch again.

---

## 11. Run on Google Colab

### Option A — Open directly

Click the button at the top of this README or open:

https://colab.research.google.com/github/NVTruong473/DIP/blob/feature/yolo-traffic-safety/END_DIP/colab_demo.ipynb

Then select:

```text
Runtime
→ Change runtime type
→ T4 GPU
→ Save
```

Finally:

```text
Runtime
→ Run all
```

### Option B — Open from GitHub

1. Open the repository.
2. Switch to branch `feature/yolo-traffic-safety`.
3. Open `END_DIP/colab_demo.ipynb`.
4. Click **Open in Colab**.
5. Enable a GPU runtime.
6. Run all cells from top to bottom.

---

## 12. What the notebook does automatically

A normal `Run all` performs the complete workflow:

```text
Mount Google Drive
        ↓
Check GPU + input video
        ↓
Clean obsolete runtime artifacts
        ↓
Download/reuse YOLO11s weights
        ↓
Build/reuse template cache
        ↓
Run smoke test
        ↓
Read video frame-by-frame
        ↓
Global + sliced inference
        ↓
Box filtering + fusion
        ↓
Ambiguous crop re-inference
        ↓
DIP validation
        ↓
Temporal stabilization
        ↓
Write annotated video + CSV
        ↓
Encode H.264/yuv420p
        ↓
Create visual audit montage
        ↓
Show a lightweight preview in Colab
```

No separate `.py` execution is required.

---

## 13. Output files

### Annotated video

```text
MyDrive/DIP/outputs/video1_result.mp4
```

The final video is encoded as H.264/yuv420p for good compatibility with Google Drive, Chrome, and Colab.

### Detection log

```text
MyDrive/DIP/outputs/video1_result.csv
```

The CSV stores machine-readable information such as frame/time, class, confidence, and bounding-box coordinates. Confidence values are kept in the CSV instead of filling the video with long text.

### Audit image

```text
MyDrive/DIP/outputs/video1_audit.jpg
```

The audit image contains representative:

```text
ORIGINAL | RESULT
```

pairs sampled across the video. It is useful for quickly reviewing:

- false positives,
- missed signs,
- incorrect labels,
- box stability,
- distant-sign performance,
- difficult lighting conditions.

This makes future tuning evidence-driven rather than based on random threshold changes.

---

## 14. How to evaluate the result

For a serious evaluation, do not judge the project only by whether boxes "look good".

Recommended criteria include:

### Detection quality

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95

### Small/far-sign performance

Evaluate signs separately by approximate bounding-box size, for example:

- small
- medium
- large

This reveals whether sliced inference is actually improving the distant-sign problem.

### Video stability

Useful project-specific measurements include:

- class-switch count per tracked sign,
- percentage of detected frames with the stable final class,
- average bounding-box center displacement after smoothing,
- duplicate-box rate,
- false-positive persistence length.

### Qualitative audit

Inspect difficult segments involving:

- very distant signs,
- strong sunlight,
- dark shadows,
- motion blur,
- partial occlusion,
- signs near advertisements or other colorful objects.

---

## 15. Important limitations

This repository should not be treated as a production autonomous-driving perception system.

Current limitations include:

- far signs with too few source pixels may be impossible to classify reliably,
- motion blur can destroy fine symbols inside a sign,
- the current far-sign slicing geometry is tuned for road/dashcam-style scenes,
- unusual sign designs or classes absent from the training data can still fail,
- template evidence covers only a subset of sign types,
- strong domain shift between cameras, cities, weather, or nighttime conditions may reduce accuracy,
- public model-card metrics are not the same as independently measured performance on this video.

The system intentionally prefers **uncertainty over hallucination**: if a physical sign is consistently visible but its category is not sufficiently supported, displaying `Sign` is considered better than forcing an incorrect specific label.

---

## 16. Potential future improvements

Strong next steps include:

- manually label a representative subset of the current dashcam video,
- fine-tune YOLO11s on difficult Vietnamese far-sign examples,
- use hard-negative mining for advertisements, traffic lights, logos, and other confusing objects,
- train with stronger blur, glare, shadow, rain, and nighttime augmentation,
- compare standard inference against formal SAHI integration,
- evaluate Soft-NMS / Weighted Boxes Fusion variants quantitatively,
- add a dedicated traffic-sign classifier after the detector for very ambiguous classes,
- calibrate class probabilities,
- evaluate with per-distance/per-size metrics,
- create a small benchmark set from the most difficult frames in `video1.mp4`.

The most valuable improvement would be **better domain-specific labeled data**, not simply adding more heuristics.

---

## 17. References and inspiration

### Small-object / sliced inference

- Akyon et al., **SAHI: Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection**  
  https://arxiv.org/abs/2202.06934

### Box fusion

- Solovyev et al., **Weighted Boxes Fusion: Ensembling Boxes from Different Object Detection Models**  
  https://arxiv.org/abs/1910.13302

### Frameworks

- Ultralytics YOLO  
  https://github.com/ultralytics/ultralytics

- OpenCV  
  https://opencv.org/

- PyTorch  
  https://pytorch.org/

### Vietnamese traffic-sign model/data

- Model  
  https://huggingface.co/star092304/traffic-sign-detection-vietnam-yolo

- Dataset  
  https://huggingface.co/datasets/star092304/Traffic-sign-detection-VietNam

---

## 18. Reproducibility notes

For repeatable runs:

1. use the same input video,
2. use the same notebook revision/commit,
3. keep the same model weights cached in Drive,
4. use a GPU runtime,
5. record the generated CSV and audit image together with the result video.

The notebook is designed so that model weights persist in Google Drive. A Colab runtime reset therefore destroys only temporary runtime state, not the downloaded model artifacts or final outputs.

---

## 19. Quick start

```text
1. Upload video1.mp4 to MyDrive/DIP/
2. Open END_DIP/colab_demo.ipynb in Google Colab
3. Select T4 GPU
4. Runtime → Run all
5. Wait for inference and H.264 encoding
6. Open MyDrive/DIP/outputs/video1_result.mp4
7. Review video1_audit.jpg for representative failures
```

**Open the notebook:**  
https://colab.research.google.com/github/NVTruong473/DIP/blob/feature/yolo-traffic-safety/END_DIP/colab_demo.ipynb

---

## Project status

This branch is an experimental/academic computer-vision implementation focused on improving robust traffic-sign recognition in a real road video. The current design favors **conservative, evidence-based detections and stable visualization** over maximizing the raw number of bounding boxes.
