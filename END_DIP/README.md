# Robust Vietnamese Traffic Sign Detection in Road Videos

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVTruong473/DIP/blob/feature/yolo-traffic-safety/END_DIP/colab_demo.ipynb)

A **Computer Vision + Digital Image Processing** project for detecting and recognizing Vietnamese traffic signs from dashcam/road video. The complete active implementation is intentionally kept inside a **single Google Colab notebook**.

> Current scope: **traffic-sign detection and recognition only**.
>
> Default input expected by the notebook: `MyDrive/DIP/video1.mp4`

---

## 1. Project overview

Traffic-sign detection in real road videos is difficult because signs may be very small, blurred, partially occluded, affected by sunlight/shadows, or visually similar to advertisements, lamps, logos and other colorful road objects.

The pipeline therefore combines:

- YOLO11s traffic-sign detection,
- global inference,
- sliced inference for small/far signs,
- geometry-first duplicate fusion,
- crop re-inference for ambiguous signs,
- HSV/color/template plausibility checks,
- class-agnostic temporal tracking,
- class voting and label hysteresis,
- conservative visualization.

The system prefers an uncertain generic `Sign` label over confidently displaying a wrong class.

---

## 2. Computer Vision or Digital Image Processing?

This project is primarily a **Computer Vision object-detection/recognition system**, while **Digital Image Processing (DIP)** is used to improve and validate the pipeline.

### Computer Vision

- YOLO11 object detection
- sliced / multi-scale inference
- object-level box fusion
- crop re-inference
- temporal tracking and class voting
- traffic-sign recognition

### Digital Image Processing

- image cropping and rescaling
- contrast enhancement
- HSV color analysis
- color-ratio validation
- edge/template comparison
- geometric filtering
- frame-by-frame video processing
- visualization and video encoding

A concise description is:

> **Traffic-sign detection in road videos using YOLO-based Computer Vision enhanced by Digital Image Processing and temporal consistency.**

---

## 3. Architecture

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

Important design rules:

- one semantic model to avoid conflicts between different class ontologies,
- global inference every frame for nearby/medium signs,
- three overlapping upper-road slices for small/far signs,
- physical geometry is resolved before the final class is selected,
- ambiguous candidates are re-checked with the same model on a larger crop,
- DIP can support or penalize a YOLO class but does not invent classes,
- no current detection means no rendered ghost box,
- uncertain but stable distant detections may be displayed simply as `Sign`.

---

## 4. Model and dataset

The project currently uses:

**YOLO11s Vietnamese Traffic Sign Detection**  
Model: `star092304/traffic-sign-detection-vietnam-yolo`

- Model: https://huggingface.co/star092304/traffic-sign-detection-vietnam-yolo
- Dataset: https://huggingface.co/datasets/star092304/Traffic-sign-detection-VietNam

The model card reports a Vietnamese traffic-sign dataset with:

- 10,157 images
- 82 classes
- Precision: 0.9642
- Recall: 0.9615
- mAP@0.5: 0.9806

These are the model author's reported metrics, not guaranteed accuracy on every user video.

Rendered labels use short English names to reduce font problems and visual clutter.

---

## 5. Why sliced inference is used

A far traffic sign can occupy only a few pixels in a 1920x1080 frame. If the entire frame is resized to the detector input size, important details may disappear.

The notebook therefore uses a **SAHI-style sliced inference idea**:

1. run one global pass,
2. crop overlapping upper-road regions,
3. run YOLO on each crop,
4. map detections back to full-frame coordinates,
5. fuse duplicate detections.

Reference:

- SAHI — *Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection*  
  https://arxiv.org/abs/2202.06934

---

## 6. How bounding boxes are stabilized

The project does not simply keep old boxes after an object disappears.

Rules include:

- spatial overlap / center-distance matching,
- current-frame box smoothing,
- accumulated class evidence,
- label hysteresis,
- no current-frame detection = no rendered box.

This is intentionally conservative to reduce false boxes and label flickering.

---

## 7. DIP validation layer

DIP is used as a soft plausibility layer.

Examples:

- prohibition signs often contain red,
- mandatory direction signs are often blue,
- no-parking/no-stopping signs often combine red and blue,
- warning signs often contain red/yellow structures.

```text
YOLO proposes candidate classes
          |
          v
DIP checks visual plausibility
          |
          v
DIP re-ranks / validates
          |
          X
DIP does NOT independently invent a class
```

This helps reduce false positives from advertisements, lamps, logos and other colorful objects.

---

## 8. Repository structure

```text
END_DIP/
├── .gitignore
├── 521H0461_521H0324.pdf
├── README.md
└── colab_demo.ipynb
```

All active runtime logic is inside `colab_demo.ipynb`.

---

## 9. Requirements

Recommended environment:

- Google Colab
- NVIDIA T4 GPU or better
- Google Drive

Main libraries are installed automatically by the notebook:

- `ultralytics`
- `torch`
- `opencv-python-headless`
- `numpy`
- `huggingface_hub`
- `requests`
- `tqdm`

FFmpeg is used to encode the final H.264 video.

---

# 10. TEST WITH YOUR OWN VIDEO

## Important: the test video is NOT stored in this GitHub repository

Video files can be large, so this repository does **not** require users to commit their test video to GitHub.

The recommended workflow is:

```text
Your computer
     |
     | upload video
     v
Google Drive
My Drive/DIP/video1.mp4
     |
     | Colab reads this file
     v
colab_demo.ipynb
     |
     v
My Drive/DIP/outputs/video1_result.mp4
```

### Method A — easiest and recommended

Take any road/dashcam video from your computer and rename it to:

```text
video1.mp4
```

Then upload it to Google Drive at exactly:

```text
My Drive/
└── DIP/
    └── video1.mp4     <-- PUT YOUR TEST VIDEO HERE
```

In Google Colab this becomes:

```text
/content/drive/MyDrive/DIP/video1.mp4
```

With this method **no code change is required**. Open the notebook, enable T4 GPU, and use `Runtime -> Run all`.

### Method B — keep your original filename

For example, if your video is called:

```text
hanoi_dashcam.mp4
```

upload it here:

```text
My Drive/DIP/hanoi_dashcam.mp4
```

Then open the **first code cell** in `colab_demo.ipynb` and change:

```python
R=Path('/content/drive/MyDrive/DIP'); V=R/'video1.mp4'; M=R/'models'; O=R/'outputs'
```

to:

```python
R=Path('/content/drive/MyDrive/DIP'); V=R/'hanoi_dashcam.mp4'; M=R/'models'; O=R/'outputs'
```

That is the only input path that must be changed.

### Recommended video format

For the smoothest Colab/Drive workflow:

- `.mp4` is recommended,
- H.264/AVC video is preferred,
- 720p or 1080p dashcam/road video is a good starting point,
- landscape road scenes are best suited to the current far-sign slicing geometry.

### Where is the output?

The notebook writes results to:

```text
My Drive/DIP/outputs/
```

Current default filenames are:

```text
video1_result.mp4
video1_result.csv
video1_audit.jpg
```

So even if you test another input filename using Method B, check the `outputs` folder for these result files.

---

## 11. Google Drive layout

A normal setup looks like this:

```text
MyDrive/DIP/
├── video1.mp4                  # YOUR INPUT VIDEO
│
├── models/
│   ├── traffic_sign_yolo11s/
│   └── sign_templates_v3/
│
└── outputs/
    ├── video1_result.mp4       # annotated result video
    ├── video1_result.csv       # detection log
    └── video1_audit.jpg        # ORIGINAL | RESULT audit
```

Model files persist on Google Drive, so a Colab disconnect does not require downloading/training the model from scratch again.

---

## 12. Run on Google Colab

### Step 1 — prepare the input video

Use either Method A or Method B from **Section 10**.

For the easiest setup:

```text
Upload your video as:
My Drive/DIP/video1.mp4
```

### Step 2 — open the notebook

Click the badge at the top of this README or open:

https://colab.research.google.com/github/NVTruong473/DIP/blob/feature/yolo-traffic-safety/END_DIP/colab_demo.ipynb

### Step 3 — enable GPU

```text
Runtime
-> Change runtime type
-> T4 GPU
-> Save
```

### Step 4 — run

```text
Runtime
-> Run all
```

The notebook mounts Google Drive and reads the input video from the path described above.

---

## 13. What `Run all` does

```text
Mount Google Drive
        ↓
Check T4 GPU + input video path
        ↓
Clean obsolete runtime artifacts
        ↓
Download/reuse YOLO11s weights
        ↓
Build/reuse template cache
        ↓
Smoke test
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
Create audit montage
        ↓
Show preview in Colab
```

No separate `.py` file needs to be executed.

---

## 14. Output files

### Annotated video

```text
MyDrive/DIP/outputs/video1_result.mp4
```

### Detection log

```text
MyDrive/DIP/outputs/video1_result.csv
```

The CSV contains frame/time, stable class, raw class, confidence and bounding-box coordinates.

### Audit image

```text
MyDrive/DIP/outputs/video1_audit.jpg
```

The audit image contains representative:

```text
ORIGINAL | RESULT
```

pairs for quickly checking missed signs, false positives, class errors and box stability.

---

## 15. Evaluation

Recommended criteria:

### Detection quality

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95

### Small/far-sign performance

Evaluate signs separately by bounding-box size:

- small
- medium
- large

### Video stability

Useful project-specific metrics include:

- class switches per tracked sign,
- percentage of detected frames with the stable class,
- average bounding-box center displacement,
- duplicate-box rate,
- false-positive persistence length.

Also inspect difficult segments with distant signs, sunlight, shadows, motion blur and partial occlusion.

---

## 16. Limitations

This is an academic/experimental traffic-sign perception system, not a production autonomous-driving stack.

Limitations include:

- very distant signs may have too few pixels for reliable classification,
- motion blur can destroy internal symbols,
- the current slice geometry is designed mainly for road/dashcam scenes,
- unusual signs or classes absent from training data may fail,
- domain shift across cameras, cities, weather and nighttime conditions may reduce accuracy,
- public model-card metrics are not the same as independently measured performance on a new user video.

The system intentionally prefers **uncertainty over hallucination**.

---

## 17. Future improvements

Strong next steps include:

- label a representative subset of difficult road-video frames,
- fine-tune on hard Vietnamese far-sign examples,
- use hard-negative mining for advertisements, traffic lights and logos,
- train with blur, glare, shadow, rain and nighttime augmentation,
- compare the custom slicing implementation with formal SAHI,
- quantitatively evaluate Soft-NMS / Weighted Boxes Fusion variants,
- add a dedicated classifier for ambiguous sign crops,
- calibrate class probabilities,
- benchmark performance by sign size/distance.

The most valuable future improvement is better **domain-specific labeled data**, not simply adding more heuristics.

---

## 18. References

### Small-object detection

- Akyon et al., **SAHI: Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection**  
  https://arxiv.org/abs/2202.06934

### Box fusion

- Solovyev et al., **Weighted Boxes Fusion: Ensembling Boxes from Different Object Detection Models**  
  https://arxiv.org/abs/1910.13302

### Frameworks

- Ultralytics YOLO — https://github.com/ultralytics/ultralytics
- OpenCV — https://opencv.org/
- PyTorch — https://pytorch.org/

### Vietnamese traffic-sign model/data

- Model — https://huggingface.co/star092304/traffic-sign-detection-vietnam-yolo
- Dataset — https://huggingface.co/datasets/star092304/Traffic-sign-detection-VietNam

---

## 19. Quick start for a new user

```text
1. Choose your own road/dashcam .mp4 video
2. Rename it to video1.mp4
3. Upload it to: Google Drive -> My Drive -> DIP -> video1.mp4
4. Open END_DIP/colab_demo.ipynb in Google Colab
5. Select T4 GPU
6. Runtime -> Run all
7. Wait for inference and H.264 encoding
8. Open: My Drive -> DIP -> outputs -> video1_result.mp4
9. Inspect video1_audit.jpg for representative failures
```

**Open notebook:**  
https://colab.research.google.com/github/NVTruong473/DIP/blob/feature/yolo-traffic-safety/END_DIP/colab_demo.ipynb

---

## Project status

This branch is an experimental/academic Computer Vision + Digital Image Processing implementation focused on robust Vietnamese traffic-sign recognition in real road videos. The current design favors **conservative, evidence-based detections and stable visualization** over maximizing the raw number of bounding boxes.
