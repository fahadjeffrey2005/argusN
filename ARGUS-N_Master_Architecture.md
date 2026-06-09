# ARGUS-N — Master Architecture Document
### FOD Detection Research | Comparative Three-Model Study
---

> **How to use this document:**
> This document is divided into four self-contained sections.
> Section 0 provides shared context that applies to all three models.
> Sections 1, 2, and 3 are fully independent design briefs — each can be fed into a separate chat session to implement that model in isolation.
> Always include Section 0 when feeding any section into a new chat.

---

# SECTION 0 — SHARED CONTEXT

## What ARGUS-N Is

ARGUS-N is a real-time Foreign Object Debris (FOD) detection system for airport runways. A downward-facing camera is mounted on an inspection vehicle. As the vehicle drives across the runway, the system detects debris that does not belong on the surface and alerts the operator with a visual overlay and saved frames.

## Research Goal

This project conducts a structured comparative study of three progressively sophisticated detection approaches:

| Directory | Model Name | Approach |
|---|---|---|
| `yolofinetune/` | YOLOFINETUNE | Fine-tuned YOLO — supervised baseline |
| `hawkeye/` | HAWKEYE | Multi-stack — YOLO + physics + unsupervised anomaly |
| `prime/` | PRIME | Novel stack — YOLO + physics + semantic CNN classifier |

The goal is to produce quantitative and graphical evidence of the performance difference between all three, culminating in a publishable novel contribution from PRIME.

## Hardware and Workflow

```
MAC (Apple M4, T72 SSD at /Volumes/T72/argusN/)
  — Write all code here
  — All architectural decisions made here
        |
        | git push
        v
    GITHUB
  (https://github.com/fahadjeffrey2005/argusN)
        |
        | git pull
        v
UBUNTU (koushik-test@maveric, NVIDIA GPU, CUDA)
  — All training runs here
  — All inference and benchmarking runs here
```

**Mac path:** `/Volumes/T72/argusN/`
**Ubuntu path:** `~/argusN/`
**Python env:** `venv` inside project root (already set up)

## Directory Structure After Setup

```
argusN/
├── yolofinetune/        ← Model 1 — baseline
├── hawkeye/             ← Model 2 — multi-stack
├── prime/               ← Model 3 — novel
├── src/                 ← original argusN source (reference only)
├── models/              ← shared pretrained weights
├── data/                ← shared raw data
└── scripts/             ← original scripts (reference only)
```

Each subdirectory (`yolofinetune/`, `hawkeye/`, `prime/`) is a fully independent project with its own `src/`, `data/`, `models/`, `scripts/`, `config/`, and `venv/`.

## Shared Dataset Strategy

All three models are trained and evaluated on the same underlying data so results are directly comparable.

### Data Collection Plan

**Step 1 — Clean tarmac video**
Drive the inspection vehicle across the runway with zero FODs present. Record at least 5-10 minutes of footage. This gives you the ground truth of what a perfect runway looks like.

**Step 2 — FOD placement sessions**
Physically place objects on the tarmac one at a time and record a pass over each:
- Metal bolt / nut
- Small rock / gravel cluster
- Piece of cloth / rag
- Piece of metal sheet
- Plastic bag / wrapper
- Bird feather
- Any airport-specific debris available

Minimum target: 50-100 frames per FOD type before augmentation.

**Step 3 — False positive category recording (PRIME only, but collect now)**
While driving, identify and record:
- Frames with clear tarmac shadows (from equipment, vehicles, overhead structures)
- Runway markings (centerline, numbers, threshold bars, chevrons)
- Strobe / approach lights in frame
- Wet tarmac patches
- Skid marks

**Step 4 — Annotation**
Use LabelImg or Roboflow to annotate FOD frames in YOLO format:
`class_id cx cy w h` (normalised 0-1)

Only one class needed: `0 = fod`

**Step 5 — Augmentation**
For each annotated FOD frame, generate augmented copies:
- Horizontal flip
- Brightness shift ±30%
- Gaussian noise
- Copy-paste FOD crops onto different clean tarmac backgrounds

Target after augmentation: 2000+ annotated FOD instances.

**Step 6 — Split**
- 70% training
- 15% validation
- 15% test (held out, never touched during training)

Keep the same split across all three models. Same test set = fair comparison.

## Shared Evaluation Metrics

Every model is evaluated on the identical test set and reported with the same metrics:

| Metric | Description |
|---|---|
| mAP50 | Mean average precision at IoU 0.5 |
| mAP50-95 | Mean average precision across IoU 0.5-0.95 |
| Precision | True positives / (true positives + false positives) |
| Recall | True positives / (true positives + false negatives) |
| F1 Score | Harmonic mean of precision and recall |
| False Positive Rate | False alerts per minute of clean runway footage |
| Inference FPS | Frames processed per second end-to-end |
| Latency ms | End-to-end time per frame |

The false positive rate on clean runway footage is the most operationally important metric. An alert on a clean runway costs time and money.

---
---

# SECTION 1 — YOLOFINETUNE

> **Self-contained design brief.**
> Feed Section 0 + Section 1 into a new chat to implement this model independently.

## What It Is

YOLOFINETUNE is the baseline model. It is a single YOLOv8 model fine-tuned on a custom runway FOD dataset. No flow, no anomaly detection, no secondary models. Pure supervised detection.

Its purpose in this research is to establish the performance floor — the best a single fine-tuned detector can do on its own, before any additional components are added.

## Architecture

```
┌──────────────────────────────┐
│         Video Frame          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│        Preprocessing         │
│   Resize to 640x640          │
│   Crop ROI (top 22% removed, │
│   bottom 15% removed)        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     YOLOv8n (fine-tuned)     │
│   Backbone: frozen (10 layers)│
│   Head: trained on FOD data  │
│   Classes: 1 (fod)           │
│   Input size: 640x640        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     NMS + Confidence Filter  │
│   conf_threshold: 0.35       │
│   iou_threshold: 0.45        │
└──────────────┬───────────────┘
               │
        ┌──────┴──────┐
        │  FOD found? │
        └──────┬──────┘
       YES ◄───┤├───► NO
        │      └┘      │
        ▼              ▼
┌──────────────┐  ┌──────────┐
│  Draw boxes  │  │  Next    │
│  Save frame  │  │  Frame   │
│  Log alert   │  └──────────┘
└──────────────┘
```

## File Structure

```
yolofinetune/
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── config_loader.py     ← KEEP from argusN
│   │   └── logger.py            ← KEEP from argusN
│   └── ingestion/
│       ├── __init__.py
│       └── camera.py            ← KEEP from argusN
├── data/
│   ├── raw/
│   │   ├── videos/              ← raw runway footage
│   │   └── images/
│   ├── annotated/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── labels/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   └── augmented/               ← generated augmented copies
├── models/
│   └── yolo/
│       ├── yolov8n.pt           ← KEEP — pretrained starting point
│       └── finetuned/           ← trained weights saved here
│           └── best.pt
├── config/
│   └── config.yaml              ← NEW — create fresh
├── scripts/
│   ├── extract_frames.py        ← NEW
│   ├── augment_dataset.py       ← NEW
│   ├── train_yolo.py            ← NEW
│   ├── run_inference.py         ← NEW
│   └── evaluate.py              ← NEW
├── outputs/
│   ├── detections/
│   └── alerts/
├── logs/
├── setup.sh                     ← KEEP from argusN (update paths)
└── requirements.txt             ← NEW
```

## Files to Keep from argusN Copy

| File | Action | Reason |
|---|---|---|
| `src/utils/config_loader.py` | KEEP | Reuse as-is |
| `src/utils/logger.py` | KEEP | Reuse as-is |
| `src/ingestion/camera.py` | KEEP | Reuse as-is |
| `models/yolo/yolov8n.pt` | KEEP | Starting point for fine-tune |
| `setup.sh` | KEEP, MODIFY | Update drive paths |

## Files to Remove from argusN Copy

| File / Directory | Action |
|---|---|
| `src/flow/` | DELETE entirely |
| `src/tracking/` | DELETE entirely |
| `src/fusion/` | DELETE entirely |
| `src/learning/` | DELETE entirely |
| `src/detection/yolo_detector.py` | DELETE — rewrite as simpler script |
| `src/ingestion/multi_camera.py` | DELETE |
| `src/ingestion/nir_simulator.py` | DELETE |
| `models/raft/` | DELETE |
| `scripts/run_pipeline_rt.py` | DELETE — replace with run_inference.py |
| `data/replay_buffer/` | DELETE |
| `data/synthetic/` | DELETE |
| `Poster FOD-1.pdf` | DELETE |

## config/config.yaml

```yaml
device: cuda

camera:
  input_mode: video_file
  video_file_path: data/raw/videos/recording.mp4
  resolution:
    width: 1920
    height: 1080
  fps: 60

yolo:
  model_path: models/yolo/finetuned/best.pt
  pretrained_path: models/yolo/yolov8n.pt
  confidence_threshold: 0.35
  iou_threshold: 0.45
  input_size: 640
  classes: 1
  class_names: ["fod"]
  freeze_layers: 10
  epochs: 50
  batch_size: 16
  learning_rate: 0.001

pipeline:
  top_crop: 0.22
  bot_crop: 0.15
  warmup_frames: 30

outputs:
  detections_path: outputs/detections
  alerts_path: outputs/alerts

logging:
  log_path: logs/yolofinetune.log
  level: INFO
```

## Dataset Requirements

| Split | Minimum Instances | Notes |
|---|---|---|
| Train | 1400 FOD instances | After augmentation |
| Val | 300 FOD instances | No augmentation |
| Test | 300 FOD instances | Never touched, identical across all 3 models |

Clean tarmac (background) frames are included implicitly — YOLO learns background from images where no label file exists.

## Training Plan

**Step 1 — Extract frames from raw video**
```bash
python scripts/extract_frames.py \
  --video data/raw/videos/fod_recording.mp4 \
  --output data/annotated/images/train \
  --fps 2
```
Extract at 2fps to avoid near-identical frames.

**Step 2 — Annotate with LabelImg**
```bash
pip install labelImg
labelImg data/annotated/images/train data/annotated/labels/train
```
Label every FOD instance. Class 0 = fod.

**Step 3 — Augment dataset**
```bash
python scripts/augment_dataset.py \
  --input data/annotated \
  --output data/augmented \
  --factor 5
```

**Step 4 — Fine-tune YOLO**
```bash
python scripts/train_yolo.py
```
This runs:
- Load YOLOv8n pretrained
- Freeze first 10 layers (backbone)
- Train detection head on FOD dataset
- Save best weights to `models/yolo/finetuned/best.pt`

**Step 5 — Evaluate**
```bash
python scripts/evaluate.py \
  --model models/yolo/finetuned/best.pt \
  --data data/annotated/test
```

## Implementation Plan (in order)

1. Copy argusN into `yolofinetune/`, delete files per table above
2. Create `config/config.yaml`
3. Create `requirements.txt`
4. Write `scripts/extract_frames.py`
5. Collect and annotate raw footage
6. Write `scripts/augment_dataset.py`
7. Write `scripts/train_yolo.py`
8. Run training on Ubuntu (push → pull → run)
9. Write `scripts/run_inference.py` (live video with MJPEG stream)
10. Write `scripts/evaluate.py`
11. Run evaluation, record all metrics

## Expected Output

- `models/yolo/finetuned/best.pt` — fine-tuned weights
- `outputs/detections/` — annotated frames with bounding boxes
- `logs/eval_results.json` — precision, recall, mAP, FPS, false positive rate

This model's results are the baseline. Every number HAWKEYE and PRIME produce is compared against these.

---
---

# SECTION 2 — HAWKEYE

> **Self-contained design brief.**
> Feed Section 0 + Section 2 into a new chat to implement this model independently.
> YOLOFINETUNE must be completed first — HAWKEYE reuses its trained YOLO weights.

## What It Is

HAWKEYE is the multi-stack model. It combines three independent detection mechanisms — a fine-tuned YOLO detector, physics-based optical flow residual analysis, and unsupervised PatchCore anomaly detection — into a single pipeline with confidence-weighted voting fusion.

Each component operates on a different principle. They fail in different situations. The fusion layer requires at least two of three to agree before raising an alert.

## Architecture

```
┌─────────────────┐   ┌─────────────────┐
│   Frame (T)     │   │   Frame (T-1)   │
└────────┬────────┘   └────────┬────────┘
         │                     │
         ├─────────────────────┤
         │                     │
         ▼                     ▼
┌─────────────────┐   ┌──────────────────────────┐
│  YOLO Detection │   │   Farneback Optical Flow  │
│  (fine-tuned)   │   │   Egomotion Subtraction   │
│  Single forward │   │   (IMU speed → expected   │
│  pass, 640x640  │   │    flow → residual map)   │
└────────┬────────┘   └──────────────┬────────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐   ┌──────────────────────────┐
│  YOLO Candidate │   │  Flow Residual Anomaly    │
│  Bounding Boxes │   │  Regions (connected       │
│                 │   │  components above thresh) │
└────────┬────────┘   └──────────────┬────────────┘
         │                           │
         └──────────┬────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   Candidate Union    │
         │   All boxes from     │
         │   both sources       │
         │   merged by IoU      │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   PatchCore Scoring  │
         │   Per candidate crop │
         │   Score: 0-1         │
         │   (trained on clean  │
         │    tarmac frames)    │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────────────┐
         │   Confidence-Weighted Voting │
         │                              │
         │   YOLO fired?   → +1 vote   │
         │   Flow flagged? → +1 vote   │
         │   PatchCore > 0.6? → +1 vote│
         │                              │
         │   2 or more votes → ALERT    │
         │   0 or 1 vote  → DISCARD     │
         └──────────┬───────────────────┘
                    │
           ┌────────┴────────┐
           │  Alert raised?  │
           └────────┬────────┘
       YES ◄────────┤├────────► NO
        │           └┘          │
        ▼                       ▼
┌──────────────┐          ┌──────────┐
│  Draw boxes  │          │  Next    │
│  Save frame  │          │  Frame   │
│  Log alert   │          └──────────┘
└──────────────┘
```

## Component Details

### Component 1 — YOLO (fine-tuned)
Identical to YOLOFINETUNE. Weights copied directly from `yolofinetune/models/yolo/finetuned/best.pt`. No retraining. Detects known FOD classes from visual appearance alone.

### Component 2 — Farneback Flow + Egomotion
Computes dense optical flow between consecutive frames using OpenCV Farneback algorithm. Subtracts expected flow (derived from simulated vehicle speed via IMU) to produce a residual map. Pixels with residual magnitude above threshold are stationary anomalies — they moved differently from how the vehicle's motion predicts they should. These become flow candidate regions.

Physics parameters:
- Camera height: 0.325m above ground
- Focal length: ~1200px
- Vehicle speed: read from config (simulated) or IMU serial feed
- Residual threshold: 2.5px (tunable)

### Component 3 — PatchCore
Unsupervised anomaly detector. Trained on clean tarmac frames only — no labels required. Builds a memory bank of normal tarmac feature vectors. At inference, scores each candidate patch by its distance to the nearest normal feature in the bank. High score = anomalous = not normal tarmac.

PatchCore parameters:
- Backbone: WideResNet50 (pretrained ImageNet, frozen)
- Layer: layer2 + layer3 features
- Bank size: 60-100 clean frames
- Anomaly threshold: 0.6 (tunable)

### Fusion Layer
Simple voting. Each component casts a vote (0 or 1) per candidate region. Total votes ≥ 2 raises an alert. This prevents any single noisy component from causing a false alarm.

## File Structure

```
hawkeye/
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── config_loader.py     ← KEEP from argusN
│   │   └── logger.py            ← KEEP from argusN
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── camera.py            ← KEEP from argusN
│   ├── detection/
│   │   ├── __init__.py
│   │   └── yolo_detector.py     ← KEEP from argusN, minor edits
│   ├── flow/
│   │   ├── __init__.py
│   │   ├── farneback.py         ← NEW (extract from raft_flow.py fallback)
│   │   ├── egomotion.py         ← KEEP from argusN
│   │   └── residual.py          ← KEEP from argusN
│   ├── anomaly/
│   │   ├── __init__.py
│   │   └── patchcore.py         ← NEW
│   └── fusion/
│       ├── __init__.py
│       └── hawkeye_fusion.py    ← NEW
├── data/
│   ├── raw/                     ← symlink or copy from yolofinetune
│   ├── annotated/               ← same split as yolofinetune
│   ├── clean_frames/            ← NEW — for PatchCore bank
│   └── patchcore_bank/          ← NEW — saved memory bank
├── models/
│   ├── yolo/
│   │   └── finetuned/
│   │       └── best.pt          ← COPY from yolofinetune — do not retrain
│   └── patchcore/
│       └── bank.pt              ← saved after bank build
├── config/
│   └── config.yaml              ← NEW
├── scripts/
│   ├── build_patchcore_bank.py  ← NEW
│   ├── run_hawkeye.py           ← NEW
│   └── evaluate_hawkeye.py      ← NEW
├── outputs/
│   ├── detections/
│   └── alerts/
├── logs/
└── setup.sh
```

## Files to Keep from argusN Copy

| File | Action | Reason |
|---|---|---|
| `src/utils/config_loader.py` | KEEP | Reuse as-is |
| `src/utils/logger.py` | KEEP | Reuse as-is |
| `src/ingestion/camera.py` | KEEP | Reuse as-is |
| `src/detection/yolo_detector.py` | KEEP, MINOR EDIT | Remove patch-only logic |
| `src/flow/egomotion.py` | KEEP | Reuse as-is |
| `src/flow/residual.py` | KEEP | Reuse as-is |
| `src/flow/raft_flow.py` | PARTIAL — extract Farneback only | Rename to `farneback.py` |
| `setup.sh` | KEEP, MODIFY | Update paths |

## Files to Remove from argusN Copy

| File / Directory | Action |
|---|---|
| `src/flow/raft_flow.py` | DELETE after extracting Farneback — no RAFT needed |
| `src/tracking/` | DELETE — ByteTrack not used in HAWKEYE |
| `src/fusion/` | DELETE — replace with `hawkeye_fusion.py` |
| `src/learning/` | DELETE entirely |
| `src/ingestion/multi_camera.py` | DELETE |
| `src/ingestion/nir_simulator.py` | DELETE |
| `models/raft/` | DELETE |
| `models/yolo/yolov8n.pt` | REPLACE with copy of yolofinetune best.pt |
| `scripts/run_pipeline_rt.py` | DELETE — replace with run_hawkeye.py |
| `data/replay_buffer/` | DELETE |
| `data/synthetic/` | DELETE |

## config/config.yaml

```yaml
device: cuda

camera:
  input_mode: video_file
  video_file_path: data/raw/videos/recording.mp4
  fps: 60

yolo:
  model_path: models/yolo/finetuned/best.pt
  confidence_threshold: 0.35
  iou_threshold: 0.45
  input_size: 640

flow:
  residual_threshold: 2.5
  min_anomaly_area_px: 10
  max_anomaly_area_px: 50000

imu:
  enabled: false
  simulated_speed_kmh: 30.0

egomotion:
  camera_height_m: 0.325
  focal_length_px: 1200.0

patchcore:
  bank_path: models/patchcore/bank.pt
  anomaly_threshold: 0.6
  backbone: wide_resnet50_2
  layers: [layer2, layer3]

fusion:
  votes_required: 2
  yolo_weight: 1
  flow_weight: 1
  patchcore_weight: 1

pipeline:
  top_crop: 0.22
  bot_crop: 0.15
  warmup_frames: 30

outputs:
  detections_path: outputs/detections
  alerts_path: outputs/alerts

logging:
  log_path: logs/hawkeye.log
  level: INFO
```

## Dataset Requirements

| Data | Purpose | Labels Needed |
|---|---|---|
| Same annotated FOD set as YOLOFINETUNE | YOLO component evaluation | Yes (already done) |
| 60-100 clean tarmac frames | PatchCore memory bank | No |
| Same test set as YOLOFINETUNE | Fair comparison | Yes (same annotations) |

The clean frames for PatchCore come from the beginning of your clean tarmac video. Extract 60-100 frames with no FOD present.

## Training Plan

**Step 1 — Copy YOLO weights**
```bash
cp ../yolofinetune/models/yolo/finetuned/best.pt models/yolo/finetuned/best.pt
```
No retraining. Same weights as YOLOFINETUNE for fair comparison.

**Step 2 — Extract clean tarmac frames for PatchCore**
```bash
python scripts/build_patchcore_bank.py \
  --video data/raw/videos/clean_runway.mp4 \
  --frames 100 \
  --output data/clean_frames
```

**Step 3 — Build PatchCore memory bank**
```bash
python scripts/build_patchcore_bank.py \
  --images data/clean_frames \
  --save models/patchcore/bank.pt
```
One-time build. Takes ~2-3 minutes. No training, no labels.

**Step 4 — Tune fusion threshold on validation set**
Run HAWKEYE on validation set, sweep `votes_required` (1 or 2) and `patchcore_threshold` (0.4-0.8). Pick combination that maximises F1.

**Step 5 — Evaluate on test set**
```bash
python scripts/evaluate_hawkeye.py \
  --video data/raw/videos/test_recording.mp4 \
  --annotations data/annotated/test
```

## Implementation Plan (in order)

1. Copy argusN into `hawkeye/`, delete files per table above
2. Copy YOLO weights from `yolofinetune/models/yolo/finetuned/best.pt`
3. Create `config/config.yaml`
4. Extract Farneback-only flow module from `raft_flow.py` → `src/flow/farneback.py`
5. Write `src/anomaly/patchcore.py`
6. Write `src/fusion/hawkeye_fusion.py`
7. Write `scripts/build_patchcore_bank.py`
8. Collect clean tarmac frames, build bank
9. Write `scripts/run_hawkeye.py`
10. Tune thresholds on validation set
11. Write `scripts/evaluate_hawkeye.py`
12. Run evaluation on test set, record all metrics

## Expected Weaknesses (by design — for comparative study)

- Shadows on sunny days cause both flow AND PatchCore to flag simultaneously → false alerts pass the 2/3 vote
- Wet tarmac causes PatchCore to score everything as anomalous → noisy results
- Strobe lights trigger flow component (periodic brightness change) → false votes
- Unknown FOD types: flow catches them, PatchCore catches them — good generalisation
- Inference FPS limited by PatchCore nearest-neighbour search (~15-30ms overhead)

These weaknesses are the motivation for PRIME.

---
---

# SECTION 3 — PRIME

> **Self-contained design brief.**
> Feed Section 0 + Section 3 into a new chat to implement this model independently.
> YOLOFINETUNE must be completed first — PRIME reuses its trained YOLO weights.
> Review HAWKEYE architecture before implementing — PRIME shares the flow component.

## What It Is

PRIME is the novel model. It replaces HAWKEYE's unsupervised PatchCore component with a learned semantic CNN classifier that explicitly understands airport-specific false positive categories.

Where HAWKEYE asks "is this statistically different from normal tarmac?", PRIME asks "what is this thing specifically?" The CNN is trained to recognise five categories: FOD, shadow, runway marking, strobe light, and clean tarmac. Only confirmed FODs raise an alert.

Additionally, PRIME feeds a 4-channel input to the CNN — the RGB patch combined with the optical flow magnitude map for that region. This gives the CNN both the visual appearance AND the physics signal simultaneously, making it significantly more accurate than appearance alone.

## Architecture

```
┌─────────────────┐   ┌─────────────────┐
│   Frame (T)     │   │   Frame (T-1)   │
└────────┬────────┘   └────────┬────────┘
         │                     │
         ├─────────────────────┤
         │                     │
         ▼                     ▼
┌─────────────────┐   ┌──────────────────────────┐
│  YOLO Detection │   │   Farneback Optical Flow  │
│  (fine-tuned)   │   │   Egomotion Subtraction   │
│  1 forward pass │   │   → Flow Residual Map     │
│  640x640        │   │   → Anomaly Regions       │
└────────┬────────┘   └──────────────┬────────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐   ┌──────────────────────────┐
│  YOLO Candidate │   │  Flow Anomaly Regions     │
│  Bounding Boxes │   │  (connected components)   │
└────────┬────────┘   └──────────────┬────────────┘
         │                           │
         └──────────┬────────────────┘
                    │
                    ▼
         ┌──────────────────────────────┐
         │   Candidate Merge            │
         │                              │
         │   IoU match YOLO ↔ flow      │
         │   Overlap → merge, tag=both  │
         │   YOLO only → tag=yolo_only  │
         │   Flow only → tag=flow_only  │
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │   Per-candidate crop:        │
         │                              │
         │   Channel 1-3: BGR patch     │
         │   Channel 4:   flow mag map  │
         │   → 4-channel, 128x128      │
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │   MobileNetV3-Small          │
         │   (pretrained ImageNet)      │
         │   First conv: 3 → 4 channels │
         │   Head: 5-class output       │
         └──────────────┬───────────────┘
                        │
          ┌─────────────┼──────────────────────────────┐
          ▼             ▼             ▼                 ▼
     ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐
     │   FOD   │  │  Shadow  │  │  Runway  │  │Strobe / Clean│
     └────┬────┘  │ Discard  │  │ Marking  │  │   Discard    │
          │       └──────────┘  │ Discard  │  └──────────────┘
          ▼                     └──────────┘
   ┌─────────────┐
   │  Draw boxes │
   │  Save frame │
   │  Log alert  │
   └─────────────┘
```

## Component Details

### Component 1 — YOLO (fine-tuned)
Identical to YOLOFINETUNE and HAWKEYE. Same weights, no retraining.

### Component 2 — Farneback Flow + Egomotion
Identical to HAWKEYE. Produces flow residual map and anomaly candidate regions.

### Component 3 — Candidate Merge with Source Tagging
Different from HAWKEYE's union approach. Each candidate carries metadata about which component detected it:

```
tag = "both"       → YOLO and flow both flagged this region (high prior confidence)
tag = "yolo_only"  → Only YOLO flagged it (visual match, no physics confirmation)
tag = "flow_only"  → Only flow flagged it (physics anomaly, YOLO didn't recognise it)
```

The tag is passed to the CNN as context. Candidates tagged `both` get a confidence bonus — when visual and physics signals agree, the classification threshold is slightly lower.

### Component 4 — 4-Channel CNN Classifier (MobileNetV3-Small)

**Input construction:**
For each candidate bounding box:
1. Crop the RGB patch from frame (T) with 20px padding
2. Crop the corresponding region from the flow magnitude map
3. Resize both to 128x128
4. Stack: `[B, G, R, flow_magnitude]` → shape `(4, 128, 128)`

**Architecture modification:**
MobileNetV3-Small pretrained on ImageNet. The first convolutional layer accepts 3 channels by default. Modify it to accept 4:
- Copy existing 3-channel weights
- Initialise the 4th channel weights near zero
- The model learns the contribution of the flow channel during fine-tuning

**Output head:**
5-class softmax:
- Class 0: `fod`
- Class 1: `shadow`
- Class 2: `runway_marking`
- Class 3: `strobe_light`
- Class 4: `clean_tarmac`

**Inference:**
Only class 0 (`fod`) raises an alert. All other classes are discarded.

## File Structure

```
prime/
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── config_loader.py       ← KEEP from argusN
│   │   └── logger.py              ← KEEP from argusN
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── camera.py              ← KEEP from argusN
│   ├── detection/
│   │   ├── __init__.py
│   │   └── yolo_detector.py       ← KEEP from argusN
│   ├── flow/
│   │   ├── __init__.py
│   │   ├── farneback.py           ← SAME as hawkeye
│   │   ├── egomotion.py           ← KEEP from argusN
│   │   └── residual.py            ← KEEP from argusN
│   ├── semantic/
│   │   ├── __init__.py
│   │   ├── cnn_classifier.py      ← NEW — MobileNetV3-Small 4-channel
│   │   └── crop_builder.py        ← NEW — builds 4-channel crops
│   └── fusion/
│       ├── __init__.py
│       └── prime_fusion.py        ← NEW — merge + source tagging
├── data/
│   ├── raw/                       ← same as hawkeye
│   ├── annotated/                 ← same test set as all models
│   └── crops/
│       ├── raw_crops/             ← untagged crops from collect step
│       ├── fod/                   ← labeled FOD crops
│       ├── shadow/                ← labeled shadow crops
│       ├── runway_marking/        ← labeled marking crops
│       ├── strobe_light/          ← labeled strobe crops
│       └── clean_tarmac/          ← labeled clean crops
├── models/
│   ├── yolo/
│   │   └── finetuned/
│   │       └── best.pt            ← COPY from yolofinetune
│   └── cnn/
│       ├── mobilenetv3_small.pth  ← pretrained ImageNet weights
│       └── prime_classifier.pth   ← trained CNN weights saved here
├── config/
│   └── config.yaml                ← NEW
├── scripts/
│   ├── collect_crops.py           ← NEW — run YOLO+flow, save all candidates
│   ├── label_crops.py             ← NEW — simple CLI labelling tool
│   ├── train_cnn.py               ← NEW — MobileNetV3 training script
│   ├── run_prime.py               ← NEW — full pipeline inference
│   └── evaluate_prime.py          ← NEW
├── outputs/
│   ├── detections/
│   └── alerts/
├── logs/
└── setup.sh
```

## Files to Keep from argusN Copy

| File | Action | Reason |
|---|---|---|
| `src/utils/config_loader.py` | KEEP | Reuse as-is |
| `src/utils/logger.py` | KEEP | Reuse as-is |
| `src/ingestion/camera.py` | KEEP | Reuse as-is |
| `src/detection/yolo_detector.py` | KEEP | Reuse as-is |
| `src/flow/egomotion.py` | KEEP | Reuse as-is |
| `src/flow/residual.py` | KEEP | Reuse as-is |
| `setup.sh` | KEEP, MODIFY | Update paths |

## Files to Remove from argusN Copy

| File / Directory | Action |
|---|---|
| `src/flow/raft_flow.py` | DELETE — replace with farneback.py |
| `src/tracking/` | DELETE entirely |
| `src/fusion/` | DELETE — replace with prime_fusion.py |
| `src/learning/` | DELETE entirely |
| `src/ingestion/multi_camera.py` | DELETE |
| `src/ingestion/nir_simulator.py` | DELETE |
| `models/raft/` | DELETE |
| `scripts/run_pipeline_rt.py` | DELETE — replace with run_prime.py |
| `data/replay_buffer/` | DELETE |
| `data/synthetic/` | DELETE |
| `Poster FOD-1.pdf` | DELETE |

## config/config.yaml

```yaml
device: cuda

camera:
  input_mode: video_file
  video_file_path: data/raw/videos/recording.mp4
  fps: 60

yolo:
  model_path: models/yolo/finetuned/best.pt
  confidence_threshold: 0.35
  iou_threshold: 0.45
  input_size: 640

flow:
  residual_threshold: 2.5
  min_anomaly_area_px: 10
  max_anomaly_area_px: 50000

imu:
  enabled: false
  simulated_speed_kmh: 30.0

egomotion:
  camera_height_m: 0.325
  focal_length_px: 1200.0

cnn:
  model_path: models/cnn/prime_classifier.pth
  pretrained_path: models/cnn/mobilenetv3_small.pth
  input_size: 128
  channels: 4
  num_classes: 5
  class_names: [fod, shadow, runway_marking, strobe_light, clean_tarmac]
  fod_class_id: 0
  confidence_threshold: 0.6
  both_tag_bonus: 0.1
  epochs: 30
  batch_size: 32
  learning_rate: 0.0005
  early_stopping_patience: 5

fusion:
  iou_match_threshold: 0.3
  patch_padding_px: 20

pipeline:
  top_crop: 0.22
  bot_crop: 0.15
  warmup_frames: 30

outputs:
  detections_path: outputs/detections
  alerts_path: outputs/alerts

logging:
  log_path: logs/prime.log
  level: INFO
```

## Dataset Requirements

### Part A — YOLO (same as YOLOFINETUNE, already done)
No additional work. Copy weights.

### Part B — CNN Classifier (new, specific to PRIME)

Target: 200+ labeled crops per class before augmentation → 1000+ per class after.

| Class | How to collect |
|---|---|
| `fod` | Crop YOLO/flow detections from FOD recording sessions |
| `shadow` | Run YOLO+flow on clean sunny day footage, manually confirm shadow regions |
| `runway_marking` | Run YOLO+flow near runway markings, manually confirm |
| `strobe_light` | Run YOLO+flow during evening/night footage, manually confirm light flashes |
| `clean_tarmac` | Random crops from clean tarmac frames with no flagged regions |

**Practical crop collection workflow:**
1. Run `scripts/collect_crops.py` on all available runway footage
2. Script runs YOLO + flow, saves every flagged candidate patch to `data/crops/raw_crops/`
3. Run `scripts/label_crops.py` — shows each crop, press 0-4 to label or `s` to skip
4. Labeled crops go into their class folder
5. Run augmentation to reach 1000 per class

## Training Plan

**Step 1 — Copy YOLO weights**
```bash
cp ../yolofinetune/models/yolo/finetuned/best.pt models/yolo/finetuned/best.pt
```

**Step 2 — Collect CNN training crops**
```bash
python scripts/collect_crops.py \
  --source data/raw/videos/ \
  --output data/crops/raw_crops \
  --save-flow-map
```
This runs the YOLO + flow pipeline and saves every candidate region as a 4-channel crop (RGB + flow magnitude).

**Step 3 — Label crops**
```bash
python scripts/label_crops.py \
  --input data/crops/raw_crops \
  --output data/crops
```
Label each crop as one of 5 classes. Target 200+ per class. Takes 1-2 hours.

**Step 4 — Augment to 1000 per class**
Flip, rotate, brightness, noise. Balance classes.

**Step 5 — Train CNN**
```bash
python scripts/train_cnn.py
```
Loads MobileNetV3-Small pretrained weights, modifies first conv 3→4 channels, trains 5-class head. Early stopping on validation loss. Saves best weights to `models/cnn/prime_classifier.pth`.

**Step 6 — Evaluate full pipeline on test set**
```bash
python scripts/evaluate_prime.py \
  --video data/raw/videos/test_recording.mp4 \
  --annotations data/annotated/test
```

## Implementation Plan (in order)

1. Copy argusN into `prime/`, delete files per table above
2. Copy YOLO weights from `yolofinetune/models/yolo/finetuned/best.pt`
3. Extract Farneback module → `src/flow/farneback.py`
4. Write `src/fusion/prime_fusion.py` (merge + source tagging)
5. Write `src/semantic/crop_builder.py` (4-channel crop construction)
6. Write `src/semantic/cnn_classifier.py` (MobileNetV3-Small 4-channel)
7. Write `scripts/collect_crops.py`
8. Write `scripts/label_crops.py`
9. Collect and label CNN training crops
10. Write `scripts/train_cnn.py`
11. Run CNN training on Ubuntu
12. Write `scripts/run_prime.py` (full pipeline)
13. Write `scripts/evaluate_prime.py`
14. Run full evaluation, record all metrics
15. Compare against YOLOFINETUNE and HAWKEYE

## Why PRIME Beats HAWKEYE

| Situation | HAWKEYE | PRIME |
|---|---|---|
| Sunny day shadow | False alert (flow + PatchCore both vote yes) | Correctly discarded (CNN: shadow) |
| Runway centerline | PatchCore may flag texture edge | Correctly discarded (CNN: runway_marking) |
| Strobe light at night | False alert (flow detects flash + PatchCore flags it) | Correctly discarded (CNN: strobe_light) |
| Wet tarmac | Mass false positives (PatchCore sees everything as anomalous) | Handles if wet tarmac in training |
| Unknown FOD type | Flow + PatchCore catch it (generalises well) | May miss if CNN never trained on that appearance |
| Inference speed | ~28-43ms/frame (~25fps) | ~15-20ms/frame (~50fps) |
| Edge compute fit | Moderate (PatchCore bank in memory) | Excellent (3MB fixed model) |

The key research result: PRIME's false positive rate on clean runway footage is measurably lower than HAWKEYE's, at higher FPS, with lower memory footprint.

---
---

# APPENDIX — Comparative Evaluation Framework

## Running the Full Comparison

After all three models are trained, run the same evaluation script on each using the identical test set.

```bash
# From argusN root
python eval/compare_all.py \
  --yolofinetune yolofinetune/models/yolo/finetuned/best.pt \
  --hawkeye hawkeye/ \
  --prime prime/ \
  --test-video data/test/test_recording.mp4 \
  --test-annotations data/test/labels/ \
  --output eval/results/
```

## Output: Results Table

| Metric | YOLOFINETUNE | HAWKEYE | PRIME |
|---|---|---|---|
| mAP50 | — | — | — |
| mAP50-95 | — | — | — |
| Precision | — | — | — |
| Recall | — | — | — |
| F1 | — | — | — |
| False Positive Rate (per min) | — | — | — |
| Inference FPS | — | — | — |
| Latency (ms/frame) | — | — | — |

*Fill in during evaluation phase.*

## Graphical Outputs Required

1. Precision-Recall curve for all three models on the same axes
2. Bar chart: False positive rate comparison
3. Bar chart: FPS comparison
4. Confusion matrix for PRIME's CNN classifier
5. Side-by-side detection frames: same FOD detected by all three
6. Side-by-side false positive frames: shadow flagged by YOLOFINETUNE and HAWKEYE, correctly dismissed by PRIME

## The Research Narrative

```
YOLOFINETUNE establishes what a fine-tuned detector can do alone.

HAWKEYE shows that adding physics and anomaly detection improves recall
but introduces new false positive categories it cannot suppress.

PRIME demonstrates that semantic understanding of those specific false
positive categories is the correct solution — achieving lower false
positive rate, higher FPS, and better edge compute fit simultaneously.
```

This is your three-act comparative study. Each model exists to set up the next one.

---
*ARGUS-N Master Architecture Document — Generated June 2026*
