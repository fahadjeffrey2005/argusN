# YOLOFINETUNE — Baseline Results
### ARGUS-N Model 1 | Generated 2026-06-10 10:27

---

## Model Summary

| Field | Value |
|---|---|
| Model | YOLOFINETUNE — YOLOv8n fine-tuned |
| Architecture | YOLOv8n, backbone frozen (10 layers) |
| Classes | 1 — `fod` |
| Input size | 640 × 640 |
| Confidence threshold | 0.35 |
| IoU threshold | 0.45 |
| Epochs | 50 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Training device | CUDA (Ubuntu NVIDIA GPU) |

---

## Dataset

| Split | Images | Labels |
|---|---|---|
| Train (augmented ×5) | ~360 | ~360 |
| Validation | 15 | 15 |
| Test | 16 | 16 |
| Total annotated | 103 | 103 |

Source: Roboflow — `durvas-workspace-ihhkq/hawkeye-ap3a8 v1`
Original classes remapped → single class `0 = fod`

---

## Detection Metrics (Test Set)

| Metric | Value |
|---|---|
| **mAP50** | **0.8297** |
| **mAP50-95** | **0.3646** |
| Precision | 0.7972 |
| Recall | 0.7778 |
| F1 Score | 0.7873 |

---

## Speed

| Metric | Value |
|---|---|
| Inference latency | 10.9 ms/frame |
| Inference FPS | 91.4 fps |

---

## False Positive Rate (Clean Runway)

| Metric | Value |
|---|---|
| Clean video duration | — |
| Total false alerts | — |
| **False positive rate** | **—** |

> The false positive rate on clean tarmac is the primary operational metric.
> A false alert stops the vehicle and wastes inspection time.

---

## Role in Comparative Study

YOLOFINETUNE is **Model 1 — the baseline**.
All metrics above are the performance floor.
HAWKEYE (Model 2) and PRIME (Model 3) are evaluated on the identical test set
and their results are compared directly against these numbers.

| Metric | YOLOFINETUNE | HAWKEYE | PRIME |
|---|---|---|---|
| mAP50 | 0.8297 | — | — |
| mAP50-95 | 0.3646 | — | — |
| Precision | 0.7972 | — | — |
| Recall | 0.7778 | — | — |
| F1 | 0.7873 | — | — |
| FP rate (alerts/min) | — | — | — |
| FPS | 91.4 | — | — |
| Latency (ms) | 10.9 | — | — |

---

## Demo Video

`outputs/demo_yolofinetune_fod1.mp4`
Produced by running the trained model on `fod1.mp4`.
Used for side-by-side visual comparison with HAWKEYE and PRIME outputs.

---

## Weights

`models/yolo/finetuned/best.pt`
HAWKEYE and PRIME copy this file directly — no retraining.
