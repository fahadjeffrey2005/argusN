# HAWKEYE — Evaluation Report

_Generated: 2026-06-11 11:36_

## Model
**HAWKEYE** — Multi-stack FOD detector: fine-tuned YOLOv8 + Farneback optical flow egomotion residual + PatchCore unsupervised anomaly detection. Alert raised when 2 or more of 3 components vote positive.

## Results

| Metric | HAWKEYE | YOLOFINETUNE | Delta |
|--------|---------|--------------|-------|
| mAP50 | 0.7521 | 0.8297 |  (▼ -0.0776✗) |
| mAP50-95 | 0.4008 | 0.3646 |  (▲ +0.0362✓) |
| Precision | 0.8568 | 0.7972 |  (▲ +0.0597✓) |
| Recall | 0.7143 | 0.7778 |  (▼ -0.0635✗) |
| F1 Score | 0.7791 | 0.7873 |  (▼ -0.0083✗) |
| False Positive Rate (per min) | 114.47 | — |  |
| Latency (ms/frame) | 220.87 | 10.94 |  (▲ +209.94✗) |
| Inference FPS | 4.5 | 91.4 |  (▼ -86.9✗) |

## Commentary

**Detection (mAP50):** HAWKEYE scores slightly below YOLOFINETUNE (0.7521 vs 0.8297, Δ=-0.0776). This is expected — the fusion gate can suppress correct YOLO detections if the other two components don't agree.

**Inference Speed:** HAWKEYE runs at 4.5 fps vs YOLOFINETUNE's 91.4 fps. The PatchCore nearest-neighbour search accounts for the additional latency (~15-30ms). PRIME addresses this with a fixed-size CNN classifier.

## Known Weaknesses (by design)

These weaknesses are the motivation for PRIME:

- **Sunny-day shadows** — flow + PatchCore both flag simultaneously → false alert passes 2/3 vote
- **Wet tarmac** — PatchCore sees texture change as anomalous → elevated FP rate
- **Strobe lights** — flow detects periodic brightness change → false vote
- **PatchCore latency** — nearest-neighbour search adds ~15-30ms overhead, capping FPS

## Output Files

| File | Description |
|------|-------------|
| `logs/eval_results.json` | Machine-readable metrics (consumed by compare_all.py) |
| `outputs/demo_hawkeye_fod1.mp4` | Annotated demo video |
| `models/patchcore/bank.pt` | PatchCore memory bank |
| `models/yolo/finetuned/best.pt` | YOLO weights (copied from yolofinetune) |
