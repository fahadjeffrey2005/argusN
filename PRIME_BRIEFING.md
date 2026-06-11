# PRIME (Model 3) — Briefing Document
### Context for the PRIME implementation chat

---

## What This Document Is

This is a complete handoff from the HAWKEYE (Model 2) implementation chat.
Read this fully before starting PRIME. It contains the guiding architecture,
everything learned from building HAWKEYE, and exactly what PRIME needs to do
differently to be better.

Always reference the ARGUS-N_Master_Architecture.md (Section 0 + Section 3)
alongside this document.

---

## Current State of the Repository

```
argusN/
├── yolofinetune/    ← COMPLETE — trained, evaluated, demo video done
├── hawkeye/         ← COMPLETE — trained, evaluated, demo video done
├── prime/           ← YOU ARE BUILDING THIS
├── data/            ← shared raw data
└── models/          ← shared pretrained weights
```

**Mac path:** `/Volumes/T72/argusN/`
**Ubuntu path:** `~/argusN/`
**GitHub:** https://github.com/fahadjeffrey2005/argusN
**Python env:** venv inside each model directory

---

## Hardware

- **Mac (Apple M4):** Write all code here at `/Volumes/T72/argusN/`
- **Ubuntu (koushik-test@maveric, NVIDIA Thor GPU, CUDA):** All training/inference runs here
- **Workflow:** Write on Mac → git push → Ubuntu git pull → run → push results → Mac git pull

---

## Model 1 — YOLOFINETUNE (Complete)

Fine-tuned YOLOv8n on 103 annotated FOD images (Roboflow dataset).
Single-frame detection, no temporal awareness.

**Results:**
- mAP50: 0.7521
- Precision: 0.8568
- Recall: 0.7143
- F1: 0.7791
- FP rate: ~0.40/min (shadows, runway texture false flags)

**Key weakness:** Fires on every frame independently. Transient false flags
(shadows moving, texture variations) trigger alerts because there is no
confirmation mechanism.

---

## Model 2 — HAWKEYE (Complete)

### What the Architecture Doc Says HAWKEYE Should Be

The architecture doc specifies YOLO + Farneback optical flow + PatchCore
anomaly detection with 2-of-3 voting fusion.

### What We Actually Built (and Why We Changed It)

We built the full three-component stack per the spec. It failed on the test
footage for the following reasons:

1. **The test video (fod1.mp4) is static/near-static footage** — the camera
   is not mounted on a moving vehicle. The optical flow component was designed
   for a moving inspection vehicle. On static footage, Farneback flow between
   consecutive frames is pure noise, generating dozens of false flow candidates
   per frame.

2. **PatchCore bank was too narrow** — 100 clean frames from one patch of
   tarmac. The runway has varied gravel texture and colour gradients that
   PatchCore had never seen, scoring them as anomalous. Scale calibration
   (scale=10.0 → scale=2.0) helped but couldn't fix the fundamental coverage
   problem.

3. **Multi-component voting amplified noise** — with flow generating noise
   candidates and PatchCore over-firing on texture, the 2-of-3 vote was
   constantly reached without a real FOD present.

### The Final HAWKEYE Architecture

After testing, we replaced the three-component stack with:

```
YOLO detects candidates every frame
            ↓
TemporalTracker — IoU matching across consecutive frames
            ↓
Confirm only if same object detected in >= 2 consecutive frames
            ↓
Alert with confirmed track
```

**Why this works:**
- Real FODs are stationary — YOLO detects them consistently across frames
- Transient false flags (shadows, texture hits) appear for 1 frame and vanish
- The 2-frame confirmation window eliminates transient noise without hurting recall

**Final config:**
```yaml
yolo:
  confidence_threshold: 0.28
  iou_threshold: 0.45
  input_size: 640

tracker:
  confirm_frames: 2
  iou_threshold: 0.20
  max_miss_frames: 3

pipeline:
  top_crop: 0.50    # cut top half — distant runway with shadows/gradients
  bot_crop: 0.05
```

### HAWKEYE File Structure

```
hawkeye/
├── src/
│   ├── utils/config_loader.py
│   ├── utils/logger.py
│   ├── ingestion/camera.py
│   ├── detection/yolo_detector.py      ← runs YOLO on full cropped frame
│   ├── tracking/temporal_tracker.py   ← THE CORE — new for HAWKEYE
│   ├── flow/                           ← kept but not used in final pipeline
│   ├── anomaly/patchcore.py            ← kept but not used in final pipeline
│   └── fusion/hawkeye_fusion.py        ← legacy, replaced by tracker
├── scripts/
│   ├── prepare_dataset.py
│   ├── split_dataset.py
│   ├── build_patchcore_bank.py         ← legacy
│   ├── make_demo_video.py              ← YOLO + temporal tracker pipeline
│   ├── run_hawkeye.py
│   ├── evaluate_hawkeye.py
│   └── generate_report.py
└── config/config.yaml
```

### Dataset

- **Source:** Roboflow workspace `durvas-workspace-ihhkq`, project `hawkeye-ap3a8`, version 1
- **API key:** WCIxuet94KXWxzmAgRSQ
- **103 images**, single class (originally 'Foreign Object' → remapped to class 0 = fod)
- **Split:** 73 train / 15 val / 15 test (70/15/15, seed=42)
- **Download note:** Always use absolute path in `--output`. Roboflow silently fails with relative paths.

---

## Known Issues / Lessons Learned (Critical for PRIME)

1. **`--` em-dash on Ubuntu terminal** — when pasting commands, `--` often
   becomes an em-dash (—). Always type `--arguments` manually on Ubuntu.

2. **Roboflow download requires absolute path** — use
   `/home/koushik-test/argusN/prime/data/annotated` not `data/annotated`.

3. **Roboflow structure is split-first** — downloads as `train/images/`,
   `valid/images/`, `test/images/`. Our `prepare_dataset.py` handles this
   automatically (Layout A detection).

4. **`git add -f` for gitignored files** — `best.pt`, `bank.pt`,
   `eval_results.json` need `-f` flag. Never use `-f` for videos (`.mp4`).
   Videos are blocked in `.gitignore` — transfer via `scp` or Google Drive.

5. **`ultralytics` saves training runs to `runs/detect/{name}/`** — not to
   the `project=` path. Use `results.save_dir` to locate `best.pt`.

6. **Run all scripts from inside the model directory** — `cd ~/argusN/prime`
   then `python scripts/...`, not from argusN root.

7. **git diverged branches** — when Ubuntu pushes and Mac also has commits,
   `git pull --no-rebase` then `git push` on whichever machine gets the error.

8. **Test video is static** — `fod1.mp4` is near-static footage (camera not
   on moving vehicle). Any flow-based component will not work as designed.
   Design PRIME to work on static footage.

9. **PatchCore scale calibration** — observed WideResNet50 L2 distances on
   this dataset are ~2.1-2.7 for FOD patches. Scale=10.0 (default) gives
   scores ~0.22, far below any useful threshold. Scale=2.0 maps L2=2.5 → 0.71.

10. **Videos are large (~865MB each)** — never push to git. Transfer via
    Google Drive or scp.

---

## What PRIME Needs to Do

Per the architecture doc, PRIME replaces PatchCore with a trained
**MobileNetV3-Small semantic CNN classifier** that explicitly recognises:

- Class 0: `fod`
- Class 1: `shadow`
- Class 2: `runway_marking`
- Class 3: `strobe_light`
- Class 4: `clean_tarmac`

HAWKEYE's temporal tracker reduces transient false flags but cannot explain
WHY something is a false flag. PRIME's CNN learns the specific visual
categories of false positives and explicitly discards them.

**Key difference from HAWKEYE:**
- HAWKEYE: "Did YOLO see this consistently?" (temporal)
- PRIME: "What IS this thing specifically?" (semantic)

**4-channel input:**
Each candidate patch is passed as [BGR + flow_magnitude] — a 4-channel
128x128 tensor. The flow channel gives the CNN physics context alongside
appearance.

**Given the static footage issue:**
Since the flow channel will be near-zero on static footage, the CNN must
still work on RGB alone. The 4-channel input is the architecture but the
CNN should be robust when flow ≈ 0.

---

## Shared Resources PRIME Can Use

All of the following already exist in the repo and should be copied into
`prime/` with minimal changes:

| File | Location | Usage in PRIME |
|------|----------|----------------|
| `config_loader.py` | `hawkeye/src/utils/` | Copy as-is |
| `logger.py` | `hawkeye/src/utils/` | Copy as-is |
| `camera.py` | `hawkeye/src/ingestion/` | Copy as-is |
| `yolo_detector.py` | `hawkeye/src/detection/` | Copy as-is |
| `farneback.py` | `hawkeye/src/flow/` | Copy — flow for 4-channel input |
| `egomotion.py` | `hawkeye/src/flow/` | Copy as-is |
| `residual.py` | `hawkeye/src/flow/` | Copy as-is |
| `temporal_tracker.py` | `hawkeye/src/tracking/` | Copy — still used for confirmation |
| `prepare_dataset.py` | `hawkeye/scripts/` | Copy — same Roboflow logic |
| `split_dataset.py` | `hawkeye/scripts/` | Copy as-is |
| `augment_dataset.py` | `yolofinetune/scripts/` | Copy — same augmentation logic |

**YOLO weights:** Copy from yolofinetune — do NOT retrain.
```bash
cp ~/argusN/yolofinetune/models/yolo/finetuned/best.pt \
   ~/argusN/prime/models/yolo/finetuned/best.pt
```

---

## PRIME Roboflow Dataset

Use the **same dataset** as HAWKEYE for fair comparison:
- Workspace: `durvas-workspace-ihhkq`
- Project: `hawkeye-ap3a8`
- Version: 1
- API key: `WCIxuet94KXWxzmAgRSQ`

For the CNN classifier, you need ADDITIONAL labeled crops per class.
See architecture doc Section 3 for the crop collection workflow.

---

## Evaluation Target

PRIME must beat HAWKEYE on false positive rate while maintaining comparable
recall. The research narrative:

```
YOLOFINETUNE  → detects FODs, false flags on transient events
HAWKEYE       → adds temporal confirmation, reduces transient false flags
PRIME         → adds semantic understanding, reduces ALL false flag categories
```

All three models evaluated on the identical test set with identical metrics.

---

*Generated from HAWKEYE implementation chat — June 2026*
