# HAWKEYE — Ubuntu End-to-End Command Sequence

Run everything from inside `~/argusN/hawkeye/` on koushik-test@maveric.

---

## 0. Pull latest code from Mac

```bash
cd ~/argusN
git pull
cd hawkeye
```

---

## 1. Set up virtual environment (first time only)

```bash
cd ~/argusN/hawkeye
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `roboflow` is included in requirements.txt. If pip install fails on
> any package, install manually: `pip install roboflow ultralytics opencv-python tqdm torchvision`

---

## 2. Download and prepare dataset

```bash
source venv/bin/activate
cd ~/argusN/hawkeye

python scripts/prepare_dataset.py \
  --api-key WCIxuet94KXWxzmAgRSQ \
  --workspace durvas-workspace-ihhkq \
  --project hawkeye-ap3a8 \
  --version 1 \
  --output /home/koushik-test/argusN/hawkeye/data/annotated
```

> **IMPORTANT:** `--output` must be an absolute path. Relative paths silently
> fail with the Roboflow SDK.
>
> **Note on `--` (em-dash issue):** If pasting into the terminal converts `--`
> to an em-dash (—), type each `--argument` manually.

This creates:
- `data/annotated/` with train/val/test splits
- `config/dataset.yaml` with absolute paths

---

## 3. Copy YOLO weights from yolofinetune (do NOT retrain)

```bash
cp /home/koushik-test/argusN/yolofinetune/models/yolo/finetuned/best.pt \
   /home/koushik-test/argusN/hawkeye/models/yolo/finetuned/best.pt
```

---

## 4. Build PatchCore memory bank

Clean frames are already in `data/clean_frames/` (100 frames committed from Mac).

```bash
python scripts/build_patchcore_bank.py \
  --images data/clean_frames \
  --save models/patchcore/bank.pt
```

Expected output:
```
Bank built: 100 vectors, dim=3072
Bank saved to models/patchcore/bank.pt
```

Takes ~2-3 minutes on CPU. Runs on CUDA if available (faster).

---

## 5. Run evaluation

```bash
python scripts/evaluate_hawkeye.py \
  --data config/dataset.yaml \
  --clean-video /home/koushik-test/argusN/yolofinetune/data/raw/videos/clean_runway.mp4 \
  --device cuda
```

Saves results to `logs/eval_results.json`.

---

## 6. Make demo video

```bash
python scripts/make_demo_video.py \
  --video /home/koushik-test/argusN/yolofinetune/data/raw/videos/fod_sessions/fod1.mp4 \
  --output outputs/demo_hawkeye_fod1.mp4
```

Headless (no display) — writes annotated MP4 directly.

---

## 7. Generate report

```bash
python scripts/generate_report.py
```

Reads `logs/eval_results.json` + `../yolofinetune/logs/eval_results.json`.
Writes `logs/hawkeye_results.md`.

---

## 8. Commit and push results

```bash
cd ~/argusN/hawkeye

git add -f models/yolo/finetuned/best.pt
git add -f models/patchcore/bank.pt
git add -f outputs/demo_hawkeye_fod1.mp4
git add logs/eval_results.json
git add logs/hawkeye_results.md
git add config/dataset.yaml

git commit -m "hawkeye: evaluation results, demo video, patchcore bank"
git push
```

> **`-f` flag is required** for files in `.gitignore` (best.pt, bank.pt, demo video).

---

## Full sequence (copy-paste for reference)

```bash
cd ~/argusN && git pull && cd hawkeye
source venv/bin/activate

python scripts/prepare_dataset.py \
  --api-key WCIxuet94KXWxzmAgRSQ \
  --workspace durvas-workspace-ihhkq \
  --project hawkeye-ap3a8 \
  --version 1 \
  --output /home/koushik-test/argusN/hawkeye/data/annotated

cp /home/koushik-test/argusN/yolofinetune/models/yolo/finetuned/best.pt \
   models/yolo/finetuned/best.pt

python scripts/build_patchcore_bank.py \
  --images data/clean_frames \
  --save models/patchcore/bank.pt

python scripts/evaluate_hawkeye.py \
  --data config/dataset.yaml \
  --clean-video /home/koushik-test/argusN/yolofinetune/data/raw/videos/clean_runway.mp4 \
  --device cuda

python scripts/make_demo_video.py \
  --video /home/koushik-test/argusN/yolofinetune/data/raw/videos/fod_sessions/fod1.mp4 \
  --output outputs/demo_hawkeye_fod1.mp4

python scripts/generate_report.py

git add -f models/yolo/finetuned/best.pt models/patchcore/bank.pt outputs/demo_hawkeye_fod1.mp4
git add logs/eval_results.json logs/hawkeye_results.md config/dataset.yaml
git commit -m "hawkeye: evaluation results, demo video, patchcore bank"
git push
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `-- becomes em-dash` | Type `--arguments` manually on Ubuntu |
| `Roboflow download silent fail` | Use absolute path in `--output` |
| `YOLO weights not found` | Run step 3 first (cp from yolofinetune) |
| `PatchCore bank not loaded` | Run step 4 first (build_patchcore_bank.py) |
| `ultralytics saves to runs/detect/` | Use `results.save_dir` to find best.pt after training |
| `git add fails` | Add `-f` flag for files in .gitignore |
| `CUDA out of memory` | Add `--device cpu` or reduce batch size |
