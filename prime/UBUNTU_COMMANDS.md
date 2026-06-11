# PRIME — Ubuntu Command Sequence
### End-to-end from fresh pull to results

---

## CRITICAL NOTES (read before typing anything)

1. **`--` em-dash bug** — never paste `--` arguments into Ubuntu terminal.
   `--` becomes an em-dash when pasted. Type all `--arguments` manually.

2. **Roboflow absolute paths** — always use full paths like
   `/home/koushik-test/argusN/prime/data/annotated`, not `data/annotated`.

3. **git add -f** — `best.pt`, `eval_results.json`, and other gitignored
   files need `-f`. Never use `-f` for `.mp4` videos (blocked intentionally).

4. **Run all scripts from inside prime/** — `cd ~/argusN/prime` first, always.

5. **Static footage** — `fod1.mp4` has a near-static camera. Flow magnitude
   will be near-zero. This is expected and handled by the CNN architecture.

---

## 0. One-time setup

```bash
cd ~/argusN/prime
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 1. Pull latest code

```bash
cd ~/argusN
git pull origin main
cd prime
source venv/bin/activate
```

---

## 2. Copy YOLO weights from yolofinetune (do NOT retrain)

```bash
mkdir -p models/yolo/finetuned
cp ../yolofinetune/models/yolo/finetuned/best.pt models/yolo/finetuned/best.pt
echo "YOLO weights ready:"
ls -lh models/yolo/finetuned/best.pt
```

---

## 3. Prepare the annotated dataset

```bash
python scripts/prepare_dataset.py \
    --dest /home/koushik-test/argusN/prime/data/annotated \
    --api-key WCIxuet94KXWxzmAgRSQ \
    --workspace durvas-workspace-ihhkq \
    --project hawkeye-ap3a8 \
    --version 1
```

If Roboflow puts everything in train/ only, run the splitter:
```bash
python scripts/split_dataset.py
```

Verify:
```bash
ls data/annotated/images/
# Should show: train/  val/  test/
ls data/annotated/images/test/ | wc -l
# Should be ~15
```

Write absolute-path dataset.yaml for YOLO val():
```bash
cat > config/dataset.yaml << 'EOF'
path: /home/koushik-test/argusN/prime/data/annotated
train: images/train
val:   images/val
test:  images/test
nc: 1
names: ['fod']
EOF
```

---

## 4. Collect CNN training crops

YOLO runs on all footage. Every detection becomes a 4-channel crop.
Flow will be near-zero on static footage — that's fine and expected.

```bash
# FOD footage — will generate mostly fod + false-positive crops
python scripts/collect_crops.py \
    --source /home/koushik-test/argusN/yolofinetune/data/raw/videos/fod_sessions/fod1.mp4 \
    --output data/crops/raw_crops

# Clean runway — will generate clean_tarmac + shadow + marking crops
python scripts/collect_crops.py \
    --source /home/koushik-test/argusN/yolofinetune/data/raw/videos/clean_runway.mp4 \
    --output data/crops/raw_crops

echo "Total crops collected:"
ls data/crops/raw_crops/ | wc -l
```

Push crops to git so you can label them on Mac:
```bash
git add -f data/crops/raw_crops/
git commit -m "Add raw CNN training crops"
git push origin main
```

---

## 5. Label crops — DO THIS ON MAC (needs display)

On your Mac:
```bash
cd /Volumes/T72/argusN/prime
source venv/bin/activate

python scripts/label_crops.py \
    --input data/crops/raw_crops \
    --output data/crops
```

Keys:
- **0** = fod
- **1** = shadow
- **2** = runway_marking
- **3** = strobe_light
- **4** = clean_tarmac
- **s** = skip
- **q** = quit

Target: **200+ per class** before training.
The script shows a count summary when you quit — check all classes.

After labeling, push to git:
```bash
git add -f data/crops/fod/ data/crops/shadow/ data/crops/runway_marking/
git add -f data/crops/strobe_light/ data/crops/clean_tarmac/
git commit -m "Add labeled CNN training crops"
git push origin main
```

Then on Ubuntu:
```bash
cd ~/argusN && git pull origin main && cd prime
source venv/bin/activate
echo "Labeled crop counts:"
for cls in fod shadow runway_marking strobe_light clean_tarmac; do
    echo "  $cls: $(ls data/crops/$cls/ 2>/dev/null | wc -l)"
done
```

---

## 6. Train the CNN classifier

```bash
cd ~/argusN/prime
source venv/bin/activate

python scripts/train_cnn.py \
    --crops-dir data/crops \
    --epochs 30
```

Watch for `✓ Best saved (val_loss=...)` lines.
Stops early if validation loss plateaus (patience=5).

Check history:
```bash
cat logs/train_history.json | python3 -m json.tool | tail -20
```

---

## 7. Sanity check: run on FOD footage

```bash
python scripts/run_prime.py \
    --source /home/koushik-test/argusN/yolofinetune/data/raw/videos/fod_sessions/fod1.mp4 \
    --save

echo "Alert frames saved:"
ls outputs/alerts/ | wc -l
```

---

## 8. Evaluate on test set

```bash
python scripts/evaluate_prime.py \
    --model  models/yolo/finetuned/best.pt \
    --data   config/dataset.yaml \
    --clean-video /home/koushik-test/argusN/yolofinetune/data/raw/videos/clean_runway.mp4 \
    --device cuda \
    --output logs/eval_results.json

cat logs/eval_results.json
```

Expected output keys: mAP50, mAP50_95, precision, recall, f1,
latency_ms, fps, fp_per_minute.

---

## 9. Generate demo video

```bash
python scripts/make_demo_video.py \
    --video  /home/koushik-test/argusN/yolofinetune/data/raw/videos/fod_sessions/fod1.mp4 \
    --output outputs/demo_prime_fod1.mp4

ls -lh outputs/demo_prime_fod1.mp4
```

---

## 10. Generate results report

```bash
python scripts/generate_report.py \
    --eval-results  logs/eval_results.json \
    --train-history logs/train_history.json \
    --output prime_results.md

cat prime_results.md
```

---

## 11. Push results to Mac

```bash
cd ~/argusN/prime

# Model weights
git add -f models/cnn/prime_classifier.pth
git add -f models/yolo/finetuned/best.pt

# Results
git add -f logs/eval_results.json
git add logs/train_history.json
git add prime_results.md
git add config/dataset.yaml

# Demo video (large — only push if < 100MB, else use scp/Drive)
ls -lh outputs/demo_prime_fod1.mp4
# If < 100MB:
git add -f outputs/demo_prime_fod1.mp4

git commit -m "PRIME: trained CNN + evaluation results + demo video"
git push origin main
```

If git diverged (Mac also has commits):
```bash
git pull --no-rebase origin main
# resolve any conflicts, then:
git push origin main
```

On Mac after push:
```bash
cd /Volumes/T72/argusN && git pull origin main
```

---

## Troubleshooting

**`models/cnn/prime_classifier.pth not found`**
Run step 6 (train_cnn.py). Pipeline still runs — CNN outputs random logits until trained.

**`No labeled crops in data/crops`**
Run steps 4 (collect) then 5 (label on Mac).

**`CUDA out of memory`**
Reduce batch size in config.yaml: `batch_size: 16`

**`VideoWriter: output file is empty`**
Try codec: change `fourcc = cv2.VideoWriter_fourcc(*"mp4v")` to `*"avc1"` or use `.avi`.

**`dataset.yaml not found` during val()**
Run the `cat > config/dataset.yaml << 'EOF' ...` command in step 3 on Ubuntu.
The path must be absolute — `/home/koushik-test/argusN/prime/...` not `data/...`.

**`git add: pathspec does not match`**
The file doesn't exist yet. Run the relevant step first.

**Flow channel is all zeros**
Expected on static footage. The CNN is trained to handle near-zero flow.
This is PRIME's design — it works on appearance when physics signal is absent.
