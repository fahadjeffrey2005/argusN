"""
ARGUS-N — NDJSON Dataset Downloader + YOLO Converter

Downloads images from Ultralytics-format NDJSON export files and writes
a properly structured YOLOv8 dataset ready for training.

Supports multiple NDJSON files — merges them into one dataset, remapping
class IDs so there are no collisions.

Output structure:
    data/yolo_dataset/
        images/  train/  val/  test/
        labels/  train/  val/  test/
        dataset.yaml   <- pass this to: yolo train data=...

Usage:
    python scripts/download_ndjson_dataset.py \
        --files runway-fod-2.ndjson runway-fod-3.ndjson \
        --out data/yolo_dataset \
        --workers 8
"""

import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request


def download_image(url, dest, retries=3):
    if Path(dest).exists():
        return True
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, str(dest))
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
            else:
                print(f"  [FAIL] {Path(dest).name}: {e}")
    return False


def boxes_to_yolo(boxes, class_offset=0):
    lines = []
    for box in boxes:
        cls = int(box[0]) + class_offset
        cx, cy, bw, bh = float(box[1]), float(box[2]), float(box[3]), float(box[4])
        cx  = max(0.0, min(1.0, cx))
        cy  = max(0.0, min(1.0, cy))
        bw  = max(0.001, min(1.0, bw))
        bh  = max(0.001, min(1.0, bh))
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files",   nargs="+", required=True,
                        help="One or more .ndjson files")
    parser.add_argument("--out",     default="data/yolo_dataset",
                        help="Output dataset root")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download threads (default 8)")
    args = parser.parse_args()

    out_root = Path(args.out)
    for split in ["train", "val", "test"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_classes   = {}
    download_jobs = []
    class_offset  = 0

    for ndjson_path in args.files:
        print(f"\nParsing {ndjson_path} ...")
        records = []
        with open(ndjson_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)

        meta      = records[0]
        raw_cls   = meta.get("class_names", {})
        n_classes = len(raw_cls)

        for k, v in raw_cls.items():
            all_classes[int(k) + class_offset] = v

        print(f"  Classes: {raw_cls}  (offset={class_offset})")

        images = [r for r in records[1:] if isinstance(r, dict)
                  and r.get("type") == "image"]
        annotated = sum(1 for i in images
                        if i.get("annotations", {}).get("boxes"))
        print(f"  Images: {len(images)}  Annotated: {annotated}")

        for img in images:
            split    = img.get("split", "train")
            stem     = Path(img["file"]).stem
            ext      = Path(img["file"]).suffix or ".jpg"
            dest_img = out_root / "images" / split / f"{stem}{ext}"
            dest_lbl = out_root / "labels" / split / f"{stem}.txt"

            anns  = img.get("annotations", {})
            boxes = anns.get("boxes", []) if isinstance(anns, dict) else []
            label_lines = boxes_to_yolo(boxes, class_offset=class_offset)

            download_jobs.append({
                "url":         img["url"],
                "dest_img":    dest_img,
                "dest_lbl":    dest_lbl,
                "label_lines": label_lines,
            })

        class_offset += n_classes

    print(f"\nDownloading {len(download_jobs)} images ({args.workers} workers) ...")
    ok = fail = 0

    def process(job):
        success = download_image(job["url"], job["dest_img"])
        if success:
            job["dest_lbl"].write_text("\n".join(job["label_lines"]))
        return success

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, job): job for job in download_jobs}
        for i, future in enumerate(as_completed(futures)):
            if future.result():
                ok += 1
            else:
                fail += 1
            if (i + 1) % 200 == 0 or (i + 1) == len(download_jobs):
                print(f"  [{i+1}/{len(download_jobs)}]  ok={ok}  fail={fail}")

    # Write dataset.yaml
    class_list = [all_classes[i] for i in sorted(all_classes)]
    yaml_path  = out_root / "dataset.yaml"
    yaml_path.write_text(f"""# ARGUS-N YOLOv8 dataset
# Auto-generated by download_ndjson_dataset.py

path: {out_root.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: {len(all_classes)}
names: {class_list}
""")

    print(f"\n{'='*55}")
    print(f"Dataset ready : {out_root}")
    print(f"Classes ({len(all_classes)}) : {class_list}")
    for split in ["train", "val", "test"]:
        n = len(list((out_root / "images" / split).glob("*")))
        print(f"  {split:5s} : {n} images")
    print(f"Downloaded: {ok}   Failed: {fail}")
    print(f"\nTo fine-tune YOLOv8:")
    print(f"  yolo train model=yolov8n.pt data={yaml_path} epochs=50 imgsz=640")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
