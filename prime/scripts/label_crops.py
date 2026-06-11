"""
PRIME — label_crops.py
Interactive CLI labelling tool for raw_crops/.
Shows each crop (BGR panel + flow channel colourised), key 0-4 to label, s to skip.

Classes:
  0 = fod
  1 = shadow
  2 = runway_marking
  3 = strobe_light
  4 = clean_tarmac

Usage (from inside prime/):
    python scripts/label_crops.py
    python scripts/label_crops.py --input data/crops/raw_crops --output data/crops
"""

import argparse
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

CLASS_NAMES = ["fod", "shadow", "runway_marking", "strobe_light", "clean_tarmac"]
KEY_MAP = {
    ord("0"): 0, ord("1"): 1, ord("2"): 2,
    ord("3"): 3, ord("4"): 4,
    ord("s"): -1,   # skip
    ord("q"): -2,   # quit
    27: -2,         # ESC
}


def make_display(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Handle both 3-channel and 4-channel saves
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        flow_ch = np.zeros_like(img[:, :, 0])
    elif img.shape[2] == 3:
        flow_ch = np.zeros(img.shape[:2], dtype=np.uint8)
    else:
        flow_ch = img[:, :, 3]
        img = img[:, :, :3]

    bgr_disp  = cv2.resize(img, (256, 256))
    flow_norm = cv2.normalize(flow_ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    flow_disp = cv2.resize(
        cv2.applyColorMap(flow_norm, cv2.COLORMAP_JET), (256, 256)
    )

    panel = np.hstack([bgr_disp, flow_disp])

    # Legend bar
    bar = np.zeros((50, 512, 3), dtype=np.uint8)
    cv2.putText(bar, "0=fod  1=shadow  2=marking  3=strobe  4=clean  s=skip  q=quit",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    return np.vstack([panel, bar])


def main():
    parser = argparse.ArgumentParser(description="Label PRIME CNN training crops")
    parser.add_argument("--input",  default="data/crops/raw_crops")
    parser.add_argument("--output", default="data/crops")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    for cls in CLASS_NAMES:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)

    crop_files = sorted(input_dir.glob("*.png"))
    if not crop_files:
        print(f"No crops in {input_dir}")
        sys.exit(0)

    print(f"\n{len(crop_files)} crops to label — window: BGR (left) | Flow magnitude (right)")
    labeled = skipped = 0

    for i, path in enumerate(crop_files):
        display = make_display(path)
        if display is None:
            print(f"  [skip] unreadable: {path.name}")
            continue

        # Add filename bar
        info = np.zeros((22, display.shape[1], 3), dtype=np.uint8)
        cv2.putText(info, f"[{i+1}/{len(crop_files)}] {path.name}",
                    (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        display = np.vstack([info, display])

        cv2.imshow("PRIME — Label Crops", display)

        while True:
            key = cv2.waitKey(0)
            if key in KEY_MAP:
                action = KEY_MAP[key]
                break

        if action == -2:
            print(f"\nQuit. Labeled: {labeled}, skipped: {skipped}")
            break

        if action == -1:
            skipped += 1
            continue

        dest = output_dir / CLASS_NAMES[action] / path.name
        shutil.move(str(path), str(dest))
        labeled += 1
        print(f"  [{i+1}/{len(crop_files)}] {path.name} → {CLASS_NAMES[action]}")

    cv2.destroyAllWindows()

    print(f"\n── Labeling complete ──")
    print(f"  Labeled: {labeled}   Skipped: {skipped}")
    for cls in CLASS_NAMES:
        n = len(list((output_dir / cls).glob("*.png")))
        status = "✓" if n >= 200 else f"⚠ need {200-n} more"
        print(f"  {cls:<20}: {n:4d}  {status}")


if __name__ == "__main__":
    main()
