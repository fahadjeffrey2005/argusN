"""
PRIME — label_crops.py
Simple CLI labelling tool for raw_crops.
Shows each crop (RGB + flow channel), press 0-4 to assign a class or s to skip.

Classes:
  0 = fod
  1 = shadow
  2 = runway_marking
  3 = strobe_light
  4 = clean_tarmac

Usage:
    python scripts/label_crops.py \
      --input data/crops/raw_crops \
      --output data/crops \
      --config config/config.yaml
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
    27: -2          # ESC = quit
}


def load_crop_display(path: Path) -> np.ndarray:
    """
    Load 4-channel crop and create a side-by-side display:
    left = BGR, right = flow magnitude (colourised).
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 2 or img.shape[2] == 3:
        # Fallback for 3-channel saves
        return cv2.resize(img[:, :, :3], (256, 128))

    bgr = img[:, :, :3]
    flow_ch = img[:, :, 3]

    # Colourise flow channel (grayscale → heatmap)
    flow_norm = cv2.normalize(flow_ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    flow_colour = cv2.applyColorMap(flow_norm, cv2.COLORMAP_JET)

    # Resize both to 256x256 and place side by side
    bgr_disp = cv2.resize(bgr, (256, 256))
    flow_disp = cv2.resize(flow_colour, (256, 256))
    display = np.hstack([bgr_disp, flow_disp])
    return display


def main():
    parser = argparse.ArgumentParser(description="Label raw crops for PRIME CNN training")
    parser.add_argument("--input", default="data/crops/raw_crops", help="Directory of unlabelled crops")
    parser.add_argument("--output", default="data/crops", help="Root output directory (class subdirs created here)")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_root = Path(args.output)

    # Create class subdirectories
    for cls in CLASS_NAMES:
        (output_root / cls).mkdir(parents=True, exist_ok=True)

    crop_files = sorted(input_dir.glob("*.png"))
    if not crop_files:
        print(f"No crops found in {input_dir}")
        sys.exit(0)

    print(f"\nFound {len(crop_files)} crops to label.")
    print("Keys: 0=fod  1=shadow  2=runway_marking  3=strobe_light  4=clean_tarmac  s=skip  q/ESC=quit\n")

    labeled = 0
    skipped = 0

    for i, crop_path in enumerate(crop_files):
        display = load_crop_display(crop_path)
        if display is None:
            print(f"[skip] Cannot read {crop_path.name}")
            continue

        # Add label bar at bottom
        bar = np.zeros((40, display.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            bar,
            f"[{i+1}/{len(crop_files)}] {crop_path.name}  |  0-4 label  s skip  q quit",
            (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
        )
        display = np.vstack([display, bar])

        window_title = "PRIME — Label Crops (BGR | Flow)"
        cv2.imshow(window_title, display)

        while True:
            key = cv2.waitKey(0)
            if key in KEY_MAP:
                action = KEY_MAP[key]
                break

        if action == -2:  # quit
            print(f"\nQuitting. Labeled {labeled}, skipped {skipped}.")
            cv2.destroyAllWindows()
            sys.exit(0)

        if action == -1:  # skip
            skipped += 1
            continue

        # Move to class directory
        dest = output_root / CLASS_NAMES[action] / crop_path.name
        shutil.move(str(crop_path), str(dest))
        labeled += 1

        print(f"  [{i+1}/{len(crop_files)}] {crop_path.name} → {CLASS_NAMES[action]}")

    cv2.destroyAllWindows()
    print(f"\nDone. Labeled: {labeled}, skipped: {skipped}")
    for cls in CLASS_NAMES:
        count = len(list((output_root / cls).glob("*.png")))
        print(f"  {cls}: {count} crops")


if __name__ == "__main__":
    main()
