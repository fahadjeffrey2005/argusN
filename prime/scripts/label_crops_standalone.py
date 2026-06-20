"""
ARGUS-N FOD Dataset Labeler
============================
Standalone script — no other code needed.

SETUP (one time):
    pip install opencv-python

RUN:
    python label_crops_standalone.py

    # Or point to a specific folder:
    python label_crops_standalone.py --input path/to/raw_crops --output path/to/output

WHAT YOU'LL SEE:
    A window showing two panels side by side:
      Left  = the actual image patch (BGR colour)
      Right = the flow magnitude channel (heatmap — usually near-zero/dark on static footage)

    For each image, press a key to classify it:

        0  →  fod              (actual foreign object debris — bolt, rock, cloth, plastic, metal)
        1  →  shadow           (dark shadow cast by equipment, vehicle, overhead structure)
        2  →  runway_marking   (painted line, number, threshold bar, chevron, arrow)
        3  →  strobe_light     (bright light flash, beacon, approach light)
        4  →  clean_tarmac     (empty runway surface, gravel, normal texture — nothing there)
        s  →  skip             (genuinely unsure — skip it, don't guess)
        q  →  quit and save progress

TIPS:
    - When in doubt between fod and anything else, lean toward the other class.
      False positives are more damaging than missed FODs.
    - If it looks like tarmac texture with no distinct object, press 4.
    - Runway numbers, centerline stripes, threshold markings → press 2.
    - A shadow with a sharp edge but no physical object underneath → press 1.
    - You can quit and resume — already-labeled crops are skipped automatically.

OUTPUT:
    Labeled crops go into subfolders:
        output/fod/
        output/shadow/
        output/runway_marking/
        output/strobe_light/
        output/clean_tarmac/

    Zip up the output/ folder and send it back.
    Unlabeled crops stay in raw_crops/ — don't delete them.
"""

import argparse
import shutil
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: opencv-python not installed.")
    print("Run:  pip install opencv-python")
    sys.exit(1)

CLASS_NAMES = ["fod", "shadow", "runway_marking", "strobe_light", "clean_tarmac"]
CLASS_DESCRIPTIONS = [
    "FOD — bolt, rock, cloth, plastic, metal fragment",
    "Shadow — dark patch from overhead object",
    "Runway marking — painted line, number, chevron",
    "Strobe / approach light",
    "Clean tarmac — empty surface, normal texture",
]
KEY_MAP = {
    ord("0"): 0,
    ord("1"): 1,
    ord("2"): 2,
    ord("3"): 3,
    ord("4"): 4,
    ord("s"): -1,   # skip
    ord("q"): -2,   # quit
    27:        -2,  # ESC = quit
}
PANEL_SIZE = 280   # px per panel


def load_display(path: Path):
    """Load 4-channel crop PNG and build a display image."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Extract BGR and flow channel
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        flow_ch = np.zeros(img.shape[:2], dtype=np.uint8)
    elif img.shape[2] == 3:
        bgr = img
        flow_ch = np.zeros(img.shape[:2], dtype=np.uint8)
    else:
        bgr = img[:, :, :3]
        flow_raw = img[:, :, 3].astype(np.float32)
        if flow_raw.max() > 0:
            flow_ch = cv2.normalize(flow_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            flow_ch = np.zeros(img.shape[:2], dtype=np.uint8)

    # Resize both panels
    bgr_panel  = cv2.resize(bgr, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_NEAREST)
    flow_color = cv2.applyColorMap(
        cv2.resize(flow_ch, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_NEAREST),
        cv2.COLORMAP_JET
    )

    # Panel labels
    cv2.putText(bgr_panel,  "IMAGE",      (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(flow_color, "FLOW (heat)",(8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    return np.hstack([bgr_panel, flow_color])


def build_legend(current_file: str, index: int, total: int,
                 labeled: int, skipped: int) -> np.ndarray:
    """Build the instruction panel below the image."""
    h, w = 160, PANEL_SIZE * 2
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    # Progress
    pct = int(labeled / max(total, 1) * 100)
    cv2.putText(panel, f"[{index}/{total}]  labeled={labeled}  skipped={skipped}  ({pct}% done)",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # Filename
    fname = current_file[:60] + ("..." if len(current_file) > 60 else "")
    cv2.putText(panel, fname, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)

    # Key guide
    keys = [
        ("0", "fod",             (0, 0, 220)),
        ("1", "shadow",          (0, 180, 255)),
        ("2", "marking",         (0, 220, 0)),
        ("3", "strobe",          (0, 220, 220)),
        ("4", "clean tarmac",    (180, 180, 180)),
        ("s", "skip",            (100, 100, 100)),
        ("q", "quit",            (80, 80, 80)),
    ]
    x = 8
    for key, label, colour in keys:
        text = f"[{key}] {label}"
        cv2.putText(panel, text, (x, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        x += tw + 14
        if x > w - 100:
            x = 8

    # Tips
    cv2.putText(panel, "TIP: If unsure between fod and anything else -> lean toward the other class.",
                (8, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 60), 1)
    cv2.putText(panel, "     Genuinely ambiguous -> press s (skip). Don't guess.",
                (8, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 60), 1)

    # Class counts
    return panel


def already_labeled(path: Path, output_root: Path) -> bool:
    """Check if this crop was already labeled in a previous session."""
    for cls in CLASS_NAMES:
        if (output_root / cls / path.name).exists():
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="ARGUS-N FOD crop labeler — standalone"
    )
    parser.add_argument("--input",  default="raw_crops",
                        help="Folder of raw crop PNGs (default: raw_crops/)")
    parser.add_argument("--output", default="labeled_crops",
                        help="Output folder for labeled crops (default: labeled_crops/)")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"ERROR: Input folder not found: {input_dir}")
        print(f"Expected a folder of .png crop files.")
        sys.exit(1)

    # Create class output folders
    for cls in CLASS_NAMES:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)

    all_crops = sorted(input_dir.glob("*.png"))
    if not all_crops:
        print(f"No .png files found in {input_dir}")
        sys.exit(0)

    # Filter out already-labeled ones (resume support)
    remaining = [p for p in all_crops if not already_labeled(p, output_dir)]
    pre_labeled = len(all_crops) - len(remaining)

    print(f"\nARGUS-N FOD Labeler")
    print(f"{'='*40}")
    print(f"Total crops  : {len(all_crops)}")
    print(f"Already done : {pre_labeled}")
    print(f"Remaining    : {len(remaining)}")
    print(f"Output       : {output_dir}/")
    print(f"\nPress 0-4 to label, s to skip, q to quit.")
    print(f"Close the terminal to stop at any time — progress is saved automatically.\n")

    if not remaining:
        print("All crops already labeled!")
        _print_summary(output_dir)
        return

    cv2.namedWindow("ARGUS-N Labeler", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ARGUS-N Labeler", PANEL_SIZE * 2, PANEL_SIZE + 160)

    labeled = skipped = 0
    total   = len(remaining)

    for i, path in enumerate(remaining):
        panels = load_display(path)
        if panels is None:
            print(f"  [skip] Cannot read: {path.name}")
            continue

        legend = build_legend(path.name, i + 1, total, labeled, skipped)
        display = np.vstack([panels, legend])
        cv2.imshow("ARGUS-N Labeler", display)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in KEY_MAP:
                action = KEY_MAP[key]
                break

        if action == -2:  # quit
            print(f"\nStopped at {i+1}/{total}. Progress saved.")
            break

        if action == -1:  # skip
            skipped += 1
            continue

        # Move to class folder
        dest = output_dir / CLASS_NAMES[action] / path.name
        shutil.copy2(str(path), str(dest))
        labeled += 1

    cv2.destroyAllWindows()

    print(f"\n{'='*40}")
    print(f"Session complete — labeled {labeled}, skipped {skipped}")
    _print_summary(output_dir)
    print(f"\nZip the '{output_dir}/' folder and send it back.")


def _print_summary(output_dir: Path):
    print(f"\nCurrent counts in {output_dir}/:")
    total = 0
    for cls in CLASS_NAMES:
        n = len(list((output_dir / cls).glob("*.png")))
        bar = "█" * (n // 10)
        status = "✓ good" if n >= 200 else f"⚠  need {200 - n} more"
        print(f"  {cls:<20} {n:4d}  {bar}  {status}")
        total += n
    print(f"  {'TOTAL':<20} {total:4d}")


if __name__ == "__main__":
    main()
