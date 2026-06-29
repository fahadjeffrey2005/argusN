"""
download_roboflow.py — Download the Hawkeye annotated dataset from Roboflow.

Before running:
  1. Create Version 1 in Roboflow (if not done):
     app.roboflow.com -> hawkeye-zyzkb -> Versions -> Generate New Version
     Add augmentations: flip, rotation ±15, brightness ±25, blur 2px — 3x multiplier
     Click Create.

  2. Get your API key:
     app.roboflow.com -> top-right avatar -> Roboflow API -> copy key

Run from argusN/ on Ubuntu:
    python download_roboflow.py --key YOUR_API_KEY

Downloads to: data/roboflow/
"""

import argparse
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key",     required=True, help="Roboflow API key")
    p.add_argument("--version", type=int, default=1, help="Dataset version number (default 1)")
    p.add_argument("--out",     default="data/roboflow", help="Download destination")
    args = p.parse_args()

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow",
                               "--break-system-packages", "-q"])
        from roboflow import Roboflow

    print(f"\n{'='*55}")
    print(f"  Roboflow Dataset Download")
    print(f"{'='*55}")
    print(f"  Workspace : rishabhs-workspace-anssy")
    print(f"  Project   : hawkeye-zyzkb")
    print(f"  Version   : {args.version}")
    print(f"  Format    : YOLOv8")
    print(f"  Output    : {args.out}")
    print(f"{'='*55}\n")

    rf      = Roboflow(api_key=args.key)
    project = rf.workspace("rishabhs-workspace-anssy").project("hawkeye-zyzkb")
    version = project.version(args.version)
    dataset = version.download("yolov8", location=args.out)

    print(f"\n{'='*55}")
    print(f"  Download complete!")
    print(f"  Location: {args.out}")

    # Show what was downloaded
    out = Path(args.out)
    for split in ["train", "valid", "test"]:
        img_dir = out / split / "images"
        lbl_dir = out / split / "labels"
        n_img = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        print(f"  {split:6s}: {n_img:4d} images, {n_lbl:4d} labels")

    # Show dataset.yaml
    yaml_path = out / "data.yaml"
    if yaml_path.exists():
        print(f"\n  Dataset YAML ({yaml_path}):")
        print("  " + open(yaml_path).read().replace("\n", "\n  "))

    print(f"{'='*55}\n")
    print("  NEXT STEP:")
    print(f"  python generate_synthetic.py --data {args.out}")
    print()


if __name__ == "__main__":
    main()
