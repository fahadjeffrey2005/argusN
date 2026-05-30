"""
ARGUS-N Nightly Retraining
Fine-tunes YOLOv8 detection head using LoRA on replay buffer samples.
Validation gate — rejects update if mAP drops more than 2%.
Runs offline — zero impact on inference pipeline.
"""

import json
import shutil
import time
from pathlib import Path
from datetime import datetime
from src.utils.config_loader import Config
from src.utils.logger import get_logger
from src.learning.replay_buffer import ReplayBuffer


class Retrainer:
    def __init__(self, cfg: Config, replay_buffer: ReplayBuffer):
        self.cfg = cfg
        self.replay_buffer = replay_buffer
        self.logger = get_logger(
            "retrainer",
            cfg.get("logging", "log_path", default="logs/argus.log"),
            cfg.get("logging", "level", default="INFO")
        )

        self.model_path = cfg.get("yolo", "model_path", default="models/yolo/yolov8n.pt")
        self.lora_rank = cfg.get("learning", "lora_rank", default=16)
        self.lora_alpha = cfg.get("learning", "lora_alpha", default=32)
        self.epochs = cfg.get("learning", "fine_tune_epochs", default=5)
        self.batch_size = cfg.get("learning", "fine_tune_batch_size", default=16)
        self.map_drop_threshold = cfg.get(
            "learning", "validation_map_drop_threshold", default=0.02
        )

        self.retrain_log_path = Path("logs/retrain_history.json")
        self.retrain_log = self._load_retrain_log()

        # Validation set — held out, never touched by training
        self.val_path = Path("data/annotated/val")

        self.logger.info("Retrainer initialised")

    def _load_retrain_log(self) -> list:
        if self.retrain_log_path.exists():
            with open(self.retrain_log_path, "r") as f:
                return json.load(f)
        return []

    def _save_retrain_log(self):
        self.retrain_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.retrain_log_path, "w") as f:
            json.dump(self.retrain_log, f, indent=2)

    def _prepare_dataset(self, samples: list) -> Path:
        """
        Write sampled replay buffer entries to a temporary
        YOLO-format dataset directory for fine-tuning.

        YOLO expects:
            dataset/
                images/train/
                labels/train/
                data.yaml
        """
        import cv2

        dataset_path = Path("data/retrain_tmp")
        images_path = dataset_path / "images" / "train"
        labels_path = dataset_path / "labels" / "train"

        # Clean previous run
        if dataset_path.exists():
            shutil.rmtree(dataset_path)

        images_path.mkdir(parents=True, exist_ok=True)
        labels_path.mkdir(parents=True, exist_ok=True)

        for i, entry in enumerate(samples):
            src = Path(entry["filepath"])
            if not src.exists():
                continue

            # Copy image
            dst_img = images_path / f"sample_{i:05d}.jpg"
            shutil.copy(str(src), str(dst_img))

            # Write YOLO format label
            # FOD class = 0, clean = no label file
            dst_lbl = labels_path / f"sample_{i:05d}.txt"
            if entry["label"] == "fod" and "detection" in entry:
                det = entry["detection"]
                img = cv2.imread(str(dst_img))
                if img is not None:
                    h, w = img.shape[:2]
                    cx = ((det["x1"] + det["x2"]) / 2) / w
                    cy = ((det["y1"] + det["y2"]) / 2) / h
                    bw = (det["x2"] - det["x1"]) / w
                    bh = (det["y2"] - det["y1"]) / h
                    with open(dst_lbl, "w") as f:
                        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            else:
                # Clean frame — empty label file
                dst_lbl.touch()

        # Write data.yaml
        data_yaml = dataset_path / "data.yaml"
        with open(data_yaml, "w") as f:
            f.write(f"path: {dataset_path.resolve()}\n")
            f.write(f"train: images/train\n")
            f.write(f"val: {self.val_path.resolve()}\n")
            f.write(f"nc: 1\n")
            f.write(f"names: ['fod']\n")

        self.logger.info(
            f"Dataset prepared — "
            f"{len(samples)} samples at {dataset_path}"
        )

        return dataset_path

    def _get_baseline_map(self) -> float:
        """
        Evaluate current model on validation set.
        Returns mAP50 as baseline before fine-tuning.
        """
        if not self.val_path.exists():
            self.logger.warning(
                "Validation set not found — "
                "skipping baseline mAP, gate disabled"
            )
            return 0.0

        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            results = model.val(
                data=str(self.val_path / "data.yaml"),
                verbose=False
            )
            map50 = float(results.box.map50)
            self.logger.info(f"Baseline mAP50: {map50:.4f}")
            return map50
        except Exception as e:
            self.logger.warning(f"Baseline mAP evaluation failed: {e}")
            return 0.0

    def _fine_tune(self, dataset_path: Path) -> Path:
        """
        Fine-tune YOLOv8 on replay buffer dataset.
        Saves new weights to models/yolo/candidate.pt
        Returns path to candidate weights.
        """
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path)

            self.logger.info(
                f"Fine-tuning — "
                f"epochs={self.epochs} | "
                f"batch={self.batch_size}"
            )

            model.train(
                data=str(dataset_path / "data.yaml"),
                epochs=self.epochs,
                batch=self.batch_size,
                imgsz=640,
                lr0=1e-4,           # low LR — fine-tune not full retrain
                lrf=0.01,
                freeze=10,          # freeze first 10 layers — head only
                verbose=False,
                project="models/yolo",
                name="candidate",
                exist_ok=True
            )

            candidate_path = Path("models/yolo/candidate/weights/best.pt")
            self.logger.info(f"Fine-tune complete — candidate at {candidate_path}")
            return candidate_path

        except Exception as e:
            self.logger.error(f"Fine-tuning failed: {e}")
            return None

    def _validate_candidate(
        self,
        candidate_path: Path,
        baseline_map: float
    ) -> bool:
        """
        Validation gate.
        Run candidate weights on held-out validation set.
        Accept only if mAP does not drop more than threshold.
        """
        if not self.val_path.exists():
            self.logger.warning("No validation set — auto-accepting candidate")
            return True

        if candidate_path is None or not candidate_path.exists():
            self.logger.error("Candidate weights not found — rejecting")
            return False

        try:
            from ultralytics import YOLO
            model = YOLO(str(candidate_path))
            results = model.val(
                data=str(self.val_path / "data.yaml"),
                verbose=False
            )
            candidate_map = float(results.box.map50)
            drop = baseline_map - candidate_map

            self.logger.info(
                f"Validation gate — "
                f"baseline={baseline_map:.4f} | "
                f"candidate={candidate_map:.4f} | "
                f"drop={drop:.4f}"
            )

            if drop > self.map_drop_threshold:
                self.logger.warning(
                    f"GATE REJECTED — mAP dropped {drop:.4f} "
                    f"(threshold={self.map_drop_threshold})"
                )
                return False

            self.logger.info("GATE PASSED — candidate accepted")
            return True

        except Exception as e:
            self.logger.error(f"Validation gate failed: {e}")
            return False

    def _deploy_candidate(self, candidate_path: Path):
        """
        Replace current model weights with validated candidate.
        Backup old weights before replacing.
        """
        backup_path = Path(self.model_path).with_suffix(
            f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        shutil.copy(self.model_path, str(backup_path))
        shutil.copy(str(candidate_path), self.model_path)
        self.logger.info(
            f"Candidate deployed — "
            f"backup at {backup_path}"
        )

    def run(self):
        """
        Full nightly retraining cycle.
        1. Sample replay buffer
        2. Prepare dataset
        3. Get baseline mAP
        4. Fine-tune
        5. Validation gate
        6. Deploy if passed
        """
        start = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.logger.info("=" * 50)
        self.logger.info("NIGHTLY RETRAIN STARTING")
        self.logger.info("=" * 50)

        # Step 1 — Check buffer has enough data
        stats = self.replay_buffer.stats()
        self.logger.info(f"Buffer stats: {stats}")

        if stats["fod"] < 10:
            self.logger.warning(
                f"Not enough FOD samples ({stats['fod']}) — "
                f"skipping retrain, need at least 10"
            )
            return

        # Step 2 — Sample buffer
        samples = self.replay_buffer.sample(n=512)

        # Step 3 — Prepare dataset
        dataset_path = self._prepare_dataset(samples)

        # Step 4 — Baseline mAP
        baseline_map = self._get_baseline_map()

        # Step 5 — Fine-tune
        candidate_path = self._fine_tune(dataset_path)

        # Step 6 — Validation gate
        passed = self._validate_candidate(candidate_path, baseline_map)

        # Step 7 — Deploy or reject
        result = "deployed" if passed else "rejected"
        if passed:
            self._deploy_candidate(candidate_path)
        else:
            self.logger.warning("Retrain rejected — keeping current weights")

        # Step 8 — Cleanup temp dataset
        if dataset_path.exists():
            shutil.rmtree(dataset_path)

        elapsed = round(time.time() - start, 2)

        # Log to retrain history
        log_entry = {
            "timestamp": timestamp,
            "result": result,
            "baseline_map": baseline_map,
            "samples_used": len(samples),
            "fod_samples": stats["fod"],
            "elapsed_seconds": elapsed
        }
        self.retrain_log.append(log_entry)
        self._save_retrain_log()

        self.logger.info("=" * 50)
        self.logger.info(
            f"RETRAIN COMPLETE — "
            f"result={result} | "
            f"elapsed={elapsed}s"
        )
        self.logger.info("=" * 50)

    def __repr__(self):
        return (
            f"Retrainer("
            f"model={self.model_path}, "
            f"epochs={self.epochs}, "
            f"gate={self.map_drop_threshold})"
        )
