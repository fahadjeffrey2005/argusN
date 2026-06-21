"""
enhanced/src/pipeline/enhanced_pipeline.py
--------------------------------------------
ENHANCED end-to-end pipeline: Detector → CMC Tracker → PostFilter → Output

Upgrades over SHARP:
  ✓ imgsz=1280 (from config — double SHARP's resolution)
  ✓ Camera motion compensation in tracker (optical flow warps tracks before matching)
  ✓ device=cuda (from config)
  ✓ Full frame passed to tracker every frame for CMC
"""

import time
from pathlib import Path
from typing import Optional
import cv2
import yaml

from ..detection.detector import EnhancedDetector
from ..tracking.tracker import EnhancedTracker
from ..filtering.post_filter import PostFilter


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class EnhancedPipeline:
    def __init__(self, config_path: str):
        cfg = load_config(config_path)

        m   = cfg["model"]
        roi = cfg["roi"]
        trk = cfg["tracker"]
        pf  = cfg["post_filter"]
        self.cfg = cfg

        self.detector = EnhancedDetector(
            model_path  = m["weights"],
            conf        = m["conf"],
            imgsz       = m["imgsz"],
            device      = m["device"],
            top_crop    = roi["top_crop"],
            bot_crop    = roi["bot_crop"],
            left_crop   = roi.get("left_crop", 0.0),
            right_crop  = roi.get("right_crop", 0.0),
        )

        self.tracker = EnhancedTracker(
            confirm_frames       = trk["confirm_frames"],
            small_confirm_frames = trk.get("small_confirm_frames", 3),
            small_area_px        = trk.get("small_area_px", 1000),
            iou_threshold        = trk["iou_threshold"],
            max_miss             = trk["max_miss"],
            max_dist_frac        = trk.get("max_dist_frac", 0.05),
            peak_conf_min        = trk.get("peak_conf_min", 0.35),
            camera_compensation  = trk.get("camera_compensation", True),
            comp_max_corners     = trk.get("comp_max_corners", 200),
            comp_quality         = trk.get("comp_quality", 0.01),
            comp_min_distance    = trk.get("comp_min_distance", 30),
        )

        self.post_filter = PostFilter(
            min_box_area     = pf["min_box_area"],
            max_box_fraction = pf["max_box_fraction"],
            min_aspect       = pf["min_aspect"],
            max_aspect       = pf["max_aspect"],
        )

        self.class_names: dict = self.detector.class_names

    def run_video(self,
                  video_path: str,
                  out_video: Optional[str] = None,
                  out_images: Optional[str] = None,
                  out_labels: Optional[str] = None,
                  extract_fps: Optional[float] = None,
                  show_rejected: bool = False,
                  max_frames: Optional[int] = None) -> dict:

        out_cfg = self.cfg["output"]
        efps    = extract_fps or out_cfg.get("extract_fps", 2.0)
        box_bgr = tuple(out_cfg.get("box_color_bgr", [0, 255, 180]))

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")

        src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        interval = max(1, round(src_fps / efps))

        writer = None
        if out_video:
            Path(out_video).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h)
            )

        if out_images:
            Path(out_images).mkdir(parents=True, exist_ok=True)
        if out_labels:
            Path(out_labels).mkdir(parents=True, exist_ok=True)

        self.tracker.reset(frame_w=w, frame_h=h)

        y_start, y_end, x_start, x_end = self.detector.roi_bounds(h, w)
        roi_area = (y_end - y_start) * (x_end - x_start)
        stem     = Path(video_path).stem

        frame_idx = total_raw = total_confirmed = total_passed = frames_with_dets = saved = 0
        t_start = time.perf_counter()

        print(f"  Processing {total_frames} frames @ {src_fps:.0f}fps"
              + (f" (capped at {max_frames})" if max_frames else "") + "...", flush=True)
        print(f"  imgsz={self.cfg['model']['imgsz']}  conf={self.cfg['model']['conf']}  "
              f"CMC={'ON' if self.cfg['tracker'].get('camera_compensation') else 'OFF'}  "
              f"confirm={self.cfg['tracker']['confirm_frames']}f  "
              f"peak_conf≥{self.cfg['tracker'].get('peak_conf_min', 0.35)}", flush=True)

        while True:
            if max_frames and frame_idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 100 == 0:
                pct = 100 * frame_idx / max(total_frames, 1)
                print(f"  {pct:5.1f}%  frame {frame_idx}  "
                      f"raw={total_raw}  confirmed={total_confirmed}  passed={total_passed}", flush=True)

            raw_dets   = self.detector.detect(frame)
            total_raw += len(raw_dets)

            # Pass full frame to tracker for camera motion compensation
            confirmed  = self.tracker.update(raw_dets, frame=frame, frame_w=w, frame_h=h)
            total_confirmed += len(confirmed)

            final = self.post_filter.apply(confirmed, roi_area)
            total_passed += len(final)

            if final:
                frames_with_dets += 1

            save_this = (frame_idx % interval == 0)

            if writer or save_this:
                vis = frame.copy()

                if show_rejected:
                    rejected = [d for d in confirmed if d not in final]
                    for d in rejected:
                        cv2.rectangle(vis, (d.x1, d.y1), (d.x2, d.y2), (100, 100, 100), 1)

                for d in final:
                    cv2.rectangle(vis, (d.x1, d.y1), (d.x2, d.y2), box_bgr, 2)
                    label = f"{self.class_names.get(d.cls, str(d.cls))} {d.conf:.2f}"
                    cv2.putText(vis, label, (d.x1, max(d.y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_bgr, 2, cv2.LINE_AA)

                if final:
                    cv2.putText(vis, f"!! ENHANCED DETECTION x{len(final)} !!",
                                (12, y_start + 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, box_bgr, 3, cv2.LINE_AA)

                if out_cfg.get("draw_roi_lines", True):
                    cv2.line(vis, (x_start, y_start), (x_end, y_start), (255, 165, 0), 1)
                    cv2.line(vis, (x_start, y_end),   (x_end, y_end),   (255, 165, 0), 1)

                ts = frame_idx / src_fps
                mm, ss = divmod(int(ts), 60)
                hud = (f"ENHANCED  |  {stem}  |  {mm:02d}:{ss:02d}  "
                       f"|  raw={len(raw_dets)}  confirmed={len(confirmed)}  passed={len(final)}")
                cv2.rectangle(vis, (0, 0), (w, 26), (20, 20, 20), -1)
                cv2.putText(vis, hud, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (200, 200, 200), 1, cv2.LINE_AA)

                if writer:
                    writer.write(vis)

                if save_this:
                    fname = f"{stem}_{frame_idx:06d}"
                    if out_images:
                        cv2.imwrite(str(Path(out_images) / f"{fname}.jpg"), frame)
                    if out_labels:
                        lines = []
                        for d in final:
                            cx = ((d.x1 + d.x2) / 2) / w
                            cy = ((d.y1 + d.y2) / 2) / h
                            bw = (d.x2 - d.x1) / w
                            bh = (d.y2 - d.y1) / h
                            lines.append(f"{d.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                        (Path(out_labels) / f"{fname}.txt").write_text("\n".join(lines))
                    saved += 1

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        t_elapsed  = time.perf_counter() - t_start
        duration_s = total_frames / src_fps
        fps        = frame_idx / t_elapsed if t_elapsed > 0 else 0
        latency_ms = (t_elapsed / max(frame_idx, 1)) * 1000

        return {
            "video":             Path(video_path).name,
            "frames":            frame_idx,
            "duration_s":        round(duration_s, 1),
            "total_raw":         total_raw,
            "total_confirmed":   total_confirmed,
            "total_passed":      total_passed,
            "frames_with_dets":  frames_with_dets,
            "dets_per_min":      round(total_passed / (duration_s / 60), 2) if duration_s > 0 else 0,
            "tracker_reduction": f"{100*(1 - total_confirmed/max(total_raw,1)):.0f}%",
            "filter_reduction":  f"{100*(1 - total_passed/max(total_confirmed,1)):.0f}%",
            "total_reduction":   f"{100*(1 - total_passed/max(total_raw,1)):.0f}%",
            "frames_saved":      saved,
            "fps":               round(fps, 1),
            "latency_ms":        round(latency_ms, 2),
            "processing_time_s": round(t_elapsed, 1),
        }
