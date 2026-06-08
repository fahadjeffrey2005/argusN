"""
ARGUS-N — Real-Time Pipeline (v3)

Two-tier architecture for genuine edge-compute real-time performance:

HOT PATH  (every frame, target <15ms on Jetson Orin):
  - Read frame
  - Batched tiled YOLO (8 tiles → 1 GPU batch → NMS)
  - Flow residual (Farneback, CPU, parallel thread)
  - Alert if YOLO fires

BACKGROUND (async threads, never block hot path):
  - PatchCore runs every 30 frames
  - CLIP runs when YOLO fires, on detected crops
  - Results posted to queue → enrich next alert
  - MJPEG HTTP stream — open browser to view live output (no GUI needed)

Target: 25-30 fps on Jetson Orin / Thor
Usage:
    python scripts/run_pipeline_rt.py --source "raw data/recording.mp4"
    python scripts/run_pipeline_rt.py --source "raw data/recording.mp4" --tiles 2x4
    python scripts/run_pipeline_rt.py --source "raw data/recording.mp4" --yolo-only
    python scripts/run_pipeline_rt.py --source "raw data/recording.mp4" --yolo-only --stream-port 8080
"""

import argparse
import json
import sys
import time
import threading
import queue
import socket
from pathlib import Path
from datetime import datetime
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
_CACHE = str(ROOT / "models" / "cache")
os.makedirs(_CACHE, exist_ok=True)
os.environ["TORCH_HOME"]         = _CACHE
os.environ["TRANSFORMERS_CACHE"] = _CACHE
os.environ["HF_HOME"]            = _CACHE

from ultralytics import YOLO as UltralyticsYOLO
from src.utils.config_loader import Config
from src.utils.logger import get_logger
from src.ingestion.multi_camera import MultiCameraIngestion
from src.ingestion.nir_simulator import NIRSimulator


# ── MJPEG HTTP Streamer ────────────────────────────────────────────────────
class MJPEGStreamer:
    """
    Headless real-time display via MJPEG over HTTP.
    Open http://<machine-ip>:<port> in any browser to see the live feed.
    push() is non-blocking — drops frame if consumer is slow, never stalls hot path.
    """

    _HTML = b"""<!DOCTYPE html>
<html><head><title>ARGUS-N Live</title>
<style>body{margin:0;background:#111;display:flex;flex-direction:column;
align-items:center;justify-content:center;min-height:100vh;font-family:monospace}
img{max-width:100%;height:auto;border:2px solid #0f0}
h1{color:#0f0;margin:8px 0 4px}p{color:#888;margin:0 0 8px;font-size:12px}
</style></head>
<body><h1>ARGUS-N RT</h1>
<p>Live detection stream &mdash; refresh if stream stalls</p>
<img src="/stream"></body></html>"""

    def __init__(self, port: int = 8089, quality: int = 75, scale: float = 0.5, ngrok: bool = False):
        self._port    = port
        self._quality = quality
        self._scale   = scale
        self._ngrok   = ngrok
        self._jpeg    = None
        self._lock    = threading.Lock()
        self._server  = None
        self._encode_q = queue.Queue(maxsize=1)  # always latest frame, never blocks hot path

    def push(self, frame: np.ndarray):
        """
        Hand frame off to encode thread. Hot-path safe — returns in microseconds.
        Drops frame if encoder is still busy (always shows latest, never queues up).
        """
        try:
            self._encode_q.get_nowait()   # evict stale frame if present
        except queue.Empty:
            pass
        try:
            self._encode_q.put_nowait(frame)
        except queue.Full:
            pass

    def _encode_loop(self):
        """Background thread: resize + JPEG encode, never touches the hot path."""
        while True:
            frame = self._encode_q.get()
            try:
                if self._scale != 1.0:
                    h, w = frame.shape[:2]
                    frame = cv2.resize(
                        frame,
                        (int(w * self._scale), int(h * self._scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )
                _, buf = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality]
                )
                with self._lock:
                    self._jpeg = buf.tobytes()
            except Exception:
                pass

    def _get_jpeg(self):
        with self._lock:
            return self._jpeg

    def start(self):
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # suppress per-request logs

            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()
                    self.wfile.write(streamer._HTML)

                elif self.path == '/stream':
                    self.send_response(200)
                    self.send_header(
                        'Content-Type',
                        'multipart/x-mixed-replace; boundary=frame'
                    )
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    try:
                        while True:
                            jpeg = streamer._get_jpeg()
                            if jpeg:
                                self.wfile.write(
                                    b'--frame\r\n'
                                    b'Content-Type: image/jpeg\r\n\r\n'
                                    + jpeg + b'\r\n'
                                )
                            time.sleep(0.033)  # ~30fps max to browser
                    except (BrokenPipeError, ConnectionResetError):
                        pass

                else:
                    self.send_error(404)

        # Start background encode thread
        threading.Thread(target=self._encode_loop, daemon=True).start()

        class _Server(HTTPServer):
            allow_reuse_address = True  # must be set before bind, not after

        self._server = _Server(('0.0.0.0', self._port), Handler)
        t = threading.Thread(target=self._server.serve_forever, daemon=True)
        t.start()
        print(f"[STREAM] Local feed  → http://0.0.0.0:{self._port}", flush=True)

        if self._ngrok:
            try:
                from pyngrok import ngrok as _ngrok
                _ngrok.kill()                          # kill any stale tunnel first
                tunnel = _ngrok.connect(self._port, 'http')
                print(f"[STREAM] ngrok feed  → {tunnel.public_url}", flush=True)
            except Exception as e:
                print(f"[STREAM] ngrok failed: {e}  (pip install pyngrok)", flush=True)

    def stop(self):
        if self._server:
            self._server.shutdown()


_fisheye_maps = {}
def undistort_fisheye(frame, strength=0.4):
    h, w = frame.shape[:2]
    key = (h, w)
    if key not in _fisheye_maps:
        K = np.array([[w*0.8, 0, w/2],[0, w*0.8, h/2],[0, 0, 1]], dtype=np.float64)
        D = np.array([strength, 0.0, 0.0, 0.0], dtype=np.float64)
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w,h), cv2.CV_16SC2)
        _fisheye_maps[key] = (m1, m2)
    return cv2.remap(frame, _fisheye_maps[key][0], _fisheye_maps[key][1], cv2.INTER_LINEAR)




# ── Batched tiled YOLO ─────────────────────────────────────────────────────
def yolo_batch_tiles(
    model, frame, conf=0.15,
    tile_rows=2, tile_cols=4,
    overlap=0.15, device="cuda",
    max_box_frac=0.03,
    max_aspect=3.5,
):
    """
    Extract tiles, run ONE batched GPU inference call, map back to frame coords.
    8 tiles → 1 batch → NMS.  Much faster than 8 sequential calls.
    """
    h, w = frame.shape[:2]
    tiles, origins = [], []

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * w / tile_cols)
            y1 = int(row * h / tile_rows)
            x2 = min(w, int((col + 1) * w / tile_cols + w * overlap / tile_cols))
            y2 = min(h, int((row + 1) * h / tile_rows + h * overlap / tile_rows))
            tile = frame[y1:y2, x1:x2]
            if tile.size == 0:
                continue
            tiles.append(tile)
            origins.append((x1, y1))

    if not tiles:
        return []

    # Batch inference — works with dynamic batch engine, falls back to sequential
    try:
        results = model.predict(tiles, conf=conf, verbose=False, device=device)
    except Exception:
        results = [model.predict(t, conf=conf, verbose=False, device=device)[0] for t in tiles]

    all_boxes, all_scores, all_cls = [], [], []
    max_box_area = max_box_frac * w * h

    for tile, result, (ox, oy) in zip(tiles, results if isinstance(results, list) else [results], origins):
        if result.boxes is None:
            continue
        for box in result.boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
            bx1 += ox; by1 += oy; bx2 += ox; by2 += oy
            bw = bx2 - bx1; bh = by2 - by1
            area = bw * bh
            if area > max_box_area:
                continue
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > max_aspect:
                continue
            # Shadow filter: extract tile crop and check contrast
            tx1=max(0,bx1-ox); ty1=max(0,by1-oy); tx2=min(tile.shape[1],bx2-ox); ty2=min(tile.shape[0],by2-oy)
            tile_crop = tile[ty1:ty2, tx1:tx2]
            if tile_crop.size > 0 and cv2.cvtColor(tile_crop, cv2.COLOR_BGR2GRAY).std() < 12:
                continue  # shadow or flat surface — skip
            # Shrink box 15% toward centre for tighter localisation
            cbx = (bx1+bx2)//2; cby = (by1+by2)//2
            bw2 = max(4, int((bx2-bx1)*0.85)//2)
            bh2 = max(4, int((by2-by1)*0.85)//2)
            bx1 = cbx-bw2; bx2 = cbx+bw2
            by1 = cby-bh2; by2 = cby+bh2
            all_boxes.append([bx1, by1, bx2, by2])
            all_scores.append(float(box.conf[0]))
            all_cls.append(int(box.cls[0]))

    if not all_boxes:
        return []

    boxes_t  = torch.tensor(all_boxes,  dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    keep = torchvision.ops.nms(boxes_t, scores_t, iou_threshold=0.45)

    return [
        {"box": tuple(all_boxes[i]), "conf": all_scores[i],
         "cls_name": model.names[all_cls[i]]}
        for i in keep.tolist()
    ]


# ── Background PatchCore worker ────────────────────────────────────────────
class PatchCoreWorker(threading.Thread):
    """Runs PatchCore every N frames in a background thread. Non-blocking."""

    def __init__(self, patchcore, stride=30):
        super().__init__(daemon=True)
        self._pc      = patchcore
        self._stride  = stride
        self._in_q    = queue.Queue(maxsize=2)
        self._out_q   = queue.Queue(maxsize=10)
        self._frame_n = 0

    def submit(self, frame_idx, frame):
        if frame_idx % self._stride != 0:
            return
        if not self._in_q.full():
            self._in_q.put_nowait((frame_idx, frame.copy()))

    def latest_result(self):
        """Returns latest (frame_idx, boxes, score) or None."""
        result = None
        while not self._out_q.empty():
            result = self._out_q.get_nowait()
        return result

    def run(self):
        while True:
            frame_idx, frame = self._in_q.get()
            try:
                score, _ = self._pc.score(frame, return_heatmap=False)
                boxes    = self._pc.get_candidate_regions(frame)
                self._out_q.put_nowait((frame_idx, boxes, score))
            except Exception:
                pass


# ── Background CLIP worker ─────────────────────────────────────────────────
class CLIPWorker(threading.Thread):
    """Runs CLIP on YOLO crops in a background thread. Non-blocking."""

    def __init__(self, clip_clf):
        super().__init__(daemon=True)
        self._clf  = clip_clf
        self._in_q = queue.Queue(maxsize=4)
        self._out_q= queue.Queue(maxsize=20)

    def submit(self, frame_idx, frame, boxes):
        if not self._in_q.full() and boxes:
            self._in_q.put_nowait((frame_idx, frame.copy(), boxes))

    def latest_result(self):
        result = None
        while not self._out_q.empty():
            result = self._out_q.get_nowait()
        return result

    def run(self):
        while True:
            frame_idx, frame, boxes = self._in_q.get()
            try:
                results = self._clf.score_regions(frame, [b["box"] for b in boxes])
                self._out_q.put_nowait((frame_idx, results))
            except Exception:
                pass


# ── Overlay ────────────────────────────────────────────────────────────────
def draw_overlay(frame, detections, fps, frame_idx):
    vis = frame.copy()
    for det in detections:
        x1,y1,x2,y2 = det["box"]
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,0,255), 2)
        label = f"{det['cls_name']} {det['conf']:.2f}"
        cv2.putText(vis, label, (x1, max(0,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    info = f"frame={frame_idx}  FOD={len(detections)}  {fps:.1f}fps"
    cv2.putText(vis, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    return vis


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",       required=True)
    parser.add_argument("--tiles",        default="2x4",
                        help="Tile grid rows x cols (default 2x4)")
    parser.add_argument("--conf",         type=float, default=0.15)
    parser.add_argument("--yolo-only",    action="store_true",
                        help="Skip PatchCore and CLIP (maximum speed)")
    parser.add_argument("--build-bank",   action="store_true")
    parser.add_argument("--bank-frames",  type=int, default=60)
    parser.add_argument("--no-display",   action="store_true",
                        help="Disable MJPEG stream entirely")
    parser.add_argument("--ngrok",        action="store_true",
                        help="Expose stream via ngrok tunnel (requires: pip install pyngrok)")
    parser.add_argument("--stream-port",  type=int, default=8089,
                        help="Port for MJPEG HTTP stream (default 8089)")
    parser.add_argument("--stream-scale", type=float, default=0.5,
                        help="Downscale before encoding for stream (default 0.5)")
    parser.add_argument("--device",       default=None)
    parser.add_argument("--top-crop",     type=float, default=0.22)
    parser.add_argument("--bot-crop",     type=float, default=0.15)
    args = parser.parse_args()

    tile_rows, tile_cols = map(int, args.tiles.lower().split("x"))

    cfg    = Config(str(ROOT / "config" / "config.yaml"))
    log    = get_logger("pipeline_rt",
                        cfg.get("logging","log_path",default="logs/argus.log"),
                        "WARNING")
    device = args.device or cfg.get("device", default="cuda")
    log.warning(f"ARGUS-N RT starting — device={device}  tiles={tile_rows}x{tile_cols}")

    # Output dirs
    out_det  = ROOT / cfg.get("outputs","detections_path",  default="outputs/detections")
    out_anom = ROOT / cfg.get("outputs","anomaly_frames_path",default="outputs/anomaly_frames")
    out_det.mkdir(parents=True, exist_ok=True)
    out_anom.mkdir(parents=True, exist_ok=True)

    # Camera
    camera = MultiCameraIngestion(cfg)
    camera.set_source(args.source)
    nir_sim = NIRSimulator()

    # YOLO — hot path
    yolo_path = str(ROOT / cfg.get("yolo","model_path",
                    default="models/yolo_runs/fod_v3/weights/best.engine"))
    yolo_d = UltralyticsYOLO(yolo_path)
    log.warning(f"YOLO loaded: {yolo_path}")

    # PatchCore + CLIP — background (optional)
    pc_worker   = None
    clip_worker = None

    if not args.yolo_only:
        try:
            from src.anomaly.patchcore import PatchCoreDetector
            bank_path = str(ROOT / "models" / "patchcore_bank.pt")
            patchcore = PatchCoreDetector(device=device, bank_path=bank_path)
            pc_worker = PatchCoreWorker(patchcore, stride=30)
            pc_worker.start()
            log.warning("PatchCore worker started")
        except Exception as e:
            log.warning(f"PatchCore skipped: {e}")

        try:
            from src.semantic.clip_classifier import CLIPClassifier
            clip_clf    = CLIPClassifier(device=device)
            clip_worker = CLIPWorker(clip_clf)
            clip_worker.start()
            log.warning("CLIP worker started")
        except Exception as e:
            log.warning(f"CLIP skipped: {e}")

    # MJPEG stream (replaces cv2.imshow — works headless)
    streamer = None
    if not args.no_display:
        streamer = MJPEGStreamer(
            port=args.stream_port,
            quality=75,
            scale=args.stream_scale,
            ngrok=args.ngrok,
        )
        streamer.start()

    # Warmup + bank build
    warmup_n   = cfg.get("pipeline","warmup_frames",default=60)
    warm_buf   = []
    frame_idx  = 0
    alert_log  = []

    # FPS tracker
    fps_buf = deque(maxlen=30)
    t_last  = time.perf_counter()

    for frames_rgb, frame_nir in camera:
        if not frames_rgb:
            continue
        primary = undistort_fisheye(frames_rgb[0])
        h = primary.shape[0]
        t = int(h * args.top_crop)
        b = int(h * (1 - args.bot_crop))
        roi = primary[t:b, :]
        roi_offset = t

        # Warmup
        if frame_idx < warmup_n:
            warm_buf.append(primary.copy())
            frame_idx += 1
            continue

        # Build bank once
        if frame_idx == warmup_n and pc_worker and (
                args.build_bank or pc_worker._pc.bank is None):
            print(f"Building PatchCore bank from {args.bank_frames} frames...")
            pc_worker._pc.build_memory_bank(warm_buf[:args.bank_frames])
            frame_idx += 1

        # ── HOT PATH ────────────────────────────────────────────────────
        t0 = time.perf_counter()

        _dev = "cuda" if "cuda" in device else device
        detections = yolo_batch_tiles(
            yolo_d, roi,
            conf=args.conf,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            device=_dev,
        )

        hot_ms = (time.perf_counter() - t0) * 1000

        # Map ROI-relative boxes back to full frame coords (applied once)
        detections = [
            {**d, "box": (d["box"][0], d["box"][1] + roi_offset,
                           d["box"][2], d["box"][3] + roi_offset)}
            for d in detections
        ]

        # Submit to background workers (non-blocking)
        if pc_worker:
            pc_worker.submit(frame_idx, primary)
        if clip_worker and detections:
            clip_worker.submit(frame_idx, primary, detections)

        # Collect background results (non-blocking)
        pc_result   = pc_worker.latest_result()   if pc_worker   else None
        clip_result = clip_worker.latest_result()  if clip_worker else None

        # ── FPS ─────────────────────────────────────────────────────────
        now = time.perf_counter()
        fps_buf.append(1.0 / max(now - t_last, 0.001))
        t_last = now
        fps    = sum(fps_buf) / len(fps_buf)

        # ── Progress ─────────────────────────────────────────────────────
        if frame_idx % 30 == 0:
            pc_score = f"{pc_result[2]:.2f}" if pc_result else "-"
            print(f"[{frame_idx:05d}]  {fps:.1f}fps  "
                  f"YOLO={len(detections)}  hot={hot_ms:.1f}ms  "
                  f"PC={pc_score}", flush=True)

        # ── Draw once, reuse for both alert save and stream ───────────────
        vis = None
        if detections or streamer is not None:
            vis = draw_overlay(primary, detections, fps, frame_idx)

        # ── Alert ─────────────────────────────────────────────────────────
        if detections:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log.warning(f"FOD ALERT  frame={frame_idx}  "
                        f"detections={len(detections)}  fps={fps:.1f}")
            for det in detections:
                alert_log.append({
                    "timestamp": ts, "frame": frame_idx,
                    "box": det["box"], "conf": det["conf"],
                    "cls": det["cls_name"], "fps": round(fps,1),
                })
            frame_path = out_anom / f"fod_{ts}.jpg"
            cv2.imwrite(str(frame_path), vis)

        # ── Stream — push() returns in microseconds, encode is background ─
        if streamer is not None:
            streamer.push(vis)

        frame_idx += 1

    # ── Teardown ──────────────────────────────────────────────────────────
    camera.release()
    if streamer:
        streamer.stop()

    log_path = out_det / f"alerts_rt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(alert_log, indent=2))
    avg_fps = sum(fps_buf)/len(fps_buf) if fps_buf else 0
    print(f"\nDone.  Frames={frame_idx}  Alerts={len(alert_log)}  "
          f"Avg FPS={avg_fps:.1f}")
    print(f"Alert log → {log_path}")


if __name__ == "__main__":
    main()
