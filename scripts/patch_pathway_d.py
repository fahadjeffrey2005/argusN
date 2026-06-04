"""
Patches run_pipeline.py and fusion.py to add YOLO as Pathway D.
Run from the argusN root: python scripts/patch_pathway_d.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── 1. fusion.py ──────────────────────────────────────────────────────────
fusion_path = ROOT / "src/fusion/fusion.py"
fusion = fusion_path.read_text()

# Add _normalise_yolo after _normalise_flow
fusion = fusion.replace(
    '    def _normalise_flow(self, flow_boxes) -> List[Dict]:\n'
    '        return [{"box": b, "pathway": "flow", "confidence": 1.0} for b in flow_boxes]',

    '    def _normalise_flow(self, flow_boxes) -> List[Dict]:\n'
    '        return [{"box": b, "pathway": "flow", "confidence": 1.0} for b in flow_boxes]\n'
    '\n'
    '    def _normalise_yolo(self, yolo_detections) -> List[Dict]:\n'
    '        return [\n'
    '            {"box": d["box"], "pathway": "yolo", "confidence": d["conf"]}\n'
    '            for d in yolo_detections\n'
    '        ]'
)

# Add yolo_detections param to fuse()
fusion = fusion.replace(
    '        flow_boxes:       List[tuple],\n'
    '        flow_discarded:   bool = False,\n',

    '        flow_boxes:       List[tuple],\n'
    '        flow_discarded:   bool = False,\n'
    '        yolo_detections:  List[Dict] = None,\n'
)

# Include pd in all_regions
fusion = fusion.replace(
    '        pa = self._normalise_patchcore(patchcore_boxes, patchcore_score)\n'
    '        pb = self._normalise_clip(clip_results)\n'
    '        pc = [] if flow_discarded else self._normalise_flow(flow_boxes)\n'
    '\n'
    '        all_regions = pa + pb + pc',

    '        pa = self._normalise_patchcore(patchcore_boxes, patchcore_score)\n'
    '        pb = self._normalise_clip(clip_results)\n'
    '        pc = [] if flow_discarded else self._normalise_flow(flow_boxes)\n'
    '        pd = self._normalise_yolo(yolo_detections or [])\n'
    '\n'
    '        all_regions = pa + pb + pc + pd'
)

fusion_path.write_text(fusion)
print("OK  src/fusion/fusion.py")

# ── 2. run_pipeline.py ────────────────────────────────────────────────────
pipeline_path = ROOT / "scripts/run_pipeline.py"
pipeline = pipeline_path.read_text()

# Add ultralytics import
pipeline = pipeline.replace(
    'from src.fusion.fusion import AdaptiveFusion',
    'from src.fusion.fusion import AdaptiveFusion\n'
    'from ultralytics import YOLO as UltralyticsYOLO'
)

# Add YOLO init before fusion gate
pipeline = pipeline.replace(
    '    log.info("Initialising fusion gate ...")',
    '    log.info("Initialising YOLO detector (Pathway D) ...")\n'
    '    _yolo_path = str(ROOT / cfg.get("yolo", "model_path",\n'
    '                     default="models/yolo_runs/fod_v2/weights/best.pt"))\n'
    '    try:\n'
    '        yolo_d = UltralyticsYOLO(_yolo_path)\n'
    '        log.info(f"YOLO Pathway D: {_yolo_path}")\n'
    '    except Exception as _e:\n'
    '        yolo_d = None\n'
    '        log.warning(f"YOLO Pathway D not loaded: {_e}")\n'
    '\n'
    '    log.info("Initialising fusion gate ...")'
)

# Add Pathway D detection block
pipeline = pipeline.replace(
    '        # -- Remap floor-crop boxes back to full-frame coords',
    '        # -- Pathway D: YOLO (every frame, ~3ms, supervised) ---------------\n'
    '        pd_result = {"detections": [], "ms": 0.0}\n'
    '        if yolo_d is not None:\n'
    '            _t0 = time.perf_counter()\n'
    '            _dev = "cuda" if "cuda" in device else device\n'
    '            _raw = yolo_d.predict(floor, conf=0.25, verbose=False, device=_dev)\n'
    '            if _raw and _raw[0].boxes is not None:\n'
    '                for _b in _raw[0].boxes:\n'
    '                    _x1,_y1,_x2,_y2 = map(int, _b.xyxy[0].tolist())\n'
    '                    pd_result["detections"].append({\n'
    '                        "box":      (_x1, _y1, _x2, _y2),\n'
    '                        "conf":     float(_b.conf[0]),\n'
    '                        "cls_name": yolo_d.names[int(_b.cls[0])],\n'
    '                    })\n'
    '            pd_result["ms"] = (time.perf_counter() - _t0) * 1000\n'
    '\n'
    '        # -- Remap floor-crop boxes back to full-frame coords'
)

# Remap YOLO boxes to full-frame coords
pipeline = pipeline.replace(
    '        pa_boxes_full     = remap(pa_result["boxes"])\n'
    '        flow_boxes_full   = remap(flow_boxes)\n'
    '        clip_results_full = remap_results(pb_result["results"])',

    '        pa_boxes_full     = remap(pa_result["boxes"])\n'
    '        flow_boxes_full   = remap(flow_boxes)\n'
    '        clip_results_full = remap_results(pb_result["results"])\n'
    '        yolo_dets_full    = [\n'
    '            {**d, "box": (d["box"][0], d["box"][1] + floor_y0,\n'
    '                          d["box"][2], d["box"][3] + floor_y0)}\n'
    '            for d in pd_result["detections"]\n'
    '        ]'
)

# Pass yolo_detections to fusion
pipeline = pipeline.replace(
    '            flow_boxes      = flow_boxes_full,\n'
    '            flow_discarded  = flow_discarded,\n'
    '        )',

    '            flow_boxes      = flow_boxes_full,\n'
    '            flow_discarded  = flow_discarded,\n'
    '            yolo_detections = yolo_dets_full,\n'
    '        )'
)

# Add timing
pipeline = pipeline.replace(
    '        meta["pa_ms"]   = pa_result.get("ms", 0)\n'
    '        meta["pb_ms"]   = pb_result.get("ms", 0)\n'
    '        meta["flow_ms"] = flow_ms',

    '        meta["pa_ms"]   = pa_result.get("ms", 0)\n'
    '        meta["pb_ms"]   = pb_result.get("ms", 0)\n'
    '        meta["pd_ms"]   = pd_result.get("ms", 0)\n'
    '        meta["flow_ms"] = flow_ms'
)

pipeline_path.write_text(pipeline)
print("OK  scripts/run_pipeline.py")
print("\nAll done. Now run:")
print("  python scripts/run_pipeline.py --source 'raw data/recording_20250521_141904.mp4' --build-bank --floor-crop 0.35 --pc-threshold 0.3")
