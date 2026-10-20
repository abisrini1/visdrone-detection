"""Run detection + ByteTrack on a video and report simple tracking statistics.

    python -m src.track --weights runs/s_640/weights/best.pt --source drone.mp4 --imgsz 960

Writes an annotated video and a JSON summary to results/. The statistics are proxies
(unique IDs, track lengths, short tracks), not MOTA/IDF1: for those, run TrackEval on a
labelled sequence from VisDrone-MOT.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from src.utils import RESULTS_DIR, parse_device


def main():
    import cv2
    import numpy as np
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="video file")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default=None, help="annotated video path (default: results/<video>_tracked.mp4)")
    args = ap.parse_args()

    model = YOLO(args.weights, task="detect")
    src = Path(args.source)
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{src.stem}_tracked.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    lengths = Counter()
    writer = None
    frames = 0
    t0 = time.perf_counter()
    for r in model.track(source=str(src), tracker=args.tracker, persist=True, stream=True,
                         imgsz=args.imgsz, conf=args.conf, device=parse_device(args.device), verbose=False):
        frame = r.plot()
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
        writer.write(frame)
        if r.boxes is not None and r.boxes.id is not None:
            for tid in r.boxes.id.int().cpu().tolist():
                lengths[tid] += 1
        frames += 1
        if args.max_frames and frames >= args.max_frames:
            break
    if writer is not None:
        writer.release()
    elapsed = time.perf_counter() - t0

    lens = np.array(list(lengths.values())) if lengths else np.zeros(0)
    summary = {
        "video": str(src), "frames": frames, "processing_fps": round(frames / max(elapsed, 1e-9), 2),
        "unique_track_ids": int(len(lens)),
        "mean_track_len_frames": round(float(lens.mean()), 1) if len(lens) else 0.0,
        "median_track_len_frames": float(np.median(lens)) if len(lens) else 0.0,
        "short_tracks_lt_5_frames": int((lens < 5).sum()) if len(lens) else 0,
        "tracker": args.tracker, "imgsz": args.imgsz, "conf": args.conf,
    }
    (RESULTS_DIR / f"{src.stem}_tracking.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("annotated video:", out_path)


if __name__ == "__main__":
    main()
