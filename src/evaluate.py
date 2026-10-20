"""Evaluate weights with Ultralytics' validator and save per-class results.

    python -m src.evaluate --weights runs/s_640/weights/best.pt --imgsz 640
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils import RESULTS_DIR, parse_device


def evaluate_weights(weights, data="VisDrone.yaml", imgsz=640, split="val", device=0, batch=8,
                     conf=0.001, iou=0.6, half=False) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights), task="detect")
    extra = {"half": True} if half else {}  # `half` is deprecated in recent Ultralytics
    m = model.val(data=data, imgsz=imgsz, split=split, device=device, batch=batch,
                  conf=conf, iou=iou, plots=False, verbose=False, **extra)
    names = m.names
    out = {
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "inference_ms": float(m.speed.get("inference", float("nan"))),
        "per_class_mAP50_95": {names[i]: float(v) for i, v in enumerate(m.box.maps)},
    }
    if str(weights).endswith(".pt"):
        out["params_M"] = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="VisDrone.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None, help="JSON output path (default: results/eval_<name>.json)")
    args = ap.parse_args()

    res = evaluate_weights(args.weights, args.data, args.imgsz, args.split,
                           parse_device(args.device), args.batch)
    print(f"mAP50 {res['mAP50']:.4f} | mAP50-95 {res['mAP50_95']:.4f} | "
          f"P {res['precision']:.3f} R {res['recall']:.3f} | {res['inference_ms']:.1f} ms/img")
    for k, v in res["per_class_mAP50_95"].items():
        print(f"  {k:<18}{v:.4f}")
    out = Path(args.out) if args.out else RESULTS_DIR / f"eval_{Path(args.weights).stem}_{args.imgsz}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
