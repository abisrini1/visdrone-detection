"""Export dynamic-batch ONNX / TensorRT models for serving.

    python -m serve.export_dynamic --weights runs/s_960/weights/best.pt --imgsz 960 \
        --formats onnx trt-fp16 trt-int8 --max-batch 16

Why a second export script: src/export.py uses a fixed batch of 1, which is right for the
single-image latency study but means an engine cannot take a batch of tiles. Here the models
accept 1..max-batch images, so tiled inference can send all tiles in one call.

INT8 calibrates on images from VisDrone (--data). --calib-fraction below 1.0 calibrates on a
subset; it needs a recent Ultralytics release that supports `fraction` in export, so drop the
flag if your version rejects it. Report the INT8 accuracy drop with src.benchmark --val.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.export import run_name
from src.utils import REPO_ROOT, parse_device

PRESETS = {
    "onnx": dict(format="onnx", simplify=True),
    "trt-fp16": dict(format="engine", half=True),
    "trt-int8": dict(format="engine", int8=True),
}


def export_one(weights, tag, imgsz, max_batch, device, data, out_dir, calib_fraction=1.0):
    from ultralytics import YOLO

    weights = Path(weights)
    kwargs = dict(PRESETS[tag])
    if kwargs.get("int8"):
        kwargs["data"] = data
        if calib_fraction < 1.0:
            kwargs["fraction"] = calib_fraction
    # For TensorRT, `batch` is the maximum batch size of the dynamic profile.
    exported = Path(YOLO(str(weights)).export(imgsz=imgsz, device=device, batch=max_batch,
                                              dynamic=True, **kwargs))
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{run_name(weights)}_{imgsz}_{tag}_dyn{max_batch}{exported.suffix}"
    shutil.move(str(exported), dst)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--formats", nargs="+", default=["onnx", "trt-fp16"], choices=list(PRESETS))
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--data", default="VisDrone.yaml")
    ap.add_argument("--calib-fraction", type=float, default=1.0)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "exports"))
    args = ap.parse_args()

    for tag in args.formats:
        try:
            dst = export_one(args.weights, tag, args.imgsz, args.max_batch, parse_device(args.device),
                             args.data, Path(args.out_dir), args.calib_fraction)
            print(f"[{tag}] -> {dst}")
        except Exception as err:  # keep going so one missing dependency doesn't stop the rest
            print(f"[{tag}] FAILED: {err}")


if __name__ == "__main__":
    main()
