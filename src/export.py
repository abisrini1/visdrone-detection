"""Export trained weights to ONNX / TensorRT for the speed-vs-accuracy study.

    python -m src.export --weights runs/s_640/weights/best.pt --imgsz 640 --formats onnx trt-fp16 trt-int8

TensorRT needs the NVIDIA TensorRT package. If an export fails, the script prints why
and continues with the next format.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.utils import REPO_ROOT, parse_device

PRESETS = {
    "onnx": dict(format="onnx", simplify=True),
    "trt-fp16": dict(format="engine", half=True),
    "trt-int8": dict(format="engine", int8=True),
}


def run_name(weights: Path) -> str:
    return weights.parent.parent.name if weights.parent.name == "weights" else weights.stem


def export_one(weights, tag, imgsz, device, data, out_dir):
    from ultralytics import YOLO

    weights = Path(weights)
    kwargs = dict(PRESETS[tag])
    if kwargs.get("int8"):
        kwargs["data"] = data  # INT8 calibrates on images from this dataset
    exported = Path(YOLO(str(weights)).export(imgsz=imgsz, device=device, batch=1, **kwargs))
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{run_name(weights)}_{imgsz}_{tag}{exported.suffix}"
    shutil.move(str(exported), dst)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--formats", nargs="+", default=["onnx", "trt-fp16"], choices=list(PRESETS))
    ap.add_argument("--device", default="0")
    ap.add_argument("--data", default="VisDrone.yaml")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "exports"))
    args = ap.parse_args()

    for tag in args.formats:
        try:
            dst = export_one(args.weights, tag, args.imgsz, parse_device(args.device), args.data,
                             Path(args.out_dir))
            print(f"[{tag}] -> {dst}")
        except Exception as err:  # keep going so one missing dependency doesn't stop the rest
            print(f"[{tag}] FAILED: {err}")


if __name__ == "__main__":
    main()
