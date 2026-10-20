"""Benchmark latency (and optionally accuracy) of PyTorch / ONNX / TensorRT weights.

    python -m src.benchmark --weights runs/s_640/weights/best.pt --imgsz 640 --val
    python -m src.benchmark --weights exports/s_640_640_trt-fp16.engine --imgsz 640 --val
    python -m src.benchmark --weights exports/s_640_640_onnx.onnx --imgsz 640 --device cpu
    python -m src.benchmark --weights runs/s_640/weights/best.pt --mode tiled+full --merge-metric ios

Latency is measured end to end per image (preprocess + inference + postprocess, plus tiling
and merging in the tiled modes) on images already in memory. A warm-up comes first, then
several repeated passes over the images. Laptop GPUs throttle, so the spread (std, p95)
matters as much as the mean: keep the laptop plugged in and other apps closed.

Deployment settings are used by default (conf 0.25, NMS IoU 0.7), unlike the accuracy
evaluation (conf 0.001), so the latency reflects what a real pipeline would pay.
Results are appended to results/benchmark.csv. Batch size is always 1.
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np

from src.tiling import predict_tiled, tile_grid
from src.utils import RESULTS_DIR, append_csv, list_images, parse_device, visdrone_images_dir


def infer_labels(weights: str, half: bool):
    p = Path(weights)
    fmt = {".pt": "pytorch", ".onnx": "onnx", ".engine": "tensorrt"}.get(p.suffix, p.suffix.lstrip("."))
    name = p.stem.lower()
    if "int8" in name:
        prec = "int8"
    elif "fp16" in name or (fmt == "pytorch" and half):
        prec = "fp16"
    else:
        prec = "fp32"
    return fmt, prec


def gpu_free_mib():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.mem_get_info()[0] / 2**20
    except Exception:
        pass
    return None


def time_calls(fn, images, warmup: int = 10, repeats: int = 3) -> np.ndarray:
    """Per-call wall time in ms: `warmup` untimed calls, then `repeats` timed passes over images."""
    if not images:
        raise ValueError("no images to benchmark")
    for i in range(warmup):
        fn(images[i % len(images)])
    wall = []
    for _ in range(repeats):
        for img in images:
            t0 = time.perf_counter()
            fn(img)
            wall.append((time.perf_counter() - t0) * 1000.0)
    return np.asarray(wall)


def summarize(wall: np.ndarray) -> dict:
    mean = float(wall.mean())
    return {
        "latency_ms_mean": round(mean, 2),
        "latency_ms_p50": round(float(np.percentile(wall, 50)), 2),
        "latency_ms_p95": round(float(np.percentile(wall, 95)), 2),
        "latency_ms_std": round(float(wall.std()), 2),
        "fps": round(1000.0 / mean, 1),
    }


def benchmark(weights, imgsz, device, n=100, warmup=10, split="val", data_dir=None, half=False,
              mode="full", conf=0.25, iou=0.7, max_det=300, tile=640, overlap=0.25,
              merge_metric="iou", merge_thr=0.6, repeats=3):
    import cv2
    from ultralytics import YOLO

    paths = list_images(visdrone_images_dir(split, data_dir), n)
    images = [cv2.imread(str(p)) for p in paths]

    free_before = gpu_free_mib()
    model = YOLO(str(weights), task="detect")
    use_half = half and str(weights).endswith(".pt")
    kw = dict(imgsz=imgsz, device=device, verbose=False, conf=conf, iou=iou, max_det=max_det)
    if use_half:
        kw["half"] = True  # only passed when requested: `half` is deprecated in recent Ultralytics

    inference_ms = []
    if mode == "full":
        def fn(img):
            inference_ms.append(model.predict(img, **kw)[0].speed["inference"])
    else:
        def fn(img):
            predict_tiled(model, img, tile=tile, overlap=overlap, imgsz=imgsz, conf=conf, iou=iou,
                          max_det=max_det, include_full=(mode == "tiled+full"),
                          merge_metric=merge_metric, merge_thr=merge_thr, device=device, half=use_half)

    wall = time_calls(fn, images, warmup, repeats)
    free_after = gpu_free_mib()

    if mode == "full":
        passes = 1.0
    else:
        extra = 1 if mode == "tiled+full" else 0
        passes = float(np.mean([len(tile_grid(im.shape[0], im.shape[1], tile, overlap)) + extra for im in images]))

    fmt, prec = infer_labels(str(weights), half)
    gpu_name = ""
    try:
        import torch

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except Exception:
        pass
    row = {
        "weights": str(weights), "format": fmt, "precision": prec, "imgsz": imgsz,
        "device": str(device), "batch": 1, "images": len(images), "repeats": repeats,
        "mode": mode, "conf": conf, "model_passes_per_image": round(passes, 1),
        **summarize(wall),
        "inference_ms_mean": round(float(np.mean(inference_ms[-len(wall):])), 2) if mode == "full" else None,
        "gpu_mem_used_mib": round(free_before - free_after) if free_before and free_after else None,
        "gpu": gpu_name, "cpu": platform.processor(),
    }
    if mode != "full":
        row.update(tile=tile, overlap=overlap, merge_metric=merge_metric, merge_thr=merge_thr)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--mode", default="full", choices=["full", "tiled", "tiled+full"])
    ap.add_argument("--n", type=int, default=100, help="number of images to time")
    ap.add_argument("--repeats", type=int, default=3, help="timed passes over the images")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--split", default="val")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--merge-metric", default="iou", choices=["iou", "ios"])
    ap.add_argument("--merge-thr", type=float, default=0.6)
    ap.add_argument("--half", action="store_true", help="FP16 for .pt weights (exports bake in their precision)")
    ap.add_argument("--val", action="store_true", help="also measure mAP (whole-image mode only)")
    ap.add_argument("--data", default="VisDrone.yaml")
    args = ap.parse_args()
    device = parse_device(args.device)

    row = benchmark(args.weights, args.imgsz, device, args.n, args.warmup, args.split, args.data_dir,
                    args.half, args.mode, args.conf, args.iou, args.max_det, args.tile, args.overlap,
                    args.merge_metric, args.merge_thr, args.repeats)
    if args.val:
        if args.mode != "full":
            print("note: --val only applies to whole-image mode; use src.eval_tiled for tiled accuracy.")
        else:
            from src.evaluate import evaluate_weights

            res = evaluate_weights(args.weights, args.data, args.imgsz, args.split, device, batch=1,
                                   half=args.half and args.weights.endswith(".pt"))
            row["mAP50"] = round(res["mAP50"], 4)
            row["mAP50_95"] = round(res["mAP50_95"], 4)
    append_csv(RESULTS_DIR / "benchmark.csv", row)
    for key, value in row.items():
        print(f"{key:<22}{value}")


if __name__ == "__main__":
    main()
