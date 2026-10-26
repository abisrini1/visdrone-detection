"""Does batching tiles actually make tiled inference faster? Measure it.

    python -m serve.bench_tiles --weights runs/s_960/weights/best.pt --imgsz 960 --batches 1 4 8 16
    python -m serve.bench_tiles --weights exports/s_960_960_trt-fp16_dyn16.engine --imgsz 960

Runs predict_tiled with a different number of tiles per model call and records latency for
each. A fixed-batch-1 engine (from src.export) is expected to fail for batch > 1; that failure
is recorded in the CSV rather than hidden. Rows go to results/serve_tile_batch.csv.
"""
from __future__ import annotations

import argparse

from src.benchmark import infer_labels, summarize, time_calls
from src.tiling import predict_tiled
from src.utils import RESULTS_DIR, append_csv, list_images, parse_device, visdrone_images_dir


def main():
    import cv2
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--mode", default="tiled+full", choices=["tiled", "tiled+full"])
    ap.add_argument("--n", type=int, default=50, help="number of images to time")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--split", default="val")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--half", action="store_true", help="FP16 for .pt weights")
    args = ap.parse_args()

    device = parse_device(args.device)
    paths = list_images(visdrone_images_dir(args.split, args.data_dir), args.n)
    images = [cv2.imread(str(p)) for p in paths]
    model = YOLO(args.weights, task="detect")
    use_half = args.half and args.weights.endswith(".pt")
    fmt, prec = infer_labels(args.weights, args.half)

    for tb in args.batches:
        row = {"weights": args.weights, "format": fmt, "precision": prec, "imgsz": args.imgsz,
               "mode": args.mode, "tile_batch": tb, "images": len(images), "repeats": args.repeats}
        try:
            def fn(img, tb=tb):
                predict_tiled(model, img, tile=args.tile, overlap=args.overlap, imgsz=args.imgsz,
                              conf=args.conf, iou=args.iou, max_det=300,
                              include_full=(args.mode == "tiled+full"), merge_metric="ios",
                              merge_thr=0.6, device=device, half=use_half, tile_batch=tb)

            row.update(summarize(time_calls(fn, images, args.warmup, args.repeats)))
        except Exception as err:
            row["error"] = str(err)[:200]
        append_csv(RESULTS_DIR / "serve_tile_batch.csv", row)
        print({k: v for k, v in row.items() if k in ("tile_batch", "latency_ms_mean", "latency_ms_p95", "error")})


if __name__ == "__main__":
    main()
