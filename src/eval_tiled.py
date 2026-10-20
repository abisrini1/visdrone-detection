"""Compare whole-image vs tiled inference on VisDrone with one shared evaluator.

    python -m src.eval_tiled --weights runs/s_640/weights/best.pt --imgsz 640 --tile 640 --overlap 0.25
    python -m src.eval_tiled --weights ... --limit 50      # quick check on 50 images

Modes: full (normal), tiled (tiles only), tiled+full (tiles plus a downscaled full pass).
Reports mAP, per-class AP, recall by object size (small/medium/large) and seconds per image.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.metrics_np import DetectionEvaluator
from src.tiling import predict_full, predict_tiled
from src.utils import (RESULTS_DIR, append_csv, label_path_for, list_images, load_yolo_labels,
                       parse_device, visdrone_images_dir)


def evaluate_mode(model, mode, images, args):
    import cv2

    ev = DetectionEvaluator(len(model.names), model.names)

    def infer(img):
        if mode == "full":
            return predict_full(model, img, args.imgsz, args.conf, args.iou,
                                args.max_det, args.device, args.half)
        return predict_tiled(
            model, img, tile=args.tile, overlap=args.overlap, imgsz=args.imgsz, conf=args.conf,
            iou=args.iou, max_det=args.max_det, include_full=(mode == "tiled+full"),
            merge_metric=args.merge_metric, merge_thr=args.merge_thr, device=args.device,
            half=args.half)

    # Warm-up (CUDA kernels, cuDNN autotune) so the first mode isn't charged for it.
    # Not timed and not scored.
    warm = cv2.imread(str(images[0]))
    for _ in range(3):
        infer(warm)

    t_total = 0.0
    for n, path in enumerate(images, 1):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_boxes, gt_cls = load_yolo_labels(label_path_for(path), w, h)

        t0 = time.perf_counter()
        boxes, scores, cls = infer(img)
        t_total += time.perf_counter() - t0

        ev.update(boxes, scores, cls, gt_boxes, gt_cls)
        if n % 50 == 0:
            print(f"  {mode}: {n}/{len(images)} images")
    res = ev.compute()
    res["sec_per_image"] = round(t_total / max(ev.n_images, 1), 4)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--data-dir", default=None, help="the VisDrone folder (default: Ultralytics datasets_dir/VisDrone)")
    ap.add_argument("--modes", nargs="+", default=["full", "tiled", "tiled+full"],
                    choices=["full", "tiled", "tiled+full"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6, help="per-pass NMS IoU inside the detector")
    ap.add_argument("--merge-metric", default="iou", choices=["iou", "ios"])
    ap.add_argument("--merge-thr", type=float, default=0.6)
    ap.add_argument("--max-det", type=int, default=1000,
                    help="VisDrone images hold up to ~900 objects, so the usual 300 would cap recall")
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only evaluate the first N images")
    ap.add_argument("--tag", default="", help="label stored with the results")
    args = ap.parse_args()
    args.device = parse_device(args.device)

    from ultralytics import YOLO

    model = YOLO(args.weights, task="detect")
    images = list_images(visdrone_images_dir(args.split, args.data_dir), args.limit)
    print(f"{len(images)} images, modes: {args.modes}")

    all_results = {}
    for mode in args.modes:
        res = evaluate_mode(model, mode, images, args)
        all_results[mode] = res
        rs = res["recall_by_size"]
        print(f"{mode:<11} mAP50 {res['mAP50']:.4f} | mAP50-95 {res['mAP50_95']:.4f} | "
              f"recall small {rs['small']['recall@0.5IoU']} | {res['sec_per_image']:.3f} s/img")
        append_csv(RESULTS_DIR / "tiled.csv", {
            "weights": args.weights, "tag": args.tag, "split": args.split, "mode": mode,
            "imgsz": args.imgsz,
            "tile": args.tile, "overlap": args.overlap, "merge_metric": args.merge_metric,
            "merge_thr": args.merge_thr, "images": res["n_images"],
            "mAP50": round(res["mAP50"], 4), "mAP50_95": round(res["mAP50_95"], 4),
            "recall_small": rs["small"]["recall@0.5IoU"], "recall_medium": rs["medium"]["recall@0.5IoU"],
            "recall_large": rs["large"]["recall@0.5IoU"], "sec_per_image": res["sec_per_image"],
        })

    # One file per configuration, so a later run never overwrites an earlier one.
    parts = [Path(args.weights).parent.parent.name, f"tile{args.tile}", f"ov{args.overlap}",
             f"{args.merge_metric}{args.merge_thr}"]
    if args.split != "val":
        parts.append(args.split)
    if args.tag:
        parts.append(args.tag)
    out = RESULTS_DIR / ("tiled_" + "_".join(parts) + ".json")
    out.write_text(json.dumps(all_results, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
