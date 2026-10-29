"""Side-by-side example images: ground truth vs whole-image detection vs tiled detection.

    python -m src.show_examples --weights runs/s_960/weights/best.pt --imgsz 960 --n 4

For each chosen val image this crops the most crowded 640x640 window (so small objects are
visible at README size) and writes docs/examples/example_<k>.png with three panels:

    ground truth | whole image | tiled + full (IoS merge)

Each panel title says how many ground-truth boxes in the window were found (same class,
IoU >= 0.5). Predictions are drawn at a display confidence (default 0.25), not the
evaluation confidence of 0.001, so the picture shows what a real pipeline would output.

Pick specific images with --images 0000001_02999_d_0000005.jpg ..., or let it choose:
--pick crowded (default) spreads over the busier images, --pick random uses --seed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.metrics_np import box_iou_np
from src.utils import (REPO_ROOT, label_path_for, list_images, load_yolo_labels, parse_device,
                       visdrone_images_dir)

GT_COLOR = (80, 200, 80)      # BGR green
PRED_COLOR = (214, 120, 42)   # BGR blue (matches the report palette's first slot)
MISS_COLOR = (52, 73, 232)    # BGR red, ground truth that was not found


def densest_window(boxes: np.ndarray, height: int, width: int, size: int = 640, step: int = 64):
    """Top-left corner (x0, y0) of the size x size window containing the most box centres."""
    size_x, size_y = min(size, width), min(size, height)
    if len(boxes) == 0:
        return 0, 0
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    xs = sorted(set(list(range(0, max(width - size_x, 0) + 1, step)) + [max(width - size_x, 0)]))
    ys = sorted(set(list(range(0, max(height - size_y, 0) + 1, step)) + [max(height - size_y, 0)]))
    best, best_n = (0, 0), -1
    for y0 in ys:
        in_y = (cy >= y0) & (cy < y0 + size_y)
        for x0 in xs:
            n = int((in_y & (cx >= x0) & (cx < x0 + size_x)).sum())
            if n > best_n:
                best, best_n = (x0, y0), n
    return best


def crop_boxes(boxes, classes, x0, y0, size_x, size_y):
    """Keep boxes whose centre is inside the window; shift to window coordinates and clip."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    classes = np.asarray(classes).reshape(-1)
    if len(boxes) == 0:
        return boxes, classes
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    keep = (cx >= x0) & (cx < x0 + size_x) & (cy >= y0) & (cy < y0 + size_y)
    out = boxes[keep] - np.array([x0, y0, x0, y0])
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, size_x)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, size_y)
    return out, classes[keep]


def found_mask(gt_boxes, gt_cls, pred_boxes, pred_cls, iou_thr: float = 0.5) -> np.ndarray:
    """Boolean per ground-truth box: is there a prediction of the same class with IoU >= thr?"""
    gt_boxes = np.asarray(gt_boxes).reshape(-1, 4)
    if len(gt_boxes) == 0:
        return np.zeros(0, dtype=bool)
    pred_boxes = np.asarray(pred_boxes).reshape(-1, 4)
    if len(pred_boxes) == 0:
        return np.zeros(len(gt_boxes), dtype=bool)
    iou = box_iou_np(gt_boxes, pred_boxes)
    same = np.asarray(gt_cls).reshape(-1, 1) == np.asarray(pred_cls).reshape(1, -1)
    return ((iou >= iou_thr) & same).any(axis=1)


def select_images(counts: dict, n: int, pick: str = "crowded", seed: int = 0):
    """counts: {image name: number of ground-truth boxes}. Returns n names, deterministic."""
    names = sorted(counts, key=lambda k: (counts[k], k))
    if not names:
        return []
    n = min(n, len(names))
    if pick == "random":
        rng = np.random.default_rng(seed)
        return [names[i] for i in sorted(rng.choice(len(names), n, replace=False))]
    lo, hi = int(0.70 * (len(names) - 1)), int(0.95 * (len(names) - 1))
    idx = np.linspace(lo, hi, n).round().astype(int)
    return [names[i] for i in dict.fromkeys(idx.tolist())]


def draw_boxes(image, boxes, color, thickness: int = 1, mask=None, miss_color=MISS_COLOR):
    import cv2

    out = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes).reshape(-1, 4)):
        c = color if mask is None or mask[i] else miss_color
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, thickness)
    return out


def panel(image, title: str, banner: int = 34):
    import cv2

    h, w = image.shape[:2]
    canvas = np.full((h + banner, w, 3), 252, dtype=np.uint8)
    canvas[banner:] = image
    cv2.putText(canvas, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (11, 11, 11), 1, cv2.LINE_AA)
    return canvas


def build_example(image, gt_boxes, gt_cls, full_pred, tiled_pred, window: int = 640):
    """Return the 3-panel BGR image for one frame. *_pred are (boxes, scores, classes)."""
    h, w = image.shape[:2]
    x0, y0 = densest_window(gt_boxes, h, w, window)
    sx, sy = min(window, w), min(window, h)
    crop = image[y0:y0 + sy, x0:x0 + sx]
    g_boxes, g_cls = crop_boxes(gt_boxes, gt_cls, x0, y0, sx, sy)

    panels = [panel(draw_boxes(crop, g_boxes, GT_COLOR), f"Ground truth ({len(g_boxes)} objects in view)")]
    for label, (pb, _, pc) in (("Whole image", full_pred), ("Tiled + full (IoS)", tiled_pred)):
        p_boxes, p_cls = crop_boxes(pb, pc, x0, y0, sx, sy)
        found = found_mask(g_boxes, g_cls, p_boxes, p_cls)
        shown = draw_boxes(crop, p_boxes, PRED_COLOR)
        shown = draw_boxes(shown, g_boxes[~found], MISS_COLOR)  # missed ground truth in red
        panels.append(panel(shown, f"{label}: found {int(found.sum())}/{len(g_boxes)}"))
    return np.hstack(panels), (x0, y0)


def main():
    import cv2
    from ultralytics import YOLO

    from src.tiling import predict_full, predict_tiled

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--split", default="val")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--images", nargs="*", default=None, help="image file names to use instead of auto-pick")
    ap.add_argument("--pick", default="crowded", choices=["crowded", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="display confidence")
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--merge-metric", default="ios", choices=["iou", "ios"])
    ap.add_argument("--merge-thr", type=float, default=0.6)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "examples"))
    args = ap.parse_args()
    device = parse_device(args.device)

    model = YOLO(args.weights, task="detect")
    paths = list_images(visdrone_images_dir(args.split, args.data_dir))
    by_name = {p.name: p for p in paths}
    labels = {}
    for p in paths:
        lp = label_path_for(p)
        labels[p.name] = sum(1 for _ in open(lp)) if lp.exists() else 0
    names = args.images or select_images(labels, args.n, args.pick, args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, name in enumerate(names, 1):
        image = cv2.imread(str(by_name[name]))
        h, w = image.shape[:2]
        gt_boxes, gt_cls = load_yolo_labels(label_path_for(by_name[name]), w, h)
        full = predict_full(model, image, args.imgsz, args.conf, args.iou, args.max_det, device)
        tiled = predict_tiled(model, image, args.tile, args.overlap, args.imgsz, args.conf, args.iou,
                              args.max_det, include_full=True, merge_metric=args.merge_metric,
                              merge_thr=args.merge_thr, device=device)
        sheet, (x0, y0) = build_example(image, gt_boxes, gt_cls, full, tiled, args.window)
        path = out_dir / f"example_{k}.png"
        cv2.imwrite(str(path), sheet)
        print(f"{path}  <- {name} (window at x={x0}, y={y0}, {labels[name]} objects in the whole image)")
    print("Green = ground truth, blue = predictions, red = ground-truth objects the method missed.")


if __name__ == "__main__":
    main()
