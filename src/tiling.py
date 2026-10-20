"""Tiled (sliced) inference for small-object detection.

Drone images are large and objects are tiny. Resizing a 2000x1500 image down to 640
shrinks a 20 px pedestrian to ~6 px. Instead we run the detector on overlapping
native-resolution tiles, shift the boxes back into full-image coordinates, and merge
duplicates with class-aware NMS. Implemented from scratch (inspired by SAHI).
"""
from __future__ import annotations

import numpy as np


def tile_grid(height: int, width: int, tile: int = 640, overlap: float = 0.25):
    """Return (x0, y0, x1, y1) tiles covering the whole image, last tile aligned to the edge."""
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    stride = max(int(tile * (1.0 - overlap)), 1)

    def starts(length: int):
        if length <= tile:
            return [0]
        s = list(range(0, length - tile + 1, stride))
        if s[-1] != length - tile:
            s.append(length - tile)
        return s

    return [
        (x, y, min(x + tile, width), min(y + tile, height))
        for y in starts(height)
        for x in starts(width)
    ]


def nms_np(boxes: np.ndarray, scores: np.ndarray, thr: float, metric: str = "iou") -> np.ndarray:
    """Greedy NMS. metric 'iou' = intersection/union; 'ios' = intersection/smaller area
    (suppresses boxes cut by a tile border that sit inside a full box). Returns kept indices."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if metric not in ("iou", "ios"):
        raise ValueError("metric must be 'iou' or 'ios'")
    areas = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        lt = np.maximum(boxes[i, :2], boxes[rest, :2])
        rb = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        wh = (rb - lt).clip(0)
        inter = wh[:, 0] * wh[:, 1]
        if metric == "iou":
            denom = areas[i] + areas[rest] - inter
        else:
            denom = np.minimum(areas[i], areas[rest])
        overlap = inter / np.maximum(denom, 1e-9)
        order = rest[overlap <= thr]
    return np.asarray(keep, dtype=int)


def merge_detections(boxes, scores, classes, thr: float = 0.6, metric: str = "iou", max_det: int = 300):
    """Class-aware NMS over detections gathered from all tiles; keep the top max_det by score."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    classes = np.asarray(classes, dtype=int).reshape(-1)
    if len(boxes) == 0:
        return boxes, scores, classes
    kept = []
    for c in np.unique(classes):
        idx = np.nonzero(classes == c)[0]
        kept.extend(idx[nms_np(boxes[idx], scores[idx], thr, metric)].tolist())
    kept = np.asarray(kept, dtype=int)
    kept = kept[np.argsort(-scores[kept], kind="stable")][:max_det]
    return boxes[kept], scores[kept], classes[kept]


def _collect(result, dx, dy, boxes_all, scores_all, cls_all):
    b = getattr(result, "boxes", None)
    if b is None or len(b) == 0:
        return
    xyxy = b.xyxy.cpu().numpy().astype(np.float64)
    xyxy[:, [0, 2]] += dx
    xyxy[:, [1, 3]] += dy
    boxes_all.append(xyxy)
    scores_all.append(b.conf.cpu().numpy().astype(np.float64))
    cls_all.append(b.cls.cpu().numpy().astype(int))


def _half_kw(half):
    # `half` is deprecated in recent Ultralytics releases, so only pass it when requested.
    return {"half": True} if half else {}


def predict_full(model, image, imgsz=640, conf=0.001, iou=0.6, max_det=1000, device=0, half=False):
    """Standard whole-image inference. Returns (boxes xyxy, scores, classes) as NumPy arrays."""
    r = model.predict(image, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                      device=device, verbose=False, **_half_kw(half))[0]
    boxes_all, scores_all, cls_all = [], [], []
    _collect(r, 0, 0, boxes_all, scores_all, cls_all)
    if not boxes_all:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    return boxes_all[0], scores_all[0], cls_all[0]


def predict_tiled(model, image, tile=640, overlap=0.25, imgsz=640, conf=0.001, iou=0.6,
                  max_det=1000, include_full=False, merge_metric="iou", merge_thr=0.6,
                  device=0, half=False, tile_batch=16):
    """Tiled inference. `image` is a BGR NumPy array (as from cv2.imread).

    Returns (boxes xyxy, scores, classes) in full-image pixel coordinates.
    """
    h, w = image.shape[:2]
    boxes_all, scores_all, cls_all = [], [], []
    grid = tile_grid(h, w, tile, overlap)
    for s in range(0, len(grid), tile_batch):
        chunk = grid[s:s + tile_batch]
        crops = [np.ascontiguousarray(image[y0:y1, x0:x1]) for (x0, y0, x1, y1) in chunk]
        results = model.predict(crops, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                                device=device, verbose=False, **_half_kw(half))
        for (x0, y0, _, _), r in zip(chunk, results):
            _collect(r, x0, y0, boxes_all, scores_all, cls_all)
    if include_full:
        r = model.predict(image, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                          device=device, verbose=False, **_half_kw(half))[0]
        _collect(r, 0, 0, boxes_all, scores_all, cls_all)
    if not boxes_all:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    return merge_detections(np.concatenate(boxes_all), np.concatenate(scores_all),
                            np.concatenate(cls_all), merge_thr, merge_metric, max_det)
