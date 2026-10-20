"""Detection metrics in plain NumPy.

COCO-style AP: 101-point interpolated precision, averaged over IoU 0.50:0.95.
Used to compare full-image vs tiled inference with exactly the same evaluator.
Numbers can differ slightly from Ultralytics' own `model.val()`, so always compare
methods within this evaluator, and use `model.val()` for headline training results.
"""
from __future__ import annotations

import numpy as np

IOUV = np.linspace(0.5, 0.95, 10)

# Object-size buckets by sqrt(box area) in pixels (COCO convention).
SIZE_BINS = {"small": (0.0, 32.0), "medium": (32.0, 96.0), "large": (96.0, float("inf"))}


def box_iou_np(a, b) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4) in xyxy format -> (N,M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def match_predictions(pred_boxes, pred_cls, gt_boxes, gt_cls, iouv=IOUV) -> np.ndarray:
    """For each prediction and IoU threshold, is it a true positive? -> (P, len(iouv)) bool.

    Greedy one-to-one matching by descending IoU; a prediction only matches a
    ground-truth box of the same class.
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    pred_cls = np.asarray(pred_cls, dtype=int).reshape(-1)
    gt_cls = np.asarray(gt_cls, dtype=int).reshape(-1)
    correct = np.zeros((len(pred_boxes), len(iouv)), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return correct

    iou = box_iou_np(gt_boxes, pred_boxes) * (gt_cls[:, None] == pred_cls[None, :])
    for j, thr in enumerate(iouv):
        gi, pi = np.nonzero(iou >= thr)
        if len(gi) == 0:
            continue
        order = np.argsort(-iou[gi, pi], kind="stable")
        gi, pi = gi[order], pi[order]
        _, first = np.unique(pi, return_index=True)  # best GT for each prediction
        gi, pi = gi[first], pi[first]
        order = np.argsort(-iou[gi, pi], kind="stable")
        gi, pi = gi[order], pi[order]
        _, first = np.unique(gi, return_index=True)  # best prediction for each GT
        correct[pi[first], j] = True
    return correct


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated area under the precision-recall curve."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))  # precision envelope
    x = np.linspace(0.0, 1.0, 101)
    y = np.interp(x, mrec, mpre)
    return float(((y[:-1] + y[1:]) / 2.0 * np.diff(x)).sum())


def ap_per_class(tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, target_cls: np.ndarray):
    """Per-class AP at every IoU threshold. Returns (classes, ap[len(classes), n_iou])."""
    order = np.argsort(-conf, kind="stable")
    tp, pred_cls = tp[order], pred_cls[order]
    classes = np.unique(target_cls)
    ap = np.zeros((len(classes), tp.shape[1]))
    for ci, c in enumerate(classes):
        mask = pred_cls == c
        n_gt = int((target_cls == c).sum())
        if n_gt == 0 or not mask.any():
            continue
        tpc = np.cumsum(tp[mask], axis=0)
        fpc = np.cumsum(~tp[mask], axis=0)
        recall = tpc / (n_gt + 1e-16)
        precision = tpc / np.maximum(tpc + fpc, 1e-16)
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    return classes, ap


class DetectionEvaluator:
    """Accumulate per-image detections, then compute mAP, per-class AP and recall by object size."""

    def __init__(self, num_classes: int, names: dict | None = None, recall_conf: float = 0.25):
        self.nc = num_classes
        self.names = names or {i: str(i) for i in range(num_classes)}
        self.recall_conf = recall_conf
        self._tp, self._conf, self._pred_cls, self._target_cls = [], [], [], []
        self._gt_size, self._gt_hit = [], []
        self.n_images = 0

    def update(self, pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls) -> None:
        pb = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
        ps = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
        pc = np.asarray(pred_cls, dtype=int).reshape(-1)
        gb = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
        gc = np.asarray(gt_cls, dtype=int).reshape(-1)

        self._tp.append(match_predictions(pb, pc, gb, gc))
        self._conf.append(ps)
        self._pred_cls.append(pc)
        self._target_cls.append(gc)
        self.n_images += 1

        if len(gb):
            size = np.sqrt(((gb[:, 2] - gb[:, 0]).clip(0)) * ((gb[:, 3] - gb[:, 1]).clip(0)))
            keep = ps >= self.recall_conf
            if keep.any():
                iou = box_iou_np(gb, pb[keep]) * (gc[:, None] == pc[keep][None, :])
                hit = (iou >= 0.5).any(axis=1)
            else:
                hit = np.zeros(len(gb), dtype=bool)
            self._gt_size.append(size)
            self._gt_hit.append(hit)

    def compute(self) -> dict:
        tp = np.concatenate(self._tp) if self._tp else np.zeros((0, len(IOUV)), dtype=bool)
        conf = np.concatenate(self._conf) if self._conf else np.zeros(0)
        pred_cls = np.concatenate(self._pred_cls) if self._pred_cls else np.zeros(0, dtype=int)
        target_cls = np.concatenate(self._target_cls) if self._target_cls else np.zeros(0, dtype=int)

        out = {"n_images": self.n_images, "n_gt": int(len(target_cls)), "n_pred": int(len(conf))}
        if len(target_cls) == 0:
            out.update(mAP50=0.0, mAP50_95=0.0, per_class={}, recall_by_size={})
            return out

        classes, ap = ap_per_class(tp, conf, pred_cls, target_cls)
        out["mAP50"] = float(ap[:, 0].mean())
        out["mAP50_95"] = float(ap.mean())
        out["per_class"] = {
            self.names.get(int(c), str(int(c))): {"AP50": float(ap[i, 0]), "AP50_95": float(ap[i].mean())}
            for i, c in enumerate(classes)
        }

        sizes = np.concatenate(self._gt_size)
        hits = np.concatenate(self._gt_hit)
        by_size = {}
        for label, (lo, hi) in SIZE_BINS.items():
            m = (sizes >= lo) & (sizes < hi)
            by_size[label] = {"n_gt": int(m.sum()), "recall@0.5IoU": float(hits[m].mean()) if m.any() else None}
        out["recall_by_size"] = by_size
        out["recall_note"] = f"any same-class prediction with conf>={self.recall_conf} and IoU>=0.5"
        return out
