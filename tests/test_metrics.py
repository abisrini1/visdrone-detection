import numpy as np
import pytest

from src.metrics_np import DetectionEvaluator, box_iou_np, compute_ap, match_predictions


def test_iou_basic_cases():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    assert box_iou_np(a, a)[0, 0] == pytest.approx(1.0)
    assert box_iou_np(a, [[20, 20, 30, 30]])[0, 0] == 0.0
    assert box_iou_np(a, [[0, 0, 10, 5]])[0, 0] == pytest.approx(0.5)


def test_compute_ap_perfect_and_empty_curves():
    assert compute_ap(np.array([1.0]), np.array([1.0])) == pytest.approx(1.0, abs=0.01)
    assert compute_ap(np.array([0.0]), np.array([0.0])) == pytest.approx(0.0, abs=0.01)


def test_matching_is_one_to_one():
    gt = [[0, 0, 100, 100]]
    preds = [[0, 0, 100, 100], [1, 1, 101, 101]]
    correct = match_predictions(preds, [0, 0], gt, [0])
    assert correct[:, 0].sum() == 1  # the duplicate is a false positive


def test_matching_requires_same_class():
    correct = match_predictions([[0, 0, 100, 100]], [1], [[0, 0, 100, 100]], [0])
    assert not correct.any()


def test_perfect_predictions_give_full_map():
    ev = DetectionEvaluator(2, {0: "a", 1: "b"})
    for _ in range(3):
        gt = np.array([[0, 0, 100, 100], [200, 200, 260, 280]], dtype=float)
        ev.update(gt, [0.9, 0.8], [0, 1], gt, [0, 1])
    res = ev.compute()
    assert res["mAP50"] == pytest.approx(1.0, abs=0.01)
    assert res["mAP50_95"] == pytest.approx(1.0, abs=0.01)


def test_no_predictions_give_zero_map():
    ev = DetectionEvaluator(1)
    ev.update(np.zeros((0, 4)), [], [], [[0, 0, 50, 50]], [0])
    res = ev.compute()
    assert res["mAP50"] == 0.0 and res["mAP50_95"] == 0.0


def test_loose_box_only_counts_at_low_iou_thresholds():
    # IoU = 0.67: a true positive at thresholds 0.50, 0.55, 0.60, 0.65 -> 4 of 10.
    ev = DetectionEvaluator(1)
    ev.update([[0, 0, 100, 67]], [0.9], [0], [[0, 0, 100, 100]], [0])
    res = ev.compute()
    assert res["mAP50"] == pytest.approx(1.0, abs=0.01)
    assert res["mAP50_95"] == pytest.approx(0.4, abs=0.01)


def test_recall_by_object_size():
    ev = DetectionEvaluator(1)
    small = [0, 0, 20, 20]      # sqrt(area) = 20  -> small
    large = [100, 100, 300, 300]  # sqrt(area) = 200 -> large
    # Detect only the large object.
    ev.update([large], [0.9], [0], [small, large], [0, 0])
    rs = ev.compute()["recall_by_size"]
    assert rs["small"]["recall@0.5IoU"] == 0.0
    assert rs["large"]["recall@0.5IoU"] == 1.0
    assert rs["medium"]["recall@0.5IoU"] is None
