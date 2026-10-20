import numpy as np
import pytest

from src.tiling import merge_detections, nms_np, tile_grid


def test_grid_covers_every_pixel():
    h, w = 1500, 2000
    covered = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in tile_grid(h, w, tile=640, overlap=0.25):
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        covered[y0:y1, x0:x1] = True
    assert covered.all()


def test_last_tile_is_aligned_to_the_edge_and_full_size():
    tiles = tile_grid(1000, 1000, tile=640, overlap=0.25)
    assert max(t[2] for t in tiles) == 1000 and max(t[3] for t in tiles) == 1000
    assert all((t[2] - t[0], t[3] - t[1]) == (640, 640) for t in tiles)


def test_small_image_gets_a_single_tile():
    assert tile_grid(300, 400, tile=640, overlap=0.25) == [(0, 0, 400, 300)]


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        tile_grid(100, 100, 50, overlap=1.0)


def test_nms_suppresses_duplicates_and_keeps_best():
    boxes = np.array([[0, 0, 100, 100], [2, 2, 102, 102], [300, 300, 400, 400]], dtype=float)
    scores = np.array([0.6, 0.9, 0.5])
    keep = nms_np(boxes, scores, thr=0.5)
    assert sorted(keep.tolist()) == [1, 2]


def test_ios_suppresses_a_box_cut_by_a_tile_border_but_iou_does_not():
    full = [0, 0, 100, 100]
    cut = [0, 0, 100, 40]  # same object, truncated by a tile edge: IoU = 0.4, IOS = 1.0
    boxes = np.array([full, cut], dtype=float)
    scores = np.array([0.9, 0.8])
    assert len(nms_np(boxes, scores, thr=0.5, metric="iou")) == 2
    assert nms_np(boxes, scores, thr=0.5, metric="ios").tolist() == [0]


def test_merge_is_class_aware():
    boxes = np.array([[0, 0, 100, 100], [0, 0, 100, 100]], dtype=float)
    scores = np.array([0.9, 0.8])
    classes = np.array([0, 1])
    b, s, c = merge_detections(boxes, scores, classes, thr=0.5)
    assert len(b) == 2


def test_merge_respects_max_det_and_sorts_by_score():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 5000, size=(50, 2))
    boxes = np.concatenate([xy, xy + 20], axis=1)
    scores = rng.uniform(0.1, 1.0, size=50)
    classes = np.zeros(50, dtype=int)
    b, s, c = merge_detections(boxes, scores, classes, thr=0.5, max_det=10)
    assert len(b) == 10 and np.all(np.diff(s) <= 0)


def test_merge_handles_empty_input():
    b, s, c = merge_detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))
    assert len(b) == len(s) == len(c) == 0
