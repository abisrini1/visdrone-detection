"""predict_tiled with a fake model: checks coordinate offsets and de-duplication without any GPU."""
import numpy as np

from src.tiling import predict_full, predict_tiled, tile_grid


class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _T(xyxy), _T(conf), _T(cls)

    def __len__(self):
        return len(self.xyxy.a)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """'Detects' a 20x20 object wherever a bright 20x20 square exists in the crop it is given."""

    def predict(self, source, **kwargs):
        crops = source if isinstance(source, list) else [source]
        out = []
        for crop in crops:
            ys, xs = np.nonzero(crop[..., 0] == 255)
            if len(xs) == 0:
                out.append(_Result(_Boxes(np.zeros((0, 4)), np.zeros(0), np.zeros(0))))
            else:
                box = [[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]]
                out.append(_Result(_Boxes(box, [0.9], [3])))
        return out


def _image_with_square(h, w, x, y):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[y:y + 20, x:x + 20, 0] = 255
    return img


def test_tiled_boxes_land_in_full_image_coordinates():
    img = _image_with_square(1500, 2000, x=1500, y=1100)
    boxes, scores, cls = predict_tiled(FakeModel(), img, tile=640, overlap=0.25)
    assert len(boxes) == 1  # seen by several overlapping tiles, merged into one
    assert np.allclose(boxes[0], [1500, 1100, 1520, 1120])
    assert cls[0] == 3


def test_object_on_a_tile_seam_is_found_and_deduplicated():
    tiles = tile_grid(1500, 2000, 640, 0.25)
    seam_x = tiles[1][0] + 5  # a column covered by two tiles
    img = _image_with_square(1500, 2000, x=seam_x, y=200)
    boxes, _, _ = predict_tiled(FakeModel(), img, tile=640, overlap=0.25)
    assert len(boxes) == 1
    assert np.allclose(boxes[0], [seam_x, 200, seam_x + 20, 220])


def test_no_objects_returns_empty_arrays():
    img = np.zeros((800, 800, 3), dtype=np.uint8)
    boxes, scores, cls = predict_tiled(FakeModel(), img, tile=640, overlap=0.25)
    assert boxes.shape == (0, 4) and len(scores) == 0 and len(cls) == 0


def test_include_full_adds_the_whole_image_pass_and_still_deduplicates():
    img = _image_with_square(1000, 1000, x=300, y=300)
    boxes, _, _ = predict_tiled(FakeModel(), img, tile=640, overlap=0.25, include_full=True)
    assert len(boxes) == 1


def test_predict_full_matches_tiled_on_a_single_tile_image():
    img = _image_with_square(500, 500, x=100, y=120)
    full = predict_full(FakeModel(), img)
    tiled = predict_tiled(FakeModel(), img, tile=640)
    assert np.allclose(full[0], tiled[0])
