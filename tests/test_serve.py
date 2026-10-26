"""Service tests with a fake model: no GPU, weights or dataset needed."""
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
cv2 = pytest.importorskip("cv2")

from fastapi.testclient import TestClient  # noqa: E402

from serve.app import Settings, create_app  # noqa: E402


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
    """'Detects' the bright square (red channel == 255) in whatever crop it is given."""

    names = {3: "car"}

    def predict(self, source, **kwargs):
        crops = source if isinstance(source, list) else [source]
        out = []
        for crop in crops:
            ys, xs = np.nonzero(crop[..., 2] == 255)  # cv2 decodes to BGR, so red is channel 2
            if len(xs) == 0:
                out.append(_Result(_Boxes(np.zeros((0, 4)), np.zeros(0), np.zeros(0))))
            else:
                out.append(_Result(_Boxes([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], [0.9], [3])))
        return out


def _png(h=800, w=1000, x=300, y=200, size=24):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[y:y + size, x:x + size, 2] = 255
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def client():
    app = create_app(model=FakeModel(), settings=Settings(model_path="fake.pt"), load_from_env=False)
    with TestClient(app) as c:
        yield c


def _post(client, data, **params):
    return client.post("/predict", params=params, files={"file": ("img.png", data, "image/png")})


def test_health_ok_when_model_is_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_health_is_503_without_a_model():
    app = create_app(model=None, settings=Settings(), load_from_env=False)
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503


@pytest.mark.parametrize("mode", ["full", "tiled", "tiled_full"])
def test_predict_finds_the_object_in_every_mode(client, mode):
    r = _post(client, _png(), mode=mode)
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == mode and body["num_detections"] == 1
    det = body["detections"][0]
    assert det["label"] == "car" and det["class_id"] == 3
    assert det["box"] == pytest.approx([300, 200, 324, 224])
    assert body["timings_ms"]["total_ms"] >= body["timings_ms"]["inference_ms"]
    assert (body["image_width"], body["image_height"]) == (1000, 800)


def test_non_image_upload_is_a_400(client):
    assert _post(client, b"this is not an image").status_code == 400


def test_unknown_mode_is_a_422(client):
    assert _post(client, _png(), mode="tiled+full").status_code == 422


def test_oversized_upload_is_a_413():
    app = create_app(model=FakeModel(), settings=Settings(max_upload_mb=0), load_from_env=False)
    with TestClient(app) as c:
        assert _post(c, _png()).status_code == 413
