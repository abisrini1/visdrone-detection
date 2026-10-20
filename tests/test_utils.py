import numpy as np
import pytest

from src.utils import label_path_for, list_images, load_yolo_labels, visdrone_images_dir


def test_label_path_swaps_images_for_labels(tmp_path):
    img = tmp_path / "VisDrone" / "images" / "val" / "0001.jpg"
    assert label_path_for(img) == tmp_path / "VisDrone" / "labels" / "val" / "0001.txt"


def test_label_path_only_swaps_the_last_images_folder(tmp_path):
    img = tmp_path / "images" / "data" / "VisDrone" / "images" / "train" / "a.jpg"
    expected = tmp_path / "images" / "data" / "VisDrone" / "labels" / "train" / "a.txt"
    assert label_path_for(img) == expected


def test_label_path_requires_an_images_folder(tmp_path):
    with pytest.raises(ValueError):
        label_path_for(tmp_path / "pictures" / "a.jpg")


def test_images_dir_finds_the_ultralytics_layout(tmp_path):
    (tmp_path / "images" / "val").mkdir(parents=True)
    assert visdrone_images_dir("val", str(tmp_path)) == tmp_path / "images" / "val"


def test_images_dir_falls_back_to_the_zip_layout(tmp_path):
    (tmp_path / "VisDrone2019-DET-val" / "images").mkdir(parents=True)
    assert visdrone_images_dir("val", str(tmp_path)) == tmp_path / "VisDrone2019-DET-val" / "images"


def test_images_dir_reports_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError) as err:
        visdrone_images_dir("val", str(tmp_path))
    assert str(tmp_path / "images" / "val") in str(err.value)


def test_list_images_sorts_filters_and_limits(tmp_path):
    for name in ["b.jpg", "a.jpg", "notes.txt", "c.png"]:
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in list_images(tmp_path)] == ["a.jpg", "b.jpg", "c.png"]
    assert [p.name for p in list_images(tmp_path, limit=2)] == ["a.jpg", "b.jpg"]


def test_load_yolo_labels_converts_to_pixel_xyxy(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("3 0.5 0.5 0.2 0.4\n")  # one line: np.loadtxt must still return 2-D
    boxes, cls = load_yolo_labels(f, width=200, height=100)
    assert np.allclose(boxes, [[80, 30, 120, 70]])
    assert cls.tolist() == [3]


def test_load_yolo_labels_handles_missing_and_empty_files(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    for path in (empty, tmp_path / "missing.txt"):
        boxes, cls = load_yolo_labels(path, 100, 100)
        assert boxes.shape == (0, 4) and len(cls) == 0
