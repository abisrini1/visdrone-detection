import numpy as np

from src.make_gif import frame_indices
from src.show_examples import (build_example, crop_boxes, densest_window, found_mask,
                               select_images)


def test_densest_window_finds_the_cluster():
    cluster = np.array([[1500 + i * 5, 900 + i * 5, 1510 + i * 5, 910 + i * 5] for i in range(20)], float)
    lone = np.array([[50, 50, 60, 60]], float)
    x0, y0 = densest_window(np.vstack([lone, cluster]), 1500, 2000, size=640)
    assert x0 <= 1500 < x0 + 640 and y0 <= 900 < y0 + 640


def test_densest_window_small_image_and_empty():
    assert densest_window(np.zeros((0, 4)), 300, 400, 640) == (0, 0)
    x0, y0 = densest_window(np.array([[10, 10, 20, 20.]]), 300, 400, 640)
    assert (x0, y0) == (0, 0)


def test_crop_boxes_shifts_and_filters():
    boxes = np.array([[110, 110, 130, 130], [900, 900, 920, 920.]])
    out, cls = crop_boxes(boxes, [3, 4], 100, 100, 640, 640)
    assert out.shape == (1, 4) and cls.tolist() == [3]
    assert out[0].tolist() == [10, 10, 30, 30]


def test_found_mask_needs_same_class_and_iou():
    gt = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [100, 100, 110, 110.]])
    pred = np.array([[0, 0, 10, 10], [50, 50, 60, 60.]])
    mask = found_mask(gt, [0, 1, 2], pred, [0, 7])  # second prediction has the wrong class
    assert mask.tolist() == [True, False, False]
    assert found_mask(gt, [0, 1, 2], np.zeros((0, 4)), []).tolist() == [False, False, False]


def test_select_images_is_deterministic_and_unique():
    counts = {f"img{i}.jpg": i for i in range(100)}
    a = select_images(counts, 4, "crowded")
    assert a == select_images(counts, 4, "crowded")
    assert len(a) == len(set(a)) == 4
    assert all(counts[n] >= 60 for n in a)
    assert select_images(counts, 3, "random", seed=1) == select_images(counts, 3, "random", seed=1)
    assert len(select_images({"a.jpg": 1}, 4)) == 1


def test_build_example_has_three_panels():
    image = np.full((800, 1000, 3), 120, dtype=np.uint8)
    gt = np.array([[100, 100, 130, 130], [200, 200, 230, 230.]])
    full = (gt[:1], np.array([0.9]), np.array([0]))
    tiled = (gt, np.array([0.9, 0.8]), np.array([0, 1]))
    sheet, (x0, y0) = build_example(image, gt, np.array([0, 1]), full, tiled, window=640)
    assert sheet.shape[1] == 3 * 640 and sheet.shape[0] > 640
    assert (x0, y0) == (0, 0)


def test_frame_indices():
    idx = frame_indices(total_frames=300, src_fps=30, start_s=0, seconds=2, out_fps=10)
    assert idx[0] == 0 and len(idx) == 20 and idx == sorted(idx)
    assert frame_indices(0, 30, 0, 2, 10) == []
    assert max(frame_indices(100, 30, 2, 5, 10)) < 100


def test_draw_tracks_marks_boxes_and_keeps_colours_stable():
    from src.track import draw_tracks, id_color

    frame = np.full((200, 300, 3), 100, dtype=np.uint8)
    out = draw_tracks(frame, np.array([[20, 40, 80, 100.], [150, 60, 200, 120.]]), [3, 4])
    assert out.shape == frame.shape and (out != frame).any()
    assert (frame == 100).all()  # input is not modified
    assert id_color(3) == id_color(3) and id_color(3) != id_color(4)
    assert (draw_tracks(frame, np.zeros((0, 4)), []) == frame).all()


def test_to_web_mp4_command():
    from src.to_web_mp4 import build_command

    cmd = build_command("ffmpeg", "in.mp4", "out.mp4", 1280, 27)
    assert cmd[0] == "ffmpeg" and "libx264" in cmd and "yuv420p" in cmd
    assert "scale=1280:-2" in cmd and cmd[-1] == "out.mp4"
