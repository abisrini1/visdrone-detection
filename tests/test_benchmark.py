import numpy as np
import pandas as pd
import pytest

from src.benchmark import infer_labels, summarize, time_calls
from src.report import build_report


def test_time_calls_runs_warmup_then_timed_passes():
    calls = []
    wall = time_calls(lambda img: calls.append(img), images=["a", "b", "c"], warmup=4, repeats=2)
    assert len(calls) == 4 + 2 * 3          # warm-up calls are made but not timed
    assert len(wall) == 2 * 3
    assert calls[:4] == ["a", "b", "c", "a"]  # warm-up cycles through the images
    assert (wall >= 0).all()


def test_time_calls_rejects_empty_input():
    with pytest.raises(ValueError):
        time_calls(lambda img: None, images=[])


def test_summarize_statistics():
    s = summarize(np.array([10.0, 20.0, 30.0, 40.0]))
    assert s["latency_ms_mean"] == 25.0
    assert s["latency_ms_p50"] == 25.0
    assert s["fps"] == 40.0
    assert s["latency_ms_std"] > 0


def test_infer_labels():
    assert infer_labels("a/best.pt", half=False) == ("pytorch", "fp32")
    assert infer_labels("a/best.pt", half=True) == ("pytorch", "fp16")
    assert infer_labels("exports/s_640_trt-int8.engine", half=False) == ("tensorrt", "int8")
    assert infer_labels("exports/s_640_onnx.onnx", half=False) == ("onnx", "fp32")


def test_deployment_report_keeps_full_and_tiled_rows_apart(tmp_path):
    base = {"weights": "runs/s_640/weights/best.pt", "format": "pytorch", "precision": "fp32",
            "imgsz": 640, "device": "0", "latency_ms_mean": 12.0, "latency_ms_p95": 15.0, "fps": 83.3}
    rows = [
        {**base},  # logged before the mode column existed
        {**base, "mode": "tiled+full", "conf": 0.25, "tile": 640, "overlap": 0.25,
         "merge_metric": "ios", "merge_thr": 0.6, "latency_ms_mean": 60.0},
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "benchmark.csv", index=False)
    report = build_report(tmp_path)
    assert "| full |" in report and "tiled+full" in report and "ios@0.6" in report
