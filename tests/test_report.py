import pandas as pd

from src.report import build_report, md_table


def test_md_table_formats_numbers_and_skips_missing_columns():
    df = pd.DataFrame([{"a": "x", "b": 0.123456}])
    out = md_table(df, [("A", "a", 0), ("B", "b", 3), ("C", "missing", 0)])
    assert out.splitlines()[0] == "| A | B |"
    assert out.splitlines()[2] == "| x | 0.123 |"


def test_empty_inputs_say_no_rows(tmp_path):
    report = build_report(tmp_path)
    assert report.count("_no rows yet_") == 3


def test_training_table_hides_smoke_and_keeps_latest_rerun(tmp_path):
    pd.DataFrame([
        {"name": "smoke", "model": "n", "mAP50": 0.07, "mAP50_95": 0.03},
        {"name": "s_640", "model": "s", "mAP50": 0.30, "mAP50_95": 0.20},
        {"name": "s_640", "model": "s", "mAP50": 0.386, "mAP50_95": 0.224},
    ]).to_csv(tmp_path / "results.csv", index=False)
    report = build_report(tmp_path)
    assert "smoke" not in report
    assert "0.386" in report and "0.300" not in report


def test_tiled_table_dedupes_and_labels_the_merge(tmp_path):
    base = {"weights": "w.pt", "tag": None, "split": "val", "mode": "tiled", "imgsz": 640, "tile": 640,
            "overlap": 0.25, "merge_metric": "iou", "merge_thr": 0.6, "images": 548,
            "mAP50": 0.404, "mAP50_95": 0.244, "recall_small": 0.55, "sec_per_image": 0.055}
    rows = [base, dict(base), {**base, "merge_metric": "ios", "tag": "ios_merge", "mAP50": 0.435}]
    pd.DataFrame(rows).to_csv(tmp_path / "tiled.csv", index=False)
    report = build_report(tmp_path)
    assert report.count("iou@0.6") == 1      # the duplicate run collapsed to one row
    assert "ios@0.6" in report and "0.435" in report
