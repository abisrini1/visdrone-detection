"""Turn the results CSVs into markdown tables for the README.

    python -m src.report                 # prints the tables and writes results/REPORT.md

Reads results/results.csv (training), results/tiled.csv (tiled inference) and
results/benchmark.csv (deployment). Smoke-test rows are hidden, and if you re-ran the
same configuration only the latest row is kept.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import RESULTS_DIR


def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def md_table(df: pd.DataFrame, columns) -> str:
    """columns: list of (header, dataframe column, decimals). Missing columns are skipped."""
    cols = [(h, c, d) for h, c, d in columns if c in df.columns]
    if df.empty or not cols:
        return "_no rows yet_"
    lines = ["| " + " | ".join(h for h, _, _ in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], d) for _, c, d in cols) + " |")
    return "\n".join(lines)


def _read(path: Path):
    return pd.read_csv(path) if Path(path).exists() else None


def training_table(path: Path) -> str:
    df = _read(path)
    if df is None:
        return "_no rows yet_"
    df = df[df["name"] != "smoke"].drop_duplicates(subset=["name"], keep="last")
    return md_table(df, [("Experiment", "name", 0), ("Model", "model", 0), ("imgsz", "imgsz", 0),
                         ("Epochs", "epochs", 0), ("Batch", "batch", 0), ("mAP@50", "mAP50", 3),
                         ("mAP@50-95", "mAP50_95", 3), ("Precision", "precision", 3),
                         ("Recall", "recall", 3), ("Params (M)", "params_M", 1),
                         ("Train (min)", "train_minutes", 1)])


def tiled_table(path: Path) -> str:
    df = _read(path)
    if df is None:
        return "_no rows yet_"
    key = [c for c in ["weights", "tag", "split", "mode", "imgsz", "tile", "overlap",
                       "merge_metric", "merge_thr", "images"] if c in df.columns]
    df = df.drop_duplicates(subset=key, keep="last").copy()
    if "merge_metric" in df.columns and "merge_thr" in df.columns:
        df["merge"] = df["merge_metric"].astype(str) + "@" + df["merge_thr"].astype(str)
    for col in ("tag", "split"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return md_table(df, [("Tag", "tag", 0), ("Split", "split", 0), ("Mode", "mode", 0),
                         ("Overlap", "overlap", 2), ("Merge", "merge", 0), ("Images", "images", 0),
                         ("mAP@50", "mAP50", 3), ("mAP@50-95", "mAP50_95", 3),
                         ("Recall small", "recall_small", 3), ("Recall medium", "recall_medium", 3),
                         ("Recall large", "recall_large", 3), ("s / image", "sec_per_image", 3)])


def deployment_table(path: Path) -> str:
    df = _read(path)
    if df is None:
        return "_no rows yet_"
    key = [c for c in ["weights", "imgsz", "device", "precision", "mode", "tile", "overlap",
                       "merge_metric", "merge_thr", "conf"] if c in df.columns]
    df = df.drop_duplicates(subset=key, keep="last").copy()
    if "weights" in df.columns:
        df["weights"] = df["weights"].astype(str).map(lambda p: Path(p).name)
    if "mode" in df.columns:
        df["mode"] = df["mode"].fillna("full")  # rows logged before tiled modes existed
    if "merge_metric" in df.columns and "merge_thr" in df.columns:
        df["merge"] = [f"{m}@{t}" if isinstance(m, str) else "" for m, t in zip(df["merge_metric"], df["merge_thr"])]
    return md_table(df, [("Weights", "weights", 0), ("Format", "format", 0), ("Precision", "precision", 0),
                         ("Mode", "mode", 0), ("Merge", "merge", 0), ("Device", "device", 0),
                         ("mAP@50-95", "mAP50_95", 3), ("Latency mean (ms)", "latency_ms_mean", 1),
                         ("p50 (ms)", "latency_ms_p50", 1), ("p95 (ms)", "latency_ms_p95", 1),
                         ("Std (ms)", "latency_ms_std", 1), ("FPS", "fps", 1),
                         ("GPU mem (MiB)", "gpu_mem_used_mib", 0)])


def build_report(results_dir: Path = RESULTS_DIR) -> str:
    results_dir = Path(results_dir)
    return "\n\n".join([
        "## Training ablations\n\n" + training_table(results_dir / "results.csv"),
        "## Tiled inference\n\n" + tiled_table(results_dir / "tiled.csv"),
        "## Deployment\n\n" + deployment_table(results_dir / "benchmark.csv"),
    ]) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    report = build_report(Path(args.results_dir))
    out = Path(args.results_dir) / "REPORT.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print("saved", out)


if __name__ == "__main__":
    main()
