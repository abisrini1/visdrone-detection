"""Plot accuracy vs latency from results/benchmark.csv (rows benchmarked with --val).

    python -m src.plot_pareto
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import RESULTS_DIR


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(RESULTS_DIR / "benchmark.csv"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "pareto.png"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "mAP50_95" not in df:
        raise SystemExit("No accuracy column yet: run src.benchmark with --val first.")
    df = df.dropna(subset=["mAP50_95"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fmt, grp in df.groupby("format"):
        ax.scatter(grp["latency_ms_mean"], grp["mAP50_95"], s=60, label=fmt)
    for _, r in df.iterrows():
        ax.annotate(f"{Path(r['weights']).stem}\n{r['precision']}", (r["latency_ms_mean"], r["mAP50_95"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("latency per image (ms, batch 1, lower is better)")
    ax.set_ylabel("mAP@50-95 (higher is better)")
    ax.set_title("Accuracy vs latency")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print("saved", args.out)


if __name__ == "__main__":
    main()
