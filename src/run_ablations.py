"""Run every experiment listed in configs/experiments.yaml, one after another.

    python -m src.run_ablations --only smoke          # tiny end-to-end check first
    python -m src.run_ablations --dry-run             # list what would run
    python -m src.run_ablations --skip-done           # resume a long queue after an interruption
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.train import run_experiment
from src.utils import parse_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--only", nargs="*", help="run only these experiment names")
    ap.add_argument("--skip-done", action="store_true", help="skip experiments that already have best.pt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    defaults = cfg.get("defaults", {})
    queue = []
    for exp in cfg["experiments"]:
        params = {**defaults, **exp}
        if args.only and params["name"] not in args.only:
            continue
        best = Path(params.get("project", "runs")) / params["name"] / "weights" / "best.pt"
        if args.skip_done and best.exists():
            print(f"skip {params['name']} (already trained)")
            continue
        queue.append(params)

    for i, p in enumerate(queue, 1):
        shown = {k: v for k, v in p.items() if k != "name"}
        print(f"[{i}/{len(queue)}] {p['name']}: {shown}")
    if args.dry_run:
        return

    for p in queue:
        p = dict(p)
        p["device"] = parse_device(p.get("device", 0))
        run_experiment(**p)


if __name__ == "__main__":
    main()
