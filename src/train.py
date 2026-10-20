"""Train one experiment, validate the best checkpoint, log a row to results/results.csv.

    python -m src.train --name s_640 --model yolo11s.pt --imgsz 640 --epochs 50 --batch 16
    python -m src.train --name s_640_noMosaic --mosaic 0.0 --epochs 50     # any extra train arg works
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from src.evaluate import evaluate_weights
from src.utils import RESULTS_DIR, append_csv, parse_device


def run_experiment(name, model="yolo11s.pt", imgsz=640, epochs=50, batch=8, data="VisDrone.yaml",
                   device=0, workers=4, seed=0, project="runs", notes="", **extra) -> dict:
    from ultralytics import YOLO

    project = str(Path(project).resolve())
    t0 = time.time()
    yolo = YOLO(model)
    yolo.train(data=data, imgsz=imgsz, epochs=epochs, batch=batch, device=device, workers=workers,
               seed=seed, project=project, name=name, exist_ok=True, plots=True, **extra)
    train_minutes = (time.time() - t0) / 60

    best = Path(project) / name / "weights" / "best.pt"
    if not best.exists():
        best = Path(yolo.trainer.best)
    res = evaluate_weights(best, data=data, imgsz=imgsz, device=device, batch=batch)

    per_class = res.pop("per_class_mAP50_95")
    per_class_dir = RESULTS_DIR / "per_class"
    per_class_dir.mkdir(parents=True, exist_ok=True)
    (per_class_dir / f"{name}.json").write_text(json.dumps(per_class, indent=2))

    row = {"name": name, "model": model, "imgsz": imgsz, "epochs": epochs, "batch": batch,
           "seed": seed, "train_minutes": round(train_minutes, 1), "notes": notes,
           **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()},
           **{f"arg_{k}": v for k, v in extra.items()}}
    append_csv(RESULTS_DIR / "results.csv", row)
    print(f"[{name}] mAP50 {res['mAP50']:.4f}  mAP50-95 {res['mAP50_95']:.4f}  ({train_minutes:.1f} min)")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--data", default="VisDrone.yaml")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--notes", default="")
    args, unknown = ap.parse_known_args()

    # Extra "--key value" pairs are forwarded to Ultralytics (e.g. --mosaic 0.0 --fraction 0.1).
    extra = {}
    it = iter(unknown)
    for tok in it:
        if not tok.startswith("--"):
            raise SystemExit(f"Unexpected argument: {tok}")
        key = tok[2:].replace("-", "_")
        extra[key] = yaml.safe_load(next(it))

    run_experiment(args.name, args.model, args.imgsz, args.epochs, args.batch, args.data,
                   parse_device(args.device), args.workers, args.seed, args.project, args.notes, **extra)


if __name__ == "__main__":
    main()
