# Small-Object Detection on Drone Imagery (VisDrone)

Detecting pedestrians and vehicles in drone images, where objects are tiny and crowded, and taking the detector from training to a measured, optimized deployment.

The goal is not just a high mAP number. Each step answers a question with evidence:

1. How well does a fine-tuned YOLO do, and *where* does it fail? (baseline + per-class + by object size)
2. What actually helps small objects: resolution, model size, augmentation, or tiled inference?
3. How much speed can be gained from ONNX / TensorRT, and at what accuracy cost?
4. Can detections be tracked stably across frames (ByteTrack)?

> Fill in the tables below as you run experiments. Delete this note when they are complete.

## Results

### Training ablations (VisDrone val, 50 epochs unless noted)

| Experiment | Model | imgsz | mAP@50 | mAP@50-95 | Train time | Takeaway |
|---|---|---|---|---|---|---|
| s_640 (baseline) | YOLO11s | 640 | TBD | TBD | TBD | |
| s_960 | YOLO11s | 960 | TBD | TBD | TBD | |
| s_1280 | YOLO11s | 1280 | TBD | TBD | TBD | |
| n_640 | YOLO11n | 640 | TBD | TBD | TBD | |
| m_640 | YOLO11m | 640 | TBD | TBD | TBD | |
| s_640_no_mosaic | YOLO11s | 640 | TBD | TBD | TBD | |
| s_640_scale_0.9 | YOLO11s | 640 | TBD | TBD | TBD | |

Auto-filled in `results/results.csv`; per-class numbers in `results/per_class/`.

### Tiled inference (own implementation, same evaluator for every row)

| Mode | mAP@50 | mAP@50-95 | Recall small | Recall medium | Recall large | s / image |
|---|---|---|---|---|---|---|
| full image | TBD | TBD | TBD | TBD | TBD | TBD |
| tiled | TBD | TBD | TBD | TBD | TBD | TBD |
| tiled + full | TBD | TBD | TBD | TBD | TBD | TBD |

Size buckets use sqrt(box area) in pixels: small < 32, medium 32-96, large > 96. Auto-filled in `results/tiled.csv`.

### Deployment (batch 1, hardware: TBD GPU / TBD CPU)

| Format | Precision | mAP@50-95 | Latency mean (ms) | p95 (ms) | FPS | GPU mem (MiB) |
|---|---|---|---|---|---|---|
| PyTorch | FP32 | TBD | TBD | TBD | TBD | TBD |
| PyTorch | FP16 | TBD | TBD | TBD | TBD | TBD |
| ONNX | FP32 | TBD | TBD | TBD | TBD | TBD |
| TensorRT | FP16 | TBD | TBD | TBD | TBD | TBD |
| TensorRT | INT8 | TBD | TBD | TBD | TBD | TBD |

Auto-filled in `results/benchmark.csv`; run `python -m src.plot_pareto` for the accuracy-vs-latency chart.

### Tracking

| Video | Frames | Proc. FPS | Unique IDs | Mean track length | Tracks < 5 frames |
|---|---|---|---|---|---|
| TBD | | | | | |

### Failure analysis

TBD: add a few annotated images where the model fails (tiny objects, dense crowds, occlusion, motion blur) and what you learned.

## Repo layout

```
src/
  train.py          train one experiment, validate, log a row to results/results.csv
  run_ablations.py  run the experiment queue in configs/experiments.yaml
  evaluate.py       Ultralytics validation + per-class results
  tiling.py         tiled inference + NMS merge (from scratch)
  metrics_np.py     COCO-style mAP + recall by object size (from scratch, unit tested)
  eval_tiled.py     full vs tiled vs tiled+full under one evaluator
  export.py         ONNX / TensorRT (FP16, INT8) export
  benchmark.py      latency / FPS / GPU memory (+ accuracy with --val)
  track.py          detection + ByteTrack on video, with track statistics
  plot_pareto.py    accuracy vs latency plot
  report.py         results/*.csv -> markdown tables (results/REPORT.md)
configs/experiments.yaml   the experiment plan
scripts/check_env.py       verifies PyTorch sees your GPU
tests/                     unit tests for tiling and metrics
results/                   CSV tables, JSON, plots (committed)
```

## Setup (Windows, project on D:)

```bat
cd /d D:\Projects\visdrone-detection
py -3.11 -m venv .venv
.venv\Scripts\activate
```

**Install PyTorch with CUDA first.** Open https://pytorch.org/get-started/locally/, choose *Pip / Windows / CUDA*, and run the command it gives you. Then:

```bat
pip install -r requirements.txt
python scripts/check_env.py
```

`check_env.py` should report `CUDA yes` and your GPU. If it says NO, the PyTorch install is CPU-only: redo the previous step.

Optional: keep the ~2 GB dataset on D: too:

```bat
yolo settings datasets_dir=D:/datasets
```

VisDrone downloads automatically the first time you train or validate.

Run the tests (no GPU or dataset needed):

```bat
python -m pytest -q
```

## Reproduce the experiments

```bat
:: 1. Pipeline check (minutes). Confirms training, validation and logging all work.
python -m src.run_ablations --only smoke

:: 2. Baseline, then the ablation queue (hours; resume after interruptions with --skip-done)
python -m src.run_ablations --only s_640
python -m src.run_ablations --skip-done

:: 3. Tiled inference vs full image
python -m src.eval_tiled --weights runs/s_640/weights/best.pt --imgsz 640 --tile 640 --overlap 0.25
python -m src.eval_tiled --weights runs/s_640/weights/best.pt --imgsz 640 --merge-metric ios --tag ios_merge

:: 4. Export, then benchmark every format (with accuracy)
python -m src.export --weights runs/s_640/weights/best.pt --imgsz 640 --formats onnx trt-fp16 trt-int8
python -m src.benchmark --weights runs/s_640/weights/best.pt --imgsz 640 --val
python -m src.benchmark --weights runs/s_640/weights/best.pt --imgsz 640 --val --half
python -m src.benchmark --weights exports/s_640_640_trt-fp16.engine --imgsz 640 --val
:: latency of the tiled pipeline at deployment settings (conf 0.25), several repeats
python -m src.benchmark --weights runs/s_640/weights/best.pt --mode tiled+full --merge-metric ios --repeats 3
python -m src.plot_pareto

:: 5. Tracking on a drone video
python -m src.track --weights runs/s_640/weights/best.pt --source path\to\drone.mp4 --imgsz 960

:: 6. Build the README tables from every results CSV
python -m src.report

:: 7. Final numbers: choose settings on val, then evaluate ONCE on the held-out test-dev split
python -m src.eval_tiled --weights runs/s_640/weights/best.pt --split test --tag final
```

Any extra training option can be passed straight through, e.g. `python -m src.train --name my_run --epochs 30 --mosaic 0.0`.

### Hardware notes

Developed on an 8 GB RTX 5060 laptop GPU with 16 GB RAM using mixed precision. Batch sizes in `configs/experiments.yaml` are chosen for that; 1280 px training needs a tiny batch. Ultralytics keeps the effective batch at 64 through gradient accumulation, so per-step batch size differences between experiments do not change the optimization. If Windows data loading hangs, set `workers: 0`.

VisDrone images contain up to ~900 objects. Ultralytics raises `max_det` automatically for its own training and validation; this repo's evaluation scripts (`eval_tiled`) default to `--max-det 1000` for the same reason (the usual 300 would cap recall in crowded scenes).

TensorRT needs the NVIDIA TensorRT package, which can be fiddly on Windows. If `trt-*` exports fail, ONNX and PyTorch FP16 still give a useful comparison, and WSL2 is an option for TensorRT.

## Honest limitations

TBD: write what you could not test, what you would do with more compute, and any difference between your numbers and published VisDrone results.

## Data and licensing

VisDrone2019-DET (Zhu et al., AISKYEYE team, Tianjin University). The dataset is for academic and non-commercial research; check its terms before reusing it elsewhere. Models use the Ultralytics library, which has its own license (AGPL-3.0 / enterprise): read it before deploying commercially.
